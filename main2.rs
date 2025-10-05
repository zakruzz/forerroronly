use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app::AppSink;
use image::{imageops, ImageBuffer, Rgb};
use serde::Deserialize;
use std::{fs, path::PathBuf, time::Instant};

// Tract ONNX (CPU) — simple & portable
use tract_ndarray::Array4;
use tract_onnx::prelude::*;

#[derive(Parser, Debug)]
#[command(name="jetson_yolo_count")]
struct Args {
    /// Path ONNX (hasil export/simplify)
    #[arg(long)]
    onnx: PathBuf,

    /// Path ke classes.json (array string)
    #[arg(long)]
    classes: PathBuf,

    /// Device kamera V4L2
    #[arg(long, default_value="/dev/video0")]
    device: String,

    /// Ukuran input model (persegi)
    #[arg(long, default_value_t=640)]
    imgsz: u32,

    /// Confidence threshold
    #[arg(long, default_value_t=0.25)]
    conf: f32,

    /// IoU threshold NMS
    #[arg(long, default_value_t=0.45)]
    iou: f32,

    /// Filter hanya kelas kendaraan
    #[arg(long, default_value_t=true)]
    filter_vehicle: bool,

    /// Tampilkan FPS dan info output
    #[arg(long, default_value_t=false)]
    verbose: bool,
}

#[derive(Deserialize, Debug)]
struct ClassList(Vec<String>);

#[derive(Clone, Debug)]
struct Detection {
    x1: f32, y1: f32, x2: f32, y2: f32,
    score: f32,
    class_id: usize,
}

fn is_vehicle(name: &str) -> bool {
    let keys = [
        "car","bus","truck","motorcycle","motorbike","bicycle","bike",
        "van","pickup","trailer","truk","mobil","sepeda","motor"
    ];
    let n = name.to_lowercase();
    keys.iter().any(|k| n.contains(k))
}

fn letterbox_rgb(
    rgb: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    dst_size: u32,
) -> (ImageBuffer<Rgb<u8>, Vec<u8>>, f32, u32, u32) {
    let (w, h) = (rgb.width() as f32, rgb.height() as f32);
    let s = (dst_size as f32 / w).min(dst_size as f32 / h);
    let nw = (w * s).round() as u32;
    let nh = (h * s).round() as u32;

    let resized = imageops::resize(rgb, nw, nh, imageops::FilterType::Triangle);
    let mut canvas = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_pixel(
        dst_size, dst_size, Rgb([114, 114, 114])
    );
    let dx = (dst_size - nw) / 2;
    let dy = (dst_size - nh) / 2;
    imageops::replace(&mut canvas, &resized, dx.into(), dy.into());
    (canvas, s, dx, dy)
}

fn chw_normalize(inp: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> Vec<f32> {
    // RGB HWC -> CHW f32 0..1
    let (w, h) = (inp.width() as usize, inp.height() as usize);
    let mut out = vec![0f32; 3 * w * h];
    for y in 0..h {
        for x in 0..w {
            let p = inp.get_pixel(x as u32, y as u32);
            let idx = y * w + x;
            out[idx] = p[0] as f32 / 255.0;         // R
            out[w * h + idx] = p[1] as f32 / 255.0; // G
            out[2 * w * h + idx] = p[2] as f32 / 255.0; // B
        }
    }
    out
}

fn iou(a: &Detection, b: &Detection) -> f32 {
    let (x1, y1) = (a.x1.max(b.x1), a.y1.max(b.y1));
    let (x2, y2) = (a.x2.min(b.x2), a.y2.min(b.y2));
    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area_a = (a.x2 - a.x1).max(0.0) * (a.y2 - a.y1).max(0.0);
    let area_b = (b.x2 - b.x1).max(0.0) * (b.y2 - b.y1).max(0.0);
    inter / (area_a + area_b - inter + 1e-6)
}

fn nms(mut dets: Vec<Detection>, iou_thresh: f32) -> Vec<Detection> {
    dets.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    let mut keep: Vec<Detection> = Vec::new();
    'outer: for d in dets.into_iter() {
        for k in &keep {
            if d.class_id == k.class_id && iou(&d, k) > iou_thresh {
                continue 'outer;
            }
        }
        keep.push(d);
    }
    keep
}

/// Decoder multi-format:
/// - (1, 5+nc, N)  => YOLOv5 (x,y,w,h,obj,cls...)
/// - (1, 4+nc, N)  => YOLOv8 (x,y,w,h,cls...)      score = max(cls_prob)
/// - (1, N, 5+nc)  / (1, N, 4+nc) => transpose varian
fn decode_yolo_dynamic(
    out: &[f32],
    shape: &[usize],          // [1, C, N] atau [1, N, C]
    conf_t: f32,
    class_names: &[String],
    img_w: u32, img_h: u32,
    letter_scale: f32,
    pad_x: u32, pad_y: u32,
    dst_size: u32,
    filter_vehicle: bool,
) -> Result<Vec<Detection>> {
    if shape.len() != 3 || shape[0] != 1 {
        bail!("shape output tidak didukung: {:?}", shape);
    }
    let (a, b) = (shape[1], shape[2]);

    let mut dets = Vec::new();

    let mut push_det = |mut cx: f32, mut cy: f32, mut w: f32, mut h: f32, score: f32, class_id: usize| {
        cx *= dst_size as f32;
        cy *= dst_size as f32;
        w  *= dst_size as f32;
        h  *= dst_size as f32;
        let (mut x1, mut y1, mut x2, mut y2) = (cx - w/2.0, cy - h/2.0, cx + w/2.0, cy + h/2.0);

        let fx  = (x1 - pad_x as f32).max(0.0);
        let fy  = (y1 - pad_y as f32).max(0.0);
        let fx2 = (x2 - pad_x as f32).max(0.0);
        let fy2 = (y2 - pad_y as f32).max(0.0);
        let inv = 1.0 / letter_scale;

        x1 = (fx  * inv).clamp(0.0, img_w as f32 - 1.0);
        y1 = (fy  * inv).clamp(0.0, img_h as f32 - 1.0);
        x2 = (fx2 * inv).clamp(0.0, img_w as f32 - 1.0);
        y2 = (fy2 * inv).clamp(0.0, img_h as f32 - 1.0);

        dets.push(Detection { x1, y1, x2, y2, score, class_id });
    };

    // ---- Case 1: (1, C, N)
    if a >= 4 {
        let n = b;
        let c = a;
        let is_v5 = c >= 5 && (c - 5) <= class_names.len();
        let is_v8 = c >= 4 && (c - 4) <= class_names.len();

        if is_v5 || is_v8 {
            let nc = if is_v5 { c - 5 } else { c - 4 };
            let stride = n;
            for i in 0..n {
                let bx = out[0 * stride + i];
                let by = out[1 * stride + i];
                let bw = out[2 * stride + i];
                let bh = out[3 * stride + i];

                let (score, best_c) = if is_v5 {
                    let obj = out[4 * stride + i];
                    let (mut best_c, mut best_p) = (0usize, 0f32);
                    for cc in 0..nc {
                        let p = out[(5 + cc) * stride + i];
                        if p > best_p { best_p = p; best_c = cc; }
                    }
                    (obj * best_p, best_c)
                } else {
                    let (mut best_c, mut best_p) = (0usize, 0f32);
                    for cc in 0..nc {
                        let p = out[(4 + cc) * stride + i];
                        if p > best_p { best_p = p; best_c = cc; }
                    }
                    (best_p, best_c)
                };

                if score < conf_t { continue; }
                if filter_vehicle {
                    let cname = class_names.get(best_c).map(|s| s.as_str()).unwrap_or("");
                    if !is_vehicle(cname) { continue; }
                }
                push_det(bx, by, bw, bh, score, best_c);
            }
            return Ok(dets);
        }
    }

    // ---- Case 2: (1, N, C)
    if b >= 4 {
        let n = a;
        let c = b;
        let is_v5 = c >= 5 && (c - 5) <= class_names.len();
        let is_v8 = c >= 4 && (c - 4) <= class_names.len();

        if is_v5 || is_v8 {
            let nc = if is_v5 { c - 5 } else { c - 4 };
            for i in 0..n {
                let base = i * c;
                let bx = out[base + 0];
                let by = out[base + 1];
                let bw = out[base + 2];
                let bh = out[base + 3];

                let (score, best_c) = if is_v5 {
                    let obj = out[base + 4];
                    let (mut best_c, mut best_p) = (0usize, 0f32);
                    for cc in 0..nc {
                        let p = out[base + 5 + cc];
                        if p > best_p { best_p = p; best_c = cc; }
                    }
                    (obj * best_p, best_c)
                } else {
                    let (mut best_c, mut best_p) = (0usize, 0f32);
                    for cc in 0..nc {
                        let p = out[base + 4 + cc];
                        if p > best_p { best_p = p; best_c = cc; }
                    }
                    (best_p, best_c)
                };

                if score < conf_t { continue; }
                if filter_vehicle {
                    let cname = class_names.get(best_c).map(|s| s.as_str()).unwrap_or("");
                    if !is_vehicle(cname) { continue; }
                }
                push_det(bx, by, bw, bh, score, best_c);
            }
            return Ok(dets);
        }
    }

    bail!("format output YOLO tidak dikenali. shape={shape:?}");
}

fn build_pipeline(device: &str) -> Result<(gst::Pipeline, AppSink)> {
    gst::init()?;

    // Pipeline simpel: BGR 1280x720 → appsink
    // Jika kamera MJPEG, bisa ganti:
    // v4l2src device=/dev/video0 ! image/jpeg,framerate=30/1 ! jpegdec ! videoconvert ! ...
    let pipeline_str = format!(
        "v4l2src device={} !
         videoconvert ! video/x-raw,format=BGR !
         videoscale ! video/x-raw,width=1280,height=720 !
         queue max-size-buffers=1 leaky=downstream !
         appsink name=sink emit-signals=false sync=false max-buffers=1 drop=true",
        device
    );

    let pipeline = gst::parse::launch(&pipeline_str)
        .with_context(|| format!("gagal buat pipeline: {pipeline_str}"))?
        .downcast::<gst::Pipeline>()
        .map_err(|_| anyhow!("bukan pipeline"))?;

    let appsink = pipeline
        .by_name("sink")
        .ok_or_else(|| anyhow!("appsink tidak ditemukan"))?
        .downcast::<AppSink>()
        .map_err(|_| anyhow!("appsink downcast gagal"))?;

    Ok((pipeline, appsink))
}

fn main() -> Result<()> {
    let args = Args::parse();

    // 1) load classes
    let classes: ClassList = serde_json::from_slice(
        &fs::read(&args.classes).context("gagal baca classes.json")?
    ).context("classes.json bukan array string?")?;
    if classes.0.is_empty() {
        bail!("classes.json kosong");
    }

    // 2) siapkan GStreamer
    let (pipeline, appsink) = build_pipeline(&args.device)?;
    pipeline.set_state(gst::State::Playing)?;

    // 3) siapkan ONNX (Tract)
    let mut model = tract_onnx::onnx()
        .model_for_path(&args.onnx)
        .with_context(|| format!("gagal load onnx: {:?}", &args.onnx))?;

    model.set_input_fact(
        0,
        InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 3, args.imgsz as usize, args.imgsz as usize))
    )?;

    let model = model
        .into_optimized()?
        .into_runnable()?;

    // 4) loop ambil frame → preprocess → infer → postprocess
    let mut last = Instant::now();
    let mut frame_idx: u64 = 0;

    loop {
        // AppSink::try_pull_sample butuh Option<ClockTime>
        let sample = match appsink.try_pull_sample(Some(gst::ClockTime::from_mseconds(500))) {
            Some(s) => s,
            None => {
                eprintln!("timeout ambil frame…");
                continue;
            }
        };

        let buffer = match sample.buffer() {
            Some(b) => b,
            None => { eprintln!("buffer kosong"); continue; }
        };

        let caps = match sample.caps() {
            Some(c) => c,
            None => { eprintln!("caps kosong"); continue; }
        };

        let s = match caps.structure(0) {
            Some(st) => st,
            None => { eprintln!("caps.structure(0) kosong"); continue; }
        };

        let w: u32 = s.get::<i32>("width").unwrap_or(1280) as u32;
        let h: u32 = s.get::<i32>("height").unwrap_or(720) as u32;

        let map = match buffer.map_readable() {
            Ok(m) => m,
            Err(_) => { eprintln!("map buffer gagal"); continue; }
        };
        let bgr = map.as_slice();

        if bgr.len() != (w * h * 3) as usize {
            eprintln!("ukuran buffer tidak cocok ({} vs {})", bgr.len(), w*h*3);
            continue;
        }

        // BGR -> RGB
        let mut rgb_data = vec![0u8; (w*h*3) as usize];
        for i in 0..(w*h) as usize {
            rgb_data[3*i]   = bgr[3*i + 2];
            rgb_data[3*i+1] = bgr[3*i + 1];
            rgb_data[3*i+2] = bgr[3*i + 0];
        }
        let rgb_img: ImageBuffer<Rgb<u8>, _> = match ImageBuffer::from_raw(w, h, rgb_data) {
            Some(im) => im,
            None => { eprintln!("buat RGB image gagal"); continue; }
        };

        // Letterbox ke imgsz
        let (letter, scale, pad_x, pad_y) = letterbox_rgb(&rgb_img, args.imgsz);
        let input_chw = chw_normalize(&letter);

        // Bentuk tensor [1,3,H,W]
        let arr: Array4<f32> = Array4::from_shape_vec(
            (1, 3, args.imgsz as usize, args.imgsz as usize),
            input_chw
        ).context("shape input mismatch")?;
        let input_t: TValue = Tensor::from(arr).into();

        // Inference
        let outputs = model.run(tvec!(input_t))?;
        let out = outputs[0].to_array_view::<f32>()?;
        let shape = out.shape().to_vec(); // contoh [1,84,8400] atau [1,8400,84]
        let flat: Vec<f32> = out.to_owned().into_raw_vec();

        if args.verbose && frame_idx % 60 == 0 {
            let (mut mn, mut mx) = (f32::INFINITY, f32::NEG_INFINITY);
            for &v in &flat { if v < mn { mn = v } if v > mx { mx = v } }
            eprintln!("OUTPUT SHAPE={:?}  RANGE=[{:.4}, {:.4}]", shape, mn, mx);
        }

        // Decode sesuai shape
        let dets_raw = match decode_yolo_dynamic(
            &flat, &shape, args.conf, &classes.0,
            w, h, scale, pad_x, pad_y, args.imgsz,
            args.filter_vehicle,
        ) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("decode gagal: {e}");
                continue;
            }
        };
        let dets = nms(dets_raw, args.iou);
        let count = dets.len();

        frame_idx += 1;
        if args.verbose {
            let now = Instant::now();
            let dt = now.duration_since(last).as_secs_f32();
            last = now;
            let fps = if dt > 0.0 { 1.0/dt } else { 0.0 };
            println!("[frame {frame_idx}] vehicles={count}  fps={fps:.1}");
        } else {
            println!("{count}");
        }
    }
}
