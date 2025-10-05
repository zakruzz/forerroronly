use anyhow::{Context, Result};
use async_cuda::{DeviceBuffer, HostBuffer, Stream};
use async_tensorrt::engine::{Engine, ExecutionContext, TensorIoMode};
use async_tensorrt::runtime::Runtime;
use opencv::{
    core::{self, Mat, MatTraitConst, Point, Rect, Scalar, Size, Vec3b},
    highgui,
    imgproc,
    videoio::{self, VideoCaptureTrait, VideoCaptureTraitConst},
};
use std::{collections::HashMap, fs, time::Instant};

// ===== Konfigurasi cepat =====
const INPUT_W: i32 = 640;
const INPUT_H: i32 = 640;
const CONF_THR: f32 = 0.25;
const IOU_THR: f32 = 0.45;
const DRAW_BOXES: bool = true;

// ===== Utils =====
fn load_classes(path: &str) -> Result<Vec<String>> {
    let txt = fs::read_to_string(path).context("read classes.json")?;
    let v: Vec<String> = serde_json::from_str(&txt).context("parse classes.json")?;
    Ok(v)
}

// Letterbox: pertahankan aspek, pad 114 seperti Ultralytics
fn letterbox_to_square_bgr(bgr: &Mat, dst_w: i32, dst_h: i32) -> Result<Mat> {
    let w = bgr.cols() as f32;
    let h = bgr.rows() as f32;
    let r = (dst_w as f32 / w).min(dst_h as f32 / h);
    let new_w = (w * r).round() as i32;
    let new_h = (h * r).round() as i32;

    let mut resized = Mat::default();
    imgproc::resize(bgr, &mut resized, Size::new(new_w, new_h), 0.0, 0.0, imgproc::INTER_LINEAR)?;

    let pad_w = dst_w - new_w;
    let pad_h = dst_h - new_h;
    let top = pad_h / 2;
    let bottom = pad_h - top;
    let left = pad_w / 2;
    let right = pad_w - left;

    let mut padded = Mat::default();
    core::copy_make_border(
        &resized, &mut padded,
        top, bottom, left, right,
        core::BorderTypes::BORDER_CONSTANT as i32,
        Scalar::new(114.0, 114.0, 114.0, 0.0),
    )?;
    Ok(padded)
}

// BGR u8 -> letterbox 640x640 -> CHW f32 (R,G,B) [0..1]
// (loop per-pixel: paling aman dan tanpa dependensi opencv::dnn)
fn make_input_chw(bgr_u8: &Mat) -> Result<Vec<f32>> {
    let ltb = letterbox_to_square_bgr(bgr_u8, INPUT_W, INPUT_H)?;
    let size = ltb.size()?;
    let (h, w) = (size.height, size.width);
    let hw = (h * w) as usize;

    let mut out = vec![0f32; 3 * hw];
    for y in 0..h {
        for x in 0..w {
            let p: Vec3b = *ltb.at_2d::<Vec3b>(y, x)?;
            // BGR -> RGB + normalize
            let r = p[2] as f32 / 255.0;
            let g = p[1] as f32 / 255.0;
            let b = p[0] as f32 / 255.0;
            let idx = (y * w + x) as usize;
            out[idx] = r;
            out[hw + idx] = g;
            out[2 * hw + idx] = b;
        }
    }
    Ok(out)
}

// ===== Struktur & util NMS =====
#[derive(Clone)]
struct Det { x1:f32, y1:f32, x2:f32, y2:f32, conf:f32, cls:usize }

fn iou(a:&Det, b:&Det) -> f32 {
    let xx1 = a.x1.max(b.x1);
    let yy1 = a.y1.max(b.y1);
    let xx2 = a.x2.min(b.x2);
    let yy2 = a.y2.min(b.y2);
    let w = (xx2 - xx1).max(0.0);
    let h = (yy2 - yy1).max(0.0);
    let inter = w * h;
    let area_a = (a.x2 - a.x1).max(0.0) * (a.y2 - a.y1).max(0.0);
    let area_b = (b.x2 - b.x1).max(0.0) * (b.y2 - b.y1).max(0.0);
    inter / (area_a + area_b - inter + 1e-6)
}

fn nms(mut dets: Vec<Det>, iou_thr: f32) -> Vec<Det> {
    dets.sort_by(|a, b| b.conf.total_cmp(&a.conf));
    let mut keep: Vec<Det> = Vec::new();
    'outer: for d in dets.into_iter() {
        for k in &keep {
            if d.cls == k.cls && iou(&d, k) > iou_thr { continue 'outer; }
        }
        keep.push(d);
    }
    keep
}

// Decoder YOLO single-output: [1,N,5+C] atau [1,5+C,N] atau flat
fn decode_single_output(out:&[f32], shape:&[usize], num_classes:usize, conf_thr:f32)->Vec<Det>{
    let stride = 5 + num_classes;
    if stride == 0 || out.is_empty() { return vec![]; }
    let mut dets = Vec::new();

    if shape.len() >= 3 {
        let a = shape[1];
        let b = shape[2];
        if a == stride {
            // [1, 5+C, N]
            let n = b;
            for i in 0..n {
                let cx = out[0*n + i];
                let cy = out[1*n + i];
                let w  = out[2*n + i];
                let h  = out[3*n + i];
                let obj = out[4*n + i];
                if obj < conf_thr { continue; }
                let (mut best_c, mut best_s) = (0usize, 0f32);
                for c in 0..num_classes {
                    let s = out[(5 + c)*n + i];
                    if s > best_s { best_s = s; best_c = c; }
                }
                let conf = obj * best_s;
                if conf < conf_thr { continue; }
                let (x1,y1,x2,y2) = (cx - w/2.0, cy - h/2.0, cx + w/2.0, cy + h/2.0);
                dets.push(Det { x1,y1,x2,y2, conf, cls: best_c });
            }
            return dets;
        } else if b == stride {
            // [1, N, 5+C]
            let n = a;
            for i in 0..n {
                let base = i * stride;
                if base + stride > out.len() { break; }
                let cx = out[base + 0];
                let cy = out[base + 1];
                let w  = out[base + 2];
                let h  = out[base + 3];
                let obj = out[base + 4];
                if obj < conf_thr { continue; }
                let (mut best_c, mut best_s) = (0usize, 0f32);
                for c in 0..num_classes {
                    let s = out[base + 5 + c];
                    if s > best_s { best_s = s; best_c = c; }
                }
                let conf = obj * best_s;
                if conf < conf_thr { continue; }
                let (x1,y1,x2,y2) = (cx - w/2.0, cy - h/2.0, cx + w/2.0, cy + h/2.0);
                dets.push(Det { x1,y1,x2,y2, conf, cls: best_c });
            }
            return dets;
        }
    }

    // fallback: flat N*(5+C)
    if out.len() % stride == 0 {
        let n = out.len() / stride;
        for i in 0..n {
            let base = i * stride;
            let cx = out[base + 0];
            let cy = out[base + 1];
            let w  = out[base + 2];
            let h  = out[base + 3];
            let obj = out[base + 4];
            if obj < conf_thr { continue; }
            let (mut best_c, mut best_s) = (0usize, 0f32);
            for c in 0..num_classes {
                let s = out[base + 5 + c];
                if s > best_s { best_s = s; best_c = c; }
            }
            let conf = obj * best_s;
            if conf < conf_thr { continue; }
            let (x1,y1,x2,y2) = (cx - w/2.0, cy - h/2.0, cx + w/2.0, cy + h/2.0);
            dets.push(Det { x1,y1,x2,y2, conf, cls: best_c });
        }
    }
    dets
}

#[tokio::main]
async fn main() -> Result<()> {
    // 1) kelas
    let classes = load_classes("models/classes.json")?;
    let num_classes = classes.len();

    // 2) kamera (atau ganti dari file kalau mau uji gambar)
    let mut cap = videoio::VideoCapture::new(0, videoio::CAP_V4L)
        .context("open camera /dev/video0")?;
    if !cap.is_opened()? { anyhow::bail!("camera not opened"); }
    let _ = cap.set(videoio::CAP_PROP_FRAME_WIDTH, 1280.0);
    let _ = cap.set(videoio::CAP_PROP_FRAME_HEIGHT, 720.0);
    let _ = cap.set(videoio::CAP_PROP_FPS, 30.0);

    // window
    highgui::named_window("preview", highgui::WINDOW_NORMAL)?;
    highgui::resize_window("preview", 960, 540)?;

    // 3) TensorRT engine
    let plan = fs::read("models/best_fp16.engine")
        .context("read models/best_fp16.engine")?;
    let rt = Runtime::new().await; // async, bukan Result
    let mut engine: Engine = rt.deserialize_engine(&plan).await.context("deserialize TRT")?;

    // IO names (wajib single-output; kalau multi, kita kasih pesan & exit rapi)
    let mut input_name: Option<String> = None;
    let mut outputs: Vec<String> = Vec::new();
    for i in 0..engine.num_io_tensors() {
        let name = engine.io_tensor_name(i);
        match engine.tensor_io_mode(&name) {
            TensorIoMode::Input => if input_name.is_none() { input_name = Some(name); },
            TensorIoMode::Output => outputs.push(name),
            _ => {}
        }
    }
    let input_name = input_name.context("no input tensor")?;
    if outputs.len() != 1 {
        eprintln!("Engine punya {} output tensor (kemungkinan EfficientNMS). Kode ini hanya mendukung single-output [1,N,5+C] / [1,5+C,N]. 
Silakan re-export engine TANPA EfficientNMS, atau kasih aku shape semua output biar tak sesuaikan decodernya.", outputs.len());
        return Ok(());
    }
    let output_name = &outputs[0];

    let in_shape = engine.tensor_shape(&input_name);
    let out_shape = engine.tensor_shape(output_name);
    let in_elems: usize = in_shape.iter().product();
    let out_elems: usize = out_shape.iter().product();

    let mut ctx = ExecutionContext::new(&mut engine).await.context("create exec ctx")?;
    let stream = Stream::new().await.context("create CUDA stream")?;

    // pre-alloc host & device
    let mut h_in  = HostBuffer::<f32>::new(in_elems).await;
    let mut h_out = HostBuffer::<f32>::new(out_elems).await;
    let mut d_in  = DeviceBuffer::<f32>::new(in_elems,  &stream).await;
    let mut d_out = DeviceBuffer::<f32>::new(out_elems, &stream).await;

    // 4) loop preview
    let mut frame = Mat::default();
    let mut disp  = Mat::default();

    let mut fps_cnt = 0u32;
    let mut fps_t0  = Instant::now();
    let mut fps_val = 0.0f32;

    loop {
        if !cap.read(&mut frame)? || frame.empty() { break; }

        // preview resize ringan
        imgproc::resize(&frame, &mut disp, Size::new(960, 540), 0.0, 0.0, imgproc::INTER_LINEAR)?;

        // preprocess (letterbox)
        let input = make_input_chw(&frame)?;
        h_in.copy_from_slice(&input);
        h_in.copy_to(&mut d_in, &stream).await.context("H->D input")?;

        // run TRT
        let mut io: HashMap<&str, &mut DeviceBuffer<f32>> = HashMap::new();
        io.insert(&input_name, &mut d_in);
        io.insert(&output_name, &mut d_out);
        ctx.enqueue(&mut io, &stream).await.context("enqueue")?;

        // copy out
        h_out.copy_from(&d_out, &stream).await.context("D->H output")?;
        let out_host: Vec<f32> = h_out.to_vec();

        // decode & NMS
        let mut dets = decode_single_output(&out_host, &out_shape, num_classes, CONF_THR);
        dets = nms(dets, IOU_THR);

        // filter & gambar
        let mut total = 0usize;
        for d in &dets {
            if let Some(name) = classes.get(d.cls) {
                let lname = name.to_lowercase();
                if lname == "cars" || lname == "motorcyle" || lname == "truck" {
                    total += 1;
                    if DRAW_BOXES {
                        let sx = 960.0 / INPUT_W as f32;
                        let sy = 540.0 / INPUT_H as f32;
                        let x1 = (d.x1 * sx).clamp(0.0, 959.0) as i32;
                        let y1 = (d.y1 * sy).clamp(0.0, 539.0) as i32;
                        let x2 = (d.x2 * sx).clamp(0.0, 959.0) as i32;
                        let y2 = (d.y2 * sy).clamp(0.0, 539.0) as i32;
                        let rect = Rect::new(x1, y1, (x2 - x1).max(1), (y2 - y1).max(1));
                        imgproc::rectangle(&mut disp, rect, Scalar::new(0.0, 255.0, 0.0, 0.0), 2, imgproc::LINE_8, 0)?;
                        let label = format!("{} {:.2}", name, d.conf);
                        imgproc::put_text(&mut disp, &label, Point::new(x1, (y1 - 6).max(12)),
                            imgproc::FONT_HERSHEY_SIMPLEX, 0.6, Scalar::new(0.0,0.0,0.0,0.0), 3, imgproc::LINE_8, false)?;
                        imgproc::put_text(&mut disp, &label, Point::new(x1, (y1 - 6).max(12)),
                            imgproc::FONT_HERSHEY_SIMPLEX, 0.6, Scalar::new(0.0,255.0,0.0,0.0), 1, imgproc::LINE_8, false)?;
                    }
                }
            }
        }

        // FPS
        fps_cnt += 1;
        let dt = fps_t0.elapsed().as_secs_f32();
        if dt >= 1.0 {
            fps_val = fps_cnt as f32 / dt;
            fps_cnt = 0;
            fps_t0  = Instant::now();
        }
        let overlay = format!("FPS: {:.1}   vehicles: {}", fps_val, total);
        imgproc::put_text(&mut disp, &overlay, Point::new(12, 28),
            imgproc::FONT_HERSHEY_SIMPLEX, 0.9, Scalar::new(0.0,0.0,0.0,0.0), 3, imgproc::LINE_8, false)?;
        imgproc::put_text(&mut disp, &overlay, Point::new(12, 28),
            imgproc::FONT_HERSHEY_SIMPLEX, 0.9, Scalar::new(0.0,255.0,255.0,0.0), 2, imgproc::LINE_8, false)?;

        highgui::imshow("preview", &disp)?;
        let k = highgui::wait_key(1)?;
        if k == 27 || k == 'q' as i32 { break; }
    }

    Ok(())
}
