
// main.rs — semua logic satu file
// Build: cargo build --release
// Run  : ./target/release/jetson_yolo_count --engine yolo_fp16.plan --classes classes.json --device /dev/video0

use anyhow::{bail, Context, Result};
use clap::Parser;
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app::AppSink;
use image::{imageops, ImageBuffer, Rgb};
use ndarray::Array3;
use serde::Deserialize;
use std::{fs, path::PathBuf, time::Duration};
use tokio::time::Instant;

// TensorRT async wrapper
use async_tensorrt::runtime::Runtime;
use async_tensorrt::engine::{Engine, TensorIoMode};
use async_cuda::stream::Stream;

#[derive(Parser, Debug)]
#[command(name="jetson_yolo_count")]
struct Args {
    /// Path ke TensorRT engine (.plan)
    #[arg(long)]
    engine: PathBuf,

    /// Path ke classes.json (array nama kelas)
    #[arg(long)]
    classes: PathBuf,

    /// Device kamera V4L2 (USB cam)
    #[arg(long, default_value="/dev/video0")]
    device: String,

    /// Ukuran input model (persegi)
    #[arg(long, default_value_t=640)]
    imgsz: u32,

    /// Confidence thresh
    #[arg(long, default_value_t=0.25)]
    conf: f32,

    /// IoU thresh untuk NMS
    #[arg(long, default_value_t=0.45)]
    iou: f32,

    /// Tampilkan FPS/log detail
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
    // Sesuaikan dengan classes.json kamu
    // Kunci kata umum kendaraan
    let keywords = [
        "car","bus","truck","motorcycle","motorbike","bicycle","bike","van","pickup","trailer","truk","mobil","sepeda","motor"
    ];
    let ln = name.to_lowercase();
    keywords.iter().any(|k| ln.contains(k))
}

fn letterbox_rgb(
    rgb: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    dst_size: u32,
) -> (ImageBuffer<Rgb<u8>, Vec<u8>>, f32, u32, u32) {
    // Letterbox ke persegi dst_size x dst_size
    let (w, h) = (rgb.width() as f32, rgb.height() as f32);
    let s = (dst_size as f32 / w).min(dst_size as f32 / h);
    let nw = (w * s).round() as u32;
    let nh = (h * s).round() as u32;
    let resized = imageops::resize(rgb, nw, nh, imageops::FilterType::Triangle);

    let mut canvas = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_pixel(
        dst_size, dst_size, Rgb([114u8, 114u8, 114u8])
    );
    let dx = ((dst_size - nw) / 2) as u32;
    let dy = ((dst_size - nh) / 2) as u32;

    imageops::replace(&mut canvas, &resized, dx.into(), dy.into());
    (canvas, s, dx, dy)
}

fn chw_normalize(inp: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> Vec<f32> {
    // RGB HWC -> CHW float32 [0,1]
    let (w, h) = (inp.width() as usize, inp.height() as usize);
    let mut out = vec![0f32; 3 * w * h];
    for y in 0..h {
        for x in 0..w {
            let p = inp.get_pixel(x as u32, y as u32);
            let idx = y * w + x;
            out[idx] = p[0] as f32 / 255.0; // R
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
    let mut keep = Vec::new();
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

/// Decode keluaran YOLO: diasumsikan (1, (5+nc), N) -> kita ubah jadi Vec<Detection>
/// bbox input di YOLO adalah xywh relatif ke input (640x640)
fn decode_yolo(
    out: &[f32],
    num_classes: usize,
    num_candidates: usize,
    conf_t: f32,
    img_w: u32,
    img_h: u32,
    letter_scale: f32,
    pad_x: u32,
    pad_y: u32,
    dst_size: u32,
    names: &[String],
) -> Vec<Detection> {
    // out layout: [ (5+nc) * N ], kita akses sebagai (5+nc, N)
    let stride = num_candidates; // per row
    let mut dets = Vec::new();
    for i in 0..num_candidates {
        let bx = out[i];                         // center x (0..1) * dst_size
        let by = out[stride + i];                // center y
        let bw = out[2 * stride + i];            // w
        let bh = out[3 * stride + i];            // h
        let obj = out[4 * stride + i];           // objectness (sigmoid sudah diaplikasikan oleh YOLO ONNX umumnya)
        // kelas mulai dari 5*stride
        let mut best_c = 0usize;
        let mut best_p = 0f32;
        for c in 0..num_classes {
            let p = out[(5 + c) * stride + i];
            if p > best_p {
                best_p = p;
                best_c = c;
            }
        }
        let score = obj * best_p;
        if score < conf_t {
            continue;
        }
        // filter kendaraan saja
        let cname = names.get(best_c).map(|s| s.as_str()).unwrap_or("");
        if !is_vehicle(cname) {
            continue;
        }

        // xywh (pada kanvas dst_size) -> xyxy pada kanvas -> lalu undo letterbox ke ukuran asli frame
        let (cx, cy, w, h) = (bx * dst_size as f32, by * dst_size as f32, bw * dst_size as f32, bh * dst_size as f32);
        let (mut x1, mut y1, mut x2, mut y2) = (cx - w/2.0, cy - h/2.0, cx + w/2.0, cy + h/2.0);

        // hapus padding letterbox
        let fx = (x1 - pad_x as f32).max(0.0);
        let fy = (y1 - pad_y as f32).max(0.0);
        let fx2 = (x2 - pad_x as f32).max(0.0);
        let fy2 = (y2 - pad_y as f32).max(0.0);

        // scale balik ke ukuran frame asli
        let inv = 1.0 / letter_scale;
        x1 = (fx * inv).clamp(0.0, img_w as f32 - 1.0);
        y1 = (fy * inv).clamp(0.0, img_h as f32 - 1.0);
        x2 = (fx2 * inv).clamp(0.0, img_w as f32 - 1.0);
        y2 = (fy2 * inv).clamp(0.0, img_h as f32 - 1.0);

        dets.push(Detection { x1, y1, x2, y2, score, class_id: best_c });
    }
    dets
}

#[tokio::main(flavor="multi_thread")]
async fn main() -> Result<()> {
    let args = Args::parse();

    // 1) load classes
    let classes: ClassList = serde_json::from_slice(
        &fs::read(&args.classes).context("gagal baca classes.json")?
    ).context("classes.json bukan array string?")?;
    if classes.0.is_empty() {
        bail!("classes.json kosong");
    }

    // 2) init gstreamer
    gst::init()?;

    // pipeline USB camera → BGR → appsink
    // NB: untuk zero-copy NVMM perlu jalur khusus; versi simpel ini tarik CPU memory (cukup untuk demo).
    let pipeline_str = format!(
        "v4l2src device={} ! \
         videoconvert ! video/x-raw,format=BGR ! \
         videoscale ! video/x-raw,width=1280,height=720 ! \
         queue max-size-buffers=1 leaky=downstream ! \
         appsink name=sink emit-signals=false sync=false max-buffers=1 drop=true",
        args.device
    );

    let pipeline = gst::parse::launch(&pipeline_str)
        .with_context(|| format!("gagal buat pipeline: {}", pipeline_str))?
        .downcast::<gst::Pipeline>()
        .unwrap();

    let appsink = pipeline
        .by_name("sink")
        .unwrap()
        .downcast::<AppSink>()
        .unwrap();

    pipeline.set_state(gst::State::Playing)?;

    // 3) TensorRT: load engine dari .plan
    let plan_bytes = fs::read(&args.engine)
        .with_context(|| format!("gagal baca engine: {:?}", args.engine))?;

    // Runtime & Engine
    let runtime = Runtime::new().await;
    let engine: Engine = runtime.deserialize_engine(&plan_bytes).await
        .context("deserialize engine TensorRT gagal")?;

    // Ambil nama IO tensor otomatis (asumsi 1 input, 1 output)
    let mut input_name = None;
    let mut output_name = None;
    let num_io = engine.num_io_tensors();
    for i in 0..num_io {
        let name = engine.io_tensor_name(i);
        match engine.tensor_io_mode(&name) {
            TensorIoMode::Input => input_name = Some(name),
            TensorIoMode::Output => output_name = Some(name),
        }
    }
    let input_name = input_name.context("tidak menemukan input tensor pada engine")?;
    let output_name = output_name.context("tidak menemukan output tensor pada engine")?;

    // Bentuk output untuk alokasi buffer host
    let out_shape = engine.tensor_shape(&output_name); // contoh: [1, (5+nc), N]
    if out_shape.len() != 3 {
        eprintln!("PERINGATAN: shape output bukan 3 dimensi, shape={:?}", out_shape);
    }
    let (out_b, out_c, out_n) = (out_shape[0], out_shape[1], out_shape[2]);
    let out_elems = out_b * out_c * out_n;

    // Buat execution context + stream CUDA
    // Catatan: API detail bisa sedikit beda antar versi crate; konsepnya:
    // - set alamat device buffer dengan nama tensor
    // - enqueue async (v3) pada stream
    let mut ctx = engine.create_execution_context()
        .await
        .context("gagal buat execution context")?;
    let stream = Stream::new().await;

    // Siapkan buffer host untuk output (copy dari device)
    let mut out_host: Vec<f32> = vec![0f32; out_elems];

    let mut frame_idx: u64 = 0;
    let mut last_t = Instant::now();

    loop {
        // 4) Ambil sample frame
        let sample = match appsink.try_pull_sample(Duration::from_millis(500)) {
            Some(s) => s,
            None => {
                eprintln!("timeout tarik frame…");
                continue;
            }
        };

        let buffer = sample.buffer().context("sample tanpa buffer")?;
        let caps = sample.caps().context("sample tanpa caps")?;
        let s = caps.structure(0).context("caps tanpa structure")?;
        let (w, h) = (s.get::<i32>("width").unwrap() as u32, s.get::<i32>("height").unwrap() as u32);

        // Map BGR bytes
        let map = buffer.map_readable().context("gagal map buffer")?;
        let bgr = map.as_slice();

        // BGR -> RGB image buffer
        if bgr.len() != (w * h * 3) as usize {
            eprintln!("ukuran buffer tak sesuai: {} vs {}", bgr.len(), w*h*3);
            continue;
        }
        let mut rgb_data = vec![0u8; (w * h * 3) as usize];
        for i in 0..(w * h) as usize {
            rgb_data[3*i]   = bgr[3*i + 2]; // R
            rgb_data[3*i+1] = bgr[3*i + 1]; // G
            rgb_data[3*i+2] = bgr[3*i + 0]; // B
        }
        let rgb_img: ImageBuffer<Rgb<u8>, _> = ImageBuffer::from_raw(w, h, rgb_data)
            .context("gagal buat RGB image")?;

        // 5) Preprocess (letterbox ke args.imgsz)
        let (letter, scale, pad_x, pad_y) = letterbox_rgb(&rgb_img, args.imgsz);
        let input_chw = chw_normalize(&letter);

        // 6) Inference (copy H->D, set tensor, enqueue, copy D->H)
        // NB: API alokasi device buffer/ set address pada async-tensorrt bersandar pada versi crate.
        // Skema umum: buat DeviceBuffer<f32> untuk input & output, set alamat via ctx.set_tensor_address(&name, dev_ptr),
        // lalu ctx.enqueue_async_v3(stream).
        // Di sini kita gunakan helper dari async-cuda untuk memindahkan data sinkron (sederhana untuk contoh).
        use async_cuda::memory::{DeviceBuffer, CopyDestination};

        let mut d_input: DeviceBuffer<f32> = DeviceBuffer::from_slice(&input_chw).await
            .context("alokasi/copy input ke device gagal")?;
        let mut d_output: DeviceBuffer<f32> = DeviceBuffer::new(out_elems).await
            .context("alokasi output device gagal")?;

        // Set alamat tensor (nama input/output diambil dari engine)
        ctx.set_tensor_address(&input_name, d_input.as_device_ptr() as u64)
            .await
            .context("set alamat tensor input gagal")?;
        ctx.set_tensor_address(&output_name, d_output.as_device_ptr() as u64)
            .await
            .context("set alamat tensor output gagal")?;

        // Jalankan (v3) pada stream
        ctx.enqueue_v3(&stream).await.context("enqueue_v3 gagal")?;

        // Sinkron & salin hasil ke host
        stream.synchronize().await;
        d_output.copy_to(&mut out_host).await
            .context("copy output ke host gagal")?;

        // 7) Post-process YOLO
        let num_classes = classes.0.len();
        let dst = args.imgsz;
        let dets_raw = decode_yolo(
            &out_host, num_classes, out_n, args.conf,
            w, h, scale, pad_x, pad_y, dst, &classes.0,
        );
        let dets = nms(dets_raw, args.iou);

        // 8) Hitung kendaraan per frame
        let count = dets.len();
        frame_idx += 1;

        if args.verbose {
            let now = Instant::now();
            let dt = now.duration_since(last_t).as_secs_f32();
            last_t = now;
            let fps = if dt > 0.0 { 1.0/dt } else { 0.0 };
            println!("[frame {frame_idx}] vehicles={count}  fps={fps:.1}");
        } else {
            println!("{count}");
        }
    }
}
