use anyhow::{Context, Result};
use async_cuda::{DeviceBuffer, HostBuffer, Stream};
use async_tensorrt::{runtime::Runtime};
use async_tensorrt::engine::{Engine, ExecutionContext, TensorIoMode};
use opencv::{
    core::{self, Mat, MatTraitConst, MatTrait, Size, Vec3b},
    imgproc,
    prelude::*,
    videoio,
};
use serde::Deserialize;
use std::{collections::HashMap, fs};

const INPUT_W: i32 = 640;
const INPUT_H: i32 = 640;

/// ---- utils: load classes.json ----
fn load_classes(path: &str) -> Result<Vec<String>> {
    let txt = fs::read_to_string(path).context("read classes.json")?;
    let v: Vec<String> = serde_json::from_str(&txt).context("parse classes.json")?;
    Ok(v)
}

/// ---- utils: resize+normalize to CHW f32 [0,1] ----
fn mat_to_chw_f32_bgr(mat_bgr_u8: &Mat) -> Result<Vec<f32>> {
    // resize to 640x640
    let mut resized = Mat::default();
    imgproc::resize(
        mat_bgr_u8,
        &mut resized,
        Size::new(INPUT_W, INPUT_H),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;

    // Expect CV_8UC3
    let size = resized.size()?;
    let (h, w) = (size.height, size.width);
    let mut out = vec![0f32; (3 * h * w) as usize];

    // CHW fill
    let hw = (h * w) as usize;
    for y in 0..h {
        for x in 0..w {
            // Safety: using at_2d is fine but returns Result
            let px: Vec3b = *resized.at_2d::<Vec3b>(y, x)?;
            let b = px[0] as f32 / 255.0;
            let g = px[1] as f32 / 255.0;
            let r = px[2] as f32 / 255.0;
            let idx = (y * w + x) as usize;
            out[idx] = r;            // C0 (R)
            out[hw + idx] = g;       // C1 (G)
            out[2 * hw + idx] = b;   // C2 (B)
        }
    }
    Ok(out)
}

/// ---- YOLO-ish decoder (generic). Adjust if layout berbeda) ----
/// Asumsi output = N x (5 + num_classes) [cx,cy,w,h,obj, class...] (common for ONNX exports)
/// Return jumlah per-kelas (by name)
fn decode_yolo_like(
    out: &[f32],
    num_classes: usize,
    class_names: &[String],
    score_thresh: f32,
) -> HashMap<String, usize> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for name in class_names {
        counts.entry(name.clone()).or_insert(0);
    }

    let stride = 5 + num_classes;
    if stride == 0 || out.len() % stride != 0 {
        // Layout tidak cocok — biarkan kosong; tinggal sesuaikan decoder kalau perlu.
        return counts;
    }
    let n = out.len() / stride;

    for i in 0..n {
        let base = i * stride;
        let obj = out[base + 4];
        if obj < score_thresh {
            continue;
        }
        // pilih kelas dengan skor tertinggi
        let mut best_c = 0usize;
        let mut best_s = 0f32;
        for c in 0..num_classes {
            let s = out[base + 5 + c];
            if s > best_s {
                best_s = s;
                best_c = c;
            }
        }
        let conf = obj * best_s;
        if conf >= score_thresh {
            if let Some(name) = class_names.get(best_c) {
                *counts.entry(name.clone()).or_insert(0) += 1;
            }
        }
    }
    counts
}

#[tokio::main]
async fn main() -> Result<()> {
    // ==== 1) Load class list ====
    // contoh: models/classes.json berisi:
    // ["License_Plate","cars","motorcyle","truck"]
    let classes = load_classes("models/classes.json")?;
    let num_classes = classes.len();

    // ==== 2) OpenCV capture ====
    // 0 = /dev/video0; sesuaikan kalau perlu.
    let mut cap = videoio::VideoCapture::new(0, videoio::CAP_V4L)
        .context("open camera /dev/video0")?;
    if !cap.is_opened()? {
        anyhow::bail!("camera not opened");
    }
    // (opsional) set resolusi capture — bisa saja driver mengabaikan.
    let _ = cap.set(videoio::CAP_PROP_FRAME_WIDTH, 1280.0);
    let _ = cap.set(videoio::CAP_PROP_FRAME_HEIGHT, 720.0);

    // ==== 3) TensorRT runtime + engine ====
    // Pastikan file sudah diexport: models/best_fp16.engine
    let plan = fs::read("models/best_fp16.engine")
        .context("read models/best_fp16.engine")?;
    let rt = Runtime::new(); // tidak async, tidak Result
    let mut engine: Engine = rt
        .deserialize_engine(&plan)
        .await
        .context("deserialize TRT engine")?;

    // Ambil nama input/output tensor dari engine
    let mut input_name: Option<String> = None;
    let mut output_name: Option<String> = None;
    let io_count = engine.num_io_tensors();
    for i in 0..io_count {
        let name = engine.io_tensor_name(i);
        match engine.tensor_io_mode(&name) {
            TensorIoMode::Input => input_name = Some(name),
            TensorIoMode::Output => output_name = Some(name),
        }
    }
    let input_name = input_name.context("no input tensor in engine")?;
    let output_name = output_name.context("no output tensor in engine")?;

    // Ambil shape untuk alokasi buffer
    let in_shape = engine.tensor_shape(&input_name);
    let out_shape = engine.tensor_shape(&output_name);
    let in_elems: usize = in_shape.iter().product();
    let out_elems: usize = out_shape.iter().product();

    // ==== 4) Eksekusi context + stream ====
    // Pakai &mut engine supaya engine tetap bisa dipakai kalau perlu.
    let mut ctx = ExecutionContext::new(&mut engine)
        .await
        .context("create execution context")?;
    let stream = Stream::new().await.context("create CUDA stream")?;

    // ==== 5) Pre-allocate pinned host & device buffers ====
    // host pinned (lebih cepat untuk copy)
    let mut h_in = HostBuffer::<f32>::new(in_elems).await;
    let mut h_out = HostBuffer::<f32>::new(out_elems).await;
    // device
    let mut d_in = DeviceBuffer::<f32>::new(in_elems, &stream).await;
    let mut d_out = DeviceBuffer::<f32>::new(out_elems, &stream).await;

    println!(
        "Engine IO:\n  input  '{}' shape {:?}\n  output '{}' shape {:?}",
        input_name, in_shape, output_name, out_shape
    );

    // ==== 6) Loop capture → preprocess → infer → copy back → (decode & count) ====
    // (demo: proses 200 frame; ubah sesuai kebutuhan)
    let mut frame = Mat::default();
    for _ in 0..200 {
        cap.read(&mut frame)
            .context("camera read")?;
        if frame.empty()? {
            continue;
        }

        // preprocess (BGR u8 -> CHW f32 [0,1] 640x640)
        let input_tensor = mat_to_chw_f32_bgr(&frame)?;

        // salin ke host pinned, lalu H->D
        h_in.copy_from_slice(&input_tensor);
        h_in.copy_to(&mut d_in, &stream)
            .await
            .context("H->D copy input")?;

        // siapkan io map (nama tensor -> device buffer)
        let mut io: HashMap<&str, &mut DeviceBuffer<f32>> = HashMap::new();
        io.insert(&input_name, &mut d_in);
        io.insert(&output_name, &mut d_out);

        // enqueue inference
        ctx.enqueue(&mut io, &stream)
            .await
            .context("enqueue inference")?;

        // D->H output
        h_out.copy_from(&d_out, &stream)
            .await
            .context("D->H copy output")?;
        // optional: sync stream (HostBuffer::copy_from sudah implicit sync, tapi aman)
        stream.synchronize().await.ok();

        // ambil Vec untuk diproses di CPU
        let out_host: Vec<f32> = h_out.to_vec();

        // ---- decode & contoh hitung kendaraan (abaikan 'License_Plate') ----
        let counts = decode_yolo_like(&out_host, num_classes, &classes, 0.25);
        let fossil_keys = ["cars", "motorcyle", "truck"];
        let mut total = 0usize;
        for k in fossil_keys {
            if let Some(v) = counts.get(k) {
                total += *v;
            }
        }
        println!("Count per class: {:?} | total (fossil-ish): {}", counts, total);
    }

    Ok(())
}
