use anyhow::{Context, Result};
use async_cuda::{DeviceBuffer, HostBuffer, Stream};
use async_tensorrt::{runtime::Runtime};
use async_tensorrt::engine::{Engine, ExecutionContext, TensorIoMode};
use opencv::{
    core::{Mat, MatTraitConst, Size, Vec3b},
    imgproc,
    prelude::*,
    videoio::{self, VideoCaptureTrait, VideoCaptureTraitConst},
};
use std::{collections::HashMap, fs};

const INPUT_W: i32 = 640;
const INPUT_H: i32 = 640;

fn load_classes(path: &str) -> Result<Vec<String>> {
    let txt = fs::read_to_string(path).context("read classes.json")?;
    let v: Vec<String> = serde_json::from_str(&txt).context("parse classes.json")?;
    Ok(v)
}

fn mat_to_chw_f32_bgr(mat_bgr_u8: &Mat) -> Result<Vec<f32>> {
    let mut resized = Mat::default();
    imgproc::resize(
        mat_bgr_u8,
        &mut resized,
        Size::new(INPUT_W, INPUT_H),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;

    let size = resized.size()?;
    let (h, w) = (size.height, size.width);
    let mut out = vec![0f32; (3 * h * w) as usize];
    let hw = (h * w) as usize;

    for y in 0..h {
        for x in 0..w {
            let px: Vec3b = *resized.at_2d::<Vec3b>(y, x)?;
            // BGR -> CHW (R,G,B) normalized
            let b = px[0] as f32 / 255.0;
            let g = px[1] as f32 / 255.0;
            let r = px[2] as f32 / 255.0;
            let idx = (y * w + x) as usize;
            out[idx] = r;
            out[hw + idx] = g;
            out[2 * hw + idx] = b;
        }
    }
    Ok(out)
}

/// Decoder YOLO generik (asumsi baris: [cx,cy,w,h,obj, class...])
fn decode_yolo_like(
    out: &[f32],
    num_classes: usize,
    class_names: &[String],
    score_thresh: f32,
) -> HashMap<String, usize> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    let stride = 5 + num_classes;
    if stride == 0 || out.len() % stride != 0 {
        return counts; // layout tak cocok, biarkan kosong
    }
    let n = out.len() / stride;

    for i in 0..n {
        let base = i * stride;
        let obj = out[base + 4];
        if obj < score_thresh {
            continue;
        }
        // pilih kelas terbaik
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
    // 1) classes
    let classes = load_classes("models/classes.json")?;
    let num_classes = classes.len();

    // 2) kamera
    let mut cap = videoio::VideoCapture::new(0, videoio::CAP_V4L)
        .context("open camera /dev/video0")?;
    if !cap.is_opened()? {
        anyhow::bail!("camera not opened");
    }
    let _ = cap.set(videoio::CAP_PROP_FRAME_WIDTH, 1280.0);
    let _ = cap.set(videoio::CAP_PROP_FRAME_HEIGHT, 720.0);

    // 3) TensorRT
    let plan = fs::read("models/best_fp16.engine")
        .context("read models/best_fp16.engine")?;
    let rt = Runtime::new().await; // async, bukan Result
    let mut engine: Engine = rt
        .deserialize_engine(&plan)
        .await
        .context("deserialize TRT engine")?;

    // cari nama IO
    let mut input_name: Option<String> = None;
    let mut output_name: Option<String> = None;
    for i in 0..engine.num_io_tensors() {
        let name = engine.io_tensor_name(i);
        match engine.tensor_io_mode(&name) {
            TensorIoMode::Input => if input_name.is_none() { input_name = Some(name); },
            TensorIoMode::Output => if output_name.is_none() { output_name = Some(name); },
        }
    }
    let input_name = input_name.context("no input tensor in engine")?;
    let output_name = output_name.context("no output tensor in engine")?;

    let in_shape = engine.tensor_shape(&input_name);
    let out_shape = engine.tensor_shape(&output_name);
    let in_elems: usize = in_shape.iter().product();
    let out_elems: usize = out_shape.iter().product();

    let mut ctx = ExecutionContext::new(&mut engine)
        .await
        .context("create execution context")?;
    let stream = Stream::new().await.context("create CUDA stream")?;

    // pre-alloc host & device
    let mut h_in = HostBuffer::<f32>::new(in_elems).await;
    let mut h_out = HostBuffer::<f32>::new(out_elems).await;
    let mut d_in = DeviceBuffer::<f32>::new(in_elems, &stream).await;
    let mut d_out = DeviceBuffer::<f32>::new(out_elems, &stream).await;

    // 4) loop
    let mut frame = Mat::default();
    loop {
        if !cap.read(&mut frame)? || frame.empty() {
            break;
        }

        let input_tensor = mat_to_chw_f32_bgr(&frame)?;
        h_in.copy_from_slice(&input_tensor);
        h_in.copy_to(&mut d_in, &stream).await.context("H->D input")?;

        let mut io: HashMap<&str, &mut DeviceBuffer<f32>> = HashMap::new();
        io.insert(&input_name, &mut d_in);
        io.insert(&output_name, &mut d_out);
        ctx.enqueue(&mut io, &stream).await.context("enqueue")?;

        h_out.copy_from(&d_out, &stream).await.context("D->H output")?;
        let out_host: Vec<f32> = h_out.to_vec();

        // hitung kendaraan: cars / motorcyle / truck (abaikan License_Plate)
        let counts = decode_yolo_like(&out_host, num_classes, &classes, 0.25);
        let mut total = 0usize;
        for k in ["cars", "motorcyle", "truck"] {
            if let Some(v) = counts.get(k) { total += *v; }
        }
        println!("counts={:?}  total={}", counts, total);
    }

    Ok(())
}
