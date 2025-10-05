use anyhow::{Context, Result};
use async_cuda::{DeviceBuffer, HostBuffer, Stream};
use async_tensorrt::engine::{Engine, ExecutionContext, TensorIoMode};
use async_tensorrt::runtime::Runtime;
use opencv::{
    core::{Mat, MatTraitConst, Point, Rect, Scalar, Size, Vec3b},
    highgui,
    imgproc,
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

/// BGR (u8) -> resize 640x640 -> CHW f32 [0..1] (R,G,B)
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
    let hw = (h * w) as usize;
    let mut out = vec![0f32; 3 * hw];

    for y in 0..h {
        for x in 0..w {
            let px: Vec3b = *resized.at_2d::<Vec3b>(y, x)?;
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

#[derive(Clone)]
struct Det {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    conf: f32,
    cls: usize,
}

fn iou(a: &Det, b: &Det) -> f32 {
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

/// Greedy NMS (per kelas)
fn nms(mut dets: Vec<Det>, iou_thr: f32) -> Vec<Det> {
    dets.sort_by(|a, b| b.conf.total_cmp(&a.conf));
    let mut keep: Vec<Det> = Vec::new();
    'outer: for d in dets.into_iter() {
        for k in &keep {
            if d.cls == k.cls && iou(&d, k) > iou_thr {
                continue 'outer;
            }
        }
        keep.push(d);
    }
    keep
}

/// Decode output YOLO:
/// - dukung [1, N, 5+C] (sekuensial) atau [1, 5+C, N] (CHW)
fn decode_boxes(
    out: &[f32],
    shape: &[usize],
    num_classes: usize,
    conf_thr: f32,
) -> Vec<Det> {
    let stride = 5 + num_classes;
    if stride == 0 || out.is_empty() { return vec![]; }

    let mut dets = Vec::new();

    if shape.len() >= 3 {
        // [B, A, N] atau [B, N, A]
        let a = shape[1];
        let b = shape[2];
        if a == stride {
            // [1, 5+C, N]  => CHW style
            let n = b;
            for i in 0..n {
                let cx = out[0 * n + i];
                let cy = out[1 * n + i];
                let w  = out[2 * n + i];
                let h  = out[3 * n + i];
                let obj = out[4 * n + i];
                if obj < conf_thr { continue; }
                let mut best_c = 0usize;
                let mut best_s = 0f32;
                for c in 0..num_classes {
                    let s = out[(5 + c) * n + i];
                    if s > best_s {
                        best_s = s; best_c = c;
                    }
                }
                let conf = obj * best_s;
                if conf < conf_thr { continue; }
                let (x1, y1, x2, y2) = (cx - w/2.0, cy - h/2.0, cx + w/2.0, cy + h/2.0);
                dets.push(Det { x1, y1, x2, y2, conf, cls: best_c });
            }
            return dets;
        } else if b == stride {
            // [1, N, 5+C] => sekuensial
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
                let mut best_c = 0usize;
                let mut best_s = 0f32;
                for c in 0..num_classes {
                    let s = out[base + 5 + c];
                    if s > best_s {
                        best_s = s; best_c = c;
                    }
                }
                let conf = obj * best_s;
                if conf < conf_thr { continue; }
                let (x1, y1, x2, y2) = (cx - w/2.0, cy - h/2.0, cx + w/2.0, cy + h/2.0);
                dets.push(Det { x1, y1, x2, y2, conf, cls: best_c });
            }
            return dets;
        }
    }

    // fallback: asumsikan flat N*(5+C)
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
            let mut best_c = 0usize;
            let mut best_s = 0f32;
            for c in 0..num_classes {
                let s = out[base + 5 + c];
                if s > best_s {
                    best_s = s; best_c = c;
                }
            }
            let conf = obj * best_s;
            if conf < conf_thr { continue; }
            let (x1, y1, x2, y2) = (cx - w/2.0, cy - h/2.0, cx + w/2.0, cy + h/2.0);
            dets.push(Det { x1, y1, x2, y2, conf, cls: best_c });
        }
    }
    dets
}

#[tokio::main]
async fn main() -> Result<()> {
    // 1) kelas
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

    // window preview
    highgui::named_window("preview", highgui::WINDOW_NORMAL)?;
    highgui::resize_window("preview", 960, 540)?;

    // 3) TensorRT
    let plan = fs::read("models/best_fp16.engine")
        .context("read models/best_fp16.engine")?;
    let rt = Runtime::new().await; // async, bukan Result
    let mut engine: Engine = rt
        .deserialize_engine(&plan)
        .await
        .context("deserialize TRT engine")?;

    // IO names
    let mut input_name: Option<String> = None;
    let mut output_name: Option<String> = None;
    for i in 0..engine.num_io_tensors() {
        let name = engine.io_tensor_name(i);
        match engine.tensor_io_mode(&name) {
            TensorIoMode::Input => if input_name.is_none() { input_name = Some(name); },
            TensorIoMode::Output => if output_name.is_none() { output_name = Some(name); },
            _ => { /* None / others: abaikan */ }
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

        // buat frame tampilan yg ukuran 640x480 (biar enteng) tapi input tetap 640x640
        let mut disp = Mat::default();
        imgproc::resize(&frame, &mut disp, Size::new(960, 540), 0.0, 0.0, imgproc::INTER_LINEAR)?;

        // preprocess untuk engine
        let input_tensor = mat_to_chw_f32_bgr(&frame)?;
        h_in.copy_from_slice(&input_tensor);
        h_in.copy_to(&mut d_in, &stream).await.context("H->D input")?;

        let mut io: HashMap<&str, &mut DeviceBuffer<f32>> = HashMap::new();
        io.insert(&input_name, &mut d_in);
        io.insert(&output_name, &mut d_out);
        ctx.enqueue(&mut io, &stream).await.context("enqueue")?;

        h_out.copy_from(&d_out, &stream).await.context("D->H output")?;
        let out_host: Vec<f32> = h_out.to_vec();

        // decode + NMS
        let mut dets = decode_boxes(&out_host, &out_shape, num_classes, 0.25);
        dets = nms(dets, 0.45);

        // filter hanya kendaraan target
        let mut total = 0usize;
        for d in dets.iter() {
            if let Some(name) = classes.get(d.cls) {
                let lname = name.to_lowercase();
                if lname == "cars" || lname == "motorcyle" || lname == "truck" {
                    total += 1;

                    // gambar bbox pada 'disp' yg 960x540 -> skala dari 640x640
                    let sx = 960.0 / INPUT_W as f32;
                    let sy = 540.0 / INPUT_H as f32;
                    let x1 = (d.x1 * sx).max(0.0) as i32;
                    let y1 = (d.y1 * sy).max(0.0) as i32;
                    let x2 = (d.x2 * sx).min(959.0) as i32;
                    let y2 = (d.y2 * sy).min(539.0) as i32;
                    let rect = Rect::new(x1, y1, (x2 - x1).max(1) as i32, (y2 - y1).max(1) as i32);

                    imgproc::rectangle(&mut disp, rect, Scalar::new(0.0, 255.0, 0.0, 0.0), 2, imgproc::LINE_8, 0)?;
                    let label = format!("{} {:.2}", name, d.conf);
                    imgproc::put_text(
                        &mut disp,
                        &label,
                        Point::new(x1.max(0), (y1 - 6).max(12)),
                        imgproc::FONT_HERSHEY_SIMPLEX,
                        0.6,
                        Scalar::new(0.0, 0.0, 0.0, 0.0),
                        3,
                        imgproc::LINE_8,
                        false,
                    )?;
                    imgproc::put_text(
                        &mut disp,
                        &label,
                        Point::new(x1.max(0), (y1 - 6).max(12)),
                        imgproc::FONT_HERSHEY_SIMPLEX,
                        0.6,
                        Scalar::new(0.0, 255.0, 0.0, 0.0),
                        1,
                        imgproc::LINE_8,
                        false,
                    )?;
                }
            }
        }

        // counter overlay
        let counter = format!("vehicles: {}", total);
        imgproc::put_text(
            &mut disp,
            &counter,
            Point::new(12, 28),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.9,
            Scalar::new(0.0, 0.0, 0.0, 0.0),
            3,
            imgproc::LINE_8,
            false,
        )?;
        imgproc::put_text(
            &mut disp,
            &counter,
            Point::new(12, 28),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.9,
            Scalar::new(0.0, 255.0, 255.0, 0.0),
            2,
            imgproc::LINE_8,
            false,
        )?;

        highgui::imshow("preview", &disp)?;
        let k = highgui::wait_key(1)?;
        if k == 27 || k == 'q' as i32 {
            break;
        }
    }

    Ok(())
}
