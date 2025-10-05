use anyhow::{bail, Context, Result};
use opencv::{
    core::{self, BorderTypes, Mat, MatTraitConst, MatTraitConstManual, MatTraitManual, Size, Vec3b},
    imgproc, videoio,
};
use serde::Deserialize;
use std::{collections::HashMap, fs, path::Path};

use async_cuda::{DeviceBuffer, Stream};
use async_tensorrt::{engine::TensorIoMode, Engine, ExecutionContext, Runtime};

/// Konfigurasi dasar
const ENGINE_PATH: &str = "./models/best_fp16.engine";
const CLASSES_PATH: &str = "./models/classes.json";
const INPUT_SIZE: i32 = 640; // YOLO 640x640

#[tokio::main]
async fn main() -> Result<()> {
    // 1) Load classes
    let classes = load_classes(CLASSES_PATH)?;
    println!("Loaded {} classes: {:?}", classes.len(), classes);

    // 2) TensorRT Runtime + Engine + ExecutionContext
    let mut engine = load_engine(ENGINE_PATH).await?;
    let mut ctx = ExecutionContext::new(&mut engine)
        .await
        .context("create TRT execution context")?;
    let stream = Stream::new().await.context("create CUDA stream")?;

    // 3) Auto-detect input & output tensors
    let (input_name, output_name) = first_io_names(&engine)?;
    let input_shape = engine.tensor_shape(&input_name);
    let output_shape = engine.tensor_shape(&output_name);
    println!(
        "I/O => input '{}' {:?}, output '{}' {:?}",
        input_name, input_shape, output_name, output_shape
    );

    // 4) OpenCV capture dari kamera 0
    let mut cap =
        videoio::VideoCapture::new(0, videoio::CAP_ANY).context("open camera (index=0)")?;
    if !videoio::VideoCapture::is_opened(&cap)? {
        bail!("Camera 0 tidak bisa dibuka");
    }
    // optional: set resolusi capture (bisa dikomentari jika tak perlu)
    let _ = cap.set(videoio::CAP_PROP_FRAME_WIDTH, 1280.0);
    let _ = cap.set(videoio::CAP_PROP_FRAME_HEIGHT, 720.0);

    // 5) Loop ambil frame → preprocess → infer → parse → hitung
    loop {
        let mut frame = Mat::default();
        cap.read(&mut frame)?;
        if frame.empty() {
            eprintln!("Frame kosong; kemungkinan kamera putus. Keluar.");
            break;
        }

        // a) preprocess: BGR -> RGB letterbox 640, float32 CHW [0..1]
        let (input_tensor, ratio, pad_w, pad_h, orig_w, orig_h) = preprocess_to_tensor(&frame)?;

        // b) upload ke GPU
        let in_dev = DeviceBuffer::from_slice(&input_tensor)
            .context("copy H->D input tensor")?;
        // siapkan output buffer di device
        let out_elems: usize = output_shape.iter().product();
        let mut out_host = vec![0f32; out_elems];
        // alokasikan device buffer output (nol di-host lalu H->D supaya dapat ukuran)
        let mut out_dev = DeviceBuffer::from_slice(&out_host)
            .context("alloc D output buffer")?;

        // c) jalankan TRT
        //    Bangun map IO: nama tensor -> device buffer
        let mut io_map: HashMap<&str, &mut DeviceBuffer<f32>> =
            HashMap::from([(input_name.as_str(), unsafe {
                // safe: life time local
                std::mem::transmute::<&DeviceBuffer<f32>, &mut DeviceBuffer<f32>>(&in_dev)
            }), (output_name.as_str(), &mut out_dev)]);

        ctx.enqueue(&mut io_map, &stream)
            .await
            .context("enqueue TRT")?;

        // d) copy output ke host
        out_dev
            .copy_to(&mut out_host)
            .context("copy D->H output tensor")?;

        // e) parse deteksi YOLO & hitung kendaraan
        let dets = parse_yolo_output(
            &out_host,
            &output_shape,
            classes.len(),
            0.25, // conf threshold
            0.45, // nms iou
        )?;

        let (veh_count, per_class) = count_vehicles(&dets, &classes);

        println!(
            "count={}   per_class={:?}   (orig={}x{}, ratio={:.4}, padW={}, padH={})",
            veh_count, per_class, orig_w, orig_h, ratio, pad_w, pad_h
        );
    }

    Ok(())
}

/// ---- Classes JSON ----
#[derive(Deserialize)]
struct ClassList(Vec<String>);

fn load_classes(p: &str) -> Result<Vec<String>> {
    let txt = fs::read_to_string(p)
        .with_context(|| format!("read classes json at {}", p))?;
    // mendukung baik `["a","b"]` langsung, atau {"classes":[...]}
    if txt.trim_start().starts_with('[') {
        let v: Vec<String> = serde_json::from_str(&txt)?;
        Ok(v)
    } else {
        let v: ClassList = serde_json::from_str(&txt)?;
        Ok(v.0)
    }
}

/// ---- TensorRT helpers ----
async fn load_engine(p: &str) -> Result<Engine> {
    if !Path::new(p).exists() {
        bail!("Engine file tidak ditemukan: {}", p);
    }
    let plan = fs::read(p).with_context(|| format!("read engine {}", p))?;
    let rt = Runtime::new().await.context("create TRT runtime")?;
    let engine = rt
        .deserialize_engine(&plan)
        .await
        .context("deserialize engine")?;
    Ok(engine)
}

fn first_io_names(engine: &Engine) -> Result<(String, String)> {
    let n = engine.num_io_tensors();
    if n < 2 {
        bail!("Engine IO tensor kurang dari 2");
    }
    let mut inp = None;
    let mut out = None;
    for i in 0..n {
        let name = engine.io_tensor_name(i);
        match engine.tensor_io_mode(&name) {
            TensorIoMode::Input if inp.is_none() => inp = Some(name),
            TensorIoMode::Output if out.is_none() => out = Some(name),
            _ => {}
        }
    }
    match (inp, out) {
        (Some(i), Some(o)) => Ok((i, o)),
        _ => bail!("Tidak bisa menemukan input & output tensor name"),
    }
}

/// ---- Preprocess YOLO: BGR Mat -> RGB letterbox 640 -> CHW f32 normalized ----
#[allow(clippy::type_complexity)]
fn preprocess_to_tensor(
    bgr: &Mat,
) -> Result<(Vec<f32>, f32, i32, i32, i32, i32)> {
    let orig_w = bgr.cols();
    let orig_h = bgr.rows();

    // BGR -> RGB
    let mut rgb = Mat::default();
    imgproc::cvt_color(bgr, &mut rgb, imgproc::COLOR_BGR2RGB, 0)?;

    // letterbox ke 640 x 640
    let (resized, ratio, pad_w, pad_h) = letterbox(&rgb, INPUT_SIZE, INPUT_SIZE)?;

    // ke CHW f32 0..1
    let chw = mat_rgb_to_chw_f32(&resized)?;

    Ok((chw, ratio, pad_w, pad_h, orig_w, orig_h))
}

/// scale + pad agar aspect ratio sama, pakai BORDER_CONSTANT=114 (gaya YOLO)
fn letterbox(img_rgb: &Mat, dst_w: i32, dst_h: i32) -> Result<(Mat, f32, i32, i32)> {
    let w = img_rgb.cols() as f32;
    let h = img_rgb.rows() as f32;

    let r = (dst_w as f32 / w).min(dst_h as f32 / h);
    let new_w = (w * r).round() as i32;
    let new_h = (h * r).round() as i32;

    let mut resized = Mat::default();
    imgproc::resize(
        img_rgb,
        &mut resized,
        Size::new(new_w, new_h),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;

    let pad_w = dst_w - new_w;
    let pad_h = dst_h - new_h;
    let top = pad_h / 2;
    let bottom = pad_h - top;
    let left = pad_w / 2;
    let right = pad_w - left;

    let mut padded = Mat::default();
    // copyMakeBorder ada di modul 'core' pada crate opencv (bukan imgproc)
    // BorderTypes::BORDER_CONSTANT dengan nilai (114,114,114)
    core::copy_make_border(
        &resized,
        &mut padded,
        top,
        bottom,
        left,
        right,
        BorderTypes::BORDER_CONSTANT as i32,
        core::Scalar::new(114.0, 114.0, 114.0, 0.0),
    )?;

    Ok((padded, r, left, top))
}

/// Konversi Mat RGB uint8 HxWx3 -> Vec<f32> CHW 1x3xHxW (norm 0..1)
fn mat_rgb_to_chw_f32(img_rgb: &Mat) -> Result<Vec<f32>> {
    let h = img_rgb.rows();
    let w = img_rgb.cols();
    let c = 3usize;
    let mut out = vec![0f32; (h * w) as usize * c];

    // akses piksel
    for y in 0..h {
        for x in 0..w {
            let p: Vec3b = *img_rgb.at_2d::<Vec3b>(y, x)?;
            // Vec3b order = [R,G,B] karena sudah cvt COLOR_BGR2RGB
            let r = p[0] as f32 / 255.0;
            let g = p[1] as f32 / 255.0;
            let b = p[2] as f32 / 255.0;

            let idx_hw = (y * w + x) as usize;
            out[idx_hw] = r;
            out[idx_hw + (h * w) as usize] = g;
            out[idx_hw + 2 * (h * w) as usize] = b;
        }
    }
    Ok(out)
}

/// ---- Struktur deteksi sederhana ----
#[derive(Debug, Clone)]
struct Det {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    conf: f32,
    cls: usize,
}

/// Coba parse output YOLO dari TensorRT.
/// Mendukung dua pola umum:
/// 1) [1, N, 5+C]  (tiap row: [cx,cy,w,h,obj_conf, class_scores...])
/// 2) [1, (5+C), N] (ultralytics sering output [B, A, N] = [1, 4+1+C, N])
fn parse_yolo_output(
    out: &[f32],
    shape: &[usize],
    num_classes: usize,
    conf_thres: f32,
    iou_thres: f32,
) -> Result<Vec<Det>> {
    if shape.len() < 2 {
        bail!("Output shape tidak dikenali: {:?}", shape);
    }

    // Bentukkan (attrs, num)
    let (attrs, num) = if shape[1] == (5 + num_classes) {
        // [1, 5+C, N]
        ((5 + num_classes), shape[2])
    } else if shape.last() == Some(&((5 + num_classes) as usize)) {
        // [..., N, 5+C]
        ((5 + num_classes), shape[shape.len() - 2])
    } else {
        // fallback: coba deteksi otomatis
        let total = out.len();
        // cari N yang menghasilkan attrs=5+num_classes
        let ac = 5 + num_classes;
        if total % ac != 0 {
            bail!(
                "Tidak bisa faktorkan output (total {} tidak kelipatan attrs {})",
                total,
                ac
            );
        }
        (ac, total / ac)
    };

    // Baca sebagai daftar deteksi
    let mut dets = Vec::with_capacity(num);
    let stride = attrs;
    for i in 0..num {
        let base = i * stride;
        if base + stride > out.len() {
            break;
        }
        let cx = out[base + 0];
        let cy = out[base + 1];
        let w = out[base + 2];
        let h = out[base + 3];
        let obj_conf = out[base + 4];

        // cari kelas terbaik
        let mut best_cls = 0usize;
        let mut best_score = 0f32;
        for c in 0..num_classes {
            let s = out[base + 5 + c];
            if s > best_score {
                best_score = s;
                best_cls = c;
            }
        }
        let conf = obj_conf * best_score;
        if conf < conf_thres {
            continue;
        }
        let x1 = cx - w / 2.0;
        let y1 = cy - h / 2.0;
        let x2 = cx + w / 2.0;
        let y2 = cy + h / 2.0;
        dets.push(Det {
            x1,
            y1,
            x2,
            y2,
            conf,
            cls: best_cls,
        });
    }

    // NMS sederhana (greedy)
    dets.sort_by(|a, b| b.conf.total_cmp(&a.conf));
    let mut keep = Vec::<Det>::new();
    'outer: for d in &dets {
        for k in &keep {
            if iou(d, k) > iou_thres {
                continue 'outer;
            }
        }
        keep.push(d.clone());
    }
    Ok(keep)
}

fn iou(a: &Det, b: &Det) -> f32 {
    let x1 = a.x1.max(b.x1);
    let y1 = a.y1.max(b.y1);
    let x2 = a.x2.min(b.x2);
    let y2 = a.y2.min(b.y2);
    let inter = ((x2 - x1).max(0.0)) * ((y2 - y1).max(0.0));
    let area_a = (a.x2 - a.x1).max(0.0) * (a.y2 - a.y1).max(0.0);
    let area_b = (b.x2 - b.x1).max(0.0) * (b.y2 - b.y1).max(0.0);
    inter / (area_a + area_b - inter + 1e-6)
}

/// Hitung kendaraan yang termasuk target (cars/motorcyle/truck)
fn count_vehicles(dets: &[Det], classes: &[String]) -> (usize, HashMap<String, usize>) {
    let mut per_class: HashMap<String, usize> = HashMap::new();
    let mut total = 0usize;

    // target names (lowercase)
    let targets = ["cars", "motorcyle", "truck"];

    for d in dets {
        if let Some(name) = classes.get(d.cls) {
            let lname = name.to_lowercase();
            if targets.contains(&lname.as_str()) {
                *per_class.entry(name.clone()).or_insert(0) += 1;
                total += 1;
            }
        }
    }
    (total, per_class)
}
