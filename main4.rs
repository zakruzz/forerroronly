use anyhow::{anyhow, bail, Context, Result};
use opencv::{
    core::{self, Mat, MatTraitConst, MatTraitConstManual, Rect, Scalar, Size, Vector},
    dnn,
    imgcodecs,
    imgproc,
    prelude::*,
    types,
};
use serde_json::Value;
use std::{env, fs, path::PathBuf};

#[derive(Clone, Debug)]
struct Det {
    x1: f32, y1: f32, x2: f32, y2: f32,
    score: f32,
    cls: usize,
}

fn is_vehicle(name: &str) -> bool {
    let keys = [
        "car","bus","truck","motorcycle","motorbike","bicycle","bike",
        "van","pickup","trailer","truk","mobil","sepeda","motor",
    ];
    let n = name.to_lowercase();
    keys.iter().any(|k| n.contains(k))
}

/// Letterbox ala YOLO: resize dengan aspect ratio, pad warna (114,114,114)
fn letterbox_bgr(img: &Mat, dst: i32) -> Result<(Mat, f32, i32, i32)> {
    let w = img.cols();
    let h = img.rows();
    let s = (dst as f32 / w as f32).min(dst as f32 / h as f32);
    let nw = ((w as f32) * s).round() as i32;
    let nh = ((h as f32) * s).round() as i32;

    let mut resized = Mat::default();
    imgproc::resize(img, &mut resized, Size::new(nw, nh), 0.0, 0.0, imgproc::INTER_LINEAR)?;

    let mut canvas = Mat::default();
    canvas = Mat::new_rows_cols_with_default(dst, dst, core::CV_8UC3, Scalar::new(114.0,114.0,114.0,0.0))?;

    let dx = (dst - nw) / 2;
    let dy = (dst - nh) / 2;
    let roi = Rect::new(dx, dy, nw, nh);
    let mut target = core::Mat::roi(&canvas, roi)?;
    resized.copy_to(&mut target)?;

    Ok((canvas, s, dx, dy))
}

/// NMS sederhana per-kelas
fn iou(a: &Det, b: &Det) -> f32 {
    let x1 = a.x1.max(b.x1);
    let y1 = a.y1.max(b.y1);
    let x2 = a.x2.min(b.x2);
    let y2 = a.y2.min(b.y2);
    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area_a = (a.x2 - a.x1).max(0.0) * (a.y2 - a.y1).max(0.0);
    let area_b = (b.x2 - b.x1).max(0.0) * (b.y2 - b.y1).max(0.0);
    inter / (area_a + area_b - inter + 1e-6)
}
fn nms(mut v: Vec<Det>, iou_t: f32) -> Vec<Det> {
    v.sort_by(|a,b| b.score.partial_cmp(&a.score).unwrap());
    let mut keep = Vec::new();
    'outer: for d in v {
        for k in &keep {
            if d.cls == k.cls && iou(&d, k) > iou_t { continue 'outer; }
        }
        keep.push(d);
    }
    keep
}

/// Decode YOLOv8 TANPA objectness, output [1, (4+nc), N] (kita asumsikan begitu)
fn decode_yolov8_4plusnc(
    flat: &[f32], c: usize, n: usize, conf: f32,
    class_names: &[String],
    orig_w: i32, orig_h: i32,
    scale: f32, pad_x: i32, pad_y: i32, dst: i32,
    filter_vehicle: bool,
) -> Result<Vec<Det>> {
    let nc = c.checked_sub(4).ok_or_else(|| anyhow!("C<4"))?;
    if nc != class_names.len() {
        bail!("Mismatch jumlah kelas: C-4={} vs classes.json={}", nc, class_names.len());
    }
    // deteksi logits/prob (untuk amankan exporter yang keluarkan logits)
    let (mut mn, mut mx) = (f32::INFINITY, f32::NEG_INFINITY);
    for &v in flat { if v < mn { mn = v } if v > mx { mx = v } }
    let need_sigmoid = mn < -0.01 || mx > 1.01;
    let sig = |x: f32| 1.0 / (1.0 + (-x).exp());

    let stride = n;
    let mut out = Vec::new();
    for i in 0..n {
        let mut cx = flat[0*stride + i].clamp(0.0, 1.0);
        let mut cy = flat[1*stride + i].clamp(0.0, 1.0);
        let mut w  = flat[2*stride + i].clamp(0.0, 1.0);
        let mut h  = flat[3*stride + i].clamp(0.0, 1.0);

        // cari kelas terbaik
        let (mut best_c, mut best_p) = (0usize, f32::MIN);
        for cc in 0..nc {
            let mut p = flat[(4+cc)*stride + i];
            if need_sigmoid { p = sig(p); }
            if p > best_p { best_p = p; best_c = cc; }
        }
        if best_p < conf { continue; }

        if filter_vehicle {
            let cname = class_names.get(best_c).map(|s|s.as_str()).unwrap_or("");
            if !is_vehicle(cname) { continue; }
        }

        // balik dari kanvas dst×dst ke koordinat asli
        cx *= dst as f32; cy *= dst as f32; w *= dst as f32; h *= dst as f32;
        let (mut x1, mut y1, mut x2, mut y2) = (cx - w/2.0, cy - h/2.0, cx + w/2.0, cy + h/2.0);
        let fx  = (x1 - pad_x as f32).max(0.0);
        let fy  = (y1 - pad_y as f32).max(0.0);
        let fx2 = (x2 - pad_x as f32).max(0.0);
        let fy2 = (y2 - pad_y as f32).max(0.0);
        let inv = 1.0 / scale;
        x1 = (fx  * inv).clamp(0.0, (orig_w - 1) as f32);
        y1 = (fy  * inv).clamp(0.0, (orig_h - 1) as f32);
        x2 = (fx2 * inv).clamp(0.0, (orig_w - 1) as f32);
        y2 = (fy2 * inv).clamp(0.0, (orig_h - 1) as f32);

        out.push(Det { x1, y1, x2, y2, score: best_p, cls: best_c });
    }
    Ok(out)
}

fn draw_box(img: &mut Mat, d: &Det, label: &str) -> Result<()> {
    let p1 = core::Point::new(d.x1 as i32, d.y1 as i32);
    let p2 = core::Point::new(d.x2 as i32, d.y2 as i32);
    imgproc::rectangle(img, core::Rect::new(p1.x, p1.y, (p2.x - p1.x).max(1), (p2.y - p1.y).max(1)),
        Scalar::new(0.0,255.0,0.0,0.0), 2, imgproc::LINE_8, 0)?;
    let text = format!("{} {:.2}", label, d.score);
    imgproc::put_text(img, &text, p1, imgproc::FONT_HERSHEY_SIMPLEX,
        0.5, Scalar::new(0.0,255.0,0.0,0.0), 1, imgproc::LINE_8, false)?;
    Ok(())
}

fn main() -> Result<()> {
    // Usage simpel:
    //   yolo_opencv_simple <onnx> <classes.json> <image> [imgsz=640] [conf=0.25] [iou=0.45] [filter_vehicle=1] [save=result.jpg]
    let a: Vec<String> = env::args().collect();
    if a.len() < 4 {
        eprintln!("Usage: {} <best.onnx> <classes.json> <image> [imgsz=640] [conf=0.25] [iou=0.45] [filter_vehicle=1] [save=result.jpg]", a[0]);
        std::process::exit(1);
    }
    let onnx = PathBuf::from(&a[1]);
    let classes_path = PathBuf::from(&a[2]);
    let image_path = PathBuf::from(&a[3]);
    let imgsz: i32 = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(640);
    let conf: f32  = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(0.25);
    let iou_t: f32 = a.get(6).and_then(|s| s.parse().ok()).unwrap_or(0.45);
    let filter_vehicle: bool = a.get(7).map(|s| s=="1" || s.to_lowercase()=="true").unwrap_or(true);
    let save = PathBuf::from(a.get(8).cloned().unwrap_or_else(|| "result.jpg".to_string()));

    // load classes.json (array string)
    let txt = fs::read_to_string(&classes_path).context("baca classes.json")?;
    let j: Value = serde_json::from_str(&txt).context("parse classes.json")?;
    let arr = j.as_array().ok_or_else(|| anyhow!("classes.json harus array string"))?;
    let class_names: Vec<String> = arr.iter().map(|v| v.as_str().unwrap_or("").to_string()).collect();
    if class_names.is_empty() { bail!("classes.json kosong"); }
    let c = 4 + class_names.len(); // 4 bbox + nc

    // load image (BGR)
    let mut img = imgcodecs::imread(image_path.to_str().unwrap(), imgcodecs::IMREAD_COLOR)
        .with_context(|| format!("open {:?}", image_path))?;
    if img.empty() { bail!("gagal baca gambar"); }
    let orig_w = img.cols();
    let orig_h = img.rows();

    // letterbox ke kanvas imgsz×imgsz
    let (letter, scale, pad_x, pad_y) = letterbox_bgr(&img, imgsz)?;

    // blob (NCHW, float32, 1/255, swapRB= true)
    let mut blob = dnn::blob_from_image(
        &letter, 1.0/255.0, Size::new(imgsz, imgsz),
        Scalar::default(), true, false, core::CV_32F
    )?;

    // baca onnx → Net
    let mut net = dnn::read_net_from_onnx(onnx.to_str().unwrap())
        .with_context(|| format!("read onnx {:?}", onnx))?;

    // coba pakai CUDA; kalau error → fallback ke CPU
    let mut used_cuda = false;
    if let Err(e) = net.set_preferable_backend(dnn::DNN_BACKEND_CUDA) {
        eprintln!("[warn] CUDA backend not available: {e}");
    } else if let Err(e) = net.set_preferable_target(dnn::DNN_TARGET_CUDA_FP16) {
        eprintln!("[warn] CUDA target not available: {e}");
    } else {
        used_cuda = true;
    }
    if !used_cuda {
        net.set_preferable_backend(dnn::DNN_BACKEND_OPENCV)?;
        net.set_preferable_target(dnn::DNN_TARGET_CPU)?;
    }

    // infer
    net.set_input(&blob, "", 1.0, Scalar::default())?;
    // gunakan API multiple outputs agar kompatibel
    let names = net.get_unconnected_out_layers_names()?;
    let mut outs = types::VectorOfMat::new();
    net.forward(&mut outs, &names)?;
    if outs.len() == 0 {
        bail!("tidak ada output dari model");
    }
    let out = outs.get(0)?; // ambil output pertama
    let total = out.total()? as usize;
    // asumsikan layout [1, C, N] → total = C*N
    if total % c != 0 {
        bail!("output total elemen {} tidak habis dibagi C={}", total, c);
    }
    let n = total / c;

    // ambil pointer data sebagai &[f32]
    let flat: &[f32] = unsafe { out.data_typed()? };

    // decode → nms
    let mut dets = decode_yolov8_4plusnc(
        flat, c, n, conf, &class_names,
        orig_w, orig_h, scale, pad_x, pad_y, imgsz,
        filter_vehicle,
    )?;
    dets = nms(dets, iou_t);
    println!("count: {}  (backend: {})", dets.len(), if used_cuda {"CUDA"} else {"CPU"});

    // gambar & simpan
    for d in &dets {
        let label = class_names.get(d.cls).map(|s| s.as_str()).unwrap_or("obj");
        draw_box(&mut img, d, label)?;
    }
    imgcodecs::imwrite(save.to_str().unwrap(), &img, &Vector::new())?;
    println!("saved: {:?}", save);
    Ok(())
}
