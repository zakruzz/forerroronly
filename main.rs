mod config;
mod preprocess;
mod yolo;
mod trt;

use anyhow::*;
use clap::Parser;
use opencv::{prelude::*, videoio};
use config::{load_classes_json, parse_count_arg};
use preprocess::letterbox_bgr_to_rgb_f32_nchw;
use yolo::{decode_flexible, nms};

#[derive(Parser, Debug)]
#[command(version, about="Full-Rust Vehicle Counter (TensorRT)")]
struct Opts {
    #[arg(long, default_value = "models/best_fp16.engine")]
    engine: String,
    #[arg(long, default_value_t = 0)]
    cam: i32,
    #[arg(long, default_value_t = 640)]
    size: i32,
    #[arg(long, default_value_t = 0.25)]
    conf_th: f32,
    #[arg(long, default_value_t = 0.45)]
    iou_th: f32,
    /// angka atau nama kelas, pisahkan koma. contoh: "car,motorcycle,bus,truck" atau "2,3,5,7"
    #[arg(long, default_value = "car,motorcycle,bus,truck")]
    count_classes: String,
}

fn main() -> anyhow::Result<()> {
    let o = Opts::parse();

    let names = config::load_classes_json("config/classes.json")?;
    let class_idxs = config::parse_count_arg(&o.count_classes, &names);
    eprintln!("[INFO] hitung kelas idx: {:?} ({} labels total)", class_idxs, names.len());

    // Kamera
    let mut cam = videoio::VideoCapture::new(o.cam, videoio::CAP_ANY)?;
    // di build kamu, is_opened() mengembalikan bool → JANGAN pakai ?
    if !cam.is_opened() {
        anyhow::bail!("kamera {} gagal dibuka", o.cam);
    }

    // TRT session (sementara pakai stub agar compile mulus)
    let mut sess = trt::TrtSession::new_stub(o.size as usize)?;

    let mut frame = Mat::default();
    loop {
        let ok = cam.read(&mut frame)?;
        if !ok || frame.empty()? {
            eprintln!("no frame, exit");
            break;
        }

        let input = preprocess::letterbox_bgr_to_rgb_f32_nchw(&frame, o.size)?;
        let out = sess.infer(&input)?;                // ndarray f32 (stub: kosong)
        let dets = yolo::decode_flexible(&out, o.conf_th, &class_idxs)?;
        let kept = yolo::nms(dets, o.iou_th);

        println!("Kendaraan terdeteksi (frame): {}", kept.len());
    }

    Ok(())
}
