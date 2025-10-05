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

fn main() -> Result<()> {
    let o = Opts::parse();
    let names = load_classes_json("config/classes.json")?;
    let class_idxs = parse_count_arg(&o.count_classes, &names);
    eprintln!("[INFO] hitung kelas idx: {:?} ({} labels total)", class_idxs, names.len());

    // kamera
    let mut cam = videoio::VideoCapture::new(o.cam, videoio::CAP_ANY)?;
    ensure!(cam.is_opened()?, "kamera {} gagal dibuka", o.cam);

    // TRT session
    let mut sess = trt::TrtSession::from_engine_file(&o.engine)?;
    eprintln!("[INFO] engine loaded: {}", o.engine);

    let mut frame = Mat::default();
    loop {
        if !cam.read(&mut frame)? || frame.empty()? {
            eprintln!("no frame, exit"); break;
        }

        // prepro → [1,3,640,640] f32
        let input = letterbox_bgr_to_rgb_f32_nchw(&frame, o.size)?;

        // infer
        let out = sess.infer(&input)?;           // ndarray f32 output

        // decode + NMS
        let dets = decode_flexible(&out, o.conf_th, &class_idxs)?;
        let kept = nms(dets, o.iou_th);

        println!("Kendaraan terdeteksi (frame): {}", kept.len());
    }
}
