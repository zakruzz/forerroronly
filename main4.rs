use anyhow::{anyhow, bail, Context, Result};
use opencv::{
    core::{self, Mat, MatTraitConst, MatTraitConstManual, Scalar, Size, Vector},
    dnn, imgcodecs, imgproc, prelude::*,
};
use serde_json::Value;
use std::{env, fs, path::PathBuf};

/// =====================
/// KONFIGURASI FORMAT OUTPUT MODEL
/// =====================
/// Jika ONNX-mu berbeda, ubah konstanta ini saja.
/// - Jika layout = [1, N, 8] → set NX_C = true
/// - Jika bbox = [x1,y1,x2,y2] → set BBOX_XYWH = false
/// - Jika koordinat sudah piksel (bukan 0..1) → set NORM_COORD = false
const IMG_SIZE: i32 = 640;
const C_EXPECT: usize = 8;      // 4 bbox + jumlah_kelas (4) = 8
const NX_C: bool = false;       // false: [1, C, N] (default Ultralytics ONNX non-end2end)
const BBOX_XYWH: bool = true;   // true: [cx,cy,w,h], false: [x1,y1,x2,y2]
const NORM_COORD: bool = true;  // true: koordinat ter-normalisasi 0..1 terhadap IMG_SIZE
const APPLY_SIGMOID: bool = true; // true: apply sigmoid ke skor kelas (umum untuk ONNX)

#[derive(Clone, Debug)]
struct Det { x1:f32, y1:f32, x2:f32, y2:f32, score:f32, cls:usize }

fn is_vehicle_label(label:&str)->bool {
    matches!(label, "cars" | "motorcyle" | "truck")
}

/// Letterbox tanpa ROI: resize + padding ke kanvas dst×dst
fn letterbox_bgr(img:&Mat, dst:i32)->Result<(Mat,f32,i32,i32)>{
    let w=img.cols(); let h=img.rows();
    let s=(dst as f32 / w as f32).min(dst as f32 / h as f32);
    let nw=((w as f32)*s).round() as i32; let nh=((h as f32)*s).round() as i32;

    let mut resized=Mat::default();
    imgproc::resize(img, &mut resized, Size::new(nw,nh), 0.0,0.0, imgproc::INTER_LINEAR)?;

    let left=(dst-nw)/2; let top=(dst-nh)/2;
    let right=dst-nw-left; let bottom=dst-nh-top;

    let mut canvas=Mat::default();
    core::copy_make_border(&resized,&mut canvas, top,bottom,left,right,
        core::BORDER_CONSTANT, Scalar::new(114.0,114.0,114.0,0.0))?;
    Ok((canvas, s, left, top))
}

fn iou(a:&Det,b:&Det)->f32{
    let x1=a.x1.max(b.x1); let y1=a.y1.max(b.y1);
    let x2=a.x2.min(b.x2); let y2=a.y2.min(b.y2);
    let inter=(x2-x1).max(0.0)*(y2-y1).max(0.0);
    let aa=(a.x2-a.x1).max(0.0)*(a.y2-a.y1).max(0.0);
    let ab=(b.x2-b.x1).max(0.0)*(b.y2-b_y1()).max(0.0);
    inter/(aa+ab-inter+1e-6)
}
trait BoxH { fn b_y1(&self)->f32; }
impl BoxH for Det { fn b_y1(&self)->f32 { self.y1 } } // helper
fn nms(mut v:Vec<Det>, thr:f32)->Vec<Det>{
    v.sort_by(|a,b| b.score.partial_cmp(&a.score).unwrap());
    let mut keep:Vec<Det>=Vec::new();
    'outer: for d in v {
        for k in &keep { if d.cls==k.cls && iou(&d,k)>thr { continue 'outer; } }
        keep.push(d);
    }
    keep
}

fn draw_box(img:&mut Mat, d:&Det, label:&str)->Result<()>{
    let p1=opencv::core::Point::new(d.x1 as i32, d.y1 as i32);
    let p2=opencv::core::Point::new(d.x2 as i32, d.y2 as i32);
    imgproc::rectangle(img, opencv::core::Rect::new(p1.x,p1.y,(p2.x-p1.x).max(1),(p2.y-p1.y).max(1)),
        Scalar::new(0.0,255.0,0.0,0.0), 2, imgproc::LINE_8, 0)?;
    let text=format!("{} {:.2}", label, d.score);
    imgproc::put_text(img,&text,p1,imgproc::FONT_HERSHEY_SIMPLEX,0.5,
        Scalar::new(0.0,255.0,0.0,0.0),1,imgproc::LINE_8,false)?;
    Ok(())
}

fn main()->Result<()>{
    // cargo run --release -- <best.onnx> <classes.json> <image> [conf=0.25] [iou=0.45] [save=result.jpg]
    let a:Vec<String>=env::args().collect();
    if a.len()<4{
        eprintln!("Usage: {} <best.onnx> <classes.json> <image> [conf=0.25] [iou=0.45] [save=result.jpg]", a[0]);
        std::process::exit(1);
    }
    let onnx=PathBuf::from(&a[1]);
    let classes_path=PathBuf::from(&a[2]);
    let image_path=PathBuf::from(&a[3]);
    let conf:f32=a.get(4).and_then(|s|s.parse().ok()).unwrap_or(0.25);
    let iou_t:f32=a.get(5).and_then(|s|s.parse().ok()).unwrap_or(0.45);
    let save=PathBuf::from(a.get(6).cloned().unwrap_or_else(||"result.jpg".to_string()));

    // classes.json: ["License_Plate","cars","motorcyle","truck"]
    let txt=fs::read_to_string(&classes_path).context("read classes.json")?;
    let j:Value=serde_json::from_str(&txt).context("parse classes.json")?;
    let arr=j.as_array().ok_or_else(||anyhow!("classes.json harus array string"))?;
    let class_names:Vec<String>=arr.iter().map(|v| v.as_str().unwrap_or("").to_string()).collect();
    if class_names.len()!=4 { bail!("classes.json harus 4 item (License_Plate,cars,motorcyle,truck)"); }

    // gambar
    let mut img=imgcodecs::imread(image_path.to_str().unwrap(), imgcodecs::IMREAD_COLOR)
        .with_context(||format!("open {:?}", image_path))?;
    if img.empty(){ bail!("gagal baca gambar"); }
    let (orig_w,orig_h)=(img.cols(), img.rows());

    // letterbox → blob
    let (letter, scale, pad_x, pad_y)=letterbox_bgr(&img, IMG_SIZE)?;
    let blob=dnn::blob_from_image(&letter, 1.0/255.0, Size::new(IMG_SIZE,IMG_SIZE),
                                  Scalar::default(), true,false, core::CV_32F)?;

    // load onnx (prefer CUDA Jetson)
    let mut net=dnn::read_net_from_onnx(onnx.to_str().unwrap())
        .with_context(||format!("read onnx {:?}", onnx))?;
    if net.set_preferable_backend(dnn::DNN_BACKEND_CUDA).is_ok()
        && net.set_preferable_target(dnn::DNN_TARGET_CUDA_FP16).is_ok() {
        eprintln!("[info] DNN backend: CUDA");
    } else {
        net.set_preferable_backend(dnn::DNN_BACKEND_OPENCV)?;
        net.set_preferable_target(dnn::DNN_TARGET_CPU)?;
        eprintln!("[info] DNN backend: CPU");
    }
    net.set_input(&blob, "", 1.0, Scalar::default())?;

    // forward
    let names=net.get_unconnected_out_layers_names()?;
    let mut outs:Vector<Mat>=Vector::new();
    net.forward(&mut outs, &names)?;
    if outs.len()==0 { bail!("model tidak mengembalikan output"); }
    let out=outs.get(0)?;

    // validasi bentuk & ambil buffer
    let total = out.total() as usize;
    if total % C_EXPECT != 0 { bail!("elemen output {total} tidak habis dibagi C={C_EXPECT}"); }
    let n = total / C_EXPECT;
    let flat:&[f32]=out.data_typed()?;

    let sig = |x:f32| 1.0/(1.0+(-x).exp());

    // decode SEDERHANA sesuai konstanta di atas
    let mut dets:Vec<Det>=Vec::new();
    for i in 0..n {
        // ambil channel per anchor
        let get = |ch:usize, idx:usize| -> f32 {
            if NX_C { flat[idx*C_EXPECT + ch] } else { flat[ch*n + idx] }
        };

        // bbox
        let (mut x1,mut y1,mut x2,mut y2) = if BBOX_XYWH {
            let mut cx=get(0,i); let mut cy=get(1,i);
            let mut w =get(2,i); let mut h =get(3,i);
            if NORM_COORD { cx*=IMG_SIZE as f32; cy*=IMG_SIZE as f32; w*=IMG_SIZE as f32; h*=IMG_SIZE as f32; }
            (cx - w/2.0, cy - h/2.0, cx + w/2.0, cy + h/2.0)
        } else {
            let mut ax=get(0,i); let mut ay=get(1,i);
            let mut bx=get(2,i); let mut by=get(3,i);
            if NORM_COORD { ax*=IMG_SIZE as f32; ay*=IMG_SIZE as f32; bx*=IMG_SIZE as f32; by*=IMG_SIZE as f32; }
            (ax, ay, bx, by)
        };

        // kelas terbaik (4 kelas)
        let mut best_c=0usize; let mut best_p=f32::MIN;
        for cc in 0..4 {
            let mut p=get(4+cc,i);
            if APPLY_SIGMOID { p = sig(p); }
            if p>best_p { best_p=p; best_c=cc; }
        }
        if best_p < conf { continue; }

        // mapping dari kanvas (letterbox) → koordinat gambar asli
        let fx  = (x1 - pad_x as f32).max(0.0);
        let fy  = (y1 - pad_y as f32).max(0.0);
        let fx2 = (x2 - pad_x as f32).max(0.0);
        let fy2 = (y2 - pad_y as f32).max(0.0);
        let inv = 1.0 / scale;
        x1=(fx*inv).clamp(0.0,(orig_w-1) as f32);
        y1=(fy*inv).clamp(0.0,(orig_h-1) as f32);
        x2=(fx2*inv).clamp(0.0,(orig_w-1) as f32);
        y2=(fy2*inv).clamp(0.0,(orig_h-1) as f32);
        if (x2-x1)<2.0 || (y2-y1)<2.0 { continue; }

        dets.push(Det{x1,y1,x2,y2,score:best_p,cls:best_c});
    }

    // NMS standar supaya tidak duplikat
    let dets = nms(dets, iou_t);

    // gambar hanya kendaraan
    let mut count=0usize;
    for d in &dets {
        let label = class_names[d.cls].as_str();
        if is_vehicle_label(label) {
            count += 1;
            draw_box(&mut img, d, label)?;
        }
    }
    println!("vehicle_count: {}", count);
    imgcodecs::imwrite(save.to_str().unwrap(), &img, &Vector::new())?;
    println!("saved: {:?}", save);

    Ok(())
}
