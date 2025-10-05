use anyhow::{anyhow, bail, Context, Result};
use opencv::{
    core::{self, Mat, MatTrait, MatTraitConst, MatTraitConstManual, Scalar, Size, Vector},
    dnn, imgcodecs, imgproc, prelude::*,
};
use serde_json::Value;
use std::{env, fs, path::PathBuf};

#[derive(Clone, Debug)]
struct Det { x1:f32, y1:f32, x2:f32, y2:f32, score:f32, cls:usize }

fn letterbox_bgr(img:&Mat, dst:i32)->Result<(Mat, f32, i32, i32)>{
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
    let ab=(b.x2-b.x1).max(0.0)*(b.y2-b.y1).max(0.0);
    inter/(aa+ab-inter+1e-6)
}
fn nms(mut v:Vec<Det>, thr:f32)->Vec<Det>{
    v.sort_by(|a,b| b.score.partial_cmp(&a.score).unwrap());
    let mut keep:Vec<Det>=Vec::new();
    'outer: for d in v {
        for k in &keep { if d.cls==k.cls && iou(&d,k)>thr { continue 'outer; } }
        keep.push(d);
    }
    keep
}
fn quality_score(d:&[Det], w:i32,h:i32)->(usize,f32){
    let mut valid=0usize; let mut sum=0f32;
    for b in d {
        if b.x1>=0.0 && b.y1>=0.0 && b.x2<=w as f32 && b.y2<=h as f32 &&
           (b.x2-b.x1)>2.0 && (b.y2-b.y1)>2.0 { valid+=1; sum+=b.score; }
    }
    let avg= if valid==0 {0.0} else {sum/valid as f32};
    (valid, avg)
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

#[inline] fn sigmoid(x:f32)->f32 { 1.0/(1.0+(-x).exp()) }

/// Decode 1 tampilan matrix (rows=c, cols=n) dengan asumsi bbox=XYWH (umum YOLOv8).
/// Otomatis deteksi: normalized (0..1) vs pixel (0..imgsz).
fn decode_matrix_cxn(
    m:&Mat, c:usize, n:usize, conf:f32, class_names:&[String],
    orig_w:i32, orig_h:i32, scale:f32, pad_x:i32, pad_y:i32, imgsz:i32
)->Result<Vec<Det>>{
    // ambil pointer tiap "row"
    let row = |r:usize| -> Result<&[f32]> {
        let rmat = m.row(r as i32)?;
        Ok(rmat.data_typed()?)
    };

    let x = row(0)?; let y = row(1)?; let w = row(2)?; let h = row(3)?;
    // deteksi skala bbox
    let mut mx = 0f32;
    for i in 0..n { mx = mx.max(x[i].abs().max(y[i].abs()).max(w[i].abs()).max(h[i].abs())); }
    let in_pixels = mx > 1.5;

    let nc = c - 4;
    // siapkan view kelas
    let cls_rows: Vec<&[f32]> = (0..nc).map(|k| row(4+k).unwrap()).collect();

    let mut out=Vec::new();
    for i in 0..n {
        // bbox XYWH → XYXY di kanvas
        let mut cx=x[i]; let mut cy=y[i]; let mut ww=w[i]; let mut hh=h[i];
        if !in_pixels { cx*=imgsz as f32; cy*=imgsz as f32; ww*=imgsz as f32; hh*=imgsz as f32; }
        let mut x1=cx-ww/2.0; let mut y1=cy-hh/2.0;
        let mut x2=cx+ww/2.0; let mut y2=cy+hh/2.0;

        // pilih kelas terbaik (anggap logits → sigmoid)
        let mut best_c=0usize; let mut best_p=f32::MIN;
        for cix in 0..nc {
            let p = sigmoid(cls_rows[cix][i]);
            if p>best_p { best_p=p; best_c=cix; }
        }
        if best_p < conf { continue; }

        // map balik letterbox → gambar asli
        let inv = 1.0/scale;
        x1=((x1 - pad_x as f32).max(0.0) * inv).clamp(0.0,(orig_w-1) as f32);
        y1=((y1 - pad_y as f32).max(0.0) * inv).clamp(0.0,(orig_h-1) as f32);
        x2=((x2 - pad_x as f32).max(0.0) * inv).clamp(0.0,(orig_w-1) as f32);
        y2=((y2 - pad_y as f32).max(0.0) * inv).clamp(0.0,(orig_h-1) as f32);

        if (x2-x1)>=2.0 && (y2-y1)>=2.0 {
            out.push(Det{x1,y1,x2,y2,score:best_p,cls:best_c});
        }
    }
    Ok(out)
}

fn main()->Result<()>{
    // cargo run --release -- <best.onnx> <classes.json> <image> [imgsz=640] [conf=0.25] [iou=0.45] [save=result.png]
    let a:Vec<String>=env::args().collect();
    if a.len()<4{
        eprintln!("Usage: {} <best.onnx> <classes.json> <image> [imgsz=640] [conf=0.25] [iou=0.45] [save=result.png]", a[0]);
        std::process::exit(1);
    }
    let onnx=PathBuf::from(&a[1]);
    let classes_path=PathBuf::from(&a[2]);
    let image_path=PathBuf::from(&a[3]);
    let imgsz:i32=a.get(4).and_then(|s|s.parse().ok()).unwrap_or(640);
    let conf:f32=a.get(5).and_then(|s|s.parse().ok()).unwrap_or(0.25);
    let iou_t:f32=a.get(6).and_then(|s|s.parse().ok()).unwrap_or(0.45);
    let save=PathBuf::from(a.get(7).cloned().unwrap_or_else(||"result.png".to_string()));

    // classes.json: ["License_Plate","cars","motorcyle","truck"]
    let txt=fs::read_to_string(&classes_path).context("read classes.json")?;
    let j:Value=serde_json::from_str(&txt).context("parse classes.json")?;
    let arr=j.as_array().ok_or_else(||anyhow!("classes.json harus array string"))?;
    let class_names:Vec<String>=arr.iter().map(|v| v.as_str().unwrap_or("").to_string()).collect();
    if class_names.len()!=4 { bail!("classes.json harus 4 item (License_Plate,cars,motorcyle,truck)"); }
    let c = 4 + class_names.len(); // 8

    // gambar
    let mut img=imgcodecs::imread(image_path.to_str().unwrap(), imgcodecs::IMREAD_COLOR)
        .with_context(||format!("open {:?}", image_path))?;
    if img.empty(){ bail!("gagal baca gambar"); }
    let (orig_w,orig_h)=(img.cols(), img.rows());

    // letterbox + blob
    let (letter, scale, pad_x, pad_y)=letterbox_bgr(&img, imgsz)?;
    let blob=dnn::blob_from_image(&letter, 1.0/255.0, Size::new(imgsz,imgsz),
                                  Scalar::default(), true,false, core::CV_32F)?;

    // load onnx (prefer CUDA bila tersedia)
    let mut net=dnn::read_net_from_onnx(onnx.to_str().unwrap())
        .with_context(||format!("read onnx {:?}", onnx))?;
    if net.set_preferable_backend(dnn::DNN_BACKEND_CUDA).is_ok() &&
       net.set_preferable_target(dnn::DNN_TARGET_CUDA_FP16).is_ok() {
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
    let out=outs.get(0)?; // tensor utama

    // buffer
    let total = out.total() as usize;
    let flat:&[f32]=out.data_typed()?;
    if total % c != 0 { bail!("elemen output {} tidak habis dibagi C={}", total, c); }
    let n = total / c;

    // === KUNCI: reshape 2 tampilan ===
    // Tampilan-1: rows=C, cols=N  (C×N)
    let m_cxn = out.reshape(1, c as i32)?;
    if m_cxn.cols() != (n as i32) { bail!("reshape C×N gagal"); }
    let cand1 = decode_matrix_cxn(&m_cxn, c, n, conf, &class_names, orig_w,orig_h,scale,pad_x,pad_y,imgsz)?;

    // Tampilan-2: rows=N, cols=C  (N×C)
    // Kita reshape ulang dari buffer awal supaya tampilan benar.
    // Caranya: buat Mat baru dari slice flatten lalu reshape rows=N.
    let m_all = Mat::from_slice(flat)?;               // 1 x total
    let m_nxc = m_all.reshape(1, n as i32)?;          // rows=N, cols=C
    if m_nxc.cols() != (c as i32) { bail!("reshape N×C gagal"); }
    // agar interface decode mirip C×N, kita transpose jadi C×N
    let mut m_nxc_t = Mat::default();
    core::transpose(&m_nxc, &mut m_nxc_t)?;
    let cand2 = decode_matrix_cxn(&m_nxc_t, c, n, conf, &class_names, orig_w,orig_h,scale,pad_x,pad_y,imgsz)?;

    // pilih terbaik
    let (v1,a1) = quality_score(&cand1, orig_w, orig_h);
    let (v2,a2) = quality_score(&cand2, orig_w, orig_h);
    let mut dets = if v2>v1 || (v1==v2 && a2>a1) { cand2 } else { cand1 };

    // NMS & gambar SEMUA kelas (termasuk License_Plate)
    dets = nms(dets, iou_t);
    for d in &dets {
        let label = class_names.get(d.cls).map(|s| s.as_str()).unwrap_or("obj");
        draw_box(&mut img, d, label)?;
    }

    imgcodecs::imwrite(save.to_str().unwrap(), &img, &Vector::new())?;
    println!("saved: {:?}", save);
    Ok(())
}
