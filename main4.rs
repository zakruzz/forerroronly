use anyhow::{anyhow, bail, Context, Result};
use opencv::{
    core::{self, Mat, MatTraitConst, MatTraitConstManual, Scalar, Size, Vector},
    dnn, imgcodecs, imgproc, prelude::*,
};
use serde_json::Value;
use std::{env, fs, path::PathBuf};

#[derive(Clone, Debug)]
struct Det { x1:f32, y1:f32, x2:f32, y2:f32, score:f32, cls:usize }

fn is_vehicle_label(label:&str)->bool {
    matches!(label, "cars" | "motorcyle" | "truck")
}

/// Letterbox tanpa ROI: resize + padding ke kanvas dst×dst
fn letterbox_bgr(img:&Mat, dst:i32)->Result<(Mat, f32, i32, i32)>{
    let w = img.cols();
    let h = img.rows();
    let s = (dst as f32 / w as f32).min(dst as f32 / h as f32);
    let nw = ((w as f32) * s).round() as i32;
    let nh = ((h as f32) * s).round() as i32;

    let mut resized = Mat::default();
    imgproc::resize(img, &mut resized, Size::new(nw, nh), 0.0, 0.0, imgproc::INTER_LINEAR)?;

    let left = (dst - nw)/2;
    let top  = (dst - nh)/2;
    let right = dst - nw - left;
    let bottom = dst - nh - top;

    let mut canvas = Mat::default();
    core::copy_make_border(
        &resized, &mut canvas,
        top, bottom, left, right,
        core::BORDER_CONSTANT,
        Scalar::new(114.0,114.0,114.0,0.0),
    )?;
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
           (b.x2-b.x1)>2.0 && (b.y2-b.y1)>2.0 {
            valid+=1; sum+=b.score;
        }
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

fn main()->Result<()>{
    // cargo run --release -- <best.onnx> <classes.json> <image> [imgsz=640] [conf=0.25] [iou=0.45] [save=result.jpg]
    let a:Vec<String>=env::args().collect();
    if a.len()<4{
        eprintln!("Usage: {} <best.onnx> <classes.json> <image> [imgsz=640] [conf=0.25] [iou=0.45] [save=result.jpg]", a[0]);
        std::process::exit(1);
    }
    let onnx=PathBuf::from(&a[1]);
    let classes_path=PathBuf::from(&a[2]);
    let image_path=PathBuf::from(&a[3]);
    let imgsz:i32=a.get(4).and_then(|s|s.parse().ok()).unwrap_or(640);
    let conf:f32=a.get(5).and_then(|s|s.parse().ok()).unwrap_or(0.25);
    let iou_t:f32=a.get(6).and_then(|s|s.parse().ok()).unwrap_or(0.45);
    let save=PathBuf::from(a.get(7).cloned().unwrap_or_else(||"result.jpg".to_string()));

    // classes.json (harus 4: ["License_Plate","cars","motorcyle","truck"])
    let txt=fs::read_to_string(&classes_path).context("read classes.json")?;
    let j:Value=serde_json::from_str(&txt).context("parse classes.json")?;
    let arr=j.as_array().ok_or_else(||anyhow!("classes.json harus array string"))?;
    let class_names:Vec<String>=arr.iter().map(|v| v.as_str().unwrap_or("").to_string()).collect();
    if class_names.len()!=4 { bail!("classes.json harus berisi 4 nama kelas persis (License_Plate, cars, motorcyle, truck)"); }
    let c_expect = 4 + class_names.len(); // 8

    // baca gambar
    let mut img=imgcodecs::imread(image_path.to_str().unwrap(), imgcodecs::IMREAD_COLOR)
        .with_context(||format!("open {:?}", image_path))?;
    if img.empty(){ bail!("gagal baca gambar"); }
    let (orig_w,orig_h)=(img.cols(), img.rows());

    // letterbox + blob
    let (letter, scale, pad_x, pad_y)=letterbox_bgr(&img, imgsz)?;
    let blob=dnn::blob_from_image(&letter, 1.0/255.0, Size::new(imgsz,imgsz),
                                  Scalar::default(), true,false, core::CV_32F)?;

    // load model
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
    let out=outs.get(0)?; // ambil tensor utama

    // buffer output
    let total = out.total() as usize;
    let flat:&[f32]=out.data_typed()?;
    if total % c_expect != 0 { bail!("elemen output {} tidak habis dibagi C(exp)={}", total, c_expect); }
    let n = total / c_expect;

    // helper mapping dari kanvas -> gambar asli
    let map_to_orig = |x1:f32,y1:f32,x2:f32,y2:f32, use_letterbox:bool| -> (f32,f32,f32,f32) {
        if use_letterbox {
            let fx  = (x1 - pad_x as f32).max(0.0);
            let fy  = (y1 - pad_y as f32).max(0.0);
            let fx2 = (x2 - pad_x as f32).max(0.0);
            let fy2 = (y2 - pad_y as f32).max(0.0);
            let inv = 1.0 / scale;
            (
                (fx  * inv).clamp(0.0, (orig_w - 1) as f32),
                (fy  * inv).clamp(0.0, (orig_h - 1) as f32),
                (fx2 * inv).clamp(0.0, (orig_w - 1) as f32),
                (fy2 * inv).clamp(0.0, (orig_h - 1) as f32),
            )
        } else {
            (
                x1.clamp(0.0, (orig_w - 1) as f32),
                y1.clamp(0.0, (orig_h - 1) as f32),
                x2.clamp(0.0, (orig_w - 1) as f32),
                y2.clamp(0.0, (orig_h - 1) as f32),
            )
        }
    };

    // generator kandidat (getter, xywh/xyxy, with/without letterbox mapping)
    let make_cand = |getter: &dyn Fn(usize,usize)->f32, xyxy_mode: bool, use_letterbox: bool| -> Result<Vec<Det>> {
        // deteksi skala bbox: normalized vs pixel
        let mut bbox_max=0.0f32;
        for i in 0..n { for ch in 0..4 { bbox_max=bbox_max.max(getter(ch,i).abs()); } }
        let bbox_in_pixels = bbox_max > 1.5;

        // deteksi logits / prob
        let mut mn=f32::INFINITY; let mut mx=f32::NEG_INFINITY;
        for i in 0..n { for ch in 4..c_expect { let v=getter(ch,i); if v<mn{mn=v} if v>mx{mx=v} } }
        let need_sigmoid = mn < -0.01 || mx > 1.01;
        let sig = |x:f32| 1.0/(1.0+(-x).exp());

        let mut v=Vec::new();
        for i in 0..n {
            let (mut x1,mut y1,mut x2,mut y2) = if xyxy_mode {
                (getter(0,i), getter(1,i), getter(2,i), getter(3,i))
            } else {
                let (mut cx,mut cy,mut w,mut h)=(getter(0,i), getter(1,i), getter(2,i), getter(3,i));
                if !bbox_in_pixels { cx*=imgsz as f32; cy*=imgsz as f32; w*=imgsz as f32; h*=imgsz as f32; }
                (cx-w/2.0, cy-h/2.0, cx+w/2.0, cy+h/2.0)
            };
            if xyxy_mode && !bbox_in_pixels {
                x1*=imgsz as f32; y1*=imgsz as f32; x2*=imgsz as f32; y2*=imgsz as f32;
            }

            // kelas terbaik
            let mut best_c=0usize; let mut best_p=f32::MIN;
            for cc in 0..(c_expect-4) {
                let mut p=getter(4+cc,i);
                if need_sigmoid { p = sig(p); }
                if p>best_p { best_p=p; best_c=cc; }
            }
            if best_p<conf { continue; }

            let (x1,y1,x2,y2) = map_to_orig(x1,y1,x2,y2, use_letterbox);
            if (x2-x1)>=2.0 && (y2-y1)>=2.0 {
                v.push(Det{x1,y1,x2,y2,score:best_p,cls:best_c});
            }
        }
        Ok(v)
    };

    // 8 kombinasi (layout × format × mapping)
    let cand = [
        ("C×N, XYWH, with LBOX", make_cand(&|ch,i| flat[ch*n + i], false, true)?),
        ("C×N, XYWH, no LBOX",   make_cand(&|ch,i| flat[ch*n + i], false, false)?),
        ("C×N, XYXY, with LBOX", make_cand(&|ch,i| flat[ch*n + i], true,  true)?),
        ("C×N, XYXY, no LBOX",   make_cand(&|ch,i| flat[ch*n + i], true,  false)?),
        ("N×C, XYWH, with LBOX", make_cand(&|ch,i| flat[i*c_expect + ch], false, true)?),
        ("N×C, XYWH, no LBOX",   make_cand(&|ch,i| flat[i*c_expect + ch], false, false)?),
        ("N×C, XYXY, with LBOX", make_cand(&|ch,i| flat[i*c_expect + ch], true,  true)?),
        ("N×C, XYXY, no LBOX",   make_cand(&|ch,i| flat[i*c_expect + ch], true,  false)?),
    ];

    // pilih kandidat terbaik (valid terbanyak, lalu skor rata-rata)
    let mut best_idx=0usize; let mut best_valid=0usize; let mut best_avg=0.0f32;
    for (idx, (_name, v)) in cand.iter().enumerate() {
        let (valid, avg) = quality_score(v, orig_w, orig_h);
        if valid > best_valid || (valid==best_valid && avg>best_avg) {
            best_idx=idx; best_valid=valid; best_avg=avg;
        }
    }
    let (chosen_name, mut dets) = {
        let (name, v) = &cand[best_idx];
        (name.to_string(), v.clone())
    };
    eprintln!("[info] decode chosen: {} | boxes(valid)={}", chosen_name, best_valid);

    // NMS
    dets = nms(dets, iou_t);

    // Gambar hanya kendaraan & hitung
    let mut vehicle_count=0usize;
    for d in &dets {
        let label = class_names.get(d.cls).map(|s| s.as_str()).unwrap_or("");
        if is_vehicle_label(label) {
            vehicle_count += 1;
            draw_box(&mut img, d, label)?;
        }
    }
    println!("vehicle_count: {}", vehicle_count);
    imgcodecs::imwrite(save.to_str().unwrap(), &img, &Vector::new())?;
    println!("saved: {:?}", save);
    Ok(())
}
