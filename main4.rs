use anyhow::{anyhow, bail, Context, Result};
use opencv::{
    core::{self, Mat, MatTraitConst, MatTraitConstManual, Scalar, Size, Vector},
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

/// Decode JALUR-A: asumsi output SUDAH ter-decode (xywh/xyxy, norm/pixel)
fn decode_decoded<F>(
    get:&F, c:usize, n:usize, conf:f32, class_names:&[String],
    orig_w:i32, orig_h:i32, scale:f32, pad_x:i32, pad_y:i32, dst:i32,
    xyxy:bool, normalized:bool
)->Result<Vec<Det>>
where F: Fn(usize,usize)->f32 {
    let nc=c.checked_sub(4).ok_or_else(||anyhow!("C<4"))?;
    let mut out=Vec::new();
    for i in 0..n {
        let (mut x1,mut y1,mut x2,mut y2) = if xyxy {
            (get(0,i),get(1,i),get(2,i),get(3,i))
        } else {
            let (mut cx,mut cy,mut w,mut h)=(get(0,i),get(1,i),get(2,i),get(3,i));
            if normalized { cx*=dst as f32; cy*=dst as f32; w*=dst as f32; h*=dst as f32; }
            (cx-w/2.0, cy-h/2.0, cx+w/2.0, cy+h/2.0)
        };
        if xyxy && normalized { x1*=dst as f32; y1*=dst as f32; x2*=dst as f32; y2*=dst as f32; }

        // kelas terbaik (anggap logits → sigmoid)
        let mut best_c=0usize; let mut best_p=f32::MIN;
        for cc in 0..nc { let p=sigmoid(get(4+cc,i)); if p>best_p{best_p=p;best_c=cc;} }
        if best_p<conf { continue; }

        // mapping balik letterbox
        let fx=(x1-pad_x as f32).max(0.0); let fy=(y1-pad_y as f32).max(0.0);
        let fx2=(x2-pad_x as f32).max(0.0); let fy2=(y2-pad_y as f32).max(0.0);
        let inv=1.0/scale;
        x1=(fx*inv).clamp(0.0,(orig_w-1) as f32);
        y1=(fy*inv).clamp(0.0,(orig_h-1) as f32);
        x2=(fx2*inv).clamp(0.0,(orig_w-1) as f32);
        y2=(fy2*inv).clamp(0.0,(orig_h-1) as f32);

        if (x2-x1)>=2.0 && (y2-y1)>=2.0 { out.push(Det{x1,y1,x2,y2,score:best_p,cls:best_c}); }
    }
    Ok(out)
}

/// Decode JALUR-B: output RAW-HEAD YOLOv8 (belum grid/stride)
fn decode_raw_with_grid<F>(
    get:&F, c:usize, n:usize, conf:f32, class_names:&[String],
    orig_w:i32, orig_h:i32, scale:f32, pad_x:i32, pad_y:i32, dst:i32
)->Result<Vec<Det>>
where F: Fn(usize,usize)->f32 {
    let nc=c.checked_sub(4).ok_or_else(||anyhow!("C<4"))?;

    // cek bahwa N cocok 80^2 + 40^2 + 20^2
    let n80=(dst/8) as usize; let n40=(dst/16) as usize; let n20=(dst/32) as usize;
    let s1=n80*n80; let s2=n40*n40; let s3=n20*n20;
    if n != s1+s2+s3 { bail!("N {} tidak cocok dengan 80^2+40^2+20^2 ({}).", n, s1+s2+s3); }

    let mut out=Vec::new();
    let mut base=0usize;

    for (nx, ny, stride) in [(n80,n80,8), (n40,n40,16), (n20,n20,32)] {
        for gy in 0..ny {
            for gx in 0..nx {
                let i = base + gy*nx + gx;

                // xywh decode sesuai YOLOv8 head
                let px = sigmoid(get(0,i))*2.0 - 0.5;
                let py = sigmoid(get(1,i))*2.0 - 0.5;
                let pw = (sigmoid(get(2,i))*2.0).powi(2);
                let ph = (sigmoid(get(3,i))*2.0).powi(2);

                let cx = (px + gx as f32) * stride as f32;
                let cy = (py + gy as f32) * stride as f32;
                let w  = pw * stride as f32;
                let h  = ph * stride as f32;

                let mut x1 = cx - w/2.0;
                let mut y1 = cy - h/2.0;
                let mut x2 = cx + w/2.0;
                let mut y2 = cy + h/2.0;

                // kelas terbaik (logits → sigmoid)
                let mut best_c=0usize; let mut best_p=f32::MIN;
                for cc in 0..nc { let p=sigmoid(get(4+cc,i)); if p>best_p { best_p=p; best_c=cc; } }
                if best_p < conf { continue; }

                // mapping balik letterbox
                let fx=(x1-pad_x as f32).max(0.0); let fy=(y1-pad_y as f32).max(0.0);
                let fx2=(x2-pad_x as f32).max(0.0); let fy2=(y2-pad_y as f32).max(0.0);
                let inv=1.0/scale;
                x1=(fx*inv).clamp(0.0,(orig_w-1) as f32);
                y1=(fy*inv).clamp(0.0,(orig_h-1) as f32);
                x2=(fx2*inv).clamp(0.0,(orig_w-1) as f32);
                y2=(fy2*inv).clamp(0.0,(orig_h-1) as f32);

                if (x2-x1)>=2.0 && (y2-y1)>=2.0 { out.push(Det{x1,y1,x2,y2,score:best_p,cls:best_c}); }
            }
        }
        base += nx*ny;
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

    // load onnx
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
    let out=outs.get(0)?; // [1, 8, N] atau [1, N, 8]

    // buffer
    let total = out.total() as usize;
    let flat:&[f32]=out.data_typed()?;
    if total % c != 0 { bail!("elemen output {} tidak habis dibagi C={}", total, c); }
    let n = total / c;

    // Dua layout: C×N dan N×C
    let get_cxn = |ch:usize, idx:usize| -> f32 { flat[ch*n + idx] };
    let get_nxc = |ch:usize, idx:usize| -> f32 { flat[idx*c + ch] };

    // Kandidat dari JALUR-B (raw head, grid/stride)
    let mut cands: Vec<(String, Vec<Det>)> = Vec::new();
    if let Ok(v) = decode_raw_with_grid(&get_cxn, c, n, conf, &class_names, orig_w,orig_h,scale,pad_x,pad_y,imgsz) {
        cands.push(("RAW C×N".into(), v));
    }
    if let Ok(v) = decode_raw_with_grid(&get_nxc, c, n, conf, &class_names, orig_w,orig_h,scale,pad_x,pad_y,imgsz) {
        cands.push(("RAW N×C".into(), v));
    }

    // Kandidat dari JALUR-A (sudah decoded)
    for &(xyxy, norm, name) in &[
        (false,true,  "DEC XYWH norm C×N"),
        (true, true,  "DEC XYXY norm C×N"),
        (false,false, "DEC XYWH pix  C×N"),
        (true, false, "DEC XYXY pix  C×N"),
    ]{
        if let Ok(v) = decode_decoded(&get_cxn, c, n, conf, &class_names, orig_w,orig_h,scale,pad_x,pad_y,imgsz, xyxy, norm) {
            cands.push((name.into(), v));
        }
    }
    for &(xyxy, norm, name) in &[
        (false,true,  "DEC XYWH norm N×C"),
        (true, true,  "DEC XYXY norm N×C"),
        (false,false, "DEC XYWH pix  N×C"),
        (true, false, "DEC XYXY pix  N×C"),
    ]{
        if let Ok(v) = decode_decoded(&get_nxc, c, n, conf, &class_names, orig_w,orig_h,scale,pad_x,pad_y,imgsz, xyxy, norm) {
            cands.push((name.into(), v));
        }
    }

    // pilih terbaik (valid terbanyak, lalu skor rata-rata)
    let mut best_idx=0usize; let mut best_valid=0usize; let mut best_avg=0.0f32;
    for (i, (_nm, v)) in cands.iter().enumerate() {
        let (valid, avg) = quality_score(v, orig_w, orig_h);
        if valid > best_valid || (valid==best_valid && avg > best_avg) {
            best_idx = i; best_valid = valid; best_avg = avg;
        }
    }
    let (chosen_name, mut dets) = {
        let (nm, v) = &cands[best_idx];
        (nm.clone(), v.clone())
    };
    eprintln!("[info] decode chosen: {} | valid={}, avg={:.3}", chosen_name, best_valid, best_avg);

    // NMS & gambar SEMUA kelas (License_Plate juga ikut)
    dets = nms(dets, iou_t);
    let mut per_class = vec![0usize; class_names.len()];
    for d in &dets {
        let label = class_names.get(d.cls).map(|s| s.as_str()).unwrap_or("obj");
        per_class[d.cls]+=1;
        draw_box(&mut img, d, label)?;
    }
    for (i,cnt) in per_class.iter().enumerate() {
        eprintln!("{}: {}", class_names[i], cnt);
    }

    imgcodecs::imwrite(save.to_str().unwrap(), &img, &Vector::new())?;
    println!("saved: {:?}", save);
    Ok(())
}
