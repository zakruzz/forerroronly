use anyhow::{anyhow, bail, Context, Result};
use opencv::{
    core::{self, Mat, MatTraitConst, MatTraitConstManual, Scalar, Size, Vector},
    dnn, imgcodecs, imgproc, prelude::*,
};
use serde_json::Value;
use std::{env, fs, path::PathBuf};

#[derive(Clone, Debug)]
struct Det { x1:f32, y1:f32, x2:f32, y2:f32, score:f32, cls:usize }

fn is_vehicle_label(label:&str)->bool{
    // pakai nama kelas dari classes.json kamu apa adanya
    matches!(label, "cars" | "motorcyle" | "truck")
}

/// Letterbox (tanpa ROI)
fn letterbox_bgr(img:&Mat, dst:i32)->Result<(Mat, f32, i32, i32)>{
    let w=img.cols(); let h=img.rows();
    let s=(dst as f32 / w as f32).min(dst as f32 / h as f32);
    let nw=((w as f32)*s).round() as i32; let nh=((h as f32)*s).round() as i32;

    let mut resized=Mat::default();
    imgproc::resize(img, &mut resized, Size::new(nw,nh), 0.0,0.0, imgproc::INTER_LINEAR)?;

    let left=(dst-nw)/2; let top=(dst-nh)/2;
    let right=dst-nw-left; let bottom=dst-nh-top;

    let mut canvas=Mat::default();
    core::copy_make_border(&resized, &mut canvas, top,bottom,left,right,
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

/// Decoder generik: get(ch,i) → nilai channel `ch` di anchor `i`.
/// `xyxy_mode`=true bila bbox adalah [x1,y1,x2,y2], else [cx,cy,w,h].
fn decode_generic<F>(
    get:&F, c:usize, n:usize, conf:f32, class_names:&[String],
    orig_w:i32, orig_h:i32, scale:f32, pad_x:i32, pad_y:i32, dst:i32,
    xyxy_mode:bool
)->Result<Vec<Det>>
where F: Fn(usize,usize)->f32 {
    let nc=c.checked_sub(4).ok_or_else(||anyhow!("C<4"))?;
    if nc!=class_names.len(){ bail!("C-4={} != classes.json={}", nc, class_names.len()); }

    // deteksi logits vs prob
    let mut mn=f32::INFINITY; let mut mx=f32::NEG_INFINITY;
    for i in 0..n { for ch in 4..c { let v=get(ch,i); if v<mn{mn=v}; if v>mx{mx=v}; } }
    let need_sigmoid= mn < -0.01 || mx > 1.01;
    let sig = |x:f32| 1.0/(1.0+(-x).exp());

    // deteksi skala bbox: normalized (<=1.0) atau pixel (>1.5)
    let mut bbox_max=0.0f32;
    for i in 0..n { for ch in 0..4 { bbox_max=bbox_max.max(get(ch,i).abs()); } }
    let bbox_in_pixels = bbox_max > 1.5;

    let mut out=Vec::new();
    for i in 0..n {
        let (mut x1,mut y1,mut x2,mut y2);

        if xyxy_mode {
            let mut ax=get(0,i); let mut ay=get(1,i);
            let mut bx=get(2,i); let mut by=get(3,i);
            if !bbox_in_pixels { ax*=dst as f32; ay*=dst as f32; bx*=dst as f32; by*=dst as f32; }
            x1=ax; y1=ay; x2=bx; y2=by;
        } else {
            let mut cx=get(0,i); let mut cy=get(1,i);
            let mut w =get(2,i); let mut h =get(3,i);
            if !bbox_in_pixels { cx*=dst as f32; cy*=dst as f32; w*=dst as f32; h*=dst as f32; }
            x1=cx - w/2.0; y1=cy - h/2.0; x2=cx + w/2.0; y2=cy + h/2.0;
        }

        // kelas terbaik
        let (mut best_c,mut best_p)=(0usize, f32::MIN);
        for cc in 0..nc {
            let mut p=get(4+cc, i);
            if need_sigmoid { p=sig(p); }
            if p>best_p { best_p=p; best_c=cc; }
        }
        if best_p<conf { continue; }

        // pindah dari kanvas ke koordinat gambar asli
        let fx =(x1 - pad_x as f32).max(0.0);
        let fy =(y1 - pad_y as f32).max(0.0);
        let fx2=(x2 - pad_x as f32).max(0.0);
        let fy2=(y2 - pad_y as f32).max(0.0);
        let inv=1.0/scale;

        x1=(fx *inv).clamp(0.0,(orig_w-1) as f32);
        y1=(fy *inv).clamp(0.0,(orig_h-1) as f32);
        x2=(fx2*inv).clamp(0.0,(orig_w-1) as f32);
        y2=(fy2*inv).clamp(0.0,(orig_h-1) as f32);

        // buang box degenerate
        if (x2-x1) >= 2.0 && (y2-y1) >= 2.0 {
            out.push(Det{x1,y1,x2,y2,score:best_p,cls:best_c});
        }
    }
    Ok(out)
}

fn quality_score(d:&[Det], w:i32,h:i32)->(usize,f32){
    // hitung berapa yg valid & skor rata-rata (untuk tie-break)
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
    // cargo run --release -- <best.onnx> <classes.json> <image> [imgsz=640] [conf=0.25] [iou=0.45] [filter_vehicle=1] [save=result.jpg]
    let a:Vec<String>=env::args().collect();
    if a.len()<4{
        eprintln!("Usage: {} <best.onnx> <classes.json> <image> [imgsz=640] [conf=0.25] [iou=0.45] [filter_vehicle=1] [save=result.jpg]", a[0]);
        std::process::exit(1);
    }
    let onnx=PathBuf::from(&a[1]);
    let classes_path=PathBuf::from(&a[2]);
    let image_path=PathBuf::from(&a[3]);
    let imgsz:i32=a.get(4).and_then(|s|s.parse().ok()).unwrap_or(640);
    let conf:f32=a.get(5).and_then(|s|s.parse().ok()).unwrap_or(0.25);
    let iou_t:f32=a.get(6).and_then(|s|s.parse().ok()).unwrap_or(0.45);
    let filter_vehicle:bool=a.get(7).map(|s| s=="1" || s.to_lowercase()=="true").unwrap_or(true);
    let save=PathBuf::from(a.get(8).cloned().unwrap_or_else(||"result.jpg".to_string()));

    // classes.json → nama kelas (harus 4)
    let txt=fs::read_to_string(&classes_path).context("read classes.json")?;
    let j:Value=serde_json::from_str(&txt).context("parse classes.json")?;
    let arr=j.as_array().ok_or_else(||anyhow!("classes.json harus array string"))?;
    let class_names:Vec<String>=arr.iter().map(|v| v.as_str().unwrap_or("").to_string()).collect();
    if class_names.len()!=4 { bail!("classes.json harus 4 item (License_Plate,cars,motorcyle,truck)"); }
    let c_expect=4 + class_names.len(); // 8 sesuai outputmu

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
    let mut net=dnn::read_net_from_onnx(onnx.to_str().unwrap()).with_context(||format!("read onnx {:?}", onnx))?;
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
    let out=outs.get(0)?; // asumsikan satu tensor utama

    let total = out.total() as usize;
    let flat:&[f32]=out.data_typed()?;
    if total % c_expect != 0 { bail!("elemen output {} tidak habis dibagi C(exp)={}", total, c_expect); }
    let n = total / c_expect;

    // Empat hipotesis: (CxN/NxC) × (XYWH/XYXY)
    let cand1 = decode_generic(&|ch,i| flat[ch*n + i], c_expect, n, conf, &class_names, orig_w, orig_h, scale, pad_x, pad_y, imgsz, false)?;
    let cand2 = decode_generic(&|ch,i| flat[ch*n + i], c_expect, n, conf, &class_names, orig_w, orig_h, scale, pad_x, pad_y, imgsz, true)?;
    let cand3 = decode_generic(&|ch,i| flat[i*c_expect + ch], c_expect, n, conf, &class_names, orig_w, orig_h, scale, pad_x, pad_y, imgsz, false)?;
    let cand4 = decode_generic(&|ch,i| flat[i*c_expect + ch], c_expect, n, conf, &class_names, orig_w, orig_h, scale, pad_x, pad_y, imgsz, true)?;

    let mut sets = vec![
        (nms(cand1, iou_t), "C×N, XYWH"),
        (nms(cand2, iou_t), "C×N, XYXY"),
        (nms(cand3, iou_t), "N×C, XYWH"),
        (nms(cand4, iou_t), "N×C, XYXY"),
    ];

    // pilih set terbaik
    sets.sort_by(|(a, _), (b, _)| {
        let (va, sa) = quality_score(a, orig_w, orig_h);
        let (vb, sb) = quality_score(b, orig_w, orig_h);
        // by valid count desc, then avg score desc
        vb.cmp(&va).then_with(|| sb.partial_cmp(&sa).unwrap())
    });
    let (mut dets, chosen) = sets.remove(0);
    eprintln!("[info] layout/format: {}", chosen);

    // hitung hanya kendaraan sesuai classes.json
    let mut veh_count = 0usize;
    for d in &dets {
        let label = class_names.get(d.cls).map(|s| s.as_str()).unwrap_or("");
        if is_vehicle_label(label) { veh_count += 1; }
        draw_box(&mut img, d, label)?;
    }
    println!("vehicle_count: {}", veh_count);
    imgcodecs::imwrite(save.to_str().unwrap(), &img, &Vector::new())?;
    println!("saved: {:?}", save);
    Ok(())
}
