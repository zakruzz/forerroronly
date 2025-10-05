use anyhow::{anyhow, bail, Context, Result};
use opencv::{
    core::{self, Mat, MatTraitConst, MatTraitConstManual, Scalar, Size, Vector},
    dnn, imgcodecs, imgproc, prelude::*,
};
use serde_json::Value;
use std::{env, fs, path::PathBuf};

#[derive(Clone, Debug)]
struct Det { x1:f32, y1:f32, x2:f32, y2:f32, score:f32, cls:usize }

fn is_vehicle(name:&str)->bool{
    let keys=["car","bus","truck","motorcycle","motorbike","bicycle","bike","van","pickup","trailer","truk","mobil","sepeda","motor"];
    let n=name.to_lowercase();
    keys.iter().any(|k| n.contains(k))
}

/// Letterbox via core::copy_make_border (tanpa ROI)
fn letterbox_bgr(img:&Mat, dst:i32)->Result<(Mat,f32,i32,i32)>{
    let w=img.cols(); let h=img.rows();
    let s=(dst as f32 / w as f32).min(dst as f32 / h as f32);
    let nw=((w as f32)*s).round() as i32; let nh=((h as f32)*s).round() as i32;

    let mut resized=Mat::default();
    imgproc::resize(img, &mut resized, Size::new(nw,nh), 0.0,0.0, imgproc::INTER_LINEAR)?;

    let left=(dst-nw)/2; let top=(dst-nh)/2;
    let right=dst-nw-left; let bottom=dst-nh-top;

    let mut canvas=Mat::default();
    core::copy_make_border(&resized,&mut canvas, top,bottom,left,right, core::BORDER_CONSTANT, Scalar::new(114.0,114.0,114.0,0.0))?;
    Ok((canvas,s,left,top))
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

/// Decoder generik: akses nilai via closure `get(ch,i)`
/// Auto: logits→sigmoid bila perlu; bbox normalized vs pixel terdeteksi otomatis
fn decode_generic<F>(
    get:&F, c:usize, n:usize, conf:f32, class_names:&[String],
    orig_w:i32, orig_h:i32, scale:f32, pad_x:i32, pad_y:i32, dst:i32, filter_vehicle:bool
)->Result<Vec<Det>>
where F: Fn(usize,usize)->f32 {
    let nc=c.checked_sub(4).ok_or_else(||anyhow!("C<4"))?;
    if nc!=class_names.len(){ bail!("C-4={} != classes.json={}", nc, class_names.len()); }

    // deteksi logits vs prob
    let mut mn=f32::INFINITY; let mut mx=f32::NEG_INFINITY;
    for i in 0..n { for ch in 4..c { let v=get(ch,i); if v<mn{mn=v}; if v>mx{mx=v}; } }
    let need_sigmoid = mn < -0.01 || mx > 1.01;
    let sig = |x:f32| 1.0/(1.0+(-x).exp());

    // deteksi skala bbox: kalau ada nilai >1.5 berarti skala pixel (0..dst), else normalized
    let mut bbox_max=0.0f32;
    for i in 0..n { for ch in 0..4 { bbox_max=bbox_max.max(get(ch,i).abs()); } }
    let bbox_in_pixels = bbox_max > 1.5;

    let mut out=Vec::new();
    for i in 0..n {
        let mut cx=get(0,i);
        let mut cy=get(1,i);
        let mut w =get(2,i);
        let mut h =get(3,i);

        // pilih kelas terbaik
        let (mut best_c,mut best_p)=(0usize, f32::MIN);
        for cc in 0..nc {
            let mut p=get(4+cc, i);
            if need_sigmoid { p=sig(p); }
            if p>best_p { best_p=p; best_c=cc; }
        }
        if best_p<conf { continue; }
        if filter_vehicle {
            let cname=class_names.get(best_c).map(|s|s.as_str()).unwrap_or("");
            if !is_vehicle(cname) { continue; }
        }

        // ke koordinat kanvas dst×dst
        if !bbox_in_pixels {
            cx*=dst as f32; cy*=dst as f32; w*=dst as f32; h*=dst as f32;
        }
        let (mut x1,mut y1,mut x2,mut y2)=(cx-w/2.0, cy-h/2.0, cx+w/2.0, cy+h/2.0);

        // lepaskan padding letterbox → ke koordinat gambar asli
        let fx =(x1 - pad_x as f32).max(0.0);
        let fy =(y1 - pad_y as f32).max(0.0);
        let fx2=(x2 - pad_x as f32).max(0.0);
        let fy2=(y2 - pad_y as f32).max(0.0);
        let inv=1.0/scale;

        x1=(fx *inv).clamp(0.0,(orig_w-1) as f32);
        y1=(fy *inv).clamp(0.0,(orig_h-1) as f32);
        x2=(fx2*inv).clamp(0.0,(orig_w-1) as f32);
        y2=(fy2*inv).clamp(0.0,(orig_h-1) as f32);

        out.push(Det{x1,y1,x2,y2,score:best_p,cls:best_c});
    }
    Ok(out)
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
    // Usage:
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

    // classes.json
    let txt=fs::read_to_string(&classes_path).context("read classes.json")?;
    let j:Value=serde_json::from_str(&txt).context("parse classes.json")?;
    let arr=j.as_array().ok_or_else(||anyhow!("classes.json harus array string"))?;
    let class_names:Vec<String>=arr.iter().map(|v| v.as_str().unwrap_or("").to_string()).collect();
    if class_names.is_empty(){ bail!("classes.json kosong"); }
    let c_expect=4 + class_names.len();

    // baca gambar
    let mut img=imgcodecs::imread(image_path.to_str().unwrap(), imgcodecs::IMREAD_COLOR)
        .with_context(||format!("open {:?}", image_path))?;
    if img.empty(){ bail!("gagal baca gambar"); }
    let (orig_w,orig_h)=(img.cols(), img.rows());

    // letterbox & blob
    let (letter, scale, pad_x, pad_y)=letterbox_bgr(&img, imgsz)?;
    let blob=dnn::blob_from_image(&letter, 1.0/255.0, Size::new(imgsz,imgsz), Scalar::default(), true,false, core::CV_32F)?;

    // load onnx
    let mut net=dnn::read_net_from_onnx(onnx.to_str().unwrap()).with_context(||format!("read onnx {:?}", onnx))?;
    // prefer CUDA; fallback CPU
    if net.set_preferable_backend(dnn::DNN_BACKEND_CUDA).is_ok() && net.set_preferable_target(dnn::DNN_TARGET_CUDA_FP16).is_ok() {
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
    if outs.len()==0 { bail!("tidak ada output dari model"); }
    let out=outs.get(0)?; // ambil tensor pertama

    // bentuk output: [1,C,N] ATAU [1,N,C] → auto
    let total = out.total() as usize;
    // Ambil data float
    let flat:&[f32]=out.data_typed()?;
    if total % c_expect != 0 {
        bail!("elemen output {} tidak habis dibagi C(exp)={}", total, c_expect);
    }
    let n1 = total / c_expect; // kandidat N jika layout [1,N,C]
    let c1 = c_expect;

    // Decode dengan 2 hipotesis, pilih yang paling banyak deteksi (atau skor rata-rata tertinggi)
    // H1: [1, C, N]  => idx(ch,i)= ch*N + i
    let dets_cxn = decode_generic(&|ch,i| flat[ch*n1 + i], c1, n1, conf, &class_names, orig_w, orig_h, scale, pad_x, pad_y, imgsz, filter_vehicle)?;
    let dets_cxn = nms(dets_cxn, iou_t);

    // H2: [1, N, C]  => idx(ch,i)= i*C + ch
    let dets_nxc = decode_generic(&|ch,i| flat[i*c1 + ch], c1, n1, conf, &class_names, orig_w, orig_h, scale, pad_x, pad_y, imgsz, filter_vehicle)?;
    let dets_nxc = nms(dets_nxc, iou_t);

    let (mut dets, layout) = if dets_nxc.len() > dets_cxn.len() {
        (dets_nxc, "1xN xC")
    } else if dets_cxn.len() > dets_nxc.len() {
        (dets_cxn, "1xC xN")
    } else {
        // tie-break pakai skor rata-rata
        let avg_a = if dets_cxn.is_empty(){0.0}else{dets_cxn.iter().map(|d|d.score).sum::<f32>()/dets_cxn.len() as f32};
        let avg_b = if dets_nxc.is_empty(){0.0}else{dets_nxc.iter().map(|d|d.score).sum::<f32>()/dets_nxc.len() as f32};
        if avg_b>avg_a { (dets_nxc, "1xN xC") } else { (dets_cxn, "1xC xN") }
    };

    eprintln!("[info] layout terpilih: {}, deteksi: {}", layout, dets.len());
    println!("count: {}", dets.len());

    // gambar & simpan
    for d in &dets {
        let label=class_names.get(d.cls).map(|s|s.as_str()).unwrap_or("obj");
        draw_box(&mut img, d, label)?;
    }
    imgcodecs::imwrite(save.to_str().unwrap(), &img, &Vector::new())?;
    println!("saved: {:?}", save);

    Ok(())
}
