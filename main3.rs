use std::{env, fs, path::PathBuf, process::Command};
use anyhow::{anyhow, bail, Context, Result};
use serde_json;
use image::{imageops, ImageBuffer, Rgb, RgbImage};
use ndarray::{Array4, ArrayD};
use ndarray_npy::{write_npy, read_npy};

#[derive(Clone, Debug)]
struct Det { x1:f32, y1:f32, x2:f32, y2:f32, score:f32, cls:usize }

fn is_vehicle(name:&str)->bool{
    let keys=["car","bus","truck","motorcycle","motorbike","bicycle","bike","van","pickup","trailer","truk","mobil","sepeda","motor"];
    let n=name.to_lowercase();
    keys.iter().any(|k| n.contains(k))
}

fn letterbox_rgb(rgb:&RgbImage, dst:u32)->(RgbImage,f32,u32,u32){
    let (w,h)=(rgb.width() as f32, rgb.height() as f32);
    let s=(dst as f32 / w).min(dst as f32 / h);
    let nw=(w*s).round() as u32; let nh=(h*s).round() as u32;
    let resized=imageops::resize(rgb, nw, nh, imageops::FilterType::Triangle);
    let mut canvas=ImageBuffer::<Rgb<u8>,Vec<u8>>::from_pixel(dst,dst,Rgb([114,114,114]));
    let dx=(dst-nw)/2; let dy=(dst-nh)/2;
    imageops::replace(&mut canvas,&resized, dx.into(), dy.into());
    (canvas, s, dx, dy)
}

fn chw_normalize(inp:&RgbImage)->Vec<f32>{
    let (w,h)=(inp.width() as usize, inp.height() as usize);
    let mut out=vec![0f32; 3*w*h];
    for y in 0..h {
        for x in 0..w {
            let p=inp.get_pixel(x as u32, y as u32);
            let idx=y*w + x;
            out[idx]            = p[0] as f32 / 255.0;
            out[w*h + idx]      = p[1] as f32 / 255.0;
            out[2*w*h + idx]    = p[2] as f32 / 255.0;
        }
    }
    out
}

fn iou(a:&Det,b:&Det)->f32{
    let (x1,y1)=(a.x1.max(b.x1), a.y1.max(b.y1));
    let (x2,y2)=(a.x2.min(b.x2), a.y2.min(b.y2));
    let inter=(x2-x1).max(0.0)*(y2-y1).max(0.0);
    let area_a=(a.x2-a.x1).max(0.0)*(a.y2-a.y1).max(0.0);
    let area_b=(b.x2-b.x1).max(0.0)*(b.y2-b.y1).max(0.0);
    inter / (area_a + area_b - inter + 1e-6)
}
fn nms(mut v:Vec<Det>, iou_t:f32)->Vec<Det>{
    v.sort_by(|a,b| b.score.partial_cmp(&a.score).unwrap());
    let mut keep=Vec::new();
    'outer: for d in v {
        for k in &keep {
            if d.cls==k.cls && iou(&d,k) > iou_t { continue 'outer; }
        }
        keep.push(d);
    }
    keep
}
fn sigmoid(x:f32)->f32 { 1.0/(1.0+(-x).exp()) }

fn decode_v8_4plusnc(
    flat:&[f32], c:usize, n:usize, conf:f32,
    names:&[String], img_w:u32, img_h:u32,
    scale:f32, pad_x:u32, pad_y:u32, dst:u32,
    filter_vehicle:bool
)->Result<Vec<Det>>{
    let nc = c.checked_sub(4).ok_or_else(||anyhow!("C<4"))?;
    if nc != names.len() { bail!("C-4={} != classes.json={}", nc, names.len()); }

    // deteksi logits/prob
    let (mut mn,mut mx)=(f32::INFINITY,f32::NEG_INFINITY);
    for &v in flat { if v<mn{mn=v} if v>mx{mx=v} }
    let need_sigmoid = mn < -0.01 || mx > 1.01;

    let stride = n;
    let mut out = Vec::new();

    let mut push = |mut cx:f32, mut cy:f32, mut w:f32, mut h:f32, score:f32, cls:usize| {
        cx*=dst as f32; cy*=dst as f32; w*=dst as f32; h*=dst as f32;
        let (mut x1,mut y1,mut x2,mut y2)=(cx-w/2.0, cy-h/2.0, cx+w/2.0, cy+h/2.0);

        // lepas padding letterbox
        let fx=(x1 - pad_x as f32).max(0.0);
        let fy=(y1 - pad_y as f32).max(0.0);
        let fx2=(x2 - pad_x as f32).max(0.0);
        let fy2=(y2 - pad_y as f32).max(0.0);
        let inv = 1.0/scale;

        x1=(fx*inv).clamp(0.0, img_w as f32 - 1.0);
        y1=(fy*inv).clamp(0.0, img_h as f32 - 1.0);
        x2=(fx2*inv).clamp(0.0, img_w as f32 - 1.0);
        y2=(fy2*inv).clamp(0.0, img_h as f32 - 1.0);

        out.push(Det{ x1,y1,x2,y2, score, cls });
    };

    for i in 0..n {
        let bx = flat[0*stride + i].clamp(0.0, 1.0);
        let by = flat[1*stride + i].clamp(0.0, 1.0);
        let bw = flat[2*stride + i].clamp(0.0, 1.0);
        let bh = flat[3*stride + i].clamp(0.0, 1.0);

        let (mut best_cls, mut best_p) = (0usize, f32::MIN);
        for cc in 0..nc {
            let mut p = flat[(4+cc)*stride + i];
            if need_sigmoid { p = sigmoid(p); }
            if p > best_p { best_p = p; best_cls = cc; }
        }
        if best_p < conf { continue; }
        if filter_vehicle {
            let cname = names.get(best_cls).map(|s|s.as_str()).unwrap_or("");
            if !is_vehicle(cname) { continue; }
        }
        push(bx,by,bw,bh,best_p,best_cls);
    }
    Ok(out)
}

fn draw_rect_rgb(img:&mut RgbImage, x1:i32, y1:i32, x2:i32, y2:i32, thick:i32, color:[u8;3]){
    let (w,h)=(img.width() as i32, img.height() as i32);
    let (x1,y1,x2,y2)=(x1.clamp(0,w-1), y1.clamp(0,h-1), x2.clamp(0,w-1), y2.clamp(0,h-1));
    if x2<=x1 || y2<=y1 { return; }
    let t=thick.max(1);
    for dy in 0..t { for x in x1..=x2 {
        if y1+dy>=0 && y1+dy<h { img.put_pixel(x as u32, (y1+dy) as u32, Rgb(color)); }
        if y2-dy>=0 && y2-dy<h { img.put_pixel(x as u32, (y2-dy) as u32, Rgb(color)); }
    }}
    for dx in 0..t { for y in y1..=y2 {
        if x1+dx>=0 && x1+dx<w { img.put_pixel((x1+dx) as u32, y as u32, Rgb(color)); }
        if x2-dx>=0 && x2-dx<w { img.put_pixel((x2-dx) as u32, y as u32, Rgb(color)); }
    }}
}

fn main() -> Result<()> {
    // Argumen sederhana (posisional, untuk hindari clap/time)
    // Usage:
    //   prog <engine> <input_name> <output_name> <classes.json> <image> [imgsz] [conf] [iou] [filter_vehicle] [save]
    let a: Vec<String> = env::args().collect();
    if a.len() < 6 {
        eprintln!("Usage: {} <engine> <input_name> <output_name> <classes.json> <image> [imgsz=640] [conf=0.25] [iou=0.45] [filter_vehicle=1] [save=result.jpg]", a[0]);
        std::process::exit(1);
    }
    let engine = PathBuf::from(&a[1]);
    let input_name = &a[2];
    let output_name = &a[3];
    let classes_path = PathBuf::from(&a[4]);
    let image_path = PathBuf::from(&a[5]);
    let imgsz: u32 = a.get(6).and_then(|s| s.parse().ok()).unwrap_or(640);
    let conf: f32 = a.get(7).and_then(|s| s.parse().ok()).unwrap_or(0.25);
    let iou: f32  = a.get(8).and_then(|s| s.parse().ok()).unwrap_or(0.45);
    let filter_vehicle: bool = a.get(9).map(|s| s=="1" || s.to_lowercase()=="true").unwrap_or(true);
    let save = PathBuf::from(a.get(10).cloned().unwrap_or_else(|| "result.jpg".to_string()));

    // classes.json -> Vec<String>
    let class_names: Vec<String> = serde_json::from_slice(&fs::read(&classes_path).context("read classes.json")?)
        .context("classes.json harus array string")?;
    if class_names.is_empty() { bail!("classes.json kosong"); }

    // load & preprocess image
    let img = image::open(&image_path).with_context(|| format!("open {:?}", image_path))?.to_rgb8();
    let (letter, scale, pad_x, pad_y) = letterbox_rgb(&img, imgsz);
    let chw = chw_normalize(&letter);
    let arr: Array4<f32> = Array4::from_shape_vec((1,3,imgsz as usize,imgsz as usize), chw)
        .map_err(|_| anyhow!("shape input mismatch"))?;
    let in_path = PathBuf::from("/tmp/trt_in.npy");
    write_npy(&in_path, &arr)?;

    // folder output
    let out_dir = PathBuf::from("/tmp/trt_out");
    if out_dir.exists() { let _=fs::remove_dir_all(&out_dir); }
    fs::create_dir_all(&out_dir)?;

    // jalankan trtexec (coba 'trtexec' lalu fallback ke path JetPack)
    let shapes = format!("{}:1x3x{}x{}", input_name, imgsz, imgsz);
    let mut cmd = Command::new("trtexec");
    cmd.args([
        "--loadEngine", engine.to_str().unwrap(),
        "--shapes", &shapes,
        "--loadInputs", &format!("{}:{}", input_name, in_path.to_str().unwrap()),
        "--exportOutput", out_dir.to_str().unwrap(),
        "--warmUp", "0", "--iterations", "1", "--avgRuns", "1", "--useCudaGraph", "0",
    ]);
    let status = cmd.status();
    let status = match status {
        Ok(s) if s.success() => s,
        _ => {
            // fallback absolute path (biasa di Jetson)
            let s2 = Command::new("/usr/src/tensorrt/bin/trtexec")
                .args([
                    "--loadEngine", engine.to_str().unwrap(),
                    "--shapes", &shapes,
                    "--loadInputs", &format!("{}:{}", input_name, in_path.to_str().unwrap()),
                    "--exportOutput", out_dir.to_str().unwrap(),
                    "--warmUp", "0", "--iterations", "1", "--avgRuns", "1", "--useCudaGraph", "0",
                ]).status().context("run trtexec fallback")?;
            if !s2.success() { bail!("trtexec failed (fallback)"); }
            s2
        }
    };

    if !status.success(){ bail!("trtexec failed"); }

    // baca output .npy
    let out_path = out_dir.join(format!("{}.npy", output_name));
    if !out_path.exists() { bail!("file output tidak ada: {:?}", out_path); }
    let out: ArrayD<f32> = read_npy(&out_path)?;
    let shape = out.shape().to_vec();
    if shape.len()!=3 || shape[0]!=1 { bail!("output shape bukan [1,C,N], dapat {:?}", shape); }
    let c = shape[1]; let n = shape[2];
    let flat: Vec<f32> = out.into_raw_vec();

    // decode YOLOv8 [1,4+nc,N] dan NMS
    let mut dets = decode_v8_4plusnc(
        &flat, c, n, conf, &class_names,
        img.width(), img.height(), scale, pad_x, pad_y, imgsz,
        filter_vehicle
    )?;
    dets = nms(dets, iou);

    println!("count: {}", dets.len());

    // gambar bbox & simpan
    let mut out_img = img.clone();
    for d in &dets {
        draw_rect_rgb(&mut out_img, d.x1 as i32, d.y1 as i32, d.x2 as i32, d.y2 as i32, 2, [0,255,0]);
    }
    out_img.save(&save).with_context(|| format!("save {:?}", save))?;
    println!("saved: {:?}", save);
    Ok(())
}
