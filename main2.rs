use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use image::{imageops, ImageBuffer, Rgb, RgbImage};
use serde::Deserialize;
use std::{ffi::c_void, fs, path::PathBuf};
use ndarray::Array4;

use cust::{prelude::*, memory::DeviceBuffer};
use tensorrt_rs as trt;

#[derive(Parser, Debug)]
#[command(name="jetson_trt_yolo_image")]
struct Args {
    #[arg(long)] engine: PathBuf,       // .engine dari trtexec
    #[arg(long)] classes: PathBuf,      // classes.json (array nama kelas)
    #[arg(long)] image: PathBuf,        // gambar input
    #[arg(long, default_value="result.jpg")] save: PathBuf,
    #[arg(long, default_value_t=640)] imgsz: u32,
    #[arg(long, default_value_t=0.25)] conf: f32,
    #[arg(long, default_value_t=0.45)] iou: f32,
    #[arg(long, default_value_t=true)] filter_vehicle: bool,
}

#[derive(Deserialize, Debug)]
struct ClassList(Vec<String>);

#[derive(Clone, Debug)]
struct Detection { x1:f32, y1:f32, x2:f32, y2:f32, score:f32, class_id:usize }

fn is_vehicle(name: &str) -> bool {
    let keys = ["car","bus","truck","motorcycle","motorbike","bicycle","bike","van","pickup","trailer","truk","mobil","sepeda","motor"];
    let n = name.to_lowercase();
    keys.iter().any(|k| n.contains(k))
}

fn letterbox_rgb(rgb:&RgbImage, dst:u32)->(RgbImage,f32,u32,u32){
    let (w,h)=(rgb.width() as f32,rgb.height() as f32);
    let s=(dst as f32/w).min(dst as f32/h);
    let nw=(w*s).round() as u32; let nh=(h*s).round() as u32;
    let resized=imageops::resize(rgb,nw,nh,imageops::FilterType::Triangle);
    let mut canvas=ImageBuffer::<Rgb<u8>,Vec<u8>>::from_pixel(dst,dst,Rgb([114,114,114]));
    let dx=(dst-nw)/2; let dy=(dst-nh)/2;
    imageops::replace(&mut canvas,&resized,dx.into(),dy.into());
    (canvas,s,dx,dy)
}
fn chw_normalize(inp:&RgbImage)->Vec<f32>{
    let (w,h)=(inp.width() as usize,inp.height() as usize);
    let mut out=vec![0f32;3*w*h];
    for y in 0..h { for x in 0..w {
        let p=inp.get_pixel(x as u32,y as u32); let idx=y*w+x;
        out[idx]=p[0] as f32/255.0; out[w*h+idx]=p[1] as f32/255.0; out[2*w*h+idx]=p[2] as f32/255.0;
    } }
    out
}
fn iou(a:&Detection,b:&Detection)->f32{
    let(x1,y1)=(a.x1.max(b.x1),a.y1.max(b.y1));
    let(x2,y2)=(a.x2.min(b.x2),a.y2.min(b.y2));
    let inter=(x2-x1).max(0.0)*(y2-y1).max(0.0);
    let area_a=(a.x2-a.x1).max(0.0)*(a.y2-a.y1).max(0.0);
    let area_b=(b.x2-b.x1).max(0.0)*(b.y2-b.y1).max(0.0);
    inter/(area_a+area_b-inter+1e-6)
}
fn nms(mut dets:Vec<Detection>, iou_t:f32)->Vec<Detection>{
    dets.sort_by(|a,b| b.score.partial_cmp(&a.score).unwrap());
    let mut keep=Vec::new();
    'o: for d in dets { for k in &keep {
        if d.class_id==k.class_id && iou(&d,k)>iou_t { continue 'o; } }
        keep.push(d);
    } keep
}
fn sigmoid(x:f32)->f32{1.0/(1.0+(-x).exp())}

/// Decoder YOLOv8 (tanpa objectness): output [1, 4+nc, N]
fn decode_yolo_v8(
    out:&[f32], c:usize, n:usize, conf_t:f32, class_names:&[String],
    img_w:u32, img_h:u32, scale:f32, pad_x:u32, pad_y:u32, dst:u32,
    filter_vehicle:bool,
)->Result<Vec<Detection>>{
    let nc=c.checked_sub(4).ok_or_else(||anyhow!("C<4"))?;
    if nc!=class_names.len(){ bail!("kelas tidak cocok: c-4={} vs classes.json={}",nc,class_names.len()); }
    let (mut mn,mut mx)=(f32::INFINITY,f32::NEG_INFINITY);
    for &v in out { if v<mn{mn=v} if v>mx{mx=v} }
    let need_sigmoid= mn<-0.01 || mx>1.01;
    let stride=n; let mut dets=Vec::new();

    let mut push=|mut cx:f32,mut cy:f32,mut w:f32,mut h:f32,score:f32,cls:usize|{
        cx*=dst as f32; cy*=dst as f32; w*=dst as f32; h*=dst as f32;
        let (mut x1,mut y1,mut x2,mut y2)=(cx-w/2.0,cy-h/2.0,cx+w/2.0,cy+h/2.0);
        let fx=(x1-pad_x as f32).max(0.0); let fy=(y1-pad_y as f32).max(0.0);
        let fx2=(x2-pad_x as f32).max(0.0); let fy2=(y2-pad_y as f32).max(0.0);
        let inv=1.0/scale;
        x1=(fx*inv).clamp(0.0,img_w as f32-1.0);
        y1=(fy*inv).clamp(0.0,img_h as f32-1.0);
        x2=(fx2*inv).clamp(0.0,img_w as f32-1.0);
        y2=(fy2*inv).clamp(0.0,img_h as f32-1.0);
        dets.push(Detection{x1,y1,x2,y2,score,class_id:cls});
    };

    for i in 0..n {
        let bx=out[0*stride+i].clamp(0.0,1.0);
        let by=out[1*stride+i].clamp(0.0,1.0);
        let bw=out[2*stride+i].clamp(0.0,1.0);
        let bh=out[3*stride+i].clamp(0.0,1.0);

        let(mut best_c,mut best_p)=(0usize,f32::MIN);
        for cc in 0..nc {
            let mut p=out[(4+cc)*stride+i];
            if need_sigmoid { p=sigmoid(p); }
            if p>best_p { best_p=p; best_c=cc; }
        }
        if best_p<conf_t { continue; }
        if filter_vehicle {
            let cname=class_names.get(best_c).map(|s|s.as_str()).unwrap_or("");
            if !is_vehicle(cname) { continue; }
        }
        push(bx,by,bw,bh,best_p,best_c);
    }
    Ok(dets)
}

fn draw_rect_rgb(img:&mut RgbImage,x1:i32,y1:i32,x2:i32,y2:i32,thick:i32,color:[u8;3]){
    let(w,h)=(img.width() as i32,img.height() as i32);
    let(x1,y1,x2,y2)=(x1.clamp(0,w-1),y1.clamp(0,h-1),x2.clamp(0,w-1),y2.clamp(0,h-1));
    if x2<=x1||y2<=y1{return;}
    let t=thick.max(1);
    for dy in 0..t{ for x in x1..=x2{
        if y1+dy>=0&&y1+dy<h{ img.put_pixel(x as u32,(y1+dy) as u32,Rgb(color)); }
        if y2-dy>=0&&y2-dy<h{ img.put_pixel(x as u32,(y2-dy) as u32,Rgb(color)); }
    }}
    for dx in 0..t{ for y in y1..=y2{
        if x1+dx>=0&&x1+dx<w{ img.put_pixel((x1+dx) as u32,y as u32,Rgb(color)); }
        if x2-dx>=0&&x2-dx<w{ img.put_pixel((x2-dx) as u32,y as u32,Rgb(color)); }
    }}
}

fn main() -> Result<()> {
    let args = Args::parse();

    // classes
    let classes: ClassList = serde_json::from_slice(&fs::read(&args.classes).context("read classes.json")?)
        .context("classes.json invalid")?;
    if classes.0.is_empty(){ bail!("classes.json kosong"); }

    // load image
    let img = image::open(&args.image).with_context(|| format!("open {:?}", &args.image))?.to_rgb8();
    let (letter, scale, pad_x, pad_y) = letterbox_rgb(&img, args.imgsz);
    let input_chw = chw_normalize(&letter);
    let arr: Array4<f32> = Array4::from_shape_vec((1,3,args.imgsz as usize,args.imgsz as usize), input_chw)
        .context("shape input mismatch")?;
    let input_host: Vec<f32> = arr.into_raw_vec();

    // TensorRT runtime
    let logger = trt::Logger::new(trt::LoggerSeverity::Warning);
    let rt = trt::runtime::Runtime::new(&logger).context("Runtime::new")?;
    let engine_bytes = fs::read(&args.engine).with_context(|| format!("read {:?}", &args.engine))?;
    let engine = rt.deserialize_cuda_engine(&engine_bytes).context("deserialize engine")?;
    let mut ctx = engine.create_execution_context().context("create_execution_context")?;

    // asumsi 2 binding (input/output)
    let nb = engine.get_nb_bindings();
    if nb != 2 { bail!("engine bindings != 2 (nb={})", nb); }

    // ambil dimensi output -> [1, 4+nc, N]
    let out_dims = engine.get_binding_dimensions(1).ok_or_else(|| anyhow!("get out dims"))?;
    let out_shape: Vec<i32> = out_dims.d.iter().copied().collect();
    if out_shape.len()!=3 || out_shape[0]!=1 { bail!("output harus [1, 4+nc, N], dapat {:?}", out_shape); }
    let c = out_shape[1] as usize;
    let n = out_shape[2] as usize;

    // validasi nc vs classes.json
    if c < 5 { bail!("C terlalu kecil ({}). Harus 4+nc.", c); }
    let nc = c - 4;
    if nc != classes.0.len() {
        bail!("Mismatch kelas: C-4={} tapi classes.json berisi {}", nc, classes.0.len());
    }

    // CUDA stream & buffers
    cust::quick_init()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let in_elems = 1*3*(args.imgsz as usize)*(args.imgsz as usize);
    let mut d_input: DeviceBuffer<f32> = DeviceBuffer::uninitialized(in_elems)?;
    d_input.copy_from(&input_host)?;

    let out_elems = 1*c*n;
    let mut d_output: DeviceBuffer<f32> = DeviceBuffer::uninitialized(out_elems)?;

    // binding pointers
    let mut bindings: [*mut c_void; 2] = [
        d_input.as_device_ptr().as_raw() as *mut c_void,
        d_output.as_device_ptr().as_raw() as *mut c_void,
    ];

    // enqueue
    ctx.enqueue_v2(bindings.as_mut_ptr(), &stream).context("enqueue_v2")?;
    stream.synchronize()?;

    // copy back
    let mut out_host = vec![0f32; out_elems];
    d_output.copy_to(&mut out_host)?;

    // decode + nms
    let mut dets = decode_yolo_v8(
        &out_host, c, n, args.conf, &classes.0,
        img.width(), img.height(), scale, pad_x, pad_y, args.imgsz,
        args.filter_vehicle,
    )?;
    dets = nms(dets, args.iou);

    println!("count: {}", dets.len());

    // draw & save
    let mut out_img = img.clone();
    for d in &dets {
        draw_rect_rgb(&mut out_img, d.x1 as i32, d.y1 as i32, d.x2 as i32, d.y2 as i32, 2, [0,255,0]);
    }
    out_img.save(&args.save).with_context(|| format!("save {:?}", &args.save))?;
    println!("saved: {:?}", &args.save);
    Ok(())
}
