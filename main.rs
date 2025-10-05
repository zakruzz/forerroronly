use anyhow::{Context, Result};
use async_cuda::{DeviceBuffer, HostBuffer, Stream};
use async_tensorrt::engine::{Engine, ExecutionContext, TensorIoMode};
use async_tensorrt::runtime::Runtime;
use opencv::{
    core::{self, Mat, MatTraitConst, Point, Rect, Scalar, Size, Vec3b},
    highgui, imgproc,
    videoio::{self, VideoCaptureTrait, VideoCaptureTraitConst},
};
use std::{collections::HashMap, fs, time::Instant};

// ==== Konfigurasi cepat ====
const INPUT_W: i32 = 640;
const INPUT_H: i32 = 640;
const CONF_THR: f32 = 0.20; // turunkan kalau gambar “susah”
const IOU_THR: f32 = 0.45;
const DRAW_BOXES: bool = true;

// ================= Utils =================
fn load_classes(path: &str) -> Result<Vec<String>> {
    let txt = fs::read_to_string(path).context("read classes.json")?;
    let v: Vec<String> = serde_json::from_str(&txt).context("parse classes.json")?;
    Ok(v)
}

// Letterbox: pertahankan aspek, pad 114 seperti Ultralytics
fn letterbox_to_square_bgr(bgr: &Mat, dst_w: i32, dst_h: i32) -> Result<Mat> {
    let w = bgr.cols() as f32;
    let h = bgr.rows() as f32;
    let r = (dst_w as f32 / w).min(dst_h as f32 / h);
    let new_w = (w * r).round() as i32;
    let new_h = (h * r).round() as i32;

    let mut resized = Mat::default();
    imgproc::resize(bgr, &mut resized, Size::new(new_w, new_h), 0.0, 0.0, imgproc::INTER_LINEAR)?;

    let pad_w = dst_w - new_w;
    let pad_h = dst_h - new_h;
    let top = pad_h / 2;
    let bottom = pad_h - top;
    let left = pad_w / 2;
    let right = pad_w - left;

    let mut padded = Mat::default();
    core::copy_make_border(
        &resized, &mut padded,
        top, bottom, left, right,
        core::BorderTypes::BORDER_CONSTANT as i32,
        Scalar::new(114.0,114.0,114.0,0.0),
    )?;
    Ok(padded)
}

// BGR u8 -> letterbox 640x640 BGR -> ke CHW f32 (R,G,B) [0..1]
fn make_input_chw(bgr_u8: &Mat) -> Result<Vec<f32>> {
    let ltb = letterbox_to_square_bgr(bgr_u8, INPUT_W, INPUT_H)?;
    let size = ltb.size()?;
    let (h, w) = (size.height, size.width);
    let hw = (h * w) as usize;

    let mut out = vec![0f32; 3 * hw];
    for y in 0..h {
        for x in 0..w {
            let p: Vec3b = *ltb.at_2d::<Vec3b>(y, x)?;
            // BGR->RGB + norm
            let r = p[2] as f32 / 255.0;
            let g = p[1] as f32 / 255.0;
            let b = p[0] as f32 / 255.0;
            let idx = (y * w + x) as usize;
            out[idx] = r;
            out[hw + idx] = g;
            out[2*hw + idx] = b;
        }
    }
    Ok(out)
}

#[derive(Clone)]
struct Det { x1:f32, y1:f32, x2:f32, y2:f32, conf:f32, cls:usize }

fn iou(a:&Det, b:&Det)->f32{
    let xx1=a.x1.max(b.x1); let yy1=a.y1.max(b.y1);
    let xx2=a.x2.min(b.x2); let yy2=a.y2.min(b.y2);
    let w=(xx2-xx1).max(0.0); let h=(yy2-yy1).max(0.0);
    let inter=w*h;
    let aa=(a.x2-a.x1).max(0.0)*(a.y2-a.y1).max(0.0);
    let bb=(b.x2-b.x1).max(0.0)*(b.y2-b.y1).max(0.0);
    inter/(aa+bb-inter+1e-6)
}

fn nms(mut dets:Vec<Det>, iou_thr:f32)->Vec<Det>{
    dets.sort_by(|a,b| b.conf.total_cmp(&a.conf));
    let mut keep=Vec::new();
    'o: for d in dets.into_iter(){
        for k in &keep{ if d.cls==k.cls && iou(&d,k)>iou_thr {continue 'o;} }
        keep.push(d);
    }
    keep
}

// ==== Decoder single-output: [1,N,5+C] atau [1,5+C,N] atau flat ====
fn decode_single_output(out:&[f32], shape:&[usize], num_classes:usize, conf_thr:f32)->Vec<Det>{
    let stride = 5 + num_classes;
    if stride==0 || out.is_empty(){ return vec![]; }
    let mut dets=Vec::new();

    if shape.len()>=3{
        let a=shape[1]; let b=shape[2];
        if a==stride{ // [1,5+C,N]
            let n=b;
            for i in 0..n{
                let cx=out[0*n+i]; let cy=out[1*n+i];
                let w =out[2*n+i]; let h =out[3*n+i];
                let obj=out[4*n+i];
                if obj<conf_thr {continue;}
                let (mut best_c,mut best_s)=(0usize,0f32);
                for c in 0..num_classes{
                    let s=out[(5+c)*n+i]; if s>best_s {best_s=s; best_c=c;}
                }
                let conf=obj*best_s; if conf<conf_thr {continue;}
                let (x1,y1,x2,y2)=(cx-w/2.0,cy-h/2.0,cx+w/2.0,cy+h/2.0);
                dets.push(Det{x1,y1,x2,y2,conf,cls:best_c});
            }
            return dets;
        } else if b==stride{ // [1,N,5+C]
            let n=a;
            for i in 0..n{
                let base=i*stride; if base+stride>out.len(){break;}
                let cx=out[base+0]; let cy=out[base+1];
                let w =out[base+2]; let h =out[base+3];
                let obj=out[base+4]; if obj<conf_thr {continue;}
                let (mut best_c,mut best_s)=(0usize,0f32);
                for c in 0..num_classes{
                    let s=out[base+5+c]; if s>best_s {best_s=s; best_c=c;}
                }
                let conf=obj*best_s; if conf<conf_thr {continue;}
                let (x1,y1,x2,y2)=(cx-w/2.0,cy-h/2.0,cx+w/2.0,cy+h/2.0);
                dets.push(Det{x1,y1,x2,y2,conf,cls:best_c});
            }
            return dets;
        }
    }
    // fallback flat
    if out.len()%stride==0{
        let n=out.len()/stride;
        for i in 0..n{
            let base=i*stride;
            let cx=out[base+0]; let cy=out[base+1];
            let w =out[base+2]; let h =out[base+3];
            let obj=out[base+4]; if obj<conf_thr {continue;}
            let (mut best_c,mut best_s)=(0usize,0f32);
            for c in 0..num_classes{
                let s=out[base+5+c]; if s>best_s {best_s=s; best_c=c;}
            }
            let conf=obj*best_s; if conf<conf_thr {continue;}
            let (x1,y1,x2,y2)=(cx-w/2.0,cy-h/2.0,cx+w/2.0,cy+h/2.0);
            dets.push(Det{x1,y1,x2,y2,conf,cls:best_c});
        }
    }
    dets
}

// ==== Decoder EfficientNMS (multi-output) ====
// Cari tensor yang "mirip" boxes (last dim 4), scores (last dim 1), classes (last dim 1)
fn pick_efficientnms_outputs(
    engine:&Engine, outputs:&[String]
) -> Option<(String,String,String)> {
    let mut boxes:Option<String>=None;
    let mut scores:Option<String>=None;
    let mut classes:Option<String>=None;

    for name in outputs {
        let shp = engine.tensor_shape(name);
        if shp.len()>=2 {
            let last = *shp.last().unwrap();
            if last==4 && boxes.is_none() { boxes = Some(name.clone()); continue; }
            if last==1 {
                if scores.is_none() { scores = Some(name.clone()); continue; }
                if classes.is_none() { classes = Some(name.clone()); continue; }
            }
        }
    }
    match (boxes,scores,classes){
        (Some(b),Some(s),Some(c)) => Some((b,s,c)),
        _ => None
    }
}

fn decode_efficientnms(
    boxes:&[f32], scores:&[f32], classes:&[f32], max_det:usize, conf_thr:f32
)->Vec<Det>{
    let mut dets=Vec::new();
    for i in 0..max_det {
        let sc = scores[i];
        if sc < conf_thr { continue; }
        let cls = if classes[i].is_finite() {
            classes[i].round().max(0.0) as usize
        } else { 0usize };
        // boxes layout biasanya [x1,y1,x2,y2]
        let base = i*4;
        if base+3 >= boxes.len() { break; }
        let x1=boxes[base+0]; let y1=boxes[base+1];
        let x2=boxes[base+2]; let y2=boxes[base+3];
        dets.push(Det{x1,y1,x2,y2,conf:sc,cls});
    }
    dets
}

// ================= Main =================
#[tokio::main]
async fn main() -> Result<()> {
    // classes
    let classes = load_classes("models/classes.json")?;
    let num_classes = classes.len();

    // camera
    let mut cap = videoio::VideoCapture::new(0, videoio::CAP_V4L)
        .context("open camera /dev/video0")?;
    if !cap.is_opened()? { anyhow::bail!("camera not opened"); }
    let _ = cap.set(videoio::CAP_PROP_FRAME_WIDTH, 1280.0);
    let _ = cap.set(videoio::CAP_PROP_FRAME_HEIGHT, 720.0);
    let _ = cap.set(videoio::CAP_PROP_FPS, 30.0);

    highgui::named_window("preview", highgui::WINDOW_NORMAL)?;
    highgui::resize_window("preview", 960, 540)?;

    // TensorRT
    let plan = fs::read("models/best_fp16.engine")
        .context("read models/best_fp16.engine")?;
    let rt = Runtime::new().await;
    let mut engine: Engine = rt.deserialize_engine(&plan).await.context("deserialize TRT")?;

    // IO names
    let mut input_name: Option<String>=None;
    let mut outputs: Vec<String> = Vec::new();
    for i in 0..engine.num_io_tensors() {
        let name = engine.io_tensor_name(i);
        match engine.tensor_io_mode(&name) {
            TensorIoMode::Input => if input_name.is_none(){ input_name=Some(name); },
            TensorIoMode::Output => outputs.push(name),
            _ => {}
        }
    }
    let input_name = input_name.context("no input tensor")?;
    if outputs.is_empty() { anyhow::bail!("no output tensor"); }

    // Single-output atau EfficientNMS?
    let single_output = outputs.len()==1;
    // Siapkan shape & buffer
    let in_shape = engine.tensor_shape(&input_name);
    let in_elems: usize = in_shape.iter().product();
    let mut ctx = ExecutionContext::new(&mut engine).await.context("create exec ctx")?;
    let stream = Stream::new().await.context("create CUDA stream")?;

    let mut h_in = HostBuffer::<f32>::new(in_elems).await;
    let mut d_in = DeviceBuffer::<f32>::new(in_elems, &stream).await;

    // Output buffer(s)
    // Single: satu tensor
    // EfficientNMS: boxes/scores/classes (abaikan num_dets)
    let mut d_out_single: Option<DeviceBuffer<f32>> = None;
    let mut h_out_single: Option<HostBuffer<f32>> = None;

    let mut d_boxes: Option<DeviceBuffer<f32>> = None;
    let mut d_scores: Option<DeviceBuffer<f32>> = None;
    let mut d_classes: Option<DeviceBuffer<f32>> = None;
    let mut h_boxes: Option<HostBuffer<f32>> = None;
    let mut h_scores: Option<HostBuffer<f32>> = None;
    let mut h_classes: Option<HostBuffer<f32>> = None;
    let mut max_det: usize = 0;

    if single_output {
        let out_shape = engine.tensor_shape(&outputs[0]);
        let out_elems: usize = out_shape.iter().product();
        d_out_single = Some(DeviceBuffer::<f32>::new(out_elems, &stream).await);
        h_out_single = Some(HostBuffer::<f32>::new(out_elems).await);
    } else {
        // coba tebak EfficientNMS
        if let Some((b,s,c)) = pick_efficientnms_outputs(&engine, &outputs) {
            let shp_b = engine.tensor_shape(&b); // [1,max_det,4] biasanya
            let shp_s = engine.tensor_shape(&s); // [1,max_det,1]
            let shp_c = engine.tensor_shape(&c); // [1,max_det,1]
            // ambil max_det dari tensor boxes
            max_det = if shp_b.len()>=2 { shp_b[shp_b.len()-2] } else { 0 };
            let nb = max_det * 4;
            let ns = max_det * 1;
            let nc = max_det * 1;
            d_boxes = Some(DeviceBuffer::<f32>::new(nb, &stream).await);
            d_scores = Some(DeviceBuffer::<f32>::new(ns, &stream).await);
            d_classes = Some(DeviceBuffer::<f32>::new(nc, &stream).await);
            h_boxes = Some(HostBuffer::<f32>::new(nb).await);
            h_scores = Some(HostBuffer::<f32>::new(ns).await);
            h_classes = Some(HostBuffer::<f32>::new(nc).await);
        } else {
            // fallback: treat as single-output (ambil yang pertama)
            let out_shape = engine.tensor_shape(&outputs[0]);
            let out_elems: usize = out_shape.iter().product();
            d_out_single = Some(DeviceBuffer::<f32>::new(out_elems, &stream).await);
            h_out_single = Some(HostBuffer::<f32>::new(out_elems).await);
        }
    }

    // Loop
    let mut frame = Mat::default();
    let mut disp  = Mat::default();
    let mut fps_counter = 0u32;
    let mut fps_last = Instant::now();
    let mut fps = 0.0f32;

    loop {
        if !cap.read(&mut frame)? || frame.empty() { break; }

        // Preview kecil
        imgproc::resize(&frame, &mut disp, Size::new(960, 540), 0.0, 0.0, imgproc::INTER_LINEAR)?;

        // Preprocess letterbox (match Ultralytics)
        let input = make_input_chw(&frame)?;
        h_in.copy_from_slice(&input);
        h_in.copy_to(&mut d_in, &stream).await.context("H->D input")?;

        // siapkan IO map
        let mut io_map: HashMap<&str, &mut DeviceBuffer<f32>> = HashMap::new();
        io_map.insert(&input_name, &mut d_in);

        if let (Some(d), Some(_)) = (d_out_single.as_mut(), h_out_single.as_ref()) {
            io_map.insert(&outputs[0], d);
        } else if d_boxes.is_some() && d_scores.is_some() && d_classes.is_some() {
            // Map ketiga output; cari nama yang cocok dari pick_efficientnms_outputs
            let (b,s,c) = pick_efficientnms_outputs(&engine, &outputs).unwrap();
            io_map.insert(&b, d_boxes.as_mut().unwrap());
            io_map.insert(&s, d_scores.as_mut().unwrap());
            io_map.insert(&c, d_classes.as_mut().unwrap());
        }

        // Inference
        ctx.enqueue(&mut io_map, &stream).await.context("enqueue")?;

        // Ambil hasil & decode
        let mut dets: Vec<Det> = Vec::new();
        if let (Some(d), Some(h)) = (d_out_single.as_ref(), h_out_single.as_mut()) {
            h.copy_from(d, &stream).await.context("D->H out")?;
            let out_host: Vec<f32> = h.to_vec();
            let out_shape = engine.tensor_shape(&outputs[0]);
            dets = decode_single_output(&out_host, &out_shape, num_classes, CONF_THR);
            dets = nms(dets, IOU_THR);
        } else if d_boxes.is_some() {
            h_boxes.as_mut().unwrap().copy_from(d_boxes.as_ref().unwrap(), &stream).await?;
            h_scores.as_mut().unwrap().copy_from(d_scores.as_ref().unwrap(), &stream).await?;
            h_classes.as_mut().unwrap().copy_from(d_classes.as_ref().unwrap(), &stream).await?;
            let boxes_host = h_boxes.as_ref().unwrap().to_vec();
            let scores_host = h_scores.as_ref().unwrap().to_vec();
            let classes_host = h_classes.as_ref().unwrap().to_vec();
            dets = decode_efficientnms(&boxes_host, &scores_host, &classes_host, max_det, CONF_THR);
            // umumnya EfficientNMS sudah NMS, jadi tidak perlu NMS lagi
        }

        // Filter & gambar
        let mut total = 0usize;
        for d in &dets {
            if let Some(name) = classes.get(d.cls) {
                let lname = name.to_lowercase();
                if lname == "cars" || lname == "motorcyle" || lname == "truck" {
                    total += 1;
                    if DRAW_BOXES {
                        let sx = 960.0 / INPUT_W as f32;
                        let sy = 540.0 / INPUT_H as f32;
                        let x1 = (d.x1 * sx).clamp(0.0, 959.0) as i32;
                        let y1 = (d.y1 * sy).clamp(0.0, 539.0) as i32;
                        let x2 = (d.x2 * sx).clamp(0.0, 959.0) as i32;
                        let y2 = (d.y2 * sy).clamp(0.0, 539.0) as i32;
                        let rect = Rect::new(x1, y1, (x2-x1).max(1), (y2-y1).max(1));
                        imgproc::rectangle(&mut disp, rect, Scalar::new(0.0,255.0,0.0,0.0), 2, imgproc::LINE_8, 0)?;
                        let label = format!("{} {:.2}", name, d.conf);
                        imgproc::put_text(&mut disp, &label, Point::new(x1, (y1-6).max(12)),
                            imgproc::FONT_HERSHEY_SIMPLEX, 0.6, Scalar::new(0.0,0.0,0.0,0.0), 3, imgproc::LINE_8, false)?;
                        imgproc::put_text(&mut disp, &label, Point::new(x1, (y1-6).max(12)),
                            imgproc::FONT_HERSHEY_SIMPLEX, 0.6, Scalar::new(0.0,255.0,0.0,0.0), 1, imgproc::LINE_8, false)?;
                    }
                }
            }
        }

        // FPS overlay
        let now = Instant::now();
        static mut FPS_LAST_INIT: bool = false;
        static mut FPS_LAST: Option<Instant> = None;
        static mut FPS_CNT: u32 = 0;
        unsafe {
            if !FPS_LAST_INIT { FPS_LAST = Some(now); FPS_LAST_INIT = true; }
            FPS_CNT += 1;
            if now.duration_since(FPS_LAST.unwrap()).as_secs_f32() >= 1.0 {
                let fps_val = FPS_CNT as f32 / now.duration_since(FPS_LAST.unwrap()).as_secs_f32();
                FPS_CNT = 0;
                FPS_LAST = Some(now);
                let overlay = format!("FPS: {:.1}   vehicles: {}", fps_val, total);
                imgproc::put_text(&mut disp, &overlay, Point::new(12, 28),
                    imgproc::FONT_HERSHEY_SIMPLEX, 0.9,
                    Scalar::new(0.0,0.0,0.0,0.0), 3, imgproc::LINE_8, false)?;
                imgproc::put_text(&mut disp, &overlay, Point::new(12, 28),
                    imgproc::FONT_HERSHEY_SIMPLEX, 0.9,
                    Scalar::new(0.0,255.0,255.0,0.0), 2, imgproc::LINE_8, false)?;
            } else {
                let overlay = format!("vehicles: {}", total);
                imgproc::put_text(&mut disp, &overlay, Point::new(12, 28),
                    imgproc::FONT_HERSHEY_SIMPLEX, 0.9,
                    Scalar::new(0.0,0.0,0.0,0.0), 3, imgproc::LINE_8, false)?;
                imgproc::put_text(&mut disp, &overlay, Point::new(12, 28),
                    imgproc::FONT_HERSHEY_SIMPLEX, 0.9,
                    Scalar::new(0.0,255.0,255.0,0.0), 2, imgproc::LINE_8, false)?;
            }
        }

        highgui::imshow("preview", &disp)?;
        let k = highgui::wait_key(1)?;
        if k == 27 || k == 'q' as i32 { break; }
    }

    Ok(())
}
