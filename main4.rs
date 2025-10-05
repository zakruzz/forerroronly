use anyhow::{anyhow, Context, Result};
use image::{imageops, ImageBuffer, Rgb, RgbImage};
use ndarray::Array4;
use ndarray_npy::write_npy;
use std::{env, path::PathBuf};

fn letterbox_rgb(rgb:&RgbImage, dst:u32)->(RgbImage,f32,u32,u32){
    let (w,h)=(rgb.width() as f32, rgb.height() as f32);
    let s=(dst as f32 / w).min(dst as f32 / h);
    let nw=(w*s).round() as u32; let nh=(h*s).round() as u32;
    let resized=imageops::resize(rgb,nw,nh,imageops::FilterType::Triangle);
    let mut canvas=ImageBuffer::<Rgb<u8>,Vec<u8>>::from_pixel(dst,dst,Rgb([114,114,114]));
    let dx=(dst-nw)/2; let dy=(dst-nh)/2;
    imageops::replace(&mut canvas,&resized, dx.into(), dy.into());
    (canvas, s, dx, dy)
}
fn chw_normalize(inp:&RgbImage)->Vec<f32>{
    let (w,h)=(inp.width() as usize, inp.height() as usize);
    let mut out=vec![0f32; 3*w*h];
    for y in 0..h { for x in 0..w {
        let p=inp.get_pixel(x as u32, y as u32);
        let idx=y*w + x;
        out[idx] = p[0] as f32 / 255.0;
        out[w*h + idx] = p[1] as f32 / 255.0;
        out[2*w*h + idx] = p[2] as f32 / 255.0;
    }}
    out
}

fn main() -> Result<()> {
    // Usage: gen_npy <image_path> [imgsz=640] [out=/tmp/trt_in.npy]
    let a: Vec<String> = env::args().collect();
    if a.len() < 2 {
        eprintln!("Usage: {} <image> [imgsz=640] [out=/tmp/trt_in.npy]", a[0]);
        std::process::exit(1);
    }
    let img_path = &a[1];
    let imgsz: u32 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(640);
    let out_path = PathBuf::from(a.get(3).cloned().unwrap_or_else(|| "/tmp/trt_in.npy".to_string()));

    let img = image::open(img_path).with_context(|| format!("open {}", img_path))?.to_rgb8();
    let (letter, _s, _dx, _dy) = letterbox_rgb(&img, imgsz);
    let chw = chw_normalize(&letter);
    let arr: Array4<f32> = Array4::from_shape_vec((1,3,imgsz as usize,imgsz as usize), chw)
        .map_err(|_| anyhow!("shape mismatch"))?;
    write_npy(&out_path, &arr).with_context(|| format!("write {:?}", out_path))?;
    println!("Wrote {:?}", out_path);
    Ok(())
}
