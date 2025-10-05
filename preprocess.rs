use anyhow::*;
use ndarray::{Array, IxDyn};
use opencv::{core, imgproc, prelude::*};

pub fn letterbox_bgr_to_rgb_f32_nchw(mat_bgr: &Mat, new_size: i32) -> Result<Array<f32, IxDyn>> {
    let (h, w) = (mat_bgr.rows(), mat_bgr.cols());
    let scale = (new_size as f32 / w as f32).min(new_size as f32 / h as f32);
    let nw = ((w as f32) * scale).round() as i32;
    let nh = ((h as f32) * scale).round() as i32;

    let mut resized = Mat::default();
    imgproc::resize(
        mat_bgr,
        &mut resized,
        core::Size { width: nw, height: nh },
        0.0, 0.0,
        imgproc::INTER_LINEAR,
    )?;

    let dw = new_size - nw;
    let dh = new_size - nh;
    let top = dh / 2;
    let bottom = dh - top;
    let left = dw / 2;
    let right = dw - left;

    let mut padded = Mat::default();
    // NOTE: fungsi & konstanta dari core::
    core::copy_make_border(
        &resized,
        &mut padded,
        top, bottom, left, right,
        core::BORDER_CONSTANT,
        core::Scalar::new(114.0, 114.0, 114.0, 0.0),
    )?;

    let mut rgb = Mat::default();
    imgproc::cvt_color(&padded, &mut rgb, imgproc::COLOR_BGR2RGB, 0)?;

    let mut f32img = Mat::default();
    rgb.convert_to(&mut f32img, core::CV_32F, 1.0 / 255.0, 0.0)?;

    // Ambil slice data bertipe f32 secara aman
    let data: &[f32] = f32img.data_typed::<f32>()?;
    let rows = f32img.rows() as usize;
    let cols = f32img.cols() as usize;
    let chans = f32img.channels() as usize;
    ensure!(chans == 3, "expected 3 channels");
    ensure!(data.len() == rows * cols * chans, "data length mismatch");

    // HWC -> CHW (N=1)
    let mut chw = vec![0f32; data.len()];
    let plane = rows * cols;
    for y in 0..rows {
        for x in 0..cols {
            let base = (y * cols + x) * 3;
            chw[0 * plane + (y * cols + x)] = data[base + 0];
            chw[1 * plane + (y * cols + x)] = data[base + 1];
            chw[2 * plane + (y * cols + x)] = data[base + 2];
        }
    }

    Ok(Array::from_shape_vec(IxDyn(&[1, 3, new_size as usize, new_size as usize]), chw)?)
}
