use anyhow::*;
use ndarray::{Array, IxDyn};
use opencv::{core, imgproc, prelude::*};

/// Letterbox ke NxN, BGR -> RGB f32 [0..1], HWC->NCHW (1,3,N,N)
pub fn letterbox_bgr_to_rgb_f32_nchw(mat_bgr: &Mat, new_size: i32) -> Result<Array<f32, IxDyn>> {
    let (h, w) = (mat_bgr.rows(), mat_bgr.cols());
    let scale = (new_size as f32 / w as f32).min(new_size as f32 / h as f32);
    let nw = ((w as f32) * scale).round() as i32;
    let nh = ((h as f32) * scale).round() as i32;

    let mut resized = Mat::default();
    imgproc::resize(mat_bgr, &mut resized, core::Size { width: nw, height: nh }, 0.0, 0.0, imgproc::INTER_LINEAR)?;

    let dw = new_size - nw;
    let dh = new_size - nh;
    let top = dh / 2;
    let bottom = dh - top;
    let left = dw / 2;
    let right = dw - left;

    let mut padded = Mat::default();
    imgproc::copy_make_border(
        &resized, &mut padded, top, bottom, left, right,
        imgproc::BORDER_CONSTANT, core::Scalar::new(114.0,114.0,114.0,0.0)
    )?;

    let mut rgb = Mat::default();
    imgproc::cvt_color(&padded, &mut rgb, imgproc::COLOR_BGR2RGB, 0)?;

    let mut f32img = Mat::default();
    rgb.convert_to(&mut f32img, core::CV_32F, 1.0/255.0, 0.0)?;

    // HWC -> CHW (N=1)
    let (rows, cols) = (f32img.rows(), f32img.cols());
    let chans = f32img.channels();
    ensure!(chans == 3, "expected 3 channels");
    let total = (rows * cols * chans) as usize;

    let data: &[f32] = unsafe {
        std::slice::from_raw_parts(f32img.ptr(0) as *const f32, total)
    };

    let mut chw = vec![0f32; total];
    let plane = (rows * cols) as usize;
    for y in 0..rows {
        for x in 0..cols {
            for c in 0..3 {
                let hwc_idx = (y * cols * 3 + x * 3 + c) as usize;
                let chw_idx = c as usize * plane + (y * cols + x) as usize;
                chw[chw_idx] = data[hwc_idx];
            }
        }
    }

    Ok(Array::from_shape_vec(IxDyn(&[1, 3, new_size as usize, new_size as usize]), chw)?)
}
