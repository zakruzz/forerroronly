use anyhow::*;
use ndarray::{Array, IxDyn};
use ordered_float::OrderedFloat;

#[derive(Clone, Debug)]
pub struct Detection {
    pub x1:f32, pub y1:f32, pub x2:f32, pub y2:f32,
    pub score:f32, pub class_id:i32
}

#[inline] fn sigmoid(x:f32)->f32 { 1.0 / (1.0 + (-x).exp()) }

pub fn iou(a:&Detection,b:&Detection)->f32 {
    let x1=a.x1.max(b.x1); let y1=a.y1.max(b.y1);
    let x2=a.x2.min(b.x2); let y2=a.y2.min(b.y2);
    let inter=(x2-x1).max(0.0)*(y2-y1).max(0.0);
    let area_a=(a.x2-a.x1).max(0.0)*(a.y2-a.y1).max(0.0);
    let area_b=(b.x2-b.x1).max(0.0)*(b.y2-b.y1).max(0.0);
    inter/(area_a+area_b-inter+1e-6)
}

pub fn nms(mut dets:Vec<Detection>, th:f32)->Vec<Detection>{
    dets.sort_by_key(|d| OrderedFloat(-d.score));
    let mut keep=Vec::new();
    'outer: for d in dets {
        for k in &keep {
            if d.class_id==k.class_id && iou(&d,k)>th { continue 'outer; }
        }
        keep.push(d);
    }
    keep
}

/// Decoder fleksibel:
/// - [1, N, 5+num_cls] atau [1, 5+num_cls, N]
pub fn decode_flexible(out:&Array<f32, IxDyn>, conf_th:f32, count_cls:&[i32]) -> Result<Vec<Detection>> {
    let shp = out.shape();
    ensure!(shp.len()==3, "output dims {:?}", shp);
    let (n, k, transposed) = if shp[2] >= 6 { (shp[1], shp[2], false) } else { (shp[2], shp[1], true) };
    ensure!(k>=6, "last-dim (K) must be >=6, got {k}");
    let num_cls = k - 5;

    let mut dets=Vec::new();
    for i in 0..n {
        let get = |idx| -> f32 { if transposed { out[[0, idx, i]] } else { out[[0, i, idx]] } };
        let cx=get(0); let cy=get(1); let w=get(2); let h=get(3);
        let obj=sigmoid(get(4));
        let (mut best_c, mut best_p)=(-1, 0f32);
        for c in 0..num_cls { let p=sigmoid(get(5+c)); if p>best_p { best_p=p; best_c=c as i32; } }
        let score = obj*best_p;
        if score < conf_th { continue; }
        if !count_cls.is_empty() && !count_cls.contains(&best_c) { continue; }
        let x1=cx-w/2.0; let y1=cy-h/2.0; let x2=cx+w/2.0; let y2=cy+h/2.0;
        dets.push(Detection{x1,y1,x2,y2,score,class_id:best_c});
    }
    Ok(dets)
}
