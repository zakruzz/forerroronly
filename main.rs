use anyhow::{Context, Result};
let mut matched: HashMap<u64, (f32,f32,f32,f32)> = HashMap::new();
let mut used = vec![false; dets.len()];
for (tid, tr) in tracks.iter_mut() {
// find best IoU detection
let mut best = (0usize, 0f32);
for (i, d) in dets.iter().enumerate() {
if used[i] { continue; }
let bb = (d.x1, d.y1, d.x2, d.y2);
let iou_v = iou(tr.bbox, bb);
if iou_v > best.1 { best = (i, iou_v); }
}
if best.1 > 0.3 {
let d = dets[best.0];
tr.bbox = (d.x1,d.y1,d.x2,d.y2);
tr.hits += 1;
tr.last_seen = Instant::now();
matched.insert(*tid, tr.bbox);
used[best.0] = true;
}
}
// create tracks for unmatched detections
for (i, d) in dets.iter().enumerate() {
if used[i] { continue; }
let id = next_id; next_id += 1;
tracks.insert(id, Track { id, bbox: (d.x1,d.y1,d.x2,d.y2), last_seen: Instant::now(), hits: 1 });
}
// drop stale tracks
tracks.retain(|_, tr| tr.last_seen.elapsed() < Duration::from_millis(800));


// --- simplistic counting: horizontal virtual line at 1/2 height ---
let y_line = (h as f32) * 0.5;
for (_id, tr) in tracks.iter_mut() {
let (_, y1, _, y2) = tr.bbox;
let cy = (y1 + y2) / 2.0;
// if a track just crossed the line this frame (toggle via hits)
if (cy - y_line).abs() < 5.0 && tr.hits == 1 { total_count += 1; }
}


// --- draw preview ---
if let Some(win) = &mut window {
let mut img = ImageBuffer::<Rgb<u8>, _>::from_raw(w, h, rgb.to_vec()).unwrap();
for (_, tr) in tracks.iter() {
let (x1,y1,x2,y2) = tr.bbox;
let r = Rect::at(x1 as i32, y1 as i32).of_size((x2-x1) as u32, (y2-y1) as u32);
draw_hollow_rect_mut(&mut img, r, Rgb([0,255,0]));
}
// line
let rline = Rect::at(0, y_line as i32).of_size(w, 2);
draw_filled_rect_mut(&mut img, rline, Rgb([255,0,0]));


// FPS text (quick & dirty: skip text, just update window title)
let now = Instant::now();
let dt = now.duration_since(last).as_secs_f32();
last = now;
let fps = 1.0 / dt.max(1e-3);
win.set_title(&format!("Counter — total={} — {:.1} FPS", total_count, fps));
win.update_with_buffer(img.as_raw(), w as usize, h as usize).unwrap();
}
}
}
