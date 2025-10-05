use anyhow::*;
use serde_json::Value;
use std::{collections::HashMap, fs, path::Path};

pub fn load_classes_json<P: AsRef<Path>>(path: P) -> Result<Vec<String>> {
    let raw = fs::read_to_string(&path)
        .with_context(|| format!("gagal baca classes.json: {}", path.as_ref().display()))?;
    let v: Value = serde_json::from_str(&raw).context("JSON classes tidak valid")?;

    if let Some(names) = v.get("names") {
        return extract_names(names);
    }
    if v.is_array() {
        return extract_names(&v);
    }
    if v.is_object() {
        return extract_names(&v);
    }
    bail!("Format classes.json tidak dikenali.");
}

fn extract_names(v: &Value) -> Result<Vec<String>> {
    // Array langsung: ["car","bus",...]
    if let Some(arr) = v.as_array() {
        let mut out = Vec::with_capacity(arr.len());
        for (i, it) in arr.iter().enumerate() {
            let s = it.as_str().ok_or_else(|| anyhow!("elemen ke-{i} bukan string"))?;
            out.push(s.to_string());
        }
        return Ok(out);
    }

    // Object mapping: {"0":"car","1":"bus",...} atau {"names":{"0":"car",...}}
    if let Some(map) = v.as_object() {
        if let Some(inner) = map.get("names") {
            return extract_names(inner);
        }
        let mut tmp: HashMap<usize, String> = HashMap::new();
        let mut max_idx = 0usize;
        for (k, vv) in map {
            let idx: usize = k.parse().with_context(|| format!("key '{k}' bukan indeks angka"))?;
            let name = vv.as_str().ok_or_else(|| anyhow!("nilai untuk key '{k}' bukan string"))?;
            tmp.insert(idx, name.to_string());
            if idx > max_idx {
                max_idx = idx;
            }
        }
        let mut out = vec![String::new(); max_idx + 1];
        for (i, name) in tmp {
            out[i] = name;
        }
        return Ok(out);
    }

    bail!("Schema 'names' tidak dikenali (bukan array/object).");
}

/// arg: "2,3,5,7" atau "car,bus" (case-insensitive) → indeks unik
pub fn parse_count_arg(arg: &str, names: &[String]) -> Vec<i32> {
    let mut idxs: Vec<i32> = Vec::new();
    'outer: for tok in arg.split(',') {
        let t = tok.trim();
        if t.is_empty() { continue; }

        // angka?
        if let Ok(v) = t.parse::<i32>() {
            if v >= 0 && (v as usize) < names.len() && !idxs.contains(&v) {
                idxs.push(v);
            }
            continue 'outer;
        }

        // nama?
        let t_low = t.to_lowercase();
        if let Some(pos) = names.iter().position(|n| n.to_lowercase() == t_low) {
            let v = pos as i32;
            if !idxs.contains(&v) { idxs.push(v); }
        } else {
            eprintln!("[warn] token '{t}' tidak cocok angka/nama kelas manapun; diabaikan");
        }
    }
    idxs
}
