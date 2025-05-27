use std::collections::HashMap;

#[allow(dead_code)]
pub fn build_category_map(values: &[Option<String>]) -> HashMap<String, u8> {
    let mut map = HashMap::new();
    let mut index = 0u8;
    for val in values.iter().filter_map(|v| v.as_ref()) {
        if !map.contains_key(val) && index < 255 {
            map.insert(val.clone(), index);
            index += 1;
        }
    }
    map.insert("__MISSING__".into(), 255);
    map
}

#[allow(dead_code)]
pub fn compute_bin_edges(data: &[f64], num_bins: usize) -> Vec<f64> {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut edges = Vec::new();
    for i in 1..num_bins {
        let idx = i * sorted.len() / num_bins;
        edges.push(sorted[idx]);
    }
    edges
}