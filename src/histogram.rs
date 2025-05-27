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
pub fn compute_bin_edges(mut values: Vec<f64>, num_bins: usize) -> Vec<f64> {
    if values.is_empty() || num_bins == 0 {
        return vec![];
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut edges = Vec::with_capacity(num_bins - 1);
    for i in 1..num_bins {
        let idx = i * values.len() / num_bins;
        if idx < values.len() {
            edges.push(values[idx]);
        }
    }
    edges
}

#[allow(dead_code)]
/// Convert a continuous value to a bin ID using bin edges.
pub fn bin_continuous(value: f64, edges: &[f64]) -> u8 {
    for (i, edge) in edges.iter().enumerate() {
        if value <= *edge {
            return i as u8;
        }
    }
    (edges.len().min(254)) as u8
}

#[allow(dead_code)]
/// Build gradient histograms per feature bin (simplified version).
pub fn build_histogram(
    binned_features: &[Vec<u8>],
    gradients: &[f64],
    feature_index: usize,
    num_bins: usize,
) -> Vec<f64> {
    let mut histogram = vec![0.0; num_bins];
    for (i, sample) in binned_features.iter().enumerate() {
        let bin = sample[feature_index] as usize;
        if bin < num_bins {
            histogram[bin] += gradients[i];
        }
    }
    histogram
}