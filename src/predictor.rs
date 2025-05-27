use crate::builder::TreeNode;

#[allow(dead_code)]
pub fn predict(tree: &TreeNode, sample: &[u8]) -> f64 {
    match tree {
        TreeNode::Leaf { value } => *value,
        TreeNode::Internal {
            feature_index,
            threshold_bin,
            left,
            right,
        } => {
            if sample[*feature_index] as usize <= *threshold_bin {
                predict(left, sample)
            } else {
                predict(right, sample)
            }
        }
    }
}

#[allow(dead_code)]
pub fn compute_mse(predictions: &[f64], targets: &[f64]) -> f64 {
    predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f64>()
        / predictions.len() as f64
}