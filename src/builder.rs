use crate::histogram::build_histogram;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone)]
#[allow (dead_code)]
pub struct SplitResult {
    pub split_bin: usize,
    pub gain: f64,
    pub left_sum: f64,
    pub right_sum: f64,
}

/// Finds the best split from a histogram by evaluating all possible split points.
#[allow(dead_code)]
pub fn find_best_split(histogram: &[f64], lambda: f64) -> Option<SplitResult> {
    let total_grad: f64 = histogram.iter().sum();
    let mut best_gain = f64::MIN;
    let mut best_split = None;

    let mut left_sum = 0.0;

    for i in 0..histogram.len() - 1 {
        left_sum += histogram[i];
        let right_sum = total_grad - left_sum;

        // Gain = (G_L^2 / (H_L + lambda)) + (G_R^2 / (H_R + lambda)) - (G^2 / (H + lambda))
        let gain = (left_sum.powi(2) / (left_sum.abs() + lambda))
            + (right_sum.powi(2) / (right_sum.abs() + lambda))
            - (total_grad.powi(2) / (total_grad.abs() + lambda));

        if gain > best_gain {
            best_gain = gain;
            best_split = Some(SplitResult {
                split_bin: i,
                gain,
                left_sum,
                right_sum,
            });
        }
    }
    best_split
}

#[derive(Debug, Serialize, Deserialize)]
#[allow(dead_code)]
pub enum TreeNode {
    Leaf { value: f64 },
    Internal {
        feature_index: usize,
        threshold_bin: usize,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
}

#[allow(dead_code)]
pub fn build_tree(
    binned_features: &[Vec<u8>],
    gradients: &[f64],
    depth: usize,
    max_depth: usize,
    num_bins: usize,
    lambda: f64,
) -> TreeNode {
    if depth >= max_depth || gradients.len() <= 1 {
        let leaf_val = gradients.iter().sum::<f64>() / (gradients.len() as f64 + 1e-6);
        return TreeNode::Leaf { value: leaf_val };
    }

    let mut best_gain = f64::MIN;
    let mut best_split = None;

    for feature_index in 0..binned_features[0].len() {
        let histogram = build_histogram(binned_features, gradients, feature_index, num_bins);
        if let Some(split) = find_best_split(&histogram, lambda) {
            if split.gain > best_gain {
                best_gain = split.gain;
                best_split = Some((feature_index, split));
            }
        }
    }

    if let Some((feature_index, split)) = best_split {
        let mut left_indices = vec![];
        let mut right_indices = vec![];

        for (i, sample) in binned_features.iter().enumerate() {
            if sample[feature_index] as usize <= split.split_bin {
                left_indices.push(i);
            } else {
                right_indices.push(i);
            }
        }

        let left_features = left_indices.iter().map(|&i| binned_features[i].clone()).collect::<Vec<_>>();
        let right_features = right_indices.iter().map(|&i| binned_features[i].clone()).collect::<Vec<_>>();
        let left_grads = left_indices.iter().map(|&i| gradients[i]).collect::<Vec<_>>();
        let right_grads = right_indices.iter().map(|&i| gradients[i]).collect::<Vec<_>>();

        let left_tree = build_tree(&left_features, &left_grads, depth + 1, max_depth, num_bins, lambda);
        let right_tree = build_tree(&right_features, &right_grads, depth + 1, max_depth, num_bins, lambda);

        TreeNode::Internal {
            feature_index,
            threshold_bin: split.split_bin,
            left: Box::new(left_tree),
            right: Box::new(right_tree),
        }
    } else {
        let leaf_val = gradients.iter().sum::<f64>() / (gradients.len() as f64 + 1e-6);
        TreeNode::Leaf { value: leaf_val }
    }
}