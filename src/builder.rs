use std::collections::HashMap;

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