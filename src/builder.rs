use std::collections::HashMap;

#[derive(Debug, Clone)]
#[derive(Default)]
#[allow (dead_code)]
pub struct SplitResult {
    pub split_bin: usize,
    pub gain: f64,
    pub left_sum: f64,
    pub right_sum: f64,
}