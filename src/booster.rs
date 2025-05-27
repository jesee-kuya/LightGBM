use crate::builder::TreeNode;

#[derive(Debug)]
#[allow (dead_code)]
pub struct Booster {
    pub trees: Vec<TreeNode>,
    pub learning_rate: f64,
    pub max_depth: usize,
    pub num_bins: usize,
    pub lambda: f64,
}