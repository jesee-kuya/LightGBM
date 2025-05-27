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