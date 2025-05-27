use crate::builder::{TreeNode, build_tree};
use crate::histogram::{compute_bin_edges, bin_continuous};
use crate::predictor::predict;

#[derive(Debug)]
#[allow (dead_code)]
pub struct Booster {
    pub trees: Vec<TreeNode>,
    pub learning_rate: f64,
    pub max_depth: usize,
    pub num_bins: usize,
    pub lambda: f64,
}

#[allow(dead_code)]
impl Booster {
    pub fn new(learning_rate: f64, max_depth: usize, num_bins: usize, lambda: f64) -> Self {
        Booster {
            trees: Vec::new(),
            learning_rate,
            max_depth,
            num_bins,
            lambda,
        }
    }

    pub fn train(&mut self, features: &[Vec<f64>], targets: &[f64], n_rounds: usize) {
        let mut predictions = vec![0.0; targets.len()];

        for _ in 0..n_rounds {
            let gradients: Vec<f64> = targets
                .iter()
                .zip(predictions.iter())
                .map(|(&y, &y_pred)| y - y_pred)
                .collect();

            let bin_edges: Vec<Vec<f64>> = (0..features[0].len())
                .map(|j| compute_bin_edges(features.iter().map(|row| row[j]).collect(), self.num_bins))
                .collect();

            let binned_features: Vec<Vec<u8>> = features
                .iter()
                .map(|row| {
                    row.iter()
                        .enumerate()
                        .map(|(j, &val)| bin_continuous(val, &bin_edges[j]))
                        .collect()
                })
                .collect();

            let tree = build_tree(
                &binned_features,
                &gradients,
                0,
                self.max_depth,
                self.num_bins,
                self.lambda,
            );

            // Update predictions
            for (i, pred) in predictions.iter_mut().enumerate() {
                *pred += self.learning_rate * predict(&tree, &binned_features[i]);
            }

            self.trees.push(tree);
        }
    }

    pub fn predict_batch(&self, features: &[Vec<f64>]) -> Vec<f64> {
        let bin_edges: Vec<Vec<f64>> = (0..features[0].len())
            .map(|j| compute_bin_edges(features.iter().map(|row| row[j]).collect(), self.num_bins))
            .collect();

        let binned_features: Vec<Vec<u8>> = features
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(j, &val)| bin_continuous(val, &bin_edges[j]))
                    .collect()
            })
            .collect();

        binned_features
            .iter()
            .map(|sample| {
                self.trees
                    .iter()
                    .map(|tree| self.learning_rate * predict(tree, sample))
                    .sum::<f64>()
            })
            .collect()
    }
}

