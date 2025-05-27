use crate::builder::TreeNode;
use crate::booster::Booster;
use crate::reader::DataRecord;

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

use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[allow(dead_code)]
pub enum TargetField {
    Clinician,
    GPT4_0,
    LLAMA,
    GEMINI,
    DDXSNOMED,
}

#[allow(dead_code)]
impl TargetField {
    pub fn all() -> Vec<Self> {
        vec![
            Self::Clinician,
            Self::GPT4_0,
            Self::LLAMA,
            Self::GEMINI,
            Self::DDXSNOMED,
        ]
    }

    pub fn extract(&self, record: &DataRecord) -> Option<f64> {
        use TargetField::*;
        let val = match self {
            Clinician => &record.clinician,
            GPT4_0 => &record.gpt4_0,
            LLAMA => &record.llama,
            GEMINI => &record.gemini,
            DDXSNOMED => &record.ddx_snomed,
        };
        val.as_ref().and_then(|s| s.parse().ok())
    }

    pub fn name(&self) -> &'static str {
        match self {
            TargetField::Clinician => "Clinician",
            TargetField::GPT4_0 => "GPT4.0",
            TargetField::LLAMA => "LLAMA",
            TargetField::GEMINI => "GEMINI",
            TargetField::DDXSNOMED => "DDX SNOMED",
        }
    }
}

#[allow(dead_code)]
pub fn extract_features(records: &[DataRecord]) -> Vec<Vec<f64>> {
    records
        .iter()
        .map(|r| {
            vec![
                r.county.as_ref().map_or(0.0, |s| s.len() as f64),
                r.health_level.as_ref().map_or(0.0, |s| s.len() as f64),
                r.years_experience
                    .as_ref()
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0),
                r.prompt.as_ref().map_or(0.0, |s| s.len() as f64),
                r.nursing_competency.as_ref().map_or(0.0, |s| s.len() as f64),
                r.clinical_panel.as_ref().map_or(0.0, |s| s.len() as f64),
            ]
        })
        .collect()
}


pub fn train_models(records: &[DataRecord]) -> HashMap<TargetField, Booster> {
    let features = extract_features(records);
    let mut models = HashMap::new();

    for target in TargetField::all() {
        let targets: Vec<f64> = records
            .iter()
            .map(|r| target.extract(r).unwrap_or(0.0))
            .collect();

        let mut booster = Booster::new(0.1, 3, 8, 1.0);
        booster.train(&features, &targets, 10);
        models.insert(target, booster);
    }

    models
}


pub fn evaluate_models(
    models: &HashMap<TargetField, Booster>,
    test_records: &[DataRecord],
) {
    let features = extract_features(test_records);

    for target in TargetField::all() {
        if let Some(model) = models.get(&target) {
            let predictions = model.predict_batch(&features);
            let actuals: Vec<f64> = test_records
                .iter()
                .map(|r| target.extract(r).unwrap_or(0.0))
                .collect();
            let mse = compute_mse(&predictions, &actuals);
            println!("{} - Test MSE: {:.6}", target.name(), mse);
        }
    }
}
