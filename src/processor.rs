use std::collections::HashMap;
use crate::reader::DataRecord;

#[derive(Debug)]
#[allow(dead_code)]
pub struct ProcessedSample {
    pub features: Vec<u8>, 
    pub label: f64,    
    pub weight: f64,       
}

#[derive(Debug)]
pub struct PreprocessingContext {
    pub categorical_maps: HashMap<String, HashMap<String, u8>>,
    pub feature_bins: HashMap<String, Vec<f64>>,
    pub label_source: String,                 
}

#[allow(dead_code)]
pub fn preprocess_record(
    record: &DataRecord,
    ctx: &PreprocessingContext,
) -> Option<ProcessedSample> {
    let mut features = Vec::new();

    // Process categorical features
    let cat_features = [
        ("county", &record.county),
        ("health_level", &record.health_level),
        ("nursing_competency", &record.nursing_competency),
        ("clinical_panel", &record.clinical_panel),
    ];

    for (key, val_opt) in cat_features.iter() {
        let val = val_opt.as_deref().unwrap_or("__MISSING__");
        let map = ctx
            .categorical_maps
            .get(*key)
            .expect(&format!("Missing map for {}", key));
        let bin = map.get(val).cloned().unwrap_or(255); // 255 = unknown
        features.push(bin);
    }

    // Process numeric features
    let num_features = [("years_experience", &record.years_experience)];

    for (key, val_opt) in num_features.iter() {
        let raw = val_opt.as_deref().unwrap_or("");
        let parsed = raw.parse::<f64>().ok();
        let bin_edges = ctx.feature_bins.get(*key).expect("Missing bin edges");
        let bin = match parsed {
            Some(value) => bin_edges.iter().position(|&e| value <= e).unwrap_or(254) as u8,
            None => 255, // missing
        };
        features.push(bin);
    }

    // Extract label
    let label_str = match ctx.label_source.as_str() {
        "clinician" => record.clinician.as_deref(),
        "gpt4_0" => record.gpt4_0.as_deref(),
        "llama" => record.llama.as_deref(),
        "gemini" => record.gemini.as_deref(),
        "ddx_snomed" => record.ddx_snomed.as_deref(),
        _ => None,
    };

    let label = label_str?.parse::<f64>().ok()?; // Skip if label missing or malformed

    Some(ProcessedSample {
        features,
        label,
        weight: 1.0,
    })
}
