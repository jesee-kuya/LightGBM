use std::collections::HashMap;

#[derive(Debug)]
#[allow(dead_code)]
pub struct ProcessedSample {
    pub features: Vec<u8>, 
    pub label: f64,    
    pub weight: f64,       
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct PreprocessingContext {
    pub categorical_maps: HashMap<String, HashMap<String, u8>>,
    pub feature_bins: HashMap<String, Vec<f64>>,
    pub label_source: String,                 
}