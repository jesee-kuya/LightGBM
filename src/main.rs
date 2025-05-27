use std::error::Error;
use std::collections::HashMap;

mod reader;
mod processor;
mod histogram;
mod builder;
mod booster;
mod predictor;
mod writer;

use reader::read_records;
use predictor::{TargetField, train_models, evaluate_models, extract_features};
use writer::write_predictions_to_csv;
use crate::booster::Booster;

fn main() -> Result<(), Box<dyn Error>> {
    // File paths (can be parameterized or read from CLI)
    let train_path = "data/train.csv";
    let test_path = "data/test.csv";
    let output_csv = "predictions.csv";

    println!("Loading training data from '{}'...", train_path);
    let train_records = read_records(train_path)?;

    println!("Loading test data from '{}'...", test_path);
    let test_records = read_records(test_path)?;

    // Train a separate model per target field
    println!("Training models for target fields...");
    let  models: HashMap<TargetField, Booster> = train_models(&train_records);

    // Save each trained model to disk
    for target in TargetField::all() {
        let filename = format!("model_{}.bin", target.name().replace(' ', "_"));
        if let Some(model) = models.get(&target) {
            model.save(&filename)?;
            println!("Saved model for '{}' to {}", target.name(), filename);
        }
    }

    // Evaluate on test set
    println!("Evaluating models on test data...");
    evaluate_models(&models, &test_records);

    // Generate and collect predictions
    println!("Generating predictions for test data...");
    let test_features = extract_features(&test_records);
    let mut predictions_map: HashMap<TargetField, Vec<f64>> = HashMap::new();
    for target in TargetField::all() {
        if let Some(model) = models.get(&target) {
            let preds = model.predict_batch(&test_features);
            predictions_map.insert(target.clone(), preds);
        }
    }

    // Write predictions to CSV
    println!("Writing predictions to '{}'...", output_csv);
    write_predictions_to_csv(&output_csv, &test_records, &predictions_map)?;
    println!("All done!");

    Ok(())
}
