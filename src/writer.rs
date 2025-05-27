use csv::Writer;
use std::collections::HashMap;
use crate::reader::DataRecord;
use crate::predictor::TargetField;

pub fn write_predictions_to_csv(
    path: &str,
    test_records: &[DataRecord],
    predictions: &HashMap<TargetField, Vec<f64>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut wtr = Writer::from_path(path)?;
    let header = [
        "Master_Index", "Clinician", "GPT4.0", "LLAMA", "GEMINI", "DDX SNOMED",
    ];
    wtr.write_record(&header)?;

    for (i, rec) in test_records.iter().enumerate() {
        let row = vec![
            rec.master_index.clone().unwrap_or_else(|| "NA".into()),
            predictions
                .get(&TargetField::Clinician)
                .and_then(|v| v.get(i))
                .map_or("".into(), |v| format!("{:.4}", v)),
            predictions
                .get(&TargetField::GPT4_0)
                .and_then(|v| v.get(i))
                .map_or("".into(), |v| format!("{:.4}", v)),
            predictions
                .get(&TargetField::LLAMA)
                .and_then(|v| v.get(i))
                .map_or("".into(), |v| format!("{:.4}", v)),
            predictions
                .get(&TargetField::GEMINI)
                .and_then(|v| v.get(i))
                .map_or("".into(), |v| format!("{:.4}", v)),
            predictions
                .get(&TargetField::DDXSNOMED)
                .and_then(|v| v.get(i))
                .map_or("".into(), |v| format!("{:.4}", v)),
        ];
        wtr.write_record(&row)?;
    }

    wtr.flush()?;
    Ok(())
}
