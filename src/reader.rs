use serde::Deserialize;
use std::{error::Error, path::Path};

/// Single row from CSV, all fields optional.
#[derive(Debug, Deserialize, Default, Clone)]
#[allow(dead_code)]
pub struct DataRecord {
    #[serde(rename = "Master_Index", default)]
    pub master_index: Option<String>,
    #[serde(rename = "County", default)]
    pub county: Option<String>,
    #[serde(rename = "Health level", default)]
    pub health_level: Option<String>,
    #[serde(rename = "Years of Experience", default)]
    pub years_experience: Option<String>,
    #[serde(rename = "Prompt", default)]
    pub prompt: Option<String>,
    #[serde(rename = "Nursing Competency", default)]
    pub nursing_competency: Option<String>,
    #[serde(rename = "Clinical Panel", default)]
    pub clinical_panel: Option<String>,
    #[serde(rename = "Clinician", default)]
    pub clinician: Option<String>,
    #[serde(rename = "GPT4.0", default)]
    pub gpt4_0: Option<String>,
    #[serde(rename = "LLAMA", default)]
    pub llama: Option<String>,
    #[serde(rename = "GEMINI", default)]
    pub gemini: Option<String>,
    #[serde(rename = "DDX SNOMED", default)]
    pub ddx_snomed: Option<String>,
}

/// Read and return all valid records from `path`.
/// Errors if no records found.
#[allow(dead_code)]
pub fn read_records<P: AsRef<Path>>(path: P) -> Result<Vec<DataRecord>, Box<dyn Error>> {
    let path = path.as_ref();
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .trim(csv::Trim::All)
        .flexible(true)
        .from_path(path)?;

    let mut rows = Vec::new();
    for result in rdr.deserialize::<DataRecord>() {
        match result {
            Ok(rec) => {
                if rec.clinician.is_none() && rec.gpt4_0.is_none() {
                    eprintln!(
                        "Warning: '{}' record missing both Clinician & GPT4.0 targets",
                        path.display()
                    );
                }
                rows.push(rec);
            }
            Err(e) => eprintln!("  skipped malformed row in {}: {}", path.display(), e),
        }
    }

    if rows.is_empty() {
        Err(format!("No valid records in {}", path.display()).into())
    } else {
        Ok(rows)
    }
}
