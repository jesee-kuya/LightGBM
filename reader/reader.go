package reader

import (
	"encoding/csv"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/jesee-kuya/LightGBM/model"
)

// ReadCSV reads a CSV file and returns a slice of DataRecord
func ReadCSV(path string) ([]model.DataRecord, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	reader.TrimLeadingSpace = true
	reader.FieldsPerRecord = -1 // variable column count

	// Read header
	headers, err := reader.Read()
	if err != nil {
		return nil, err
	}

	var records []model.DataRecord
	for {
		row, err := reader.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			return nil, err
		}

		rowMap := map[string]string{}
		for i, h := range headers {
			if i < len(row) {
				rowMap[strings.ToLower(strings.TrimSpace(h))] = row[i]
			}
		}

		years, _ := strconv.ParseFloat(rowMap["years of experience"], 64)

		records = append(records, model.DataRecord{
			ID:              rowMap["master_index"],
			County:          strings.ToLower(rowMap["county"]),
			HealthLevel:     strings.ToLower(rowMap["health level"]),
			YearsExperience: years,
			Prompt:          rowMap["prompt"],
			Competency:      strings.ToLower(rowMap["nursing competency"]),
			Panel:           strings.ToLower(rowMap["clinical panel"]),
			Clinician:       rowMap["clinician"],
			GPT4:            rowMap["gpt4.0"],
			LLAMA:           rowMap["llama"],
			GEMINI:          rowMap["gemini"],
			DDXSNOMED:       rowMap["ddx snomed"],
		})
	}
	return records, nil
}
