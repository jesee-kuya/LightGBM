package writer

import (
	"encoding/csv"
	"os"

	"github.com/jesee-kuya/LightGBM/booster"
	"github.com/jesee-kuya/LightGBM/model"
	"github.com/jesee-kuya/LightGBM/preprocess"
	"github.com/jesee-kuya/LightGBM/util"
)

// writePredictions outputs CSV: ID,Clinician,GPT4.0,LLAMA,GEMINI,DDX_SNOMED
func WritePredictions(
	records []model.DataRecord,
	Xall [][]float64,
	boost *booster.Booster,
	pre *preprocess.Preprocessor,
	outPath string,
) error {
	f, err := os.Create(outPath)
	if err != nil {
		return err
	}
	defer f.Close()

	w := csv.NewWriter(f)
	defer w.Flush()

	header := []string{"Master_Index", "Clinician", "GPT4.0", "LLAMA", "GEMINI", "DDX_SNOMED"}
	if err := w.Write(header); err != nil {
		return err
	}

	clinLabels := pre.ClinicianClasses()
	gpt4Labels := pre.GPT4Classes()
	llamaLabels := pre.LLAMAClasses()
	geminiLabels := pre.GEMINIClasses()
	ddxLabels := pre.DDXClasses()

	for i, rec := range records {
		pred := boost.Predict(Xall[i])
		clinIdx := util.Clamp(pred[0], len(clinLabels))
		gpt4Idx := util.Clamp(pred[1], len(gpt4Labels))
		llamaIdx := util.Clamp(pred[2], len(llamaLabels))
		geminiIdx := util.Clamp(pred[3], len(geminiLabels))
		ddxIdx := util.Clamp(pred[4], len(ddxLabels))

		row := []string{
			rec.ID,
			clinLabels[clinIdx],
			gpt4Labels[gpt4Idx],
			llamaLabels[llamaIdx],
			geminiLabels[geminiIdx],
			ddxLabels[ddxIdx],
		}
		if err := w.Write(row); err != nil {
			return err
		}
	}

	return nil
}
