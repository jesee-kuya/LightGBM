// main.go
package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"os"

	"github.com/jesee-kuya/LightGBM/booster"
	"github.com/jesee-kuya/LightGBM/model"
	"github.com/jesee-kuya/LightGBM/preprocess"
	"github.com/jesee-kuya/LightGBM/reader"
	"github.com/jesee-kuya/LightGBM/util"
)

func main() {
	// ─── 1) READ & MERGE TRAINING DATA ───
	cleanTrain, err := reader.ReadCSV("data/train.csv")
	if err != nil {
		log.Fatalf("failed to read data/train.csv: %v", err)
	}
	rawTrain, err := reader.ReadCSV("data/train_raw.csv")
	if err != nil {
		log.Fatalf("failed to read data/train_raw.csv: %v", err)
	}
	trainRecords := util.MergeByID(cleanTrain, rawTrain)
	if len(trainRecords) == 0 {
		log.Fatal("no training records after merging")
	}
	fmt.Printf("Merged training records: %d\n", len(trainRecords))

	// ─── 2) PREPROCESS ON TRAIN ───
	numBuckets := 100
	pre := preprocess.NewPreprocessor(numBuckets)
	pre.Fit(trainRecords)
	XtrainAll, YtrainAll := pre.Transform(trainRecords)

	// ─── 3) SPLIT TRAIN ↔ VALIDATION (80/20) ───
	N := len(trainRecords)
	trainIdx, valIdx := util.ShuffleSplit(N, 0.8)

	Xtrain := make([][]float64, len(trainIdx))
	Ytrain := make([][]float64, len(trainIdx))
	for i, idx := range trainIdx {
		Xtrain[i] = XtrainAll[idx]
		Ytrain[i] = YtrainAll[idx]
	}
	Xval := make([][]float64, len(valIdx))
	Yval := make([][]float64, len(valIdx))
	for i, idx := range valIdx {
		Xval[i] = XtrainAll[idx]
		Yval[i] = YtrainAll[idx]
	}

	// ─── 4) TRAIN THE BOOSTER ───
	numTargets := len(YtrainAll[0]) // should be 5
	boost := booster.NewBooster(numTargets, 0.1, 3, 5)
	boost.Fit(Xtrain, Ytrain, 50)
	fmt.Println("Training complete.")

	// ─── 5) EVALUATE ON VALIDATION ───
	evaluate(boost, Xval, Yval, pre)

	// ─── 6) READ & MERGE TEST DATA ───
	cleanTest, err := reader.ReadCSV("data/test.csv")
	if err != nil {
		log.Fatalf("failed to read data/test.csv: %v", err)
	}
	rawTest, err := reader.ReadCSV("data/test_raw.csv")
	if err != nil {
		log.Fatalf("failed to read data/test_raw.csv: %v", err)
	}
	testRecords := util.MergeByID(cleanTest, rawTest)
	if len(testRecords) == 0 {
		log.Fatal("no test records after merging")
	}
	fmt.Printf("Merged test records: %d\n", len(testRecords))

	// ─── 7) TRANSFORM TEST ───
	Xtest, _ := pre.Transform(testRecords)

	// ─── 8) WRITE TEST PREDICTIONS ───
	outTest := "data/test_prediction.csv"
	if err := writePredictions(testRecords, Xtest, boost, pre, outTest); err != nil {
		log.Fatalf("failed to write test predictions: %v", err)
	}
	fmt.Printf("Test predictions written to %s\n", outTest)
}

// evaluate prints per-target accuracy on validation set
func evaluate(
	boost *booster.Booster,
	Xval [][]float64,
	Yval [][]float64,
	pre *preprocess.Preprocessor,
) {
	Nval := len(Xval)
	if Nval == 0 {
		fmt.Println("No validation data.")
		return
	}
	numTargets := len(Yval[0])
	correct := make([]int, numTargets)
	for i := 0; i < Nval; i++ {
		pred := boost.Predict(Xval[i])
		trueRow := Yval[i]
		for j := 0; j < numTargets; j++ {
			pi := int(math.Round(pred[j]))
			if pi < 0 {
				pi = 0
			}
			var maxIdx int
			switch j {
			case 0:
				maxIdx = len(pre.ClinicianClasses()) - 1
			case 1:
				maxIdx = len(pre.GPT4Classes()) - 1
			case 2:
				maxIdx = len(pre.LLAMAClasses()) - 1
			case 3:
				maxIdx = len(pre.GEMINIClasses()) - 1
			case 4:
				maxIdx = len(pre.DDXClasses()) - 1
			}
			if pi > maxIdx {
				pi = maxIdx
			}
			if pi == int(trueRow[j]) {
				correct[j]++
			}
		}
	}
	targetNames := []string{"Clinician", "GPT4.0", "LLAMA", "GEMINI", "DDX SNOMED"}
	for j, name := range targetNames {
		acc := float64(correct[j]) / float64(Nval) * 100.0
		fmt.Printf("%s validation accuracy: %.2f%%\n", name, acc)
	}
}

// writePredictions outputs CSV: ID,Clinician,GPT4.0,LLAMA,GEMINI,DDX_SNOMED
func writePredictions(
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
		clinIdx := clamp(pred[0], len(clinLabels))
		gpt4Idx := clamp(pred[1], len(gpt4Labels))
		llamaIdx := clamp(pred[2], len(llamaLabels))
		geminiIdx := clamp(pred[3], len(geminiLabels))
		ddxIdx := clamp(pred[4], len(ddxLabels))

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

// clamp rounds to nearest int and clamps into [0, maxLen-1]
func clamp(val float64, maxLen int) int {
	idx := int(math.Round(val))
	if idx < 0 {
		return 0
	}
	if idx >= maxLen {
		return maxLen - 1
	}
	return idx
}
