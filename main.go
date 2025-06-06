// main.go
package main

import (
	"fmt"
	"log"

	"github.com/jesee-kuya/LightGBM/booster"
	"github.com/jesee-kuya/LightGBM/preprocess"
	"github.com/jesee-kuya/LightGBM/reader"
	"github.com/jesee-kuya/LightGBM/util"
	"github.com/jesee-kuya/LightGBM/writer"
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
	// Include numBins (e.g., 64) as the fifth parameter
	boost := booster.NewBooster(numTargets, 0.1, 3, 5, 64)
	boost.Fit(Xtrain, Ytrain, 50)
	fmt.Println("Training complete.")

	// ─── 5) EVALUATE ON VALIDATION ───
	util.Evaluate(boost, Xval, Yval, pre)

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
	if err := writer.WritePredictions(testRecords, Xtest, boost, pre, outTest); err != nil {
		log.Fatalf("failed to write test predictions: %v", err)
	}
	fmt.Printf("Test predictions written to %s\n", outTest)
}
