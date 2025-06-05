package util

import (
	"fmt"
	"math"

	"github.com/jesee-kuya/LightGBM/booster"
	"github.com/jesee-kuya/LightGBM/preprocess"
)

// evaluate prints per-target accuracy on validation set
func Evaluate(
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
