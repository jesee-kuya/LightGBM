// preprocess/preprocess.go
package preprocess

import (
	"hash/fnv"
	"strings"

	"github.com/jesee-kuya/LightGBM/model"
)

// Preprocessor holds state (maps) for converting DataRecord fields to numeric.
type Preprocessor struct {
	// maps from categorical value (string) → integer index
	// we will assign 0,1,2,... for each distinct category
	countyEncoder      map[string]int
	healthLevelEncoder map[string]int
	competencyEncoder  map[string]int
	panelEncoder       map[string]int

	// number of buckets for hashing the prompt text
	numPromptBuckets int
}

// NewPreprocessor returns a Preprocessor that will hash prompts into
// numPromptBuckets dimensions, and build categorical encoders:
func NewPreprocessor(numPromptBuckets int) *Preprocessor {
	return &Preprocessor{
		countyEncoder:      make(map[string]int),
		healthLevelEncoder: make(map[string]int),
		competencyEncoder:  make(map[string]int),
		panelEncoder:       make(map[string]int),
		numPromptBuckets:   numPromptBuckets,
	}
}

// Fit scans through all records once to build up each categorical encoder.
// ‣ Any unseen category gets the next integer index (0,1,2,...).
func (p *Preprocessor) Fit(records []model.DataRecord) {
	for _, r := range records {
		// COUNTY
		c := strings.ToLower(strings.TrimSpace(r.County))
		if _, ok := p.countyEncoder[c]; !ok {
			p.countyEncoder[c] = len(p.countyEncoder)
		}

		// HEALTH LEVEL
		h := strings.ToLower(strings.TrimSpace(r.HealthLevel))
		if _, ok := p.healthLevelEncoder[h]; !ok {
			p.healthLevelEncoder[h] = len(p.healthLevelEncoder)
		}

		// NURSING COMPETENCY
		comp := strings.ToLower(strings.TrimSpace(r.Competency))
		if _, ok := p.competencyEncoder[comp]; !ok {
			p.competencyEncoder[comp] = len(p.competencyEncoder)
		}

		// CLINICAL PANEL
		pnl := strings.ToLower(strings.TrimSpace(r.Panel))
		if _, ok := p.panelEncoder[pnl]; !ok {
			p.panelEncoder[pnl] = len(p.panelEncoder)
		}
	}
}

// Transform takes a slice of DataRecord and returns:
//   - X: [][]float64 where each inner []float64 is the numeric feature vector for a record
//   - Y: [][]string where each inner []string is the list of raw targets (you can label‐encode these later)
func (p *Preprocessor) Transform(records []model.DataRecord) ([][]float64, [][]string) {
	n := len(records)
	X := make([][]float64, n)
	Y := make([][]string, n)

	for i, r := range records {
		// 1) build one feature‐vector step‐by‐step
		//  Features in order:
		//   [0] county (int),
		//   [1] healthLevel (int),
		//   [2] yearsExperience (float),
		//   [3] competency (int),
		//   [4] panel (int),
		//   [5..(5+numPromptBuckets-1)] promptHash features (float)

		// Allocate a slice of length = 5 + numPromptBuckets
		featVec := make([]float64, 5+p.numPromptBuckets)

		// --- (0) county index
		c := strings.ToLower(strings.TrimSpace(r.County))
		if idx, ok := p.countyEncoder[c]; ok {
			featVec[0] = float64(idx)
		} else {
			featVec[0] = -1.0 // unseen category (shouldn’t happen if Fit saw all)
		}

		// --- (1) health level index
		h := strings.ToLower(strings.TrimSpace(r.HealthLevel))
		if idx, ok := p.healthLevelEncoder[h]; ok {
			featVec[1] = float64(idx)
		} else {
			featVec[1] = -1.0
		}

		// --- (2) years of experience (already numeric)
		featVec[2] = r.YearsExperience

		// --- (3) nursing competency index
		comp := strings.ToLower(strings.TrimSpace(r.Competency))
		if idx, ok := p.competencyEncoder[comp]; ok {
			featVec[3] = float64(idx)
		} else {
			featVec[3] = -1.0
		}

		// --- (4) clinical panel index
		pnl := strings.ToLower(strings.TrimSpace(r.Panel))
		if idx, ok := p.panelEncoder[pnl]; ok {
			featVec[4] = float64(idx)
		} else {
			featVec[4] = -1.0
		}

		// --- (5..): hash the prompt text into numPromptBuckets “bag‐of‐hashes”
		// We’ll zero‐initialize all prompt buckets, then accumulate counts.
		promptBuckets := make([]float64, p.numPromptBuckets)
		words := strings.Fields(strings.ToLower(r.Prompt))
		for _, w := range words {
			hv := hashWord(w)
			bucket := int(hv % uint32(p.numPromptBuckets))
			promptBuckets[bucket] += 1.0
		}
		// Copy promptBuckets into featVec[5:]
		for j := 0; j < p.numPromptBuckets; j++ {
			featVec[5+j] = promptBuckets[j]
		}

		X[i] = featVec

		// 2) collect Y[i] = raw text targets (5 of them, for example)
		Y[i] = []string{
			r.Clinician,
			r.GPT4,
			r.LLAMA,
			r.GEMINI,
			r.DDXSNOMED,
		}
	}

	return X, Y
}

// hashWord returns a 32‐bit FNV‐1a hash of the input string
func hashWord(s string) uint32 {
	h := fnv.New32a()
	h.Write([]byte(s))
	return h.Sum32()
}
