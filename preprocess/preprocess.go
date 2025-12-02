package preprocess

import (
	"hash/fnv"
	"strings"

	"github.com/jesee-kuya/LightGBM/model"
)

// Preprocessor holds maps for encoding both inputs and targets.
type Preprocessor struct {
	countyEncoder      map[string]int
	healthLevelEncoder map[string]int
	competencyEncoder  map[string]int
	panelEncoder       map[string]int
	numPromptBuckets int
	clinicianEncoder map[string]int
	gpt4Encoder      map[string]int
	llamaEncoder     map[string]int
	geminiEncoder    map[string]int
	ddxEncoder       map[string]int
}

// NewPreprocessor allocates a Preprocessor that will hash prompts into
// numPromptBuckets and also build encoders for five target columns.
func NewPreprocessor(numPromptBuckets int) *Preprocessor {
	return &Preprocessor{
		countyEncoder:      make(map[string]int),
		healthLevelEncoder: make(map[string]int),
		competencyEncoder:  make(map[string]int),
		panelEncoder:       make(map[string]int),

		numPromptBuckets: numPromptBuckets,

		clinicianEncoder: make(map[string]int),
		gpt4Encoder:      make(map[string]int),
		llamaEncoder:     make(map[string]int),
		geminiEncoder:    make(map[string]int),
		ddxEncoder:       make(map[string]int),
	}
}

// Fit builds all categorical‐and‐target encoders by scanning through every record.
// After calling Fit, every distinct string in each column has been assigned an integer ID.
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

		// BUILD TARGET ENCODERS 
		// Each of these maps string → unique int

		// Clinician
		cl := strings.TrimSpace(r.Clinician)
		if cl != "" {
			if _, ok := p.clinicianEncoder[cl]; !ok {
				p.clinicianEncoder[cl] = len(p.clinicianEncoder)
			}
		}

		// GPT4.0
		g4 := strings.TrimSpace(r.GPT4)
		if g4 != "" {
			if _, ok := p.gpt4Encoder[g4]; !ok {
				p.gpt4Encoder[g4] = len(p.gpt4Encoder)
			}
		}

		// LLAMA
		la := strings.TrimSpace(r.LLAMA)
		if la != "" {
			if _, ok := p.llamaEncoder[la]; !ok {
				p.llamaEncoder[la] = len(p.llamaEncoder)
			}
		}

		// GEMINI
		ge := strings.TrimSpace(r.GEMINI)
		if ge != "" {
			if _, ok := p.geminiEncoder[ge]; !ok {
				p.geminiEncoder[ge] = len(p.geminiEncoder)
			}
		}

		// DDX SNOMED
		dd := strings.TrimSpace(r.DDXSNOMED)
		if dd != "" {
			if _, ok := p.ddxEncoder[dd]; !ok {
				p.ddxEncoder[dd] = len(p.ddxEncoder)
			}
		}
	}
}

// Transform returns:
//   - X: [][]float64  (numeric feature vectors, one row per record)
//   - Y: [][]float64  (each row is a slice of five encoded‐target ints, in float64 form)
//
// The order of targets in Y[i] is exactly:
//
//	[ encoded(Clinician), encoded(GPT4.0), encoded(LLAMA), encoded(GEMINI), encoded(DDX SNOMED) ]
func (p *Preprocessor) Transform(records []model.DataRecord) ([][]float64, [][]float64) {
	n := len(records)
	X := make([][]float64, n)
	Y := make([][]float64, n)

	for i, r := range records {
		// BUILD INPUT FEATURE VECTOR 
		featVec := make([]float64, 5+p.numPromptBuckets)

		// county → float64(idx)
		c := strings.ToLower(strings.TrimSpace(r.County))
		if idx, ok := p.countyEncoder[c]; ok {
			featVec[0] = float64(idx)
		} else {
			featVec[0] = -1.0
		}

		// health level → float64(idx)
		h := strings.ToLower(strings.TrimSpace(r.HealthLevel))
		if idx, ok := p.healthLevelEncoder[h]; ok {
			featVec[1] = float64(idx)
		} else {
			featVec[1] = -1.0
		}

		// years of experience (already a float)
		featVec[2] = r.YearsExperience

		// competency → float64(idx)
		comp := strings.ToLower(strings.TrimSpace(r.Competency))
		if idx, ok := p.competencyEncoder[comp]; ok {
			featVec[3] = float64(idx)
		} else {
			featVec[3] = -1.0
		}

		// panel → float64(idx)
		pnl := strings.ToLower(strings.TrimSpace(r.Panel))
		if idx, ok := p.panelEncoder[pnl]; ok {
			featVec[4] = float64(idx)
		} else {
			featVec[4] = -1.0
		}

		// bag‐of‐hashes on Prompt
		promptBuckets := make([]float64, p.numPromptBuckets)
		words := strings.Fields(strings.ToLower(r.Prompt))
		for _, w := range words {
			hv := hashWord(w)
			bucket := int(hv % uint32(p.numPromptBuckets))
			promptBuckets[bucket] += 1.0
		}
		for j := 0; j < p.numPromptBuckets; j++ {
			featVec[5+j] = promptBuckets[j]
		}

		X[i] = featVec

		// BUILD TARGET VECTOR (as float64 of each label index) ───
		targs := make([]float64, 5)

		// Clinician
		cl := strings.TrimSpace(r.Clinician)
		if idx, ok := p.clinicianEncoder[cl]; ok {
			targs[0] = float64(idx)
		} else {
			targs[0] = -1.0
		}

		// GPT4.0
		g4 := strings.TrimSpace(r.GPT4)
		if idx, ok := p.gpt4Encoder[g4]; ok {
			targs[1] = float64(idx)
		} else {
			targs[1] = -1.0
		}

		// LLAMA
		la := strings.TrimSpace(r.LLAMA)
		if idx, ok := p.llamaEncoder[la]; ok {
			targs[2] = float64(idx)
		} else {
			targs[2] = -1.0
		}

		// GEMINI
		ge := strings.TrimSpace(r.GEMINI)
		if idx, ok := p.geminiEncoder[ge]; ok {
			targs[3] = float64(idx)
		} else {
			targs[3] = -1.0
		}

		// DDX SNOMED
		dd := strings.TrimSpace(r.DDXSNOMED)
		if idx, ok := p.ddxEncoder[dd]; ok {
			targs[4] = float64(idx)
		} else {
			targs[4] = -1.0
		}

		Y[i] = targs
	}

	return X, Y
}

func (p *Preprocessor) ClinicianClasses() []string {
	rev := make([]string, len(p.clinicianEncoder))
	for str, idx := range p.clinicianEncoder {
		rev[idx] = str
	}
	return rev
}

func (p *Preprocessor) GPT4Classes() []string {
	rev := make([]string, len(p.gpt4Encoder))
	for str, idx := range p.gpt4Encoder {
		rev[idx] = str
	}
	return rev
}

func (p *Preprocessor) LLAMAClasses() []string {
	rev := make([]string, len(p.llamaEncoder))
	for str, idx := range p.llamaEncoder {
		rev[idx] = str
	}
	return rev
}

func (p *Preprocessor) GEMINIClasses() []string {
	rev := make([]string, len(p.geminiEncoder))
	for str, idx := range p.geminiEncoder {
		rev[idx] = str
	}
	return rev
}

func (p *Preprocessor) DDXClasses() []string {
	rev := make([]string, len(p.ddxEncoder))
	for str, idx := range p.ddxEncoder {
		rev[idx] = str
	}
	return rev
}

// hashWord returns a 32‐bit FNV‐1a hash of the input string.
func hashWord(s string) uint32 {
	h := fnv.New32a()
	h.Write([]byte(s))
	return h.Sum32()
}
