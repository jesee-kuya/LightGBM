// server/server.go
package server

import (
	"encoding/json"
	"net/http"

	"github.com/jesee-kuya/LightGBM/booster"
	"github.com/jesee-kuya/LightGBM/model"
	"github.com/jesee-kuya/LightGBM/preprocess"
	"github.com/jesee-kuya/LightGBM/util"
)

// Server holds the necessary components for prediction.
type Server struct {
	Booster *booster.Booster
	Preproc *preprocess.Preprocessor
}

// PredictionRequest is the JSON structure for the incoming request.
type PredictionRequest struct {
	IllnessDescription string `json:"illness_description"`
}

// PredictionResponse is the JSON structure for the outgoing response,
// now simplified to only include the main diagnosis.
type PredictionResponse struct {
	MainDiagnosis string `json:"main_diagnosis"`
}

// NewServer creates a new Server instance.
func NewServer(b *booster.Booster, p *preprocess.Preprocessor) *Server {
	return &Server{
		Booster: b,
		Preproc: p,
	}
}

// SetCORSHeaders adds necessary headers to allow cross-origin requests.
func SetCORSHeaders(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusOK)
		return
	}
}

// PredictHandler handles incoming prediction requests.
func (s *Server) PredictHandler(w http.ResponseWriter, r *http.Request) {
	SetCORSHeaders(w, r)

	if r.Method != http.MethodPost {
		http.Error(w, "Only POST method is supported", http.StatusMethodNotAllowed)
		return
	}
	if r.Method != http.MethodPost {
		http.Error(w, "Only POST method is supported", http.StatusMethodNotAllowed)
		return
	}

	var req PredictionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body: "+err.Error(), http.StatusBadRequest)
		return
	}

	record := model.DataRecord{
		County:          "unknown",
		HealthLevel:     "unknown",
		YearsExperience: 5.0,
		Competency:      "unknown",
		Panel:           "unknown",
		Prompt:          req.IllnessDescription,
	}

	// Transform the single record into a feature vector X
	X, _ := s.Preproc.Transform([]model.DataRecord{record})
	if len(X) == 0 {
		http.Error(w, "Failed to transform input data", http.StatusInternalServerError)
		return
	}

	// Get prediction from the Booster (returns 5 predictions)
	rawPreds := s.Booster.Predict(X[0])
	ddxLabels := s.Preproc.DDXClasses()
	ddxIdx := util.Clamp(rawPreds[4], len(ddxLabels))
	mainDiagnosis := ddxLabels[ddxIdx]

	// Send JSON response with only the main diagnosis
	resp := PredictionResponse{
		MainDiagnosis: mainDiagnosis,
	}

	w.Header().Set("Content-Type", "application/json")

	if err := json.NewEncoder(w).Encode(resp); err != nil {
		http.Error(w, "Failed to encode response", http.StatusInternalServerError)
	}
}
