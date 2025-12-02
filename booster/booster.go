// booster/booster.go
package booster

import (
	"github.com/jesee-kuya/LightGBM/tree"
)

// Booster trains one histogram-based tree ensemble per target column.
type Booster struct {
	Trees        [][]*tree.Node 
	LearningRate float64
	MaxDepth     int
	MinSamples   int 
	NumBins      int 

	NumTargets int
}

// NewBooster allocates a Booster for `numTargets` outputs.
// Uses `defaultBins` as the histogram bin count.
func NewBooster(numTargets int, lr float64, maxDepth, minSamples, defaultBins int) *Booster {
	trees := make([][]*tree.Node, numTargets)
	return &Booster{
		Trees:        trees,
		LearningRate: lr,
		MaxDepth:     maxDepth,
		MinSamples:   minSamples,
		NumBins:      defaultBins,
		NumTargets:   numTargets,
	}
}

// Fit trains `nRounds` of boosting; X is N×D, Y is N×T (T = numTargets).
// Uses squared-error: gradient = pred - target, hessian = 1.
func (b *Booster) Fit(X [][]float64, Y [][]float64, nRounds int) {
	N := len(X)
	T := b.NumTargets

	// Initialize predictions ŷ to zero: preds[i][j]
	preds := make([][]float64, N)
	for i := range preds {
		preds[i] = make([]float64, T)
	}

	for round := 0; round < nRounds; round++ {
		for j := 0; j < T; j++ {
			// Compute gradients and hessians for target j
			grad := make([]float64, N)
			hess := make([]float64, N)
			for i := range N {
				grad[i] = preds[i][j] - Y[i][j] // pred - target
				hess[i] = 1.0
			}

			// Build one histogram‐based tree on (X, grad, hess)
			treeJ := tree.BuildHistogramTree(
				X,
				grad,
				hess,
				0,            
				b.MaxDepth,   
				b.MinSamples, 
				b.NumBins,    
			)
			b.Trees[j] = append(b.Trees[j], treeJ)

			// Update preds[i][j] += learningRate * tree prediction
			for i := range N {
				val := tree.PredictTree(treeJ, X[i])
				preds[i][j] += b.LearningRate * val
			}
		}
	}
}

// Predict returns a slice of length T (numTargets), giving the boosted ensemble
// output for a single feature vector x.
func (b *Booster) Predict(x []float64) []float64 {
	out := make([]float64, b.NumTargets)
	for j := 0; j < b.NumTargets; j++ {
		var sum float64
		for _, tnode := range b.Trees[j] {
			sum += b.LearningRate * tree.PredictTree(tnode, x)
		}
		out[j] = sum
	}
	return out
}
