package booster

import "github.com/jesee-kuya/LightGBM/tree"

// Booster trains one tree‐ensemble per target column.
type Booster struct {
	Trees        [][]*tree.Node // Trees[j] is slice of *Node for target j
	LearningRate float64
	MaxDepth     int
	MinSamples   int // minimum samples to split

	NumTargets int
}

// NewBooster allocates a Booster for `numTargets` outputs.
func NewBooster(numTargets int, lr float64, maxDepth, minSamples int) *Booster {
	trees := make([][]*tree.Node, numTargets)
	return &Booster{
		Trees:        trees,
		LearningRate: lr,
		MaxDepth:     maxDepth,
		MinSamples:   minSamples,
		NumTargets:   numTargets,
	}
}

// Fit trains `nRounds` of boosting; X is N×D, Y is N×T (T = numTargets).
// We assume Y[i][j] is the class index (float64) for jth target on ith row.
// We’ll do a simple squared‐error residual (i.e. treat indices as real numbers).
func (b *Booster) Fit(X [][]float64, Y [][]float64, nRounds int) {
	N := len(X)
	T := b.NumTargets

	// Initialize predictions ŷ to zero: preds[i][j] will hold the current pred for row i, target j.
	preds := make([][]float64, N)
	for i := range preds {
		preds[i] = make([]float64, T)
	}

	// For each boosting round
	for round := 0; round < nRounds; round++ {
		for j := 0; j < T; j++ {
			// 1) Compute residuals rᵢ = Y[i][j] - preds[i][j]
			residuals := make([]float64, N)
			for i := 0; i < N; i++ {
				residuals[i] = Y[i][j] - preds[i][j]
			}

			// 2) Fit one tree on (X, residuals)
			treeJ := tree.BuildRegressionTree(X, residuals, 0, b.MaxDepth, b.MinSamples)
			b.Trees[j] = append(b.Trees[j], treeJ)

			// 3) Update preds[i][j] += learningRate * tree.predict(X[i])
			for i := 0; i < N; i++ {
				predVal := tree.PredictTree(treeJ, X[i])
				preds[i][j] += b.LearningRate * predVal
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
		for _, t := range b.Trees[j] {
			sum += b.LearningRate * tree.PredictTree(t, x)
		}
		out[j] = sum
	}
	return out
}
