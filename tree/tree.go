package tree

import (
	"math"
)

// Node represents one node in our regression tree.
type Node struct {
	FeatureIdx int     // index of feature to split on
	Threshold  float64 // split: if x[FeatureIdx] ≤ Threshold → go Left; else go Right

	Left  *Node
	Right *Node

	Value  float64 // leaf prediction
	IsLeaf bool
}

// BuildRegressionTree fits a depth‐limited tree to (X, y).
//   - X: N×D feature matrix
//   - y: N‐vector of targets (residuals when used in boosting)
//   - depth: current depth (start with 0)
//   - maxDepth: maximum depth allowed (e.g. 3 or 4)
//   - minSamplesSplit: optional minimum number of samples to even consider splitting
func BuildRegressionTree(X [][]float64, y []float64, depth, maxDepth, minSamplesSplit int) *Node {
	n := len(y)
	if n == 0 {
		return &Node{IsLeaf: true, Value: 0.0}
	}

	// If reached maxDepth or too few samples, make leaf = avg(y)
	if depth >= maxDepth || n <= minSamplesSplit {
		var sum float64
		for _, val := range y {
			sum += val
		}
		return &Node{IsLeaf: true, Value: sum / float64(n)}
	}

	// Try all possible splits
	bestScore := math.Inf(1) // want to minimize SSE
	bestFeat := -1
	bestThresh := 0.0

	D := len(X[0])
	for featureIdx := 0; featureIdx < D; featureIdx++ {
		// Collect all unique values of X[i][featureIdx]
		thresholds := uniqueFeatureValues(X, featureIdx)
		for _, t := range thresholds {
			// Partition indices into left/right
			var leftIdx, rightIdx []int
			for i := 0; i < n; i++ {
				if X[i][featureIdx] <= t {
					leftIdx = append(leftIdx, i)
				} else {
					rightIdx = append(rightIdx, i)
				}
			}
			// Need at least one sample on each side
			if len(leftIdx) == 0 || len(rightIdx) == 0 {
				continue
			}

			// Compute SSE for this split
			score := splitSSE(y, leftIdx, rightIdx)
			if score < bestScore {
				bestScore = score
				bestFeat = featureIdx
				bestThresh = t
			}
		}
	}

	// If no valid split found, make leaf
	if bestFeat < 0 {
		var sum float64
		for _, val := range y {
			sum += val
		}
		return &Node{IsLeaf: true, Value: sum / float64(n)}
	}

	// Otherwise, split data into left/right subsets
	var (
		Xleft, Xright [][]float64
		yLeft, yRight []float64
	)

	for i := 0; i < n; i++ {
		if X[i][bestFeat] <= bestThresh {
			Xleft = append(Xleft, X[i])
			yLeft = append(yLeft, y[i])
		} else {
			Xright = append(Xright, X[i])
			yRight = append(yRight, y[i])
		}
	}

	// Recurse
	leftChild := BuildRegressionTree(Xleft, yLeft, depth+1, maxDepth, minSamplesSplit)
	rightChild := BuildRegressionTree(Xright, yRight, depth+1, maxDepth, minSamplesSplit)

	return &Node{
		FeatureIdx: bestFeat,
		Threshold:  bestThresh,
		Left:       leftChild,
		Right:      rightChild,
		IsLeaf:     false,
	}
}

// uniqueFeatureValues returns a slice of all distinct values of X[i][featIdx].
func uniqueFeatureValues(X [][]float64, featIdx int) []float64 {
	set := make(map[float64]struct{})
	for i := range X {
		val := X[i][featIdx]
		set[val] = struct{}{}
	}
	unique := make([]float64, 0, len(set))
	for val := range set {
		unique = append(unique, val)
	}
	return unique
}

// splitSSE computes sum of squared errors of y on leftIdx and rightIdx:
// SSE_left + SSE_right.
func splitSSE(y []float64, leftIdx, rightIdx []int) float64 {
	// Helper to compute SSE of y over indices idxs
	ev := func(idxs []int) float64 {
		var sum, sqSum float64
		for _, i := range idxs {
			sum += y[i]
			sqSum += y[i] * y[i]
		}
		n := float64(len(idxs))
		mean := sum / n
		// SSE = Σ (yᵢ - mean)² = Σ(yᵢ²) - n * mean²
		return sqSum - n*mean*mean
	}
	return ev(leftIdx) + ev(rightIdx)
}

// PredictTree returns the prediction of one tree node on feature vector x.
func PredictTree(root *Node, x []float64) float64 {
	if root.IsLeaf {
		return root.Value
	}
	if x[root.FeatureIdx] <= root.Threshold {
		return PredictTree(root.Left, x)
	}
	return PredictTree(root.Right, x)
}
