// tree/tree.go
package tree

import (
	"math"
)

// Node represents one node in histogram-based regression tree.
type Node struct {
	FeatureIdx int    
	Threshold  float64 
	Left  *Node
	Right *Node
	Value  float64 
	IsLeaf bool
}

// BuildHistogramTree fits a histogram-based regression tree to (X, grad, hess).
//   - X: N×D feature matrix
//   - grad: length-N slice of gradients
//   - hess: length-N slice of hessians (for squared-error, hess[i] = 1)
//   - depth: current depth (start at 0)
//   - maxDepth: maximum depth allowed
//   - minSamples: minimum number of samples to allow a split
//   - numBins: number of bins to discretize each feature
func BuildHistogramTree(
	X [][]float64,
	grad []float64,
	hess []float64,
	depth, maxDepth, minSamples, numBins int,
) *Node {
	N := len(grad)
	if N == 0 {
		return &Node{IsLeaf: true, Value: 0.0}
	}

	// Compute total sum of gradients and hessians for this node
	var sumGrad, sumHess float64
	for i := 0; i < N; i++ {
		sumGrad += grad[i]
		sumHess += hess[i]
	}

	// If max depth reached or too few samples, make a leaf
	if depth >= maxDepth || N <= minSamples {
		leafValue := -sumGrad / (sumHess + 1e-3)
		return &Node{IsLeaf: true, Value: leafValue}
	}

	D := len(X[0]) 
	
	// Step 1: For each feature j, compute min and max over samples
	minVals := make([]float64, D)
	maxVals := make([]float64, D)
	for j := 0; j < D; j++ {
		minVals[j] = X[0][j]
		maxVals[j] = X[0][j]
		for i := 1; i < N; i++ {
			v := X[i][j]
			if v < minVals[j] {
				minVals[j] = v
			}
			if v > maxVals[j] {
				maxVals[j] = v
			}
		}
	}

	// Step 2: Build histograms and find best split
	type histBin struct {
		sumG  float64
		sumH  float64
		count int
	}

	bestGain := math.Inf(-1)
	bestFeat := -1
	bestBinIdx := -1

	// For each feature, build a histogram of numBins bins
	for j := 0; j < D; j++ {
		minVal := minVals[j]
		maxVal := maxVals[j]
		width := (maxVal - minVal) / float64(numBins)
		if width == 0 {
			// All values identical → cannot split on this feature
			continue
		}

		// Initialize histogram bins
		hist := make([]histBin, numBins)
		for i := 0; i < N; i++ {
			v := X[i][j]
			bin := int((v - minVal) / width)
			if bin < 0 {
				bin = 0
			} else if bin >= numBins {
				bin = numBins - 1
			}
			hist[bin].sumG += grad[i]
			hist[bin].sumH += hess[i]
			hist[bin].count++
		}

		// Prefix sums to get left-side accumulations
		leftGradSum := make([]float64, numBins)
		leftHessSum := make([]float64, numBins)
		leftCount := make([]int, numBins)

		var cumG, cumH float64
		var cumC int
		for b := 0; b < numBins; b++ {
			cumG += hist[b].sumG
			cumH += hist[b].sumH
			cumC += hist[b].count
			leftGradSum[b] = cumG
			leftHessSum[b] = cumH
			leftCount[b] = cumC
		}

		totalGrad := sumGrad
		totalHess := sumHess
		totalCount := N

		// Evaluate splits at each bin boundary b (left = bins ≤ b)
		for b := 0; b < numBins-1; b++ {
			G_L := leftGradSum[b]
			H_L := leftHessSum[b]
			C_L := leftCount[b]

			G_R := totalGrad - G_L
			H_R := totalHess - H_L
			C_R := totalCount - C_L

			// Skip if either side too small
			if C_L < minSamples || C_R < minSamples {
				continue
			}

			// Gain = 0.5 * (G_L^2/(H_L+λ) + G_R^2/(H_R+λ) - G_total^2/(H_total+λ))
			le := (G_L * G_L) / (H_L + 1e-3)
			re := (G_R * G_R) / (H_R + 1e-3)
			mega := (totalGrad * totalGrad) / (totalHess + 1e-3)
			gain := 0.5 * (le + re - mega)

			if gain > bestGain {
				bestGain = gain
				bestFeat = j
				bestBinIdx = b
			}
		}
	}

	// If no valid split found, make a leaf
	if bestFeat < 0 {
		leafValue := -sumGrad / (sumHess + 1e-3)
		return &Node{IsLeaf: true, Value: leafValue}
	}

	// Compute float threshold for the best split
	minVal := minVals[bestFeat]
	maxVal := maxVals[bestFeat]
	width := (maxVal - minVal) / float64(numBins)
	// Split at boundary between bin bestBinIdx and bestBinIdx+1
	threshold := minVal + width*float64(bestBinIdx+1)

	// Partition samples into left/right by comparing to threshold
	leftIdx := make([]int, 0, N)
	rightIdx := make([]int, 0, N)
	for i := 0; i < N; i++ {
		if X[i][bestFeat] <= threshold {
			leftIdx = append(leftIdx, i)
		} else {
			rightIdx = append(rightIdx, i)
		}
	}

	// Build child datasets
	Xleft := make([][]float64, len(leftIdx))
	gradLeft := make([]float64, len(leftIdx))
	hessLeft := make([]float64, len(leftIdx))
	for i, idx := range leftIdx {
		Xleft[i] = X[idx]
		gradLeft[i] = grad[idx]
		hessLeft[i] = hess[idx]
	}

	Xright := make([][]float64, len(rightIdx))
	gradRight := make([]float64, len(rightIdx))
	hessRight := make([]float64, len(rightIdx))
	for i, idx := range rightIdx {
		Xright[i] = X[idx]
		gradRight[i] = grad[idx]
		hessRight[i] = hess[idx]
	}

	// Recurse
	leftChild := BuildHistogramTree(Xleft, gradLeft, hessLeft, depth+1, maxDepth, minSamples, numBins)
	rightChild := BuildHistogramTree(Xright, gradRight, hessRight, depth+1, maxDepth, minSamples, numBins)

	return &Node{
		FeatureIdx: bestFeat,
		Threshold:  threshold,
		Left:       leftChild,
		Right:      rightChild,
		IsLeaf:     false,
	}
}

// PredictTree traverses the tree to return a prediction for a single feature vector x.
func PredictTree(node *Node, x []float64) float64 {
	if node.IsLeaf {
		return node.Value
	}
	if x[node.FeatureIdx] <= node.Threshold {
		return PredictTree(node.Left, x)
	}
	return PredictTree(node.Right, x)
}
