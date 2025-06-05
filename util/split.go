// util/split.go
package util

import (
	"math/rand"
	"time"
)

// ShuffleSplit randomly shuffles indices [0..n-1] and returns two slices:
//   • trainIdx: first trainFrac·n shuffled indices
//   • valIdx: remaining indices
//
// trainFrac should be between 0.0 and 1.0 (e.g., 0.8 for an 80/20 split).
func ShuffleSplit(n int, trainFrac float64) (trainIdx, valIdx []int) {
	if trainFrac < 0.0 || trainFrac > 1.0 {
		trainFrac = 0.8
	}

	// Initialize index array [0,1,2,...,n-1]
	indices := make([]int, n)
	for i := 0; i < n; i++ {
		indices[i] = i
	}

	// Shuffle in place
	rand.Seed(time.Now().UnixNano())
	rand.Shuffle(n, func(i, j int) {
		indices[i], indices[j] = indices[j], indices[i]
	})

	// Compute split point
	trainCount := int(float64(n) * trainFrac)
	if trainCount < 1 {
		trainCount = 1
	}
	if trainCount > n-1 {
		trainCount = n - 1
	}

	trainIdx = make([]int, trainCount)
	copy(trainIdx, indices[:trainCount])

	valIdx = make([]int, n-trainCount)
	copy(valIdx, indices[trainCount:])

	return
}
