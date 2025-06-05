package util

import (
	"math/rand"
	"time"
)

// ShuffleSplit returns two index slices: trainIdx and valIdx,
// representing a random split of [0..n-1] with proportion trainFrac for training.
func ShuffleSplit(n int, trainFrac float64) (trainIdx, valIdx []int) {
	indices := make([]int, n)
	for i := range indices {
		indices[i] = i
	}
	rand.Seed(time.Now().UnixNano())
	rand.Shuffle(n, func(i, j int) {
		indices[i], indices[j] = indices[j], indices[i]
	})

	trainCount := int(float64(n) * trainFrac)
	trainIdx = indices[:trainCount]
	valIdx = indices[trainCount:]
	return
}
