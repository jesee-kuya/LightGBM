package util

import "math"

// clamp rounds to nearest int and clamps into [0, maxLen-1]
func Clamp(val float64, maxLen int) int {
	idx := int(math.Round(val))
	if idx < 0 {
		return 0
	}
	if idx >= maxLen {
		return maxLen - 1
	}
	return idx
}
