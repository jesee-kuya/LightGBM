package util

import "github.com/jesee-kuya/LightGBM/model"

// mergeByID: raw overrides clean for duplicate Master_Index
func MergeByID(clean, raw []model.DataRecord) []model.DataRecord {
	m := make(map[string]model.DataRecord, len(clean)+len(raw))
	for _, r := range clean {
		m[r.ID] = r
	}
	for _, r := range raw {
		m[r.ID] = r
	}
	out := make([]model.DataRecord, 0, len(m))
	for _, r := range m {
		out = append(out, r)
	}
	return out
}
