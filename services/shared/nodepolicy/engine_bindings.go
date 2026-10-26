// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nodepolicy

import (
	"encoding/json"
	"errors"
	"strings"

	"nvpair-shared/engines"
)

// EngineBindings is target-owned configuration, not inventory or engine intent.
// Absence and an empty list both disable automatic preparation. Readback always
// emits a list so a caller can distinguish support from discarded input.
type EngineBindings []Role

func (b EngineBindings) MarshalJSON() ([]byte, error) {
	if err := b.validate(); err != nil {
		return nil, err
	}
	if b == nil {
		return []byte("[]"), nil
	}
	return json.Marshal([]Role(b))
}

func (b *EngineBindings) UnmarshalJSON(data []byte) error {
	refs, err := decodeModelList("engineBindings", data)
	if err != nil {
		return err
	}
	next := EngineBindings(refs)
	if err := next.validate(); err != nil {
		return err
	}
	*b = EngineBindings(nilIfEmpty(next))
	return nil
}

func (b EngineBindings) validate() error {
	if err := validateModelList("engineBindings", b); err != nil {
		return err
	}
	seen := map[string]bool{}
	for _, ref := range b {
		if seen[ref.Model] {
			return errors.New("engineBindings must assign each native model name only once")
		}
		seen[ref.Model] = true
	}
	return nil
}

// ValidateEngineBindings checks target-dependent matching using the owner's
// current descriptor naming resolver. It keeps the saved native spelling and
// stores no descriptor registry. Unknown naming cannot authorize a new binding.
func (p Policy) ValidateEngineBindings(naming func(string) (string, bool)) error {
	if err := p.EngineBindings.validate(); err != nil {
		return err
	}
	type match struct {
		naming, canonical string
	}
	matches := make([]match, 0, len(p.EngineBindings))
	for _, ref := range p.EngineBindings {
		rule, ok := naming(ref.Engine)
		if !ok || rule != engines.ModelNamingExact && rule != engines.ModelNamingImpliedLatestTag {
			return errors.New("engineBindings require a current target engine naming descriptor")
		}
		canonical := engines.NormalizeModelName(ref.Model, rule)
		if err := checkModelRef(ref.Engine, canonical, 0, nil); err != nil {
			return err
		}
		if p.ServedModels != nil {
			served := false
			for _, allowed := range p.ServedModels {
				if allowed.Engine == ref.Engine && engines.NormalizeModelName(allowed.Model, rule) == canonical {
					served = true
					break
				}
			}
			if !served {
				return errors.New("engineBindings must name served models after target normalization")
			}
		}
		matches = append(matches, match{rule, canonical})
	}
	for i, a := range matches {
		// Under descriptor naming, a canonical name and its valid untagged
		// spelling exhaust the distinct matching classes (outer space is trimmed).
		spellings := []string{a.canonical}
		if base, ok := strings.CutSuffix(a.canonical, ":latest"); ok && engines.NormalizeModelName(base, a.naming) == a.canonical {
			spellings = append(spellings, base)
		}
		for j, b := range matches {
			if i == j {
				continue
			}
			for _, spelling := range spellings {
				if engines.NormalizeModelName(spelling, b.naming) == b.canonical {
					return errors.New("engineBindings have overlapping model matching sets")
				}
			}
		}
	}
	return nil
}
