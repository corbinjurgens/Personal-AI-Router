// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nodepolicy

import (
	"encoding/json"
	"errors"
	"unicode/utf8"
)

const MaxModelIdentifierBytes = 512

// ValidateModelIdentifier checks the original identifier before any trimming or
// engine normalization. Call it again on a normalized selectable identifier.
// Errors deliberately contain no identifier text.
func ValidateModelIdentifier(model string) error {
	if model == "" {
		return errors.New("model is required")
	}
	if len(model) > MaxModelIdentifierBytes || !utf8.ValidString(model) {
		return errors.New("model must be nonempty valid UTF-8 of at most 512 bytes")
	}
	for _, r := range model {
		if r < 0x20 || r >= 0x7f && r <= 0x9f {
			return errors.New("model must not contain control characters")
		}
	}
	return nil
}

// ValidateModelIdentifierJSON checks a raw JSON model string before the JSON
// decoder can replace invalid UTF-8, then checks the unescaped identifier with
// the same rule as typed callers.
func ValidateModelIdentifierJSON(raw json.RawMessage) error {
	if !utf8.Valid(raw) {
		return errors.New("model must be valid UTF-8")
	}
	var model string
	if json.Unmarshal(raw, &model) != nil {
		return errors.New("model must be a string")
	}
	return ValidateModelIdentifier(model)
}
