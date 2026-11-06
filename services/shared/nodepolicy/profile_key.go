// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nodepolicy

import (
	"errors"
	"fmt"
	"slices"
	"strings"
	"unicode"
	"unicode/utf8"
)

// ValidateProfileKey implements phase 06's key-grammar decision: preserve model
// identifiers and exclude routing aliases. Declared-role conflicts belong to Policy.
func ValidateProfileKey(key string) error {
	if err := ValidateModelIdentifier(key); err != nil {
		return err
	}
	if IsReservedProfileKey(key) {
		return &reservedProfileKeyError{key: key}
	}
	return nil
}

type reservedProfileKeyError struct{ key string }

func (e *reservedProfileKeyError) Error() string {
	return fmt.Sprintf("profile key %s is reserved: the text before its colon is a valid role name", e.key)
}

// ProfileIcons implements phase 06's profile-fields decision with fixed KUI names.
// Callers receive a copy so validation cannot be changed by a consumer.
func ProfileIcons() []string {
	return []string{"chat-single", "code", "cube", "document", "image"}
}

// ValidateProfileIcon accepts an omitted icon or one of the chosen KUI names.
func ValidateProfileIcon(icon string) error {
	if icon != "" && !slices.Contains(ProfileIcons(), icon) {
		return errors.New("profile icon must be a supported KUI icon name")
	}
	return nil
}

// ValidateProfileDisplayName enforces the free-text display-name decision.
func ValidateProfileDisplayName(name string) error {
	if name == "" || len(name) > 128 || !utf8.ValidString(name) || strings.ContainsFunc(name, unicode.IsControl) {
		return errors.New("profile display name must be 1-128 bytes of UTF-8 without control characters")
	}
	return nil
}
