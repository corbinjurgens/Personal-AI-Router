// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package nodename is the only validator for node display names and the owner
// of the generated last-resort name. Every Go component that accepts a name
// (the cluster manager, the node scanner, the broker) calls Normalize rather
// than checking names itself.
package nodename

import (
	"errors"
	"strings"
	"unicode"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"
)

// MaxBytes is the largest accepted node name, in UTF-8 bytes after normalization.
const MaxBytes = 128

// fallbackUUIDChars is how many characters of the UUID the fallback name shows.
const fallbackUUIDChars = 8

var (
	ErrInvalidUTF8      = errors.New("node name is not valid UTF-8")
	ErrEmpty            = errors.New("node name is empty")
	ErrInvalidCharacter = errors.New("node name contains a control or bidirectional control character")
	ErrTooLong          = errors.New("node name is longer than 128 bytes")
)

// Normalize returns the canonical form of a node name or one of the errors above.
func Normalize(raw string) (string, error) {
	if !utf8.ValidString(raw) {
		return "", ErrInvalidUTF8
	}
	name := strings.TrimSpace(norm.NFC.String(raw))
	if name == "" {
		return "", ErrEmpty
	}
	for _, r := range name {
		if rejected(r) {
			return "", ErrInvalidCharacter
		}
	}
	if len(name) > MaxBytes {
		return "", ErrTooLong
	}
	return name, nil
}

// rejected reports whether r is a control character, a line or paragraph
// separator, or a bidirectional control. The joiners U+200C and U+200D stay
// allowed because emoji and script sequences need them.
func rejected(r rune) bool {
	if unicode.IsControl(r) {
		return true
	}
	switch {
	case r == 0x2028, r == 0x2029, // line and paragraph separators
		r == 0x061C, r == 0x200E, r == 0x200F, // Arabic letter mark, LRM, RLM
		r >= 0x202A && r <= 0x202E, // embeddings and overrides
		r >= 0x2066 && r <= 0x2069: // isolates
		return true
	}
	return false
}

// Reason maps an error returned by Normalize to a stable machine-readable
// reason: "invalid-utf8", "empty", "invalid-character", "too-long"; "" for nil
// and for any other error.
func Reason(err error) string {
	switch {
	case errors.Is(err, ErrInvalidUTF8):
		return "invalid-utf8"
	case errors.Is(err, ErrEmpty):
		return "empty"
	case errors.Is(err, ErrInvalidCharacter):
		return "invalid-character"
	case errors.Is(err, ErrTooLong):
		return "too-long"
	}
	return ""
}

// Fallback returns the generated last-resort name "PAIR " + the first 8
// characters of the lowercased UUID (the whole UUID when shorter), or "PAIR"
// for an empty UUID.
func Fallback(nodeUUID string) string {
	short := []rune(strings.ToLower(nodeUUID))
	if len(short) > fallbackUUIDChars {
		short = short[:fallbackUUIDChars]
	}
	if len(short) == 0 {
		return "PAIR"
	}
	return "PAIR " + string(short)
}
