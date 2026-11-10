// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nodename

import (
	"errors"
	"strings"
	"testing"
)

func TestNormalize(t *testing.T) {
	accepts := func(name, input, want string) {
		t.Run(name, func(t *testing.T) {
			got, err := Normalize(input)
			if err != nil {
				t.Fatalf("Normalize(%q) error = %v, want nil", input, err)
			}
			if got != want {
				t.Fatalf("Normalize(%q) = %q, want %q", input, got, want)
			}
		})
	}
	rejects := func(name, input string, want error) {
		t.Run(name, func(t *testing.T) {
			got, err := Normalize(input)
			if !errors.Is(err, want) {
				t.Fatalf("Normalize(%q) error = %v, want %v", input, err, want)
			}
			if got != "" {
				t.Fatalf("Normalize(%q) = %q, want empty on error", input, got)
			}
		})
	}

	accepts("keeps a plain name", "Studio Mac", "Studio Mac")
	accepts("trims ASCII white space", "  Studio  ", "Studio")
	accepts("trims a no-break and an ideographic space", " Studio　", "Studio")
	accepts("keeps a precomposed accent", "Café", "Café")
	accepts("composes a decomposed accent to NFC", "Café", "Café")
	accepts("keeps Japanese text", "白いPC", "白いPC")
	accepts("keeps an emoji joiner sequence", "👩‍💻 desk", "👩‍💻 desk")
	accepts("keeps a dot", "Desk.A", "Desk.A")
	accepts("keeps digits only", "1234", "1234")
	accepts("keeps backslash, quote and equals", `back\slash "quote" =eq`, `back\slash "quote" =eq`)
	accepts("accepts exactly 128 ASCII bytes", strings.Repeat("a", 128), strings.Repeat("a", 128))
	accepts("accepts 126 bytes of Japanese", strings.Repeat("あ", 42), strings.Repeat("あ", 42))

	rejects("rejects the empty string", "", ErrEmpty)
	rejects("rejects only white space", "   ", ErrEmpty)
	rejects("rejects NUL", "a\x00b", ErrInvalidCharacter)
	rejects("rejects a newline", "a\nb", ErrInvalidCharacter)
	rejects("rejects NEL", "a\u0085b", ErrInvalidCharacter)
	rejects("rejects a line separator", "a b", ErrInvalidCharacter)
	rejects("rejects a paragraph separator", "a b", ErrInvalidCharacter)
	rejects("rejects the right-to-left override", "a‮b", ErrInvalidCharacter)
	rejects("rejects a left-to-right isolate", "a⁦b", ErrInvalidCharacter)
	rejects("rejects the right-to-left mark", "a‏b", ErrInvalidCharacter)
	rejects("rejects the Arabic letter mark", "a؜b", ErrInvalidCharacter)
	rejects("rejects invalid UTF-8", "\xff", ErrInvalidUTF8)
	rejects("rejects 129 ASCII bytes", strings.Repeat("a", 129), ErrTooLong)
	rejects("rejects 129 bytes of Japanese", strings.Repeat("あ", 43), ErrTooLong)
}

func TestReason(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want string
	}{
		{"nil error", nil, ""},
		{"invalid UTF-8", ErrInvalidUTF8, "invalid-utf8"},
		{"empty", ErrEmpty, "empty"},
		{"invalid character", ErrInvalidCharacter, "invalid-character"},
		{"too long", ErrTooLong, "too-long"},
		{"wrapped error", errors.Join(errors.New("context"), ErrTooLong), "too-long"},
		{"unrelated error", errors.New("other"), ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Reason(tt.err); got != tt.want {
				t.Fatalf("Reason(%v) = %q, want %q", tt.err, got, tt.want)
			}
		})
	}
}

func TestFallback(t *testing.T) {
	tests := []struct {
		name string
		uuid string
		want string
	}{
		{"full UUID is lowercased and shortened", "1B9BE97E-E214-4AAA-8BBB-000000000000", "PAIR 1b9be97e"},
		{"short UUID is kept whole", "abc", "PAIR abc"},
		{"empty UUID gives the bare prefix", "", "PAIR"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Fallback(tt.uuid); got != tt.want {
				t.Fatalf("Fallback(%q) = %q, want %q", tt.uuid, got, tt.want)
			}
		})
	}
}
