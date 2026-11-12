// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package mdns

import "strings"

// github.com/miekg/dns treats TXT strings as zone-file text: packing consumes
// backslash escapes (\X and \DDD), and unpacking escapes '"', '\' and every
// byte outside printable ASCII. EscapeTXT and UnescapeTXT undo both, so a TXT
// string the responder advertises reaches the browser byte for byte, including
// backslashes, quotes and UTF-8 text.

// EscapeTXT returns s with every backslash doubled, so packing a TXT record
// writes the original bytes to the wire.
func EscapeTXT(s string) string {
	return strings.ReplaceAll(s, `\`, `\\`)
}

// UnescapeTXT decodes a TXT string as unpacked from the wire: \DDD (three
// decimal digits, 0 to 255) becomes that byte and \X becomes X. A trailing
// lone backslash is kept as is.
func UnescapeTXT(s string) string {
	if !strings.Contains(s, `\`) {
		return s
	}
	var b strings.Builder
	b.Grow(len(s))
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c != '\\' || i+1 == len(s) {
			b.WriteByte(c)
			continue
		}
		if v, ok := decimalByte(s[i+1:]); ok {
			b.WriteByte(v)
			i += 3
			continue
		}
		b.WriteByte(s[i+1])
		i++
	}
	return b.String()
}

// decimalByte parses a leading three-digit decimal escape value of at most 255.
func decimalByte(s string) (byte, bool) {
	if len(s) < 3 {
		return 0, false
	}
	v := 0
	for _, c := range []byte(s[:3]) {
		if c < '0' || c > '9' {
			return 0, false
		}
		v = v*10 + int(c-'0')
	}
	if v > 255 {
		return 0, false
	}
	return byte(v), true
}
