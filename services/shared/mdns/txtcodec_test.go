// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package mdns

import (
	"strings"
	"testing"

	"github.com/miekg/dns"
)

func TestTXTRoundTrip(t *testing.T) {
	// A 128-byte display name (the validator's maximum) of mostly multibyte
	// text: every non-ASCII byte becomes a \DDD escape when unpacked.
	longName := strings.Repeat("白", 42) + "ab"
	if len(longName) != 128 {
		t.Fatalf("long name fixture is %d bytes, want 128", len(longName))
	}
	samples := []struct {
		name string
		txt  string
	}{
		{"dots inside", "name=Desk.A"},
		{"leading and trailing dots", "name=.Studio."},
		{"backslash", `name=back\slash`},
		{"quotes", `name="quoted"`},
		{"equals in value", "name=a=b"},
		{"numeric value", "name=1234"},
		{"japanese", "name=白いPC"},
		{"nfc accent", "name=Café"},
		{"emoji zwj sequence", "name=\U0001F469‍\U0001F4BB"},
		{"133 bytes", "name=" + longName},
	}
	for _, tc := range samples {
		t.Run(tc.name, func(t *testing.T) {
			r := testResponder([]string{"v=1", tc.txt})
			msg := &dns.Msg{MsgHdr: dns.MsgHdr{Response: true, Authoritative: true}}
			msg.Answer = append(msg.Answer, r.txtRR(true))
			packed, err := msg.Pack()
			if err != nil {
				t.Fatalf("pack: %v", err)
			}
			var got dns.Msg
			if err := got.Unpack(packed); err != nil {
				t.Fatalf("unpack: %v", err)
			}
			rr, ok := got.Answer[0].(*dns.TXT)
			if !ok {
				t.Fatalf("answer = %T, want *dns.TXT", got.Answer[0])
			}
			if len(rr.Txt) != 2 {
				t.Fatalf("unpacked TXT = %q, want two strings", rr.Txt)
			}
			if decoded := UnescapeTXT(rr.Txt[1]); decoded != tc.txt {
				t.Fatalf("round trip = %q, want %q (wire form %q)", decoded, tc.txt, rr.Txt[1])
			}
		})
	}
}

func TestEscapeTXTKeepsExistingASCIIValues(t *testing.T) {
	for _, s := range []string{
		"uuid=0f6c3c1e-8d4f-4a52-9a0e-5b1f2d3c4e5f",
		"ips=10.0.0.1,10.0.0.2",
		"ni=14320",
	} {
		if got := EscapeTXT(s); got != s {
			t.Errorf("EscapeTXT(%q) = %q, want unchanged", s, got)
		}
	}
}

func TestUnescapeTXT(t *testing.T) {
	cases := []struct {
		name, in, want string
	}{
		{"plain", "v=1", "v=1"},
		{"escaped quote and backslash", `a\"b\\c`, `a"b\c`},
		{"decimal escape", `\231\153\189`, "白"},
		{"decimal over 255 is a plain escape", `\256`, "256"},
		{"short decimal is a plain escape", `\12`, "12"},
		{"trailing lone backslash", `abc\`, `abc\`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := UnescapeTXT(tc.in); got != tc.want {
				t.Fatalf("UnescapeTXT(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}
