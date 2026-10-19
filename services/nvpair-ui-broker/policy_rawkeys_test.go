// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"nvpair-shared/nodepolicy"
)

func TestPolicySetRawDuplicateKeys(t *testing.T) {
	test := func(name string, peer bool, raw string) {
		t.Run(name, func(t *testing.T) {
			rb := newBrokerOnFile(t, `{"version":2}`)
			before := rb.baseline(t)
			expected := mustJSON(rb.b.localPolicy().Policy.Revision)
			var err error
			if peer {
				_, err = rb.b.setPeerPolicy(json.RawMessage(raw), expected)
			} else {
				_, err = rb.b.setLocalPolicy(json.RawMessage(raw), expected)
			}
			_, parseErr := nodepolicy.Parse([]byte(raw))
			if parseErr == nil || err == nil || !strings.Contains(err.Error(), "duplicate or case-variant") {
				t.Fatalf("setter error = %v, Parse error = %v; want raw duplicate rejection", err, parseErr)
			}
			var invalid *invalidParamsError
			if errors.As(err, &invalid) != peer {
				t.Fatalf("setter error class = %T, want invalidParams only for peer=%v", err, peer)
			}
			before.requireUnchanged(t)
		})
	}
	for _, setter := range []struct {
		name string
		peer bool
	}{{"local", false}, {"peer", true}} {
		test(setter.name+" served null then empty", setter.peer, `{"servedModels":null,"servedModels":[]}`)
		test(setter.name+" resident null then empty", setter.peer, `{"residentModels":null,"residentModels":[]}`)
		test(setter.name+" served empty then permission", setter.peer, `{"servedModels":[],"servedModels":[{"engine":"ollama","model":"m"}]}`)
	}
}
