// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package engines

import (
	"strings"
	"testing"
)

func TestValidID(t *testing.T) {
	for _, id := range []string{"ollama", "llamacpp-custom", "llama.cpp_2", "A1"} {
		if !ValidID(id) {
			t.Errorf("ValidID(%q) = false, want true", id)
		}
	}
	for _, id := range []string{"", ".", "..", "a/b", `a\b`, "a b", "a:b"} {
		if ValidID(id) {
			t.Errorf("ValidID(%q) = true, want false", id)
		}
	}
}

func validDescriptor() Descriptor {
	return Descriptor{
		Version:     DescriptorVersion,
		Engine:      "llamacpp-custom",
		Family:      NameLlamaCPP,
		Label:       "llama.cpp (custom)",
		Platforms:   []string{"darwin/arm64", "linux/amd64"},
		Available:   true,
		Protocols:   []string{ProtocolLlamaCPPRouter, ProtocolOpenAIV1},
		ModelNaming: ModelNamingExact,
		ModelStore:  NameLlamaCPP,
		Catalog:     NameLlamaCPP,
		Operations: map[string]OperationCapability{
			OperationLoad:   {ResponseTimeoutS: 600},
			OperationDelete: {},
		},
	}
}

func TestDescriptorValidate(t *testing.T) {
	if err := validDescriptor().Validate(); err != nil {
		t.Fatalf("valid descriptor rejected: %v", err)
	}
	cases := []struct {
		name   string
		mutate func(*Descriptor)
		want   string
	}{
		{"unknown version", func(d *Descriptor) { d.Version = 2 }, "version"},
		{"bad engine", func(d *Descriptor) { d.Engine = "../x" }, "engine"},
		{"missing family", func(d *Descriptor) { d.Family = "" }, "family"},
		{"blank label", func(d *Descriptor) { d.Label = " " }, "label"},
		{"bad platform", func(d *Descriptor) { d.Platforms = []string{"linux"} }, "platform"},
		{"nested platform", func(d *Descriptor) { d.Platforms = []string{"linux/amd64/v3"} }, "platform"},
		{"unknown protocol", func(d *Descriptor) { d.Protocols = []string{"grpc"} }, "unknown protocol"},
		{"duplicate protocol", func(d *Descriptor) { d.Protocols = []string{ProtocolOpenAIV1, ProtocolOpenAIV1} }, "twice"},
		{"unknown naming", func(d *Descriptor) { d.ModelNaming = "" }, "model naming"},
		{"bad adapter", func(d *Descriptor) { d.ModelStore = "a/b" }, "adapter"},
		{"unknown operation", func(d *Descriptor) { d.Operations["warm"] = OperationCapability{} }, "unknown operation"},
		{"budget too large", func(d *Descriptor) {
			d.Operations[OperationLoad] = OperationCapability{ResponseTimeoutS: MaxOperationResponseTimeoutS + 1}
		}, "budget"},
		{"negative budget", func(d *Descriptor) { d.Operations[OperationLoad] = OperationCapability{ResponseTimeoutS: -1} }, "budget"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			d := validDescriptor()
			tc.mutate(&d)
			if err := d.Validate(); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("Validate error = %v, want %q", err, tc.want)
			}
		})
	}
}

// The vocabularies are shared tables; a caller must not be able to edit them.
func TestVocabulariesAreCopies(t *testing.T) {
	Protocols()[0] = "changed"
	Operations()[0] = "changed"
	ModelNamings()[0] = "changed"
	if Protocols()[0] == "changed" || Operations()[0] == "changed" || ModelNamings()[0] == "changed" {
		t.Fatal("a returned vocabulary aliases the shared table")
	}
}

// Every built-in engine name is a valid descriptor ID and family.
func TestBuiltInNamesAreValidIDs(t *testing.T) {
	for _, name := range Names() {
		if !ValidID(name) {
			t.Errorf("built-in engine %q is not a valid ID", name)
		}
	}
}
