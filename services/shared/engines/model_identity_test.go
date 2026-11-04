// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package engines

import (
	"strings"
	"testing"
)

func TestModelIdentityValidate(t *testing.T) {
	valid := func() ModelIdentity {
		return ModelIdentity{Repository: "owner/repo.name-1", Revision: strings.Repeat("a", 40), Quantization: "Q4_K_M", QuantizationSource: "hf-tag", Strength: "verified", ContentID: strings.Repeat("b", 64), Files: []ModelIdentityFile{{Role: "model", Name: "model.gguf", SizeBytes: 1, SHA256: strings.Repeat("c", 64)}}}
	}
	if err := valid().Validate(); err != nil {
		t.Fatal(err)
	}
	if err := (ModelIdentity{Strength: "declared"}).Validate(); err != nil {
		t.Fatal(err)
	}
	reject := func(name string, change func(*ModelIdentity)) {
		t.Run(name, func(t *testing.T) {
			m := valid()
			change(&m)
			if err := m.Validate(); err == nil {
				t.Fatal("accepted invalid identity")
			}
		})
	}
	reject("repository format", func(m *ModelIdentity) { m.Repository = "owner/repo/extra" })
	reject("revision length", func(m *ModelIdentity) { m.Revision = "abc" })
	reject("revision uppercase", func(m *ModelIdentity) { m.Revision = strings.Repeat("A", 40) })
	reject("revision nonhex", func(m *ModelIdentity) { m.Revision = strings.Repeat("g", 40) })
	reject("content length", func(m *ModelIdentity) { m.ContentID = "abc" })
	reject("content uppercase", func(m *ModelIdentity) { m.ContentID = strings.Repeat("A", 64) })
	reject("content nonhex", func(m *ModelIdentity) { m.ContentID = strings.Repeat("g", 64) })
	reject("sha length", func(m *ModelIdentity) { m.Files[0].SHA256 = "abc" })
	reject("sha uppercase", func(m *ModelIdentity) { m.Files[0].SHA256 = strings.Repeat("A", 64) })
	reject("sha nonhex", func(m *ModelIdentity) { m.Files[0].SHA256 = strings.Repeat("g", 64) })
	reject("source enum", func(m *ModelIdentity) { m.QuantizationSource = "unknown" })
	reject("source without label", func(m *ModelIdentity) { m.Quantization = "" })
	reject("verified without content", func(m *ModelIdentity) { m.ContentID = "" })
	reject("declared with content", func(m *ModelIdentity) { m.Strength = "declared" })
	reject("strength enum", func(m *ModelIdentity) { m.Strength = "other" })
	reject("too many files", func(m *ModelIdentity) { m.Files = make([]ModelIdentityFile, 65) })
	reject("forward slash", func(m *ModelIdentity) { m.Files[0].Name = "a/b" })
	reject("backslash", func(m *ModelIdentity) { m.Files[0].Name = `a\b` })
	reject("negative total", func(m *ModelIdentity) { m.TotalBytes = -1 })
	reject("negative trained context", func(m *ModelIdentity) { m.TrainedContextTokens = -1 })
	reject("negative configured context", func(m *ModelIdentity) { m.ConfiguredContextTokens = -1 })
	reject("negative file size", func(m *ModelIdentity) { m.Files[0].SizeBytes = -1 })
	reject("file role enum", func(m *ModelIdentity) { m.Files[0].Role = "other" })
	fields := []struct {
		name string
		set  func(*ModelIdentity, string)
	}{
		{"repository", func(m *ModelIdentity, s string) { m.Repository = s }},
		{"revision", func(m *ModelIdentity, s string) { m.Revision = s }},
		{"quantization", func(m *ModelIdentity, s string) { m.Quantization = s }},
		{"source", func(m *ModelIdentity, s string) { m.QuantizationSource = s }},
		{"general name", func(m *ModelIdentity, s string) { m.GeneralName = s }},
		{"size label", func(m *ModelIdentity, s string) { m.SizeLabel = s }},
		{"architecture", func(m *ModelIdentity, s string) { m.Architecture = s }},
		{"content", func(m *ModelIdentity, s string) { m.ContentID = s }},
		{"strength", func(m *ModelIdentity, s string) { m.Strength = s }},
		{"file role", func(m *ModelIdentity, s string) { m.Files[0].Role = s }},
		{"file name", func(m *ModelIdentity, s string) { m.Files[0].Name = s }},
		{"file sha", func(m *ModelIdentity, s string) { m.Files[0].SHA256 = s }},
	}
	for _, field := range fields {
		reject(field.name+" invalid UTF8", func(m *ModelIdentity) { field.set(m, string([]byte{0xff})) })
		reject(field.name+" control", func(m *ModelIdentity) { field.set(m, "a\u0085b") })
		reject(field.name+" long", func(m *ModelIdentity) { field.set(m, strings.Repeat("x", 257)) })
	}
	for _, source := range []string{"hf-tag", "file-name", "gguf-file-type"} {
		t.Run(source, func(t *testing.T) {
			m := valid()
			m.QuantizationSource = source
			if err := m.Validate(); err != nil {
				t.Fatal(err)
			}
		})
	}
	t.Run("boundary strings and file count", func(t *testing.T) {
		m := valid()
		m.GeneralName = strings.Repeat("é", 128)
		m.Files = make([]ModelIdentityFile, 64)
		for i := range m.Files {
			m.Files[i] = ModelIdentityFile{Role: "companion", Name: "file"}
		}
		if err := m.Validate(); err != nil {
			t.Fatal(err)
		}
	})
}
