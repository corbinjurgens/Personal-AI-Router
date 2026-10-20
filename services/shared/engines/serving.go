// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package engines

import (
	"encoding/hex"
	"fmt"
	"reflect"
	"slices"
	"strconv"
	"strings"
)

// ServingConfig is an operator-approved declaration, not runtime evidence.
// Models are keyed by their exact served IDs; aliases are not normalized here.
type ServingConfig struct {
	Version int            `json:"version"`
	Models  []ServingModel `json:"models"`
}

type ServingModel struct {
	ModelID                    string            `json:"modelId"`
	MaxOutputTokens            int64             `json:"maxOutputTokens"`
	Memory                     ServingMemory     `json:"memory"`
	Artifacts                  []ServingArtifact `json:"artifacts"`
	ExpectedNativeSourceDigest string            `json:"expectedNativeSourceDigest,omitempty"`
}

// Pointers distinguish an unknown resource domain from an explicit CPU-only 0.
type ServingMemory struct {
	HostRAMBytes *int64 `json:"hostRamBytes"`
	VRAMBytes    *int64 `json:"vramBytes"`
}

type ServingArtifact struct {
	Role       string            `json:"role"`
	SHA256     string            `json:"sha256"`
	Provenance ServingProvenance `json:"provenance"`
	Binding    *ServingBinding   `json:"binding,omitempty"`
}

// Kind identifies trusted digest provenance, without assertion IDs or a
// separate attestation permission/acceptance contract.
type ServingProvenance struct {
	Kind      string `json:"kind"`
	Reference string `json:"reference,omitempty"`
}

type ServingBinding struct {
	SourcePath     string `json:"sourcePath"`
	ResolvedTarget string `json:"resolvedTarget"`
	SizeBytes      int64  `json:"sizeBytes"`
	// Decimal text preserves nanoseconds across JSON clients with IEEE doubles.
	MtimeNS      string `json:"mtimeNs"`
	SampleSHA256 string `json:"sampleSha256"`
}

// ServingEnvelope is derived by the engine adapter from the native source.
// A declaration alone must never populate LaunchValidated or Observed.
type ServingEnvelope struct {
	ModelID              string        `json:"modelId"`
	PerSlotContextTokens int64         `json:"perSlotContextTokens"`
	TotalContextTokens   int64         `json:"totalContextTokens"`
	Slots                int64         `json:"slots"`
	MaxOutputTokens      int64         `json:"maxOutputTokens"`
	Memory               ServingMemory `json:"memory"`
	NativeSourceDigest   string        `json:"nativeSourceDigest"`
	Placement            []string      `json:"placement,omitempty"`
	Retention            string        `json:"retention,omitempty"`
	RouterCapacity       int64         `json:"routerCapacity,omitempty"`
}

type ServingObservation struct {
	ModelID              string `json:"modelId"`
	PerSlotContextTokens int64  `json:"perSlotContextTokens"`
	Slots                int64  `json:"slots"`
}

type ServingReadback struct {
	Status          string               `json:"status"`
	Reason          string               `json:"reason,omitempty"`
	Configured      []ServingEnvelope    `json:"configured,omitempty"`
	LaunchValidated []ServingEnvelope    `json:"launchValidated,omitempty"`
	Observed        []ServingObservation `json:"observed,omitempty"`
}

// Validate is structural and performs no file, network or process operations.
// JSON integer decoding rejects fractional and overflowing int64 values; the
// tighter safe-integer bound keeps declarations lossless in desktop clients.
func (c *ServingConfig) Validate() map[string]string {
	errors := map[string]string{}
	if c == nil {
		return errors
	}
	if c.Version != 1 {
		errors["serving.version"] = "Unsupported serving version."
	}
	if len(c.Models) == 0 {
		errors["serving.models"] = "Declare at least one served model."
	}
	ids := map[string]bool{}
	const maxInteger = int64(1<<53 - 1)
	digest := func(key, value string) {
		decoded, err := hex.DecodeString(value)
		if err != nil || len(decoded) != 32 {
			errors[key] = "Enter a full SHA-256 digest."
		}
	}
	for i, model := range c.Models {
		key := fmt.Sprintf("serving.models.%d", i)
		if strings.TrimSpace(model.ModelID) == "" || ids[model.ModelID] {
			errors[key+".modelId"] = "Use a nonempty, unique exact served model ID."
		}
		ids[model.ModelID] = true
		if model.MaxOutputTokens <= 0 || model.MaxOutputTokens > maxInteger {
			errors[key+".maxOutputTokens"] = "Enter a positive safe integer output bound."
		}
		if model.Memory.HostRAMBytes == nil || *model.Memory.HostRAMBytes <= 0 || *model.Memory.HostRAMBytes > maxInteger {
			errors[key+".memory.hostRamBytes"] = "Declare a positive safe integer host RAM bound."
		}
		if model.Memory.VRAMBytes == nil || *model.Memory.VRAMBytes < 0 || *model.Memory.VRAMBytes > maxInteger {
			errors[key+".memory.vramBytes"] = "Declare a nonnegative safe integer VRAM bound; zero means CPU-only."
		}
		if model.ExpectedNativeSourceDigest != "" {
			digest(key+".expectedNativeSourceDigest", model.ExpectedNativeSourceDigest)
		}
		if len(model.Artifacts) == 0 {
			errors[key+".artifacts"] = "Declare the consumed model artifacts and their digest provenance."
		}
		for j, artifact := range model.Artifacts {
			field := fmt.Sprintf("%s.artifacts.%d", key, j)
			if strings.TrimSpace(artifact.Role) == "" {
				errors[field+".role"] = "Declare the artifact role."
			}
			digest(field+".sha256", artifact.SHA256)
			switch artifact.Provenance.Kind {
			case "hf-lfs-sha256", "strata-source-verification", "verified-manifest", "one-time-sha256", "supplied-sha256":
			default:
				errors[field+".provenance.kind"] = "Declare supported SHA-256 provenance."
			}
			if b := artifact.Binding; b != nil {
				if strings.TrimSpace(b.SourcePath) == "" {
					errors[field+".binding.sourcePath"] = "Declare the source path."
				}
				if strings.TrimSpace(b.ResolvedTarget) == "" {
					errors[field+".binding.resolvedTarget"] = "Declare the resolved target."
				}
				if b.SizeBytes <= 0 || b.SizeBytes > maxInteger {
					errors[field+".binding.sizeBytes"] = "Declare a positive safe integer artifact size."
				}
				if n, err := strconv.ParseInt(b.MtimeNS, 10, 64); err != nil || strconv.FormatInt(n, 10) != b.MtimeNS {
					errors[field+".binding.mtimeNs"] = "Declare an int64 nanosecond timestamp as decimal text."
				}
				digest(field+".binding.sampleSha256", b.SampleSHA256)
			}
		}
	}
	return errors
}

func (c *ServingConfig) Equal(other *ServingConfig) bool { return reflect.DeepEqual(c, other) }

func cloneMemory(m ServingMemory) ServingMemory {
	if m.HostRAMBytes != nil {
		n := *m.HostRAMBytes
		m.HostRAMBytes = &n
	}
	if m.VRAMBytes != nil {
		n := *m.VRAMBytes
		m.VRAMBytes = &n
	}
	return m
}

func (c *ServingConfig) Clone() *ServingConfig {
	if c == nil {
		return nil
	}
	copy := *c
	copy.Models = slices.Clone(c.Models)
	for i := range copy.Models {
		m := &copy.Models[i]
		m.Memory = cloneMemory(m.Memory)
		m.Artifacts = slices.Clone(m.Artifacts)
		for j := range m.Artifacts {
			if b := m.Artifacts[j].Binding; b != nil {
				value := *b
				m.Artifacts[j].Binding = &value
			}
		}
	}
	return &copy
}

func (r *ServingReadback) Clone() *ServingReadback {
	if r == nil {
		return nil
	}
	copy := *r
	cloneEnvelopes := func(values []ServingEnvelope) []ServingEnvelope {
		values = slices.Clone(values)
		for i := range values {
			values[i].Memory = cloneMemory(values[i].Memory)
			values[i].Placement = slices.Clone(values[i].Placement)
		}
		return values
	}
	copy.Configured = cloneEnvelopes(r.Configured)
	copy.LaunchValidated = cloneEnvelopes(r.LaunchValidated)
	copy.Observed = slices.Clone(r.Observed)
	return &copy
}
