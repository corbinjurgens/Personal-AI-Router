// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package modelconfig

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"slices"
	"strings"

	"nvpair-shared/nodepolicy"
)

// MeasurementIdentity projects configuration applicability, never capacity.
// The wire value lives in nodepolicy to avoid a modelconfig import cycle.
type MeasurementIdentity = nodepolicy.MeasurementIdentity

// MeasurementFromDiagnostic reuses the diagnostic owner's complete hardware
// and realization proof. semantic is the configuration owner's path-free digest
// of applied artifacts, native source and effective load/launch options. Finite
// and unpriced diagnostic keys retain their separate memory-evidence meanings.
func MeasurementFromDiagnostic(semantic string, identity DiagnosticIdentity) MeasurementIdentity {
	if !digest(semantic) || identity.Validate() != nil || identity.BuildInfo == "" || identity.TotalRAMBytes == "0" {
		return MeasurementIdentity{}
	}
	for _, gpu := range identity.GPUs {
		if gpu.TotalMemoryBytes == "0" {
			return MeasurementIdentity{}
		}
	}
	i := identity.Clone()
	slices.SortFunc(i.GPUs, func(a, b DiagnosticGPU) int { return strings.Compare(a.UUID, b.UUID) })
	slices.SortFunc(i.Placement, func(a, b DiagnosticPlacement) int { return strings.Compare(a.DomainID, b.DomainID) })
	data, err := json.Marshal(struct {
		Semantic, Build, CPU, RAM string
		GPUs                      []DiagnosticGPU
		Context, Slots            int32
		Placement                 []DiagnosticPlacement
	}{semantic, i.NativeBuildID, i.CPUModel, i.TotalRAMBytes, i.GPUs, i.PerSlotContext, i.Slots, i.Placement})
	if err != nil {
		return MeasurementIdentity{}
	}
	sum := sha256.Sum256(data)
	return MeasurementIdentity{Digest: hex.EncodeToString(sum[:]), Complete: true}
}
