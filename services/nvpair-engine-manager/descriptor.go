// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// descriptor.go projects validated manifests into the sanitized descriptors of
// nvpair-shared/engines (engine:descriptors), and owns the family-keyed
// selection of the coded adapters a family shares. The projection copies named
// safe fields only: a manifest's binaries, commands, environment, paths, URLs
// and action definitions never reach it.

import (
	"runtime"
	"slices"

	"nvpair-shared/engines"
)

const descriptorsMethod = "engine:descriptors"

// adapterFamilies are the families with a coded model-store adapter
// (modellayout.go) and a download catalogue (catalog.go). An adapter is named
// after its family.
var adapterFamilies = []string{engines.NameOllama, engines.NameLMStudio, engines.NameLlamaCPP}

// modelStoreAdapter returns the model-store adapter a family selects, or "".
func modelStoreAdapter(family string) string {
	if slices.Contains(adapterFamilies, family) {
		return family
	}
	return ""
}

// catalogAdapter returns the download catalogue a family selects, or "".
func catalogAdapter(family string) string {
	if slices.Contains(adapterFamilies, family) {
		return family
	}
	return ""
}

// familyOf returns a registered engine's family.
func (r *Registry) familyOf(engine string) (string, bool) {
	m, ok := r.Get(engine)
	if !ok {
		return "", false
	}
	return m.FamilyOrDefault(), true
}

// Descriptors returns every registered engine's descriptor for this host,
// sorted by engine ID.
func (r *Registry) Descriptors() []engines.Descriptor {
	names := r.Names()
	out := make([]engines.Descriptor, 0, len(names))
	for _, name := range names {
		m, ok := r.Get(name)
		if !ok {
			if engines.ValidID(name) {
				out = append(out, engines.Descriptor{Version: engines.DescriptorVersion,
					Engine: name, Family: name, Label: name, ModelNaming: engines.ModelNamingExact,
					Platforms: []string{}, Protocols: []string{}, Operations: map[string]engines.OperationCapability{}})
			}
			continue
		}
		out = append(out, m.descriptor(runtime.GOOS, runtime.GOARCH))
	}
	return out
}

// descriptor projects the manifest for a host.
func (m *Manifest) descriptor(goos, goarch string) engines.Descriptor {
	family := m.FamilyOrDefault()
	_, available := m.PlatformFor(goos, goarch)
	d := engines.Descriptor{
		Version:     engines.DescriptorVersion,
		Engine:      m.Engine,
		Family:      family,
		Label:       m.DisplayName,
		Platforms:   sortedKeys(m.Platforms),
		Available:   available,
		Protocols:   append([]string{}, m.Protocols...),
		ModelNaming: m.ModelNamingOrDefault(),
		ModelStore:  modelStoreAdapter(family),
		Catalog:     catalogAdapter(family),
		Operations:  map[string]engines.OperationCapability{},
	}
	for _, op := range engines.Operations() {
		if b, ok := m.operationBinding(op); ok {
			d.Operations[op] = engines.OperationCapability{ResponseTimeoutS: b.ResponseTimeoutS}
		}
	}
	return d
}
