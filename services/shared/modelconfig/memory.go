// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package modelconfig

import (
	"fmt"
	"math"
	"nvpair-shared/noderec"
	"slices"
	"strconv"
)

var MemoryComponentNames = []string{"weights", "attention-cache", "recurrent-state", "compute-workspace", "prompt-cache", "companions", "runtime-overhead"}

// DeviceBinding is stable launch authority. It deliberately excludes sample
// bytes, clocks and producer-run identities. ReviewedInputID retains historical
// observation provenance without requiring that run to recur on recovery.
type DeviceBinding struct {
	HostID   string `json:"hostId"`
	DomainID string `json:"domainId"`
	DeviceID string `json:"deviceId"`
	Selector string `json:"selector"`
	Digest   string `json:"digest"`
}

func (b DeviceBinding) Validate() error {
	for _, s := range []string{b.HostID, b.DomainID, b.DeviceID, b.Selector} {
		if boundedText(s, 256, true) != nil {
			return fmt.Errorf("deviceBinding: missing identity")
		}
	}
	if b.Selector != "CUDA0" || len(b.DeviceID) != 40 || !stringsGPUUUID(b.DeviceID) || !digest(b.Digest) {
		return fmt.Errorf("deviceBinding: unsupported exact CUDA binding")
	}
	return nil
}
func stringsGPUUUID(s string) bool {
	if len(s) != 40 || s[:4] != "GPU-" {
		return false
	}
	for i, c := range s[4:] {
		if i == 8 || i == 13 || i == 18 || i == 23 {
			if c != '-' {
				return false
			}
			continue
		}
		if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f') {
			return false
		}
	}
	return s != "GPU-00000000-0000-0000-0000-000000000000"
}

// TargetMemory travels only on the target broker's private worker connection.
// The hop is stamped immediately before dispatch, never at decode/receipt.
type TargetMemory struct {
	HostID        string                     `json:"hostId"`
	ObservationID string                     `json:"observationId"`
	MapID         string                     `json:"mapId,omitempty"`
	Observation   *noderec.MemoryObservation `json:"observation"`
	Hop           *noderec.MemoryHop         `json:"hop"`
}

func (m *TargetMemory) Clone() *TargetMemory {
	if m == nil {
		return nil
	}
	c := *m
	c.Observation = m.Observation.Clone()
	c.Hop = clonePointer(m.Hop)
	return &c
}
func (m *TargetMemory) Validate() error {
	if m == nil || boundedText(m.HostID, 256, true) != nil || boundedText(m.ObservationID, 256, true) != nil || boundedText(m.MapID, 256, false) != nil || m.Hop == nil {
		return fmt.Errorf("target memory: missing identity/hop")
	}
	if _, err := m.Hop.ElapsedAt(math.MaxUint64); err != nil {
		return err
	}
	return m.Observation.Validate()
}

// At ages a detached observation by measured same-host transit. Invalid clocks
// retire evidence; producer UTC is never used to restart the freshness budget.
func (m *TargetMemory) At(clock func() (uint64, error)) *noderec.MemoryObservation {
	if m.Validate() != nil {
		return nil
	}
	out := m.Observation.Clone()
	now, err := clock()
	ms, hopErr := m.Hop.ElapsedAt(now)
	if err != nil || hopErr != nil {
		ms = -1
	}
	out.AddAge(ms)
	out.Expire()
	return out
}

type WorkerInspectRequest struct {
	Request InspectRequest `json:"request"`
	Memory  *TargetMemory  `json:"memory,omitempty"`
}

// Generic envelopes preserve the launch owner's strict request/settings type
// without creating a modelconfig -> enginesettings dependency cycle.
type WorkerPreviewRequest[T any] struct {
	Request T             `json:"request"`
	Memory  *TargetMemory `json:"memory,omitempty"`
}
type WorkerFrozenValidation[T any] struct {
	Settings T             `json:"settings"`
	Memory   *TargetMemory `json:"memory,omitempty"`
}

func DecodeWorkerInspectRequest(data []byte) (WorkerInspectRequest, error) {
	var r WorkerInspectRequest
	if err := DecodeStrict(data, &r); err != nil {
		return r, err
	}
	if err := r.Request.Validate(); err != nil {
		return r, err
	}
	if r.Memory != nil {
		if err := r.Memory.Validate(); err != nil {
			return r, err
		}
	}
	return r, nil
}

// The launch owner additionally validates its typed request or finite settings.
func DecodeWorkerPreviewRequest[T any](data []byte) (WorkerPreviewRequest[T], error) {
	var r WorkerPreviewRequest[T]
	if err := DecodeStrict(data, &r); err != nil {
		return r, err
	}
	if r.Memory != nil {
		if err := r.Memory.Validate(); err != nil {
			return r, err
		}
	}
	return r, nil
}
func DecodeWorkerFrozenValidation[T any](data []byte) (WorkerFrozenValidation[T], error) {
	var r WorkerFrozenValidation[T]
	if err := DecodeStrict(data, &r); err != nil {
		return r, err
	}
	if r.Memory != nil {
		if err := r.Memory.Validate(); err != nil {
			return r, err
		}
	}
	return r, nil
}

func validMemoryFit(s string) bool {
	return slices.Contains([]string{"", "unknown", "exceeds", "tight", "fits"}, s)
}
func MemoryReserve(capacity uint64) uint64 {
	r := capacity / 10
	if capacity%10 != 0 {
		r++
	}
	return r + 512<<20
}

// CompareEstimate compares priced ranges and names omissions without treating
// uncertainty as a smaller cost. Selected external figures use domain slots.
// No spare capacity in another pool can cover a shortfall.
func CompareEstimate(e Estimate) Estimate {
	e = e.Clone()
	e.CapacityFit, e.FreeFit = "unknown", "unknown"
	complete := e.Validate() == nil
	summed := e.EstimateSource == nil || e.EstimateSource.Kind == "pair-estimate"
	sums := map[string]uint64{}
	uppers := map[string]uint64{}
	lows := map[string]uint64{}
	valid := map[string]bool{}
	weightsPresent := false
	for _, d := range e.Domains {
		valid[d.ID] = true
	}
	for _, c := range e.Components {
		weightsPresent = weightsPresent || c.Name == "weights"
		if c.Status == "excluded" {
			continue
		}
		if !summed {
			continue
		}
		if c.Status == "partial" || c.Status == "unknown" || c.DomainID == "" {
			for i := range e.Domains {
				if c.DomainID == "" {
					e.Domains[i].Unknowns = append(e.Domains[i].Unknowns, "not priced (any pool): "+c.Name)
				} else if e.Domains[i].ID == c.DomainID {
					e.Domains[i].Unknowns = append(e.Domains[i].Unknowns, "not priced: "+c.Name)
				}
			}
		}
		n, err := strconv.ParseUint(c.Bytes, 10, 64)
		if c.Name == "weights" && (c.Status == "unknown" || c.DomainID == "") {
			for id := range valid {
				if c.DomainID == "" || c.DomainID == id {
					valid[id] = false
				}
			}
		}
		if c.DomainID == "" || err != nil {
			continue
		}
		upper, _ := strconv.ParseUint(c.HighBytes, 10, 64)
		low, _ := strconv.ParseUint(c.LowBytes, 10, 64)
		if n > math.MaxUint64-sums[c.DomainID] || upper > math.MaxUint64-uppers[c.DomainID] || low > math.MaxUint64-lows[c.DomainID] {
			valid[c.DomainID] = false
			continue
		}
		sums[c.DomainID] += n
		uppers[c.DomainID] += upper
		lows[c.DomainID] += low
	}
	complete = complete && (!summed || weightsPresent)
	for i := range e.Domains {
		d := &e.Domains[i]
		d.CapacityFit, d.FreeFit = "unknown", "unknown"
		d.ReserveBytes = ""
		if e.EstimateSource != nil {
			for _, name := range e.EstimateSource.UnpricedCosts {
				d.Unknowns = append(d.Unknowns, "not priced (any pool): "+name)
			}
		}
		required, err := strconv.ParseUint(d.RequiredBytes, 10, 64)
		upper := required
		if summed {
			required, upper = sums[d.ID], uppers[d.ID]
			err = nil
			d.RequiredBytes, d.RequiredLowBytes, d.RequiredHighBytes = strconv.FormatUint(required, 10), strconv.FormatUint(lows[d.ID], 10), strconv.FormatUint(upper, 10)
		} else if d.RequiredHighBytes != "" {
			upper, err = strconv.ParseUint(d.RequiredHighBytes, 10, 64)
		}
		if !complete || !valid[d.ID] || err != nil {
			d.Unknowns = append(d.Unknowns, "costs incomplete or domain sum unresolved")
			continue
		}
		if d.Kind == "device" && d.DeviceID == "" {
			d.Unknowns = append(d.Unknowns, "physical device mapping unavailable")
			continue
		}
		capacity, capErr := strconv.ParseUint(d.CapacityBytes, 10, 64)
		if capErr != nil || capacity == 0 || d.RemainingValidityMS <= 0 || d.SampleID == "" || d.CapacitySource == "" || d.CapacityTimestamp == "" {
			d.Unknowns = append(d.Unknowns, "capacity unavailable or expired")
			continue
		}
		d.ReserveBytes = strconv.FormatUint(MemoryReserve(capacity), 10)
		compare := func(available uint64) string {
			if required > available {
				return "exceeds"
			}
			if upper > available {
				return "tight"
			}
			return CompareMemory(upper, capacity, available)
		}
		d.CapacityFit = compare(capacity)
		free, freeErr := strconv.ParseUint(d.FreeBytes, 10, 64)
		if freeErr == nil && free <= capacity && d.FreeSource != "" && d.FreeTimestamp != "" {
			d.FreeFit = compare(free)
		} else {
			d.Unknowns = append(d.Unknowns, "available memory unavailable")
		}
	}
	aggregate := func(capacity bool) string {
		result := "fits"
		for _, d := range e.Domains {
			s := d.FreeFit
			if capacity {
				s = d.CapacityFit
			}
			if s == "unknown" || s == "" {
				return "unknown"
			}
			if s == "exceeds" {
				result = "exceeds"
			} else if s == "tight" && result == "fits" {
				result = "tight"
			}
		}
		if len(e.Domains) == 0 {
			return "unknown"
		}
		return result
	}
	e.CapacityFit, e.FreeFit = aggregate(true), aggregate(false)
	return e
}

// CompareMemory compares already validated, complete costs against one pool.
// Callers must establish cost completeness, mapping and observation freshness.
// This arithmetic alone does not establish that an estimate is eligible to fit.
func CompareMemory(required, capacity, available uint64) string {
	if capacity == 0 || available > capacity {
		return "unknown"
	}
	if required > available {
		return "exceeds"
	}
	reserve := MemoryReserve(capacity)
	if required > math.MaxUint64-reserve {
		return "unknown"
	}
	if required+reserve > available {
		return "tight"
	}
	return "fits"
}
