// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package noderec

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"math"
	"strconv"
	"time"
	"unicode"
	"unicode/utf8"
)

// MemoryFact describes one independently sampled capacity or available value.
// Bytes is a canonical decimal string so JSON consumers do not lose precision.
type MemoryFact struct {
	Bytes     string    `json:"bytes,omitempty"`
	Source    string    `json:"source,omitempty"`
	SampledAt time.Time `json:"sampledAt,omitempty"`
	AgeMS     int64     `json:"ageMs"`
	Valid     bool      `json:"valid"`
	Reason    string    `json:"reason,omitempty"`
}

func (f *MemoryFact) UnmarshalJSON(data []byte) error {
	if err := memoryObject(data, []string{"bytes", "source", "sampledAt", "ageMs", "valid", "reason"}, []string{"ageMs", "valid"}); err != nil {
		return err
	}
	type wire MemoryFact
	var value wire
	if err := json.Unmarshal(data, &value); err != nil {
		return err
	}
	decoded := MemoryFact(value)
	if _, err := decoded.validate(); err != nil {
		return err
	}
	*f = decoded
	return nil
}

type MemoryDomain struct {
	ID        string     `json:"id"`
	Kind      string     `json:"kind"`
	DeviceIDs []string   `json:"deviceIds,omitempty"`
	Total     MemoryFact `json:"total"`
	Free      MemoryFact `json:"free"`
}

func (d *MemoryDomain) UnmarshalJSON(data []byte) error {
	if err := memoryObject(data, []string{"id", "kind", "deviceIds", "total", "free"}, []string{"id", "kind", "total", "free"}); err != nil {
		return err
	}
	type wire MemoryDomain
	var value wire
	if err := json.Unmarshal(data, &value); err != nil {
		return err
	}
	*d = MemoryDomain(value)
	return nil
}

// MemoryObservation is scoped to one node-info process run.
type MemoryObservation struct {
	RunID   string         `json:"runId"`
	Domains []MemoryDomain `json:"domains"`
}

func (m *MemoryObservation) UnmarshalJSON(data []byte) error {
	if err := memoryObject(data, []string{"runId", "domains"}, []string{"runId", "domains"}); err != nil {
		return err
	}
	type wire MemoryObservation
	var value wire
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&value); err != nil {
		return err
	}
	decoded := MemoryObservation(value)
	if err := decoded.Validate(); err != nil {
		return err
	}
	*m = decoded
	return nil
}

// A small observation budget leaves room for the rest of the node telemetry
// notification within the scanner's 1 MiB JSON-RPC frame.
const MaxMemoryObservationBytes = 64 << 10

// memoryObject checks exact keys before encoding/json can fold spelling or
// replace duplicates. Every present field is non-null; required scalars cannot
// silently become fresh zero values.
func memoryObject(data []byte, allowed, required []string) error {
	if len(data) > MaxMemoryObservationBytes || !utf8.Valid(data) {
		return errors.New("invalid memory JSON size or UTF-8")
	}
	d := json.NewDecoder(bytes.NewReader(data))
	start, err := d.Token()
	if err != nil || start != json.Delim('{') {
		return errors.New("memory JSON must be an object")
	}
	seen := make(map[string]bool, len(allowed))
	for d.More() {
		token, err := d.Token()
		if err != nil {
			return err
		}
		key, ok := token.(string)
		accepted := false
		for _, name := range allowed {
			accepted = accepted || name == key
		}
		if !ok || !accepted || seen[key] {
			return errors.New("unknown or repeated memory JSON field")
		}
		seen[key] = true
		var raw json.RawMessage
		if err := d.Decode(&raw); err != nil {
			return err
		}
		if bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
			return errors.New("null memory JSON field")
		}
	}
	if _, err := d.Token(); err != nil {
		return err
	}
	if _, err := d.Token(); err != io.EOF {
		return errors.New("trailing memory JSON")
	}
	for _, key := range required {
		if !seen[key] {
			return errors.New("missing memory JSON field")
		}
	}
	return nil
}

// MemoryHop identifies the local forwarder and its pre-serialization boot time.
// It is meaningful only on the host where the forwarder and receiver run.
type MemoryHop struct {
	ForwarderID string `json:"forwarderId"`
	BootNS      string `json:"bootNs"`
}

func (h *MemoryHop) UnmarshalJSON(data []byte) error {
	if err := memoryObject(data, []string{"forwarderId", "bootNs"}, []string{"forwarderId", "bootNs"}); err != nil {
		return err
	}
	type wire MemoryHop
	var value wire
	if err := json.Unmarshal(data, &value); err != nil {
		return err
	}
	if !validMemoryText(value.ForwarderID) {
		return errors.New("invalid memory forwarder identity")
	}
	if _, err := memoryDecimal(value.BootNS); err != nil {
		return err
	}
	*h = MemoryHop(value)
	return nil
}

// ElapsedMemoryMS rounds up, so transport time never makes a sample younger.
func ElapsedMemoryMS(start, end uint64) (int64, error) {
	if end < start {
		return 0, errors.New("boot clock regressed")
	}
	delta := end - start
	ms := delta / 1000000
	if delta%1000000 != 0 {
		ms++
	}
	if ms > math.MaxInt64 {
		return 0, errors.New("boot clock duration overflow")
	}
	return int64(ms), nil
}

// ElapsedAt validates a same-host hop stamp before calculating its age.
func (h MemoryHop) ElapsedAt(now uint64) (int64, error) {
	if !validMemoryText(h.ForwarderID) {
		return 0, errors.New("invalid forwarder identity")
	}
	start, err := memoryDecimal(h.BootNS)
	if err != nil {
		return 0, err
	}
	return ElapsedMemoryMS(start, now)
}

// AddAge advances every known fact without changing its sample provenance.
// An invalid duration or overflow makes that fact explicitly unknown.
func (m *MemoryObservation) AddAge(ms int64) {
	if m == nil {
		return
	}
	for i := range m.Domains {
		for _, fact := range []*MemoryFact{&m.Domains[i].Total, &m.Domains[i].Free} {
			if !fact.Valid {
				continue
			}
			if ms < 0 || fact.AgeMS < 0 || ms > math.MaxInt64-fact.AgeMS {
				*fact = MemoryFact{Reason: "invalid-age"}
				continue
			}
			fact.AgeMS += ms
		}
	}
}

// Clone detaches all mutable slices from the original observation.
func (m *MemoryObservation) Clone() *MemoryObservation {
	if m == nil {
		return nil
	}
	out := *m
	if m.Domains != nil {
		out.Domains = make([]MemoryDomain, len(m.Domains))
		copy(out.Domains, m.Domains)
	}
	for i := range out.Domains {
		out.Domains[i].DeviceIDs = append([]string(nil), m.Domains[i].DeviceIDs...)
	}
	return &out
}

func memoryDecimal(s string) (uint64, error) {
	n, err := strconv.ParseUint(s, 10, 64)
	if err != nil || strconv.FormatUint(n, 10) != s {
		return 0, errors.New("invalid decimal")
	}
	return n, nil
}

func validMemoryText(s string) bool {
	if len(s) == 0 || len(s) > 256 || !utf8.ValidString(s) {
		return false
	}
	for _, r := range s {
		if unicode.IsControl(r) {
			return false
		}
	}
	return true
}

const MemoryFreshMS int64 = 5000

func (m *MemoryObservation) Expire() {
	if m == nil {
		return
	}
	for i := range m.Domains {
		for _, f := range []*MemoryFact{&m.Domains[i].Total, &m.Domains[i].Free} {
			if f.Valid && f.AgeMS > MemoryFreshMS {
				*f = MemoryFact{Reason: "stale"}
			}
		}
	}
}

func (m *MemoryObservation) Advance(start uint64, clockErr error, clock func() (uint64, error)) *MemoryObservation {
	out := m.Clone()
	if out == nil {
		return nil
	}
	if out.Validate() != nil {
		return nil
	}
	end, err := clock()
	ms, ageErr := ElapsedMemoryMS(start, end)
	if clockErr != nil || err != nil || ageErr != nil {
		ms = -1
	}
	out.AddAge(ms)
	return out
}

func NewMemoryHop(forwarder string, clock func() (uint64, error)) *MemoryHop {
	now, err := clock()
	if err != nil {
		return nil
	}
	return &MemoryHop{ForwarderID: forwarder, BootNS: strconv.FormatUint(now, 10)}
}

func (f MemoryFact) validate() (uint64, error) {
	if !f.Valid {
		if !validMemoryText(f.Reason) || f.Bytes != "" || f.Source != "" || !f.SampledAt.IsZero() || f.AgeMS != 0 {
			return 0, errors.New("invalid unknown memory fact")
		}
		return 0, nil
	}
	if !validMemoryText(f.Source) || f.Reason != "" || f.SampledAt.IsZero() || f.SampledAt.Location() != time.UTC || f.AgeMS < 0 {
		return 0, errors.New("invalid memory fact provenance")
	}
	return memoryDecimal(f.Bytes)
}

// Validate rejects malformed, ambiguous and overlapping physical pools.
func (m *MemoryObservation) Validate() error {
	if m == nil || !validMemoryText(m.RunID) || m.Domains == nil || len(m.Domains) > 64 {
		return errors.New("invalid memory observation")
	}
	ids, devices := map[string]bool{}, map[string]bool{}
	shared, system := false, false
	for _, d := range m.Domains {
		if !validMemoryText(d.ID) || ids[d.ID] || len(d.DeviceIDs) > 64 {
			return errors.New("invalid memory domain identity")
		}
		ids[d.ID] = true
		switch d.Kind {
		case "system":
			if system || shared {
				return errors.New("overlapping system memory")
			}
			system = true
		case "shared":
			if shared || system {
				return errors.New("overlapping shared memory")
			}
			shared = true
		case "dedicated", "unknown":
		default:
			return errors.New("invalid memory domain kind")
		}
		for _, id := range d.DeviceIDs {
			if !validMemoryText(id) || devices[id] {
				return errors.New("duplicate memory device")
			}
			devices[id] = true
		}
		total, err := d.Total.validate()
		if err != nil {
			return err
		}
		free, err := d.Free.validate()
		if err != nil {
			return err
		}
		if d.Total.Valid && d.Free.Valid && free > total {
			return errors.New("available memory exceeds total")
		}
	}
	encoded, err := json.Marshal(m)
	if err != nil || len(encoded) > MaxMemoryObservationBytes {
		return errors.New("memory observation exceeds encoded budget")
	}
	return nil
}

func memoryClockTimespec(sec, nsec int64) (uint64, error) {
	if sec < 0 || nsec < 0 || nsec >= 1e9 || uint64(sec) > (math.MaxUint64-uint64(nsec))/1e9 {
		return 0, errors.New("invalid boot clock")
	}
	return uint64(sec)*1e9 + uint64(nsec), nil
}
