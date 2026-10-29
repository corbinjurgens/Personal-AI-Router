// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package modelconfig

import (
	"encoding/json"
	"fmt"
	"math/big"
	"slices"
	"strconv"
	"strings"
	"time"
)

// BufferObservation is optional diagnostic evidence, never allocation authority.
// Byte endpoints enclose a rounded report; they are not exact native bytes.
type BufferObservation struct {
	Status            string             `json:"status"`
	Reason            string             `json:"reason,omitempty"`
	SampleID          string             `json:"sampleId,omitempty"`
	RecordedAt        string             `json:"recordedAt,omitempty"`
	Current           bool               `json:"current"`
	Persistence       string             `json:"persistence,omitempty"`
	UpstreamPin       string             `json:"upstreamPin,omitempty"`
	BuildAssurance    string             `json:"buildAssurance,omitempty"`
	EstimatorRevision string             `json:"estimatorRevision,omitempty"`
	InputIdentity     string             `json:"inputIdentity,omitempty"`
	SourceIdentity    string             `json:"sourceIdentity,omitempty"`
	OptionsIdentity   string             `json:"optionsIdentity,omitempty"`
	TopologyIdentity  string             `json:"topologyIdentity,omitempty"`
	LoadPhase         string             `json:"loadPhase,omitempty"`
	PerSlotContext    int32              `json:"perSlotContext,omitempty"`
	Slots             int32              `json:"slots,omitempty"`
	CapturedOptions   *RequestedLoad     `json:"capturedOptions,omitempty"`
	RequestedOptions  *RequestedLoad     `json:"requestedOptions,omitempty"`
	OptionUnknowns    []string           `json:"optionUnknowns,omitempty"`
	Artifacts         []ArtifactSummary  `json:"artifacts,omitempty"`
	Baseline          []BufferBaseline   `json:"baseline,omitempty"`
	Comparisons       []BufferComparison `json:"comparisons,omitempty"`
	Rows              []BufferReportRow  `json:"rows,omitempty"`
	Reasons           []string           `json:"reasons,omitempty"`
}

// Only closed component states and numeric values cross the observation
// boundary. Estimator diagnostics, domain descriptions and paths do not.
type BufferBaseline struct {
	Component string `json:"component"`
	Status    string `json:"status"`
	Bytes     string `json:"bytes,omitempty"`
}

type BufferComparison struct {
	Component       string `json:"component"`
	Status          string `json:"status"`
	EstimatedBytes  string `json:"estimatedBytes,omitempty"`
	LowerBytes      string `json:"lowerBytes,omitempty"`
	UpperBytes      string `json:"upperBytes,omitempty"`
	DeltaLowerBytes string `json:"deltaLowerBytes,omitempty"`
	DeltaUpperBytes string `json:"deltaUpperBytes,omitempty"`
}

type BufferReportRow struct {
	Component      string `json:"component"`
	Label          string `json:"label"`
	Phase          string `json:"phase"`
	Sequence       string `json:"sequence"`
	Stream         string `json:"stream,omitempty"`
	Role           string `json:"role"`
	ReportedMiB    string `json:"reportedMiB"`
	Unit           string `json:"unit"`
	Precision      string `json:"precision"`
	SourceKind     string `json:"sourceKind"`
	IntervalStatus string `json:"intervalStatus"`
	LowerBytes     string `json:"lowerBytes,omitempty"`
	UpperBytes     string `json:"upperBytes,omitempty"`
	EstimatedBytes string `json:"estimatedBytes,omitempty"`
	DeviceBinding  string `json:"deviceBinding,omitempty"`
	PoolIdentity   string `json:"poolIdentity,omitempty"`
	Coverage       string `json:"coverage"`
}

func (o BufferObservation) Clone() BufferObservation {
	o.Rows = slices.Clone(o.Rows)
	o.Reasons = slices.Clone(o.Reasons)
	o.Artifacts = slices.Clone(o.Artifacts)
	o.Baseline = slices.Clone(o.Baseline)
	o.Comparisons = slices.Clone(o.Comparisons)
	if o.CapturedOptions != nil {
		c := o.CapturedOptions.Clone()
		o.CapturedOptions = &c
	}
	if o.RequestedOptions != nil {
		c := o.RequestedOptions.Clone()
		o.RequestedOptions = &c
	}
	o.OptionUnknowns = slices.Clone(o.OptionUnknowns)
	return o
}

// BufferReportInterval uses integer thousandths of MiB and rounds outwards.
// The two-place emitter's inclusive interval includes midpoint ties and zero.
func BufferReportInterval(value string) (string, string, error) {
	parts := strings.Split(value, ".")
	if !canonicalBufferReport(value) {
		return "", "", fmt.Errorf("buffer report: expected canonical two-place MiB")
	}
	for _, c := range parts[1] {
		if c < '0' || c > '9' {
			return "", "", fmt.Errorf("buffer report: invalid fraction")
		}
	}
	whole, err := strconv.ParseUint(parts[0], 10, 64)
	if err != nil || whole >= 1<<30 {
		return "", "", fmt.Errorf("buffer report: interval exceeds support budget")
	}
	fraction, _ := strconv.ParseUint(parts[1], 10, 64)
	q := whole*1000 + fraction*10
	low := uint64(0)
	if q >= 5 {
		low = (q - 5) * 1048576 / 1000
	}
	high := ((q+5)*1048576 + 999) / 1000
	return strconv.FormatUint(low, 10), strconv.FormatUint(high, 10), nil
}

func canonicalBufferReport(value string) bool {
	if len(value) > 256 {
		return false
	}
	parts := strings.Split(value, ".")
	if len(parts) != 2 || len(parts[0]) == 0 || len(parts[1]) != 2 || len(parts[0]) > 1 && parts[0][0] == '0' {
		return false
	}
	for _, part := range parts {
		for _, c := range part {
			if c < '0' || c > '9' {
				return false
			}
		}
	}
	return true
}

func canonicalBufferNumeric(value string) bool {
	if len(value) > 256 {
		return false
	}
	parts := strings.Split(value, ".")
	if len(parts) != 2 || len(parts[0]) == 0 || len(parts[1]) == 0 || len(parts[1]) > 16 || len(parts[0]) > 1 && parts[0][0] == '0' {
		return false
	}
	for _, part := range parts {
		for _, c := range part {
			if c < '0' || c > '9' {
				return false
			}
		}
	}
	return true
}

func validBufferLabel(label string) bool {
	if slices.Contains([]string{"CPU", "CPU_Mapped", "CUDA_Host"}, label) {
		return true
	}
	number := ""
	if strings.HasPrefix(label, "CUDA") {
		number = strings.TrimPrefix(label, "CUDA")
	}
	if strings.HasPrefix(label, "MTL") {
		number = strings.TrimPrefix(label, "MTL")
		if strings.HasSuffix(number, "_Private") {
			number = strings.TrimSuffix(number, "_Private")
		} else {
			number = strings.TrimSuffix(number, "_Mapped")
		}
	}
	if number == "" || ValidateUnsignedDecimal(number) != nil {
		return false
	}
	n, err := strconv.ParseUint(number, 10, 8)
	return err == nil && n < 64
}

func (o BufferObservation) Validate() error {
	if !slices.Contains([]string{"unknown", "partial", "reported"}, o.Status) || len(o.Rows) > 256 || len(o.Reasons) > 16 {
		return fmt.Errorf("loadObservation: invalid state or collection budget")
	}
	for _, value := range []string{o.Reason, o.SampleID, o.RecordedAt, o.Persistence, o.UpstreamPin, o.BuildAssurance, o.EstimatorRevision, o.InputIdentity} {
		if boundedText(value, 256, false) != nil {
			return fmt.Errorf("loadObservation: invalid bounded identifier")
		}
	}
	if o.Persistence != "" && !slices.Contains([]string{"pending", "saved", "uncertain", "failed", "history-unavailable"}, o.Persistence) {
		return fmt.Errorf("loadObservation: invalid persistence")
	}
	for _, id := range []string{o.SampleID, o.InputIdentity, o.SourceIdentity, o.OptionsIdentity, o.TopologyIdentity} {
		if id != "" && !digest(id) {
			return fmt.Errorf("loadObservation: expected opaque SHA256 identity")
		}
	}
	if o.LoadPhase != "" && !slices.Contains([]string{"initial", "wakeup"}, o.LoadPhase) || o.PerSlotContext < 0 || o.Slots < 0 || len(o.Artifacts) > MaxArtifacts || len(o.Baseline) > 64 || len(o.Comparisons) > 6 {
		return fmt.Errorf("loadObservation: invalid frozen load")
	}
	if o.CapturedOptions != nil {
		if err := o.CapturedOptions.Validate(); err != nil {
			return err
		}
	}
	if o.RequestedOptions != nil {
		if err := o.RequestedOptions.Validate(); err != nil {
			return err
		}
	}
	if len(o.OptionUnknowns) > 64 {
		return fmt.Errorf("loadObservation: too many option unknowns")
	}
	for _, field := range o.OptionUnknowns {
		if !slices.Contains([]string{"mmap", "load-mode", "flash-attention", "batch", "ubatch", "cache-k", "cache-v", "kv-offload", "op-offload", "placement", "speculation"}, field) {
			return fmt.Errorf("loadObservation: invalid option unknown")
		}
	}
	if o.EstimatorRevision != "" && o.EstimatorRevision != "pair-gguf/2;gguf-parser-go/v0.27.0@9257743a3c1775b882ba1b58ec7f5a856c2cc385" {
		return fmt.Errorf("loadObservation: unsupported estimator revision")
	}
	for _, a := range o.Artifacts {
		if !slices.Contains([]string{"model", "draft", "companion"}, a.Role) || !digest(a.Identity) || ValidateUnsignedDecimal(a.Size) != nil || a.Assurance != "sampled-local-binding" {
			return fmt.Errorf("loadObservation: invalid artifact reference")
		}
	}
	baselines := map[string]BufferBaseline{}
	for _, b := range o.Baseline {
		_, duplicate := baselines[b.Component]
		if duplicate || !slices.Contains(MemoryComponentNames, b.Component) || !slices.Contains([]string{"estimated", "partial", "unknown", "excluded"}, b.Status) || b.Bytes != "" && ValidateUnsignedDecimal(b.Bytes) != nil {
			return fmt.Errorf("loadObservation: invalid frozen component")
		}
		baselines[b.Component] = b
	}
	seen := map[string]bool{}
	for _, c := range o.Comparisons {
		if seen[c.Component] || !slices.Contains(MemoryComponentNames, c.Component) || !slices.Contains([]string{"compatible", "unknown"}, c.Status) {
			return fmt.Errorf("loadObservation: invalid component comparison")
		}
		seen[c.Component] = true
		for _, n := range []string{c.EstimatedBytes, c.LowerBytes, c.UpperBytes} {
			if n != "" && ValidateUnsignedDecimal(n) != nil {
				return fmt.Errorf("loadObservation: invalid comparison bytes")
			}
		}
		for _, n := range []string{c.DeltaLowerBytes, c.DeltaUpperBytes} {
			if n != "" {
				if err := ValidateUnsignedDecimal(strings.TrimPrefix(n, "-")); err != nil || n == "-0" {
					return fmt.Errorf("loadObservation: invalid signed interval")
				}
			}
		}
		if c.Status == "unknown" && (c.DeltaLowerBytes != "" || c.DeltaUpperBytes != "" || c.LowerBytes != "" || c.UpperBytes != "") {
			return fmt.Errorf("loadObservation: unknown comparison has a delta")
		}
		if c.Status == "compatible" {
			b := baselines[c.Component]
			lo, loOK := new(big.Int).SetString(c.LowerBytes, 10)
			hi, hiOK := new(big.Int).SetString(c.UpperBytes, 10)
			estimate, estimateOK := new(big.Int).SetString(c.EstimatedBytes, 10)
			if b.Status != "estimated" || b.Bytes != c.EstimatedBytes || !loOK || !hiOK || !estimateOK || lo.Cmp(hi) > 0 || new(big.Int).Sub(lo, estimate).String() != c.DeltaLowerBytes || new(big.Int).Sub(hi, estimate).String() != c.DeltaUpperBytes {
				return fmt.Errorf("loadObservation: incompatible baseline or interval reference")
			}
		}
	}
	if o.RecordedAt != "" {
		if _, err := time.Parse(time.RFC3339Nano, o.RecordedAt); err != nil {
			return fmt.Errorf("loadObservation: invalid timestamp")
		}
	}
	if o.UpstreamPin != "" && o.UpstreamPin != "7fe450e19305b828c199d602c23a8337aaa1f03b" {
		return fmt.Errorf("loadObservation: unsupported pin")
	}
	if o.BuildAssurance != "" && o.BuildAssurance != "recognized-self-reported-pin" {
		return fmt.Errorf("loadObservation: unsupported provenance assurance")
	}
	if o.Current && (o.Status == "unknown" || o.SampleID == "") {
		return fmt.Errorf("loadObservation: unknown current sample")
	}
	for _, reason := range append(slices.Clone(o.Reasons), o.Reason) {
		if !slices.Contains([]string{"", "observation-omitted", "response-budget", "history-unavailable", "no-matching-sample", "missing-boundary", "unsupported-build", "clock-reversal", "unmapped-device", "mixed-role", "incomplete-reserve", "oversized-frame", "truncated-frame", "invalid-utf8", "read-failure", "capture-budget", "unknown-baseline", "source-changed", "source-unavailable", "unsupported-buffer"}, reason) {
			return fmt.Errorf("loadObservation: invalid reason")
		}
	}
	frames := map[string]bool{}
	for _, row := range o.Rows {
		if !slices.Contains([]string{"model", "kv", "compute", "output", "recurrent"}, row.Component) || !validBufferLabel(row.Label) || !slices.Contains([]string{"load", "reserve"}, row.Phase) || !slices.Contains([]string{"model", "mixed", "unattributed"}, row.Role) || !slices.Contains([]string{"partial", "unknown", "compatible"}, row.Coverage) {
			return fmt.Errorf("loadObservation.rows: unsupported category")
		}
		if row.Stream != "" && row.Stream != "stdout" && row.Stream != "stderr" || row.Unit != "MiB" || !slices.Contains([]string{"0.01", "unknown"}, row.Precision) || row.SourceKind != "log-scraped" || ValidateUnsignedDecimal(row.Sequence) != nil || row.Sequence == "0" {
			return fmt.Errorf("loadObservation.rows: invalid provenance")
		}
		frame := row.Stream + ":" + row.Sequence
		if frames[frame] {
			return fmt.Errorf("loadObservation.rows: duplicate source frame")
		}
		frames[frame] = true
		low, high, err := BufferReportInterval(row.ReportedMiB)
		if row.IntervalStatus == "unknown" {
			if !canonicalBufferNumeric(row.ReportedMiB) || row.LowerBytes != "" || row.UpperBytes != "" || row.Coverage == "compatible" {
				return fmt.Errorf("loadObservation.rows: invalid unsupported numeric report")
			}
		} else if row.IntervalStatus != "bounded" || row.Precision != "0.01" || err != nil || row.LowerBytes != low || row.UpperBytes != high {
			return fmt.Errorf("loadObservation.rows: invalid rounded interval")
		}
		if row.EstimatedBytes != "" && ValidateUnsignedDecimal(row.EstimatedBytes) != nil {
			return fmt.Errorf("loadObservation.rows: invalid estimate")
		}
		for _, value := range []string{row.DeviceBinding, row.PoolIdentity} {
			if value != "" && !digest(value) {
				return fmt.Errorf("loadObservation.rows: invalid binding")
			}
		}
	}
	encoded, err := json.Marshal(o)
	if err != nil || len(encoded) > 64<<10 {
		return fmt.Errorf("loadObservation: sample exceeds byte budget")
	}
	return nil
}
