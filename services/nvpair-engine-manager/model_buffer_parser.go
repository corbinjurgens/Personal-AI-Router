// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"unicode/utf8"

	"nvpair-shared/modelconfig"
)

const modelBufferPin = "0ec4027a2a0183de87c007069adddc06e665f7de"

// Exact emitter forms from the pinned source, never arbitrary substrings.
// Repeated identical reports are distinct allocation occurrences.
var modelBufferEmitter = regexp.MustCompile(`^(load_tensors|llama_kv_cache|sched_reserve|llama_context|llama_memory_recurrent): +([A-Za-z0-9_]+) +(model|KV|compute|output|RS) buffer size = +([0-9]+\.[0-9]+) MiB$`)
var modelBufferCandidate = regexp.MustCompile(`^(load_tensors|llama_kv_cache|sched_reserve|llama_context|llama_memory_recurrent): +[^ ]+ +(model|KV|rotation|compute|output|RS) buffer size`)
var modelBufferPort = regexp.MustCompile(`^\[ *([0-9]{1,5})\] (.*)$`)
var modelBufferLogPrefix = regexp.MustCompile(`^(?:(?:\x1b\[34m)?[0-9]+\.[0-9]{2}\.[0-9]{3}\.[0-9]{3}(?:\x1b\[0m)? )?(?:(?:\x1b\[32m)?I (?:\x1b\[0m)?|(?:\x1b\[35m)?W |(?:\x1b\[31m)?E |(?:\x1b\[33m)?D )(.*?)(?:\x1b\[0m)?$`)
var modelBufferReserveEnd = regexp.MustCompile(`^sched_reserve: reserve took [0-9]+\.[0-9]{2} ms, sched copies = [0-9]+$`)

type modelBufferParser struct {
	rows            []modelconfig.BufferReportRow
	reserve         []modelconfig.BufferReportRow
	inReserve       bool
	partial         bool
	reasons         []string
	sequence        [2]uint64
	lastHash        [2]string
	reserveComplete bool
}

func modelBufferBuildSupported(build string) bool {
	return build == "b12020-0ec4027a2" || build == "b12020-"+modelBufferPin
}

// JSONL is an envelope around a diagnostic message, not an exact-byte API.
func modelBufferMessage(line string) (string, bool) {
	if len(line) > 64<<10 || !utf8.ValidString(line) {
		return "", false
	}
	if strings.HasPrefix(line, "{") {
		var envelope struct {
			Type    string          `json:"type"`
			Message string          `json:"msg"`
			Level   string          `json:"level,omitempty"`
			Time    json.RawMessage `json:"time,omitempty"`
		}
		if modelconfig.DecodeStrict([]byte(line), &envelope) != nil || envelope.Type != "log" || envelope.Message == "" {
			return "", false
		}
		var timestamp int64
		if len(envelope.Time) == 0 || json.Unmarshal(envelope.Time, &timestamp) != nil || timestamp < 0 || string(envelope.Time) != strconv.FormatInt(timestamp, 10) || !slices.Contains([]string{"info", "warn", "error", "debug", "cont", "none"}, envelope.Level) {
			return "", false
		}
		line = strings.TrimSuffix(envelope.Message, "\n")
	}
	if prefixed := modelBufferLogPrefix.FindStringSubmatch(line); prefixed != nil {
		line = prefixed[1]
	}
	return line, true
}

func (p *modelBufferParser) consume(frame managedOutputFrame, sourceSequence ...uint64) {
	stream := 0
	if frame.Stream == "stderr" {
		stream = 1
	} else if frame.Stream != "stdout" {
		p.partial = true
		return
	}
	hash := modelBufferHash(struct{ Line, Failure string }{frame.Line, frame.Failure})
	if frame.Sequence == p.sequence[stream] && hash == p.lastHash[stream] {
		return
	}
	if frame.Sequence <= p.sequence[stream] {
		p.partial = true
		return
	}
	if frame.Sequence != p.sequence[stream]+1 || frame.Failure != "" {
		p.partial = true
	}
	p.sequence[stream] = frame.Sequence
	p.lastHash[stream] = hash
	if frame.Failure != "" {
		return
	}
	line, ok := modelBufferMessage(frame.Line)
	if !ok {
		p.partial = true
		return
	}
	// Only the capture demultiplexer removes parent child-port prefixes.
	if modelBufferPort.MatchString(line) {
		p.partial = true
		return
	}
	if line == "sched_reserve: reserving ..." {
		if p.inReserve {
			p.partial = true
		}
		p.inReserve, p.reserve = true, nil
		p.reserveComplete = false
		return
	}
	if modelBufferReserveEnd.MatchString(line) {
		if !p.inReserve {
			p.partial = true
			return
		}
		kept := make([]modelconfig.BufferReportRow, 0, len(p.rows))
		for _, row := range p.rows {
			if row.Component != "compute" {
				kept = append(kept, row)
			}
		}
		p.rows = append(kept, p.reserve...)
		p.reserve, p.inReserve = nil, false
		p.reserveComplete = true
		return
	}
	match := modelBufferEmitter.FindStringSubmatch(line)
	if match == nil {
		if modelBufferCandidate.MatchString(line) {
			p.partial = true
		}
		return
	}
	component := ""
	switch match[1] + ":" + match[3] {
	case "load_tensors:model":
		component = "model"
	case "llama_kv_cache:KV":
		component = "kv"
	case "sched_reserve:compute":
		component = "compute"
	case "llama_context:output":
		component = "output"
	case "llama_memory_recurrent:RS":
		component = "recurrent"
	default:
		p.partial = true
		return
	}
	low, high, err := modelconfig.BufferReportInterval(match[4])
	precision, intervalStatus := "0.01", "bounded"
	if err != nil {
		p.partial = true
		intervalStatus = "unknown"
		if parts := strings.Split(match[4], "."); len(parts) != 2 || len(parts[1]) != 2 {
			precision = "unknown"
		}
	}
	phase := "load"
	if component == "compute" {
		if !p.inReserve {
			p.partial = true
			return
		}
		phase = "reserve"
	}
	row := modelconfig.BufferReportRow{Component: component, Label: strings.Clone(match[2]), Phase: phase, Sequence: strconv.FormatUint(frame.Sequence, 10), Stream: frame.Stream, Role: "unattributed", ReportedMiB: strings.Clone(match[4]), Unit: "MiB", Precision: precision, SourceKind: "log-scraped", IntervalStatus: intervalStatus, LowerBytes: low, UpperBytes: high, Coverage: "unknown"}
	if len(sourceSequence) > 0 {
		row.Sequence = strconv.FormatUint(sourceSequence[0], 10)
	}
	if (modelconfig.BufferObservation{Status: "partial", Rows: []modelconfig.BufferReportRow{row}}).Validate() != nil {
		p.partial = true
		if !slices.Contains(p.reasons, "unsupported-buffer") {
			p.reasons = append(p.reasons, "unsupported-buffer")
		}
		return
	}
	// A replacement reservation displaces the previous compute block. Its
	// budget must not accidentally add old and new reservation occurrences.
	retained := len(p.rows) + len(p.reserve)
	if p.inReserve {
		for _, old := range p.rows {
			if old.Component == "compute" {
				retained--
			}
		}
	}
	if retained >= 256 {
		p.partial = true
		return
	}
	if component == "compute" {
		p.reserve = append(p.reserve, row)
	} else {
		p.rows = append(p.rows, row)
	}
}
