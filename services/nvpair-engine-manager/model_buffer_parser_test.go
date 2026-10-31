// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"errors"
	"fmt"
	"strings"
	"testing"
)

func TestModelBufferPinnedReports(t *testing.T) {
	t.Run("finite emitters and log envelopes", func(t *testing.T) {
		for _, envelope := range []func(string) string{
			func(s string) string { return s },
			func(s string) string { return "I " + s },
			func(s string) string { return "12.34.567.890 I " + s },
			func(s string) string { return "\x1b[34m12.34.567.890\x1b[0m \x1b[32mI \x1b[0m" + s + "\x1b[0m" },
			func(s string) string { return fmt.Sprintf(`{"type":"log","time":12,"level":"info","msg":%q}`, s+"\n") },
		} {
			p := &modelBufferParser{}
			for i, message := range []string{
				"load_tensors: CPU_Mapped model buffer size = 1.00 MiB",
				"llama_kv_cache: CUDA0 KV buffer size = 2.00 MiB",
				"llama_kv_cache: CUDA0 KV buffer size = 2.00 MiB",
				"llama_memory_recurrent: CUDA_Host RS buffer size = 3.00 MiB",
				"llama_context: MTL0_Mapped output buffer size = 4.00 MiB",
				"sched_reserve: reserving ...",
				"sched_reserve: MTL63_Private compute buffer size = 5.00 MiB",
				"sched_reserve: reserve took 1.00 ms, sched copies = 1",
			} {
				p.consume(managedOutputFrame{Stream: "stdout", Sequence: uint64(i + 1), Line: envelope(message)})
			}
			if p.partial || !p.reserveComplete || len(p.rows) != 6 || p.rows[3].Component != "recurrent" || p.rows[4].Component != "output" {
				t.Fatalf("emitter/envelope categories: %+v", p)
			}
		}
	})
	// Native emitters: llama.cpp 0ec4027a2a0183de87c007069adddc06e665f7de,
	// src/llama-model.cpp:1849 and src/llama-kv-cache.cpp:274.
	p := &modelBufferParser{}
	for i, line := range []string{
		"load_tensors:          CPU model buffer size =     1.00 MiB",
		"load_tensors:          CPU model buffer size =     1.00 MiB",
		`{"type":"log","time":0,"level":"info","msg":"llama_kv_cache: CUDA0 KV buffer size = 0.00 MiB\n"}`,
	} {
		p.consume(managedOutputFrame{Stream: "stderr", Sequence: uint64(i + 1), Line: line})
	}
	if p.partial || len(p.rows) != 3 {
		t.Fatalf("capture: %+v", p)
	}
	if p.rows[0].Sequence == p.rows[1].Sequence {
		t.Fatal("identical allocation occurrences collapsed")
	}
	if p.rows[2].UpperBytes != "5243" || p.rows[2].SourceKind != "log-scraped" {
		t.Fatalf("JSONL zero report: %+v", p.rows[2])
	}
	p.consume(managedOutputFrame{Stream: "stderr", Sequence: 3, Line: "duplicate delivery"})
	if len(p.rows) != 3 {
		t.Fatal("duplicate delivery inflated allocation")
	}
}

func TestModelBufferUnsupportedBuild(t *testing.T) {
	for _, build := range []string{"", "b11147-0ec4027a2", "b12020-0ec4027a2-dirty", "0ec4027a2", "b12020-7fe450"} {
		if modelBufferBuildSupported(build) {
			t.Fatalf("accepted unsupported tuple %q", build)
		}
	}
	if !modelBufferBuildSupported("b12020-0ec4027a2") {
		t.Fatal("explicit mapped tuple unsupported")
	}
	// A modified binary can self-report the same tuple. This predicate proves
	// recognition only; it must never be represented as clean-source attestation.
}

func TestModelBufferUnmappedRotationIsPartial(t *testing.T) {
	p := &modelBufferParser{}
	p.consume(managedOutputFrame{Stream: "stderr", Sequence: 1, Line: "llama_kv_cache: CUDA0 rotation buffer size = 1.00 MiB"})
	if !p.partial || len(p.rows) != 0 {
		t.Fatalf("unmapped rotation must remain partial: %+v", p)
	}
}

type modelBufferFailedReader struct{ read bool }

func (r *modelBufferFailedReader) Read(p []byte) (int, error) {
	if r.read {
		return 0, errors.New("injected read failure")
	}
	r.read = true
	return copy(p, "load_tensors: CPU model buffer size = 1.00 MiB\n"), nil
}

func TestModelBufferMalformedFrames(t *testing.T) {
	t.Run("unsupported numeric precision preserves an unknown interval", func(t *testing.T) {
		for _, value := range []string{"1073741824.00", "184467440737095516160.00", "1.234"} {
			p := &modelBufferParser{}
			p.consume(managedOutputFrame{Stream: "stdout", Sequence: 1, Line: "load_tensors: CPU model buffer size = " + value + " MiB"})
			if !p.partial || len(p.rows) != 1 || p.rows[0].ReportedMiB != value || p.rows[0].IntervalStatus != "unknown" || p.rows[0].LowerBytes != "" {
				t.Fatalf("unsupported report fabricated interval: %+v", p)
			}
		}
	})
	t.Run("excluded families and malformed envelopes", func(t *testing.T) {
		for _, line := range []string{
			"load_tensors: CPU model buffer size = -1.00 MiB",
			"load_tensors: CPU model buffer size = Inf MiB",
			"load_tensors: CPU model buffer size = 1.00 MB",
			"load_tensors: Vulkan0 model buffer size = 1.00 MiB",
			"llama_kv_cache: MTL64 KV buffer size = 1.00 MiB",
			"llama_kv_cache: CPU model buffer size = 1.00 MiB",
			"llama_kv_cache: size = 1024.00 MiB (1024 cells, 1 layers)",
			"| model | CPU | 1024 |",
			`{"type":"fit_memory_breakdown","model":1024}`,
			`{"type":"log","time":0,"level":"info","msg":"load_tensors: CPU model buffer size = 1.00 MiB","msg":"conflict"}`,
			`{"type":"log","time":0,"level":"info","msg":"load_tensors: CPU model buffer size = 1.00 MiB","secret":"extra"}`,
			`{"type":"log","time":1.5,"level":"info","msg":"load_tensors: CPU model buffer size = 1.00 MiB"}`,
		} {
			p := &modelBufferParser{}
			p.consume(managedOutputFrame{Stream: "stdout", Sequence: 1, Line: line})
			if len(p.rows) != 0 {
				t.Fatalf("excluded diagnostic became report: %q", line)
			}
		}
	})
	for _, tc := range []struct{ name, input, failure string }{
		{"oversized", strings.Repeat("x", 70<<10) + "\nvalid\n", "oversized-frame"},
		{"truncated", "not terminated", "truncated-frame"},
		{"invalid UTF8", string([]byte{0xff, '\n'}), "invalid-utf8"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var frames []managedOutputFrame
			scanObservedLines(strings.NewReader(tc.input), "stdout", nil, func(f managedOutputFrame) { frames = append(frames, f) })
			if len(frames) == 0 || frames[0].Failure != tc.failure {
				t.Fatalf("frames: %+v", frames)
			}
			if tc.name == "oversized" && (len(frames) != 2 || frames[1].Line != "valid") {
				t.Fatalf("failed to drain past oversized frame: %+v", frames)
			}
		})
	}
	var failures []string
	scanObservedLines(&modelBufferFailedReader{}, "stderr", nil, func(f managedOutputFrame) {
		if f.Failure != "" {
			failures = append(failures, f.Failure)
		}
	})
	if len(failures) != 1 || failures[0] != "read-failure" {
		t.Fatalf("read failure: %v", failures)
	}
	for _, line := range []string{"prefix load_tensors: CPU model buffer size = 1.00 MiB", "load_tensors: Metal0 model buffer size = 1.00 MiB", "load_tensors: CUDA64 model buffer size = 1.00 MiB", "load_tensors: CPU model buffer size = NaN MiB", `{"type":"fit_memory_breakdown","msg":"load_tensors: CPU model buffer size = 1.00 MiB"}`} {
		p := &modelBufferParser{}
		p.consume(managedOutputFrame{Stream: "stdout", Sequence: 1, Line: line})
		if len(p.rows) != 0 {
			t.Fatalf("accepted diagnostic %q", line)
		}
	}
}

func TestModelBufferRoleAndPhaseCoverage(t *testing.T) {
	t.Run("a full reservation can replace a full reservation", func(t *testing.T) {
		p := &modelBufferParser{}
		var sequence uint64
		line := func(message string) {
			sequence++
			p.consume(managedOutputFrame{Stream: "stdout", Sequence: sequence, Line: message})
		}
		for _, value := range []string{"1.00", "2.00"} {
			line("sched_reserve: reserving ...")
			for range 256 {
				line("sched_reserve: CPU compute buffer size = " + value + " MiB")
			}
			line("sched_reserve: reserve took 1.00 ms, sched copies = 1")
		}
		if p.partial || len(p.rows) != 256 || p.rows[0].ReportedMiB != "2.00" {
			t.Fatal("replacement reservations were charged or added across phases")
		}
		line("load_tensors: CPU model buffer size = 1.00 MiB")
		if !p.partial || len(p.rows) != 256 {
			t.Fatal("row budget did not bound capture")
		}
	})
	t.Run("missing and conflicting stream frames cannot inflate reports", func(t *testing.T) {
		p := &modelBufferParser{}
		frame := managedOutputFrame{Stream: "stdout", Sequence: 1, Line: "load_tensors: CPU model buffer size = 1.00 MiB"}
		p.consume(frame)
		p.consume(frame)
		if len(p.rows) != 1 || p.partial {
			t.Fatal("exact frame replay changed report")
		}
		frame.Line = "load_tensors: CPU model buffer size = 2.00 MiB"
		p.consume(frame)
		if len(p.rows) != 1 || !p.partial {
			t.Fatal("conflicting replay accepted")
		}
	})
	p := &modelBufferParser{}
	lines := []string{"sched_reserve: reserving ...", "sched_reserve: CPU compute buffer size = 1.00 MiB", "sched_reserve: reserve took 1.00 ms, sched copies = 1", "sched_reserve: reserving ...", "sched_reserve: CPU compute buffer size = 2.00 MiB", "sched_reserve: reserve took 2.00 ms, sched copies = 1"}
	for i, line := range lines {
		p.consume(managedOutputFrame{Stream: "stderr", Sequence: uint64(i + 1), Line: line})
	}
	if p.partial || len(p.rows) != 1 || p.rows[0].ReportedMiB != "2.00" || p.rows[0].Role != "unattributed" {
		t.Fatalf("successive reservations: %+v", p)
	}
	p.consume(managedOutputFrame{Stream: "stderr", Sequence: 7, Line: "sched_reserve: CPU compute buffer size = 3.00 MiB"})
	if !p.partial || len(p.rows) != 1 {
		t.Fatalf("unbounded reserve accepted: %+v", p)
	}
}
