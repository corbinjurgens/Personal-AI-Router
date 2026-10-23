// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"strings"
	"sync"
	"time"
	"unicode"
	"unicode/utf8"
)

// redactServingLine removes path-bearing tokens before native serving output
// enters either the exported ring or startup diagnostics. The configured models
// directory matches case-insensitively, as Windows paths do. A quoted span
// holding a path is removed whole, so a quoted path with spaces leaves no
// fragment.
func redactServingLine(line, modelsDir string) string {
	if modelsDir != "" {
		line = replaceFold(line, modelsDir, "[redacted]")
	}
	tokens := strings.Fields(redactQuotedPaths(line))
	for i, token := range tokens {
		if pathBearing(token) {
			tokens[i] = "[redacted]"
		}
	}
	return strings.Join(tokens, " ")
}

// redactQuotedPaths treats every double quote as opening a span. A span runs to
// the last matching quote in the line, or to the end when none follows, so
// embedded or escaped quotes cannot end a path early.
func redactQuotedPaths(line string) string {
	lastQuote := map[byte]int{'"': strings.LastIndexByte(line, '"'), '\'': strings.LastIndexByte(line, '\'')}
	// A closed span found path-free covers every later quote of its type,
	// including its own closing quote, so none of them opens a span; checking
	// them again would also make a quote-heavy line quadratic.
	pathFree := map[byte]bool{}
	var b strings.Builder
	last := 0
	for i := 0; i < len(line); i++ {
		quote := line[i]
		if quote != '"' && quote != '\'' {
			continue
		}
		if quote == '\'' && !apostropheOpens(line, i) {
			continue
		}
		if pathFree[quote] {
			continue
		}
		closed := i < lastQuote[quote]
		end := len(line)
		if closed {
			end = lastQuote[quote] + 1
		}
		if !pathBearing(line[i:end]) {
			pathFree[quote] = closed
			continue
		}
		b.WriteString(line[last:i])
		b.WriteString("[redacted]")
		last = end
		i = end - 1
	}
	b.WriteString(line[last:])
	return b.String()
}

// apostropheOpens keeps contractions ordinary words: a single quote after a
// letter or digit opens a span only when a path separator follows it.
func apostropheOpens(line string, i int) bool {
	if r, _ := utf8.DecodeLastRuneInString(line[:i]); i == 0 || !(unicode.IsLetter(r) || unicode.IsNumber(r)) {
		return true
	}
	return i+1 < len(line) && (line[i+1] == '/' || line[i+1] == '\\')
}

func replaceFold(s, old, replacement string) string {
	var b strings.Builder
	for i := 0; i < len(s); {
		if i+len(old) <= len(s) && strings.EqualFold(s[i:i+len(old)], old) {
			b.WriteString(replacement)
			i += len(old)
			continue
		}
		b.WriteByte(s[i])
		i++
	}
	return b.String()
}

func pathBearing(text string) bool {
	return strings.ContainsAny(text, `/\`) || strings.Contains(text, "[redacted]")
}

// maxEngineLogLines bounds each engine's captured-output ring.
const maxEngineLogLines = 2000

// LogLine is one captured stdout/stderr line from a managed engine.
type LogLine struct {
	Time   string `json:"time"`
	Stream string `json:"stream"` // "stdout" | "stderr"
	Text   string `json:"text"`
}

// logBuffer is a per-engine bounded ring of recent output lines.
type logBuffer struct {
	mu    sync.Mutex
	lines []LogLine
}

func newLogBuffer() *logBuffer {
	return &logBuffer{lines: make([]LogLine, 0, 256)}
}

func (b *logBuffer) append(stream, text string) {
	line := LogLine{Time: time.Now().Format("15:04:05.000"), Stream: stream, Text: text}
	b.mu.Lock()
	if len(b.lines) >= maxEngineLogLines {
		// Copy the surviving half into a fresh slice so the dropped
		// portion's backing array can be reclaimed.
		keep := b.lines[len(b.lines)-maxEngineLogLines/2:]
		b.lines = append(make([]LogLine, 0, maxEngineLogLines), keep...)
	}
	b.lines = append(b.lines, line)
	b.mu.Unlock()
}

func (b *logBuffer) snapshot() []LogLine {
	b.mu.Lock()
	defer b.mu.Unlock()
	out := make([]LogLine, len(b.lines))
	copy(out, b.lines)
	return out
}
