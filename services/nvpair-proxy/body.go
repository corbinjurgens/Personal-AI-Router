// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// Request-body edits: rewriting a tier request's model to the concrete one,
// and merging a profile's requestOptions into a request.
//
// Both splice bytes rather than decode and re-encode the body. A round trip
// through a Go map reorders keys and rewrites numbers and escapes, and an
// engine is entitled to see the request its client sent; only the member being
// changed is touched, and every other byte goes through as it arrived.

import (
	"bytes"
	"encoding/json"
	"io"
	"slices"
)

// jsonMember is one top-level member of a JSON object, located by the byte
// offsets of its value.
type jsonMember struct {
	key        string
	valueStart int
	valueEnd   int
}

// objectMembers parses data as one JSON object and returns its members in
// order, plus the offset of its closing brace. ok is false when data is not
// exactly one object.
func objectMembers(data []byte) (members []jsonMember, closeBrace int, ok bool) {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	tok, err := dec.Token()
	if err != nil {
		return nil, 0, false
	}
	if d, isDelim := tok.(json.Delim); !isDelim || d != '{' {
		return nil, 0, false
	}
	for dec.More() {
		keyTok, err := dec.Token()
		if err != nil {
			return nil, 0, false
		}
		key, isString := keyTok.(string)
		if !isString {
			return nil, 0, false
		}
		var raw json.RawMessage
		if err := dec.Decode(&raw); err != nil {
			return nil, 0, false
		}
		end := int(dec.InputOffset())
		members = append(members, jsonMember{key: key, valueStart: end - len(raw), valueEnd: end})
	}
	tok, err = dec.Token()
	if err != nil {
		return nil, 0, false
	}
	if d, isDelim := tok.(json.Delim); !isDelim || d != '}' {
		return nil, 0, false
	}
	closeBrace = int(dec.InputOffset()) - 1
	// Nothing but whitespace may follow the object.
	if _, err := dec.Token(); err != io.EOF {
		return nil, 0, false
	}
	if closeBrace < 0 || closeBrace >= len(data) || data[closeBrace] != '}' {
		return nil, 0, false
	}
	return members, closeBrace, true
}

// replaceModel returns body with its top-level "model" replaced by model, and
// false when body is not an object with a model member. Every other byte is
// preserved.
func replaceModel(body []byte, model string) ([]byte, bool) {
	members, _, ok := objectMembers(body)
	if !ok {
		return nil, false
	}
	value, err := json.Marshal(model)
	if err != nil {
		return nil, false
	}
	for _, m := range members {
		if m.key != "model" {
			continue
		}
		out := make([]byte, 0, len(body)-(m.valueEnd-m.valueStart)+len(value))
		out = append(out, body[:m.valueStart]...)
		out = append(out, value...)
		out = append(out, body[m.valueEnd:]...)
		return out, true
	}
	return nil, false
}

// mergeDefaults merges a profile's requestOptions object into body. Keys
// already in the request win; where both sides hold an object the merge
// recurses, so a profile's {"options":{"num_ctx":16384}} still applies to a
// request that sets only options.temperature. Missing members are appended
// before the object's closing brace and everything else is preserved. A body
// or defaults that is not a JSON object leaves body unchanged.
func mergeDefaults(body, defaults []byte) []byte {
	if len(bytes.TrimSpace(defaults)) == 0 {
		return body
	}
	merged, ok := mergeObject(body, defaults)
	if !ok {
		return body
	}
	return merged
}

func mergeObject(body, defaults []byte) ([]byte, bool) {
	members, closeBrace, ok := objectMembers(body)
	if !ok {
		return nil, false
	}
	defMembers, _, ok := objectMembers(defaults)
	if !ok {
		return nil, false
	}
	byKey := make(map[string]jsonMember, len(members))
	for _, m := range members {
		// A duplicate key resolves to the last occurrence, as encoding/json
		// and most engines read it.
		byKey[m.key] = m
	}
	type splice struct {
		start, end int
		with       []byte
	}
	var splices []splice
	var appended [][]byte
	for _, d := range defMembers {
		defValue := defaults[d.valueStart:d.valueEnd]
		existing, present := byKey[d.key]
		if !present {
			key, err := json.Marshal(d.key)
			if err != nil {
				return nil, false
			}
			appended = append(appended, slices.Concat(key, []byte(":"), defValue))
			continue
		}
		reqValue := body[existing.valueStart:existing.valueEnd]
		if isJSONObject(reqValue) && isJSONObject(defValue) {
			inner, ok := mergeObject(reqValue, defValue)
			if !ok {
				continue
			}
			splices = append(splices, splice{existing.valueStart, existing.valueEnd, inner})
		}
	}
	if len(splices) == 0 && len(appended) == 0 {
		return body, true
	}
	var tail []byte
	if len(appended) > 0 {
		if len(members) > 0 {
			tail = append(tail, ',')
		}
		tail = append(tail, bytes.Join(appended, []byte(","))...)
	}
	out := make([]byte, 0, len(body)+len(tail)+64)
	pos := 0
	// Splices are in member order, so offsets only grow.
	slices.SortFunc(splices, func(a, b splice) int { return a.start - b.start })
	for _, s := range splices {
		out = append(out, body[pos:s.start]...)
		out = append(out, s.with...)
		pos = s.end
	}
	out = append(out, body[pos:closeBrace]...)
	out = append(out, tail...)
	out = append(out, body[closeBrace:]...)
	return out, true
}

func isJSONObject(raw []byte) bool {
	t := bytes.TrimSpace(raw)
	return len(t) > 0 && t[0] == '{'
}
