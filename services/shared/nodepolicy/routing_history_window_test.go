// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nodepolicy

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"nvpair-shared/jsonrpc"
)

func TestHistoryWindowGeneration(t *testing.T) {
	for _, value := range []string{"1", "18446744073709551616", "340282366920938463463374607431768211455"} {
		if err := ValidateRoutingHistoryGeneration(value); err != nil {
			t.Fatalf("legal generation rejected: %v", err)
		}
	}
	for _, value := range []string{"", "0", "01", "+1", "-1", "1.0", "1e1", " 1", "340282366920938463463374607431768211456", strings.Repeat("9", 40)} {
		if err := ValidateRoutingHistoryGeneration(value); err == nil || err.Error() != "invalid-value" {
			t.Fatalf("generation error=%v", err)
		}
	}
	g := RoutingHistoryGeneration{Low: ^uint64(0)}
	value, err := g.Next()
	if err != nil || value != "18446744073709551616" || g.High != 1 || g.Low != 0 {
		t.Fatalf("carry=%+v %q %v", g, value, err)
	}
	g = RoutingHistoryGeneration{High: ^uint64(0), Low: ^uint64(0) - 1}
	value, err = g.Next()
	if err != nil || value != "340282366920938463463374607431768211455" {
		t.Fatalf("maximum=%q err=%v", value, err)
	}
	before := g
	if _, err := g.Next(); err == nil || g != before {
		t.Fatal("overflow changed generation")
	}
}

func TestHistoryWindowPrivateShapes(t *testing.T) {
	valid := `{"schema":1,"generation":"1","cutoff":null,"keys":[{"request":{"originNodeId":"origin","proxyRunId":"run","requestId":"1"},"createdAt":0,"seq":1}]}`
	var window RoutingHistoryWindow
	if err := json.Unmarshal([]byte(valid), &window); err != nil {
		t.Fatal(err)
	}
	if window.Generation != "1" || len(window.Keys) != 1 || window.Cutoff != nil || window.Keys[0].Seq != 1 {
		t.Fatalf("window=%+v", window)
	}
	test := func(name, raw string) {
		t.Run(name, func(t *testing.T) {
			before := window
			if err := json.Unmarshal([]byte(raw), &window); err == nil || !reflect.DeepEqual(window, before) {
				t.Fatalf("invalid private value changed destination: %v", err)
			}
		})
	}
	test("duplicate generation", strings.Replace(valid, `"generation":"1"`, `"generation":"1","\u0067eneration":"2"`, 1))
	test("null generation", strings.Replace(valid, `"generation":"1"`, `"generation":null`, 1))
	test("missing cutoff", strings.Replace(valid, `"cutoff":null,`, "", 1))
	test("null keys", `{"schema":1,"generation":"1","cutoff":null,"keys":null}`)
	test("invalid seq", strings.Replace(valid, `"seq":1`, `"seq":0`, 1))
	test("wrong schema", strings.Replace(valid, `"schema":1`, `"schema":2`, 1))
	test("unknown duplicate subtree", strings.Replace(valid, `"schema":1`, `"schema":1,"unknown":{"secret":0,"\u0073ecret":1}`, 1))
	unknown := strings.Replace(valid, `"schema":1`, `"schema":1,"unknown":{"synthetic":"discard"}`, 1)
	if err := json.Unmarshal([]byte(unknown), &window); err != nil {
		t.Fatal(err)
	}
	for _, method := range []string{NotifyRoutingHistoryWindowEnd, NotifyRoutingHistoryAccepted} {
		schema, ok := RoutingHistoryTransportSchema(method, false)
		if !ok {
			t.Fatalf("method missing: %s", method)
		}
		out, _, err := jsonrpc.ProjectHistory([]byte(`{"schema":1,"generation":"1"}`), make([]byte, 0, 128), make([]uint32, 128), schema)
		if err != nil || string(out) != `{"schema":1,"generation":"1"}` {
			t.Fatalf("end/accepted projection=%s %v", out, err)
		}
		if _, _, err := jsonrpc.ProjectHistory([]byte(`{"schema":1,"generation":"0"}`), make([]byte, 0, 128), make([]uint32, 128), schema); err == nil {
			t.Fatal("invalid end/accepted generation accepted")
		}
	}
}

func TestHistoryWindowManifestOrderAndCutoff(t *testing.T) {
	window := RoutingHistoryWindow{Schema: 1, Generation: "340282366920938463463374607431768211455", Keys: make([]RoutingHistoryWindowKey, RoutingHistoryRecords)}
	for i := range window.Keys {
		window.Keys[i] = RoutingHistoryWindowKey{Request: RoutingRequestKey{OriginNodeID: strings.Repeat("o", 128), ProxyRunID: strings.Repeat("r", 128), RequestID: fmt.Sprintf("9223372036854775%03d", i)}, CreatedAt: routingHistorySafeInteger - int64(i), Seq: routingHistorySafeInteger}
	}
	least := window.Keys[255]
	window.Cutoff = &RoutingHistoryCutoff{Request: least.Request, CreatedAt: least.CreatedAt}
	if err := window.Validate(); err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(window)
	if err != nil {
		t.Fatal(err)
	}
	if len(encoded) > 131072 {
		t.Fatalf("maximum manifest length=%d", len(encoded))
	}
	var decoded RoutingHistoryWindow
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(decoded, window) {
		t.Fatal("manifest roundtrip changed")
	}
	window.Cutoff = nil
	if err := window.Validate(); err == nil {
		t.Fatal("full window missing cutoff accepted")
	}
	window.Cutoff = &RoutingHistoryCutoff{Request: least.Request, CreatedAt: least.CreatedAt + 1}
	if err := window.Validate(); err == nil {
		t.Fatal("wrong cutoff accepted")
	}
	window.Keys = window.Keys[:2]
	window.Cutoff = nil
	window.Keys[0], window.Keys[1] = window.Keys[1], window.Keys[0]
	if err := window.Validate(); err == nil {
		t.Fatal("unsorted keys accepted")
	}
	window.Keys[1] = window.Keys[0]
	if err := window.Validate(); err == nil {
		t.Fatal("duplicate key accepted")
	}
	window.Keys[0].Request.RequestID = "2"
	window.Keys[1].Request.RequestID = "10"
	if err := window.Validate(); err != nil {
		t.Fatalf("numeric request-ID order rejected: %v", err)
	}
}

func TestHistoryRawGuardMaximumJ2Values(t *testing.T) {
	fixture := loadRoutingHistoryFixtures(t)[0].Record
	value, _, err := DecodeRoutingHistoryRecord(fixture)
	if err != nil {
		t.Fatal(err)
	}
	value.Requested.Model = strings.Repeat("m", 512)
	encoded, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	padding := RoutingHistoryRecordBytes - len(encoded) - len(`"unknown":"",`)
	if padding < 0 {
		t.Fatal("fixture exceeds record budget")
	}
	raw := []byte(`{"unknown":"` + strings.Repeat("x", padding) + `",` + string(encoded[1:]))
	if len(raw) != RoutingHistoryRecordBytes {
		t.Fatalf("record fixture bytes=%d", len(raw))
	}
	schema, _ := RoutingHistoryTransportSchema(NotifyRoutingRequest, false)
	projected, _, err := jsonrpc.ProjectHistory(raw, make([]byte, 0, RoutingHistoryRecordBytes), make([]uint32, RoutingHistoryRecordBytes/4), schema)
	if err != nil {
		t.Fatal(err)
	}
	got, _, err := DecodeRoutingHistoryRecord(projected)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, value) {
		t.Fatal("maximum legal record changed")
	}
	initial := []byte(`{"requests":[` + strings.TrimSuffix(strings.Repeat(string(encoded)+",", 256), ",") + `]}`)
	initial = append(initial, []byte(strings.Repeat(" ", RoutingHistoryInitialBytes-len(initial)))...)
	schema, _ = RoutingHistoryTransportSchema("", true)
	projected, _, err = jsonrpc.ProjectHistory(initial, make([]byte, 0, RoutingHistoryInitialBytes), make([]uint32, RoutingHistoryInitialBytes/4), schema)
	if err != nil {
		t.Fatal(err)
	}
	decoded, _, err := DecodeRoutingHistoryInitial(projected)
	if err != nil {
		t.Fatal(err)
	}
	if len(decoded.Requests) != 256 || !reflect.DeepEqual(decoded.Requests[255], value) {
		t.Fatal("maximum legal initial changed")
	}
}

func TestHistoryRawGuardPublicCorpusProjection(t *testing.T) {
	for _, fixture := range loadRoutingHistoryFixtures(t) {
		t.Run(fixture.Name, func(t *testing.T) {
			want, _, err := DecodeRoutingHistoryRecord(fixture.Record)
			if err != nil {
				t.Fatal(err)
			}
			schema, ok := RoutingHistoryTransportSchema(NotifyRoutingRequest, false)
			if !ok {
				t.Fatal("record schema absent")
			}
			raw := strings.Replace(string(fixture.Record), `{`, `{"syntheticUnknown":{"large":"`+strings.Repeat("x", 65536)+`"},`, 1)
			projected, stripped, err := jsonrpc.ProjectHistory([]byte(raw), make([]byte, 0, RoutingHistoryRecordBytes), make([]uint32, RoutingHistoryRecordBytes/4), schema)
			if err != nil || stripped != 1 {
				t.Fatalf("raw guard stripped=%d err=%v", stripped, err)
			}
			got, _, err := DecodeRoutingHistoryRecord(projected)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatal("streaming projection changed J2 facts")
			}
		})
	}
}

func TestHistoryRawGuardInitialRecordLimits(t *testing.T) {
	schema, ok := RoutingHistoryTransportSchema("", true)
	if !ok {
		t.Fatal("initial schema absent")
	}
	fixture := loadRoutingHistoryFixtures(t)[0].Record
	oversize := strings.Replace(string(fixture), `{`, `{"unknown":"`+strings.Repeat("x", RoutingHistoryRecordBytes)+`",`, 1)
	raw := []byte(`{"requests":[` + oversize + `]}`)
	if _, _, err := jsonrpc.ProjectHistory(raw, make([]byte, 0, len(raw)), make([]uint32, len(raw)/4), schema); err == nil {
		t.Fatal("oversized original record passed after stripping")
	}
	tooDeep := strings.Replace(string(fixture), `{`, `{"unknown":`+strings.Repeat("[", 16)+`0`+strings.Repeat("]", 16)+`,`, 1)
	raw = []byte(`{"requests":[` + tooDeep + `]}`)
	if _, _, err := jsonrpc.ProjectHistory(raw, make([]byte, 0, len(raw)), make([]uint32, len(raw)/4), schema); err == nil {
		t.Fatal("record borrowed initial depth allowance")
	}
}

func TestHistoryProjectedRecordUsesSharedSemantics(t *testing.T) {
	for _, fixture := range loadRoutingHistoryFixtures(t) {
		t.Run(fixture.Name, func(t *testing.T) {
			want, _, err := DecodeRoutingHistoryRecord(fixture.Record)
			if err != nil {
				t.Fatalf("decode fixture: %v", err)
			}
			schema, _ := RoutingHistoryTransportSchema(NotifyRoutingRequest, false)
			projected, _, err := jsonrpc.ProjectHistory(fixture.Record, make([]byte, 0, RoutingHistoryRecordBytes), make([]uint32, RoutingHistoryRecordBytes/4), schema)
			if err != nil {
				t.Fatalf("project fixture: %v", err)
			}
			view, err := ReadProjectedRoutingHistoryRecord(projected)
			if err != nil {
				t.Fatalf("read view: %v", err)
			}
			if view.AttemptCount() != len(want.Attempts) {
				t.Fatalf("attempt count = %d, want %d", view.AttemptCount(), len(want.Attempts))
			}
			var current, related RoutingHistoryAttempt
			for i := range want.Attempts {
				if err := view.ReadAttempt(i, &current); err != nil {
					t.Fatalf("read current: %v", err)
				}
				if err := view.ReadAttempt(len(want.Attempts)-1, &related); err != nil {
					t.Fatalf("read related: %v", err)
				}
				if !reflect.DeepEqual(current, want.Attempts[i]) || !reflect.DeepEqual(related, want.Attempts[len(want.Attempts)-1]) {
					t.Fatal("related lookup clobbered current or changed facts")
				}
			}
			// The header is independent of both scratch slots and has no
			// full attempt slice to retain after a nested lookup.
			header := reflect.TypeOf(view.Header)
			record := reflect.TypeOf(want)
			if header.NumField()+1 != record.NumField() {
				t.Fatal("header field count differs from record prefix")
			}
			for i := 0; i < header.NumField(); i++ {
				if header.Field(i).Name != record.Field(i).Name || header.Field(i).Type != record.Field(i).Type || !reflect.DeepEqual(reflect.ValueOf(view.Header).Field(i).Interface(), reflect.ValueOf(want).Field(i).Interface()) {
					t.Fatalf("header differs at field %s", header.Field(i).Name)
				}
			}
		})
	}
}

func TestHistoryProjectedRecordSemanticRefusals(t *testing.T) {
	fixture := loadRoutingHistoryFixtures(t)[0].Record
	test := func(name, path string, value any) {
		t.Run(name, func(t *testing.T) {
			object := routingHistoryObject(t, fixture)
			parent, field := routingHistoryParent(t, object, path)
			parent[field] = value
			raw := routingHistoryJSON(t, object)
			schema, _ := RoutingHistoryTransportSchema(NotifyRoutingRequest, false)
			// Obtain shape-checked spans without running the semantic callback,
			// then compare that callback's refusal with the public decoder.
			schema.Check = nil
			projected, _, err := jsonrpc.ProjectHistory(raw, make([]byte, 0, RoutingHistoryRecordBytes), make([]uint32, RoutingHistoryRecordBytes/4), schema)
			if err != nil {
				t.Fatalf("shape fixture: %v", err)
			}
			_, _, want := DecodeRoutingHistoryRecord(raw)
			_, got := ReadProjectedRoutingHistoryRecord(projected)
			if want == nil || got == nil || got.Error() != want.Error() {
				t.Fatalf("view/public errors = %v / %v", got, want)
			}
		})
	}
	test("nonpositive sequence", "seq", 0)
	test("terminal recovery", "recoverable", true)
	test("record before creation", "endedAt", 0)
	test("attempt ordinal", "attempts.0.index", routingHistorySafeInteger)
	test("noncanonical money", "spend.heldNanoUsd", "01")
	test("destination exclusivity", "attempts.0.destination.accountId", "account")
}

func TestHistoryProjectedLongIntegerAfterOriginalGrammar(t *testing.T) {
	fixture := loadRoutingHistoryFixtures(t)[0].Record
	object := routingHistoryObject(t, fixture)
	object["seq"] = json.Number("0." + strings.Repeat("0", 66000) + "1E66001")
	raw := routingHistoryJSON(t, object)
	schema, _ := RoutingHistoryTransportSchema(NotifyRoutingRequest, false)
	out := make([]byte, 0, RoutingHistoryRecordBytes)
	offsets := make([]uint32, RoutingHistoryRecordBytes/4)
	projected, _, err := jsonrpc.ProjectHistory(raw, out, offsets, schema)
	if err != nil {
		t.Fatalf("legal long integer: %v", err)
	}
	view, err := ReadProjectedRoutingHistoryRecord(projected)
	if err != nil || view.Header.Seq != 1 {
		t.Fatalf("long integer sequence = %d / %v", view.Header.Seq, err)
	}
	malformed := strings.Replace(string(raw), "1E66001", "1E+", 1)
	if _, _, err := jsonrpc.ProjectHistory([]byte(malformed), out, offsets, schema); err == nil {
		t.Fatal("invalid original JSON reached semantic acceptance")
	}
}

func TestHistoryProjectedWindowBoundedIteration(t *testing.T) {
	window := RoutingHistoryWindow{Schema: 1, Generation: "1", Keys: make([]RoutingHistoryWindowKey, RoutingHistoryRecords)}
	for i := range window.Keys {
		window.Keys[i] = RoutingHistoryWindowKey{Request: RoutingRequestKey{OriginNodeID: strings.Repeat("o", 128), ProxyRunID: strings.Repeat("r", 128), RequestID: fmt.Sprintf("9223372036854775%03d", i)}, CreatedAt: int64(i), Seq: 1}
	}
	window.Cutoff = &RoutingHistoryCutoff{Request: window.Keys[0].Request, CreatedAt: 0}
	raw, err := json.Marshal(window)
	if err != nil {
		t.Fatalf("encode window: %v", err)
	}
	schema, _ := RoutingHistoryTransportSchema(NotifyRoutingHistoryWindow, false)
	projected, _, err := jsonrpc.ProjectHistory(raw, make([]byte, 0, RoutingHistoryRecordBytes), make([]uint32, RoutingHistoryRecordBytes/4), schema)
	if err != nil {
		t.Fatalf("project maximum manifest: %v", err)
	}
	view, err := ReadProjectedRoutingHistoryWindow(projected)
	if err != nil {
		t.Fatalf("read window view: %v", err)
	}
	if view.KeyCount() != len(window.Keys) || view.Generation != window.Generation || !reflect.DeepEqual(view.Cutoff, window.Cutoff) {
		t.Fatal("manifest header changed")
	}
	if err := view.ForEachKey(func(i int, key RoutingHistoryWindowKey) error {
		if !reflect.DeepEqual(key, window.Keys[i]) {
			t.Fatalf("manifest key %d changed", i)
		}
		return nil
	}); err != nil {
		t.Fatalf("iterate maximum manifest: %v", err)
	}
}
