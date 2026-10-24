// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package jsonrpc

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"unsafe"
)

type historyTestValue struct {
	Schema int64 `json:"schema"`
}
type historyTestInitial struct {
	Requests []historyTestValue `json:"requests"`
}

func TestHistoryGeneratedCorrelationAllocationAndExhaustion(t *testing.T) {
	var arbitrary HistoryCorrelation
	if err := arbitrary.Begin([]byte(`"jh:1"`)); err != nil {
		t.Fatal(err)
	}
	if err := arbitrary.EnableGeneratedIDs(); !errors.Is(err, ErrHistoryBusy) {
		t.Fatalf("opt-in converted a live arbitrary call: %v", err)
	}
	arbitrary.Complete()
	if err := arbitrary.EnableGeneratedIDs(); !errors.Is(err, ErrHistoryInvalid) {
		t.Fatalf("opt-in reclassified a used arbitrary-ID generation: %v", err)
	}
	if err := arbitrary.Begin([]byte(`"still-arbitrary"`)); err != nil {
		t.Fatal(err)
	}
	var correlation HistoryCorrelation
	if _, err := correlation.BeginGeneratedMethod("routing:subscribe"); !errors.Is(err, ErrHistoryInvalid) {
		t.Fatalf("generated allocation without opt-in: %v", err)
	}
	if err := correlation.EnableGeneratedIDs(); err != nil {
		t.Fatal(err)
	}
	first, err := correlation.BeginGeneratedMethod("routing:subscribe")
	if err != nil || string(first) != `"jh:1"` || !correlation.MatchesPending(first) {
		t.Fatalf("first generated allocation: %s, %v", first, err)
	}
	if _, err := correlation.BeginGeneratedMethod("routing:get-initial"); !errors.Is(err, ErrHistoryBusy) {
		t.Fatalf("second live initial: %v", err)
	}
	correlation.Retire() // Proven-unsent also consumes the ordinal.
	correlation.ReleaseRetired()
	second, err := correlation.BeginGeneratedMethod("routing:get-initial")
	if err != nil || string(second) != `"jh:2"` {
		t.Fatalf("retirement reused an ordinal: %s, %v", second, err)
	}
	correlation.Retire()
	correlation.issued = ^uint64(0) - 1
	last, err := correlation.BeginGeneratedMethod("routing:unsubscribe")
	if err != nil || string(last) != `"jh:18446744073709551615"` || len(last) != 25 || cap(last) != 25 {
		t.Fatalf("last generated token: %s, %v", last, err)
	}
	correlation.Complete()
	if err := correlation.EnableGeneratedIDs(); err != nil {
		t.Fatal(err)
	}
	if _, err := correlation.BeginGeneratedMethod("routing:subscribe"); !errors.Is(err, ErrHistoryExhausted) || err.Error() != "history-exhausted" || correlation.issued != ^uint64(0) {
		t.Fatalf("exhaustion wrapped or reset the frontier: %v", err)
	}
	if got := unsafe.Sizeof(correlation); got != 1088 {
		t.Fatalf("correlation control bytes = %d, want 1088", got)
	}
}

func TestHistoryGeneratedQuarantineWithoutLateAAllowsB(t *testing.T) {
	test := func(name, candidates string) {
		t.Run(name, func(t *testing.T) {
			var correlation HistoryCorrelation
			if err := correlation.EnableGeneratedIDs(); err != nil {
				t.Fatal(err)
			}
			if _, err := correlation.BeginGeneratedMethod("routing:subscribe"); err != nil {
				t.Fatal(err)
			}
			c := historyTestReader(t, "", &correlation)
			read := func(frame string) (*Message, error) {
				c.history.r.Reset(strings.NewReader(frame + "\n"))
				return c.Read()
			}
			_, err := read(`{"jsonrpc":"2.0",` + candidates + `,"result":{"requests":[]}}`)
			var failure *HistoryFailure
			if !errors.As(err, &failure) || !failure.FailPending || !failure.ScheduleRefetch || failure.Disposition != "quarantine" {
				t.Fatalf("A ambiguity: %+v, %v", failure, err)
			}
			_, err = read(`{"jsonrpc":"2.0",` + candidates + `,"result":{"requests":[]}}`)
			var repeated *HistoryFailure
			if !errors.As(err, &repeated) || repeated.FailPending || repeated.ScheduleRefetch {
				t.Fatalf("A failed twice or queued another refetch: %+v, %v", repeated, err)
			}
			ordinary, err := read(`{"jsonrpc":"2.0","id":7,"result":{"alive":true}}`)
			if err != nil || ordinary == nil || string(*ordinary.ID) != "7" || string(ordinary.Result) != `{"alive":true}` {
				t.Fatalf("ordinary O lost: %+v, %v", ordinary, err)
			}
			b, err := correlation.BeginGeneratedMethod("routing:subscribe")
			if err != nil || string(b) != `"jh:2"` || !correlation.Quarantined() {
				t.Fatalf("B requires absent late A: %s, %v", b, err)
			}
			msg, err := read(`{"jsonrpc":"2.0","id":"jh:2","result":{"requests":[]}}`)
			if err != nil || msg == nil || !correlation.MatchesPending(*msg.ID) || !correlation.Quarantined() {
				t.Fatalf("guard alone restored history: %+v, %v", msg, err)
			}
			correlation.Complete() // The owner has now applied B.
			if correlation.Quarantined() {
				t.Fatal("applied current B did not restore history")
			}
		})
	}
	test("history first", `"id":"jh:1","\u0069d":7`)
	test("history last escaped namespace", `"id":7,"\u0069d":"\u006ah\u003a\u0031"`)
}

func TestHistoryGeneratedRepeatedQuarantineStaleAndReservedResponses(t *testing.T) {
	var correlation HistoryCorrelation
	if err := correlation.EnableGeneratedIDs(); err != nil {
		t.Fatal(err)
	}
	c := historyTestReader(t, "", &correlation)
	read := func(frame string) (*Message, error) {
		c.history.r.Reset(strings.NewReader(frame + "\n"))
		return c.Read()
	}
	for _, candidates := range []string{`"id":"jh:1","id":7`, `"ID":7,"\u0069d":"\u006ah:2"`} {
		if _, err := correlation.BeginGeneratedMethod("routing:subscribe"); err != nil {
			t.Fatal(err)
		}
		_, err := read(`{"jsonrpc":"2.0",` + candidates + `,"result":{"requests":[]}}`)
		var failure *HistoryFailure
		if !errors.As(err, &failure) || !failure.FailPending || !failure.ScheduleRefetch {
			t.Fatalf("repeated quarantine: %+v, %v", failure, err)
		}
	}
	if _, err := correlation.BeginGeneratedMethod("routing:subscribe"); err != nil {
		t.Fatal(err)
	}
	if msg, err := read(`{"jsonrpc":"2.0","id":"jh:3","result":{"requests":[]}}`); err != nil || msg == nil {
		t.Fatalf("B recovery: %+v, %v", msg, err)
	}
	correlation.Complete()
	// A new quarantine must survive every old success/error, including invalid
	// stale values and reserved but unissued/noncanonical namespace responses.
	if _, err := read(`{"broken":`); err == nil {
		t.Fatal("malformed input accepted")
	}
	if _, err := correlation.BeginGeneratedMethod("routing:subscribe"); err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{`"jh:1"`, `"\u006ah:2"`, `"jh:3"`, `"jh:5"`, `"jh:0"`, `"jh:01"`, `"jh:+1"`, `"jh:18446744073709551616"`, `"jh:"`} {
		for _, body := range []string{`"result":{"requests":null}`, `"error":{"code":-32096,"message":"routing history busy; retry"}`} {
			msg, err := read(`{"jsonrpc":"2.0","id":` + id + `,` + body + `}`)
			var failure *HistoryFailure
			if msg != nil || !errors.As(err, &failure) || failure.Disposition != "discard" || failure.FailPending || failure.ScheduleRefetch || !correlation.Quarantined() || !correlation.MatchesPending([]byte(`"jh:4"`)) {
				t.Fatalf("old/reserved response changed current ownership: %s, %+v, %v", id, failure, err)
			}
		}
	}
}

func TestHistoryGeneratedStaleMaximumInitialUsesOnlyRawLease(t *testing.T) {
	var correlation HistoryCorrelation
	if err := correlation.EnableGeneratedIDs(); err != nil {
		t.Fatal(err)
	}
	if _, err := correlation.BeginGeneratedMethod("routing:subscribe"); err != nil {
		t.Fatal(err)
	}
	correlation.Retire()
	value := `{"requests":[],"unknown":"` + strings.Repeat("x", HistoryInitialBytes-len(`{"requests":[],"unknown":""}`)) + `"}`
	prefix := `{"jsonrpc":"2.0","id":"\u006ah:1","result":`
	frame := prefix + value + strings.Repeat(" ", HistoryEnvelopeBytes-len(prefix)-2) + "}\n"
	if len(frame) != HistoryFrameBytes {
		t.Fatal("fixture does not consume H")
	}
	c := historyTestReader(t, frame, &correlation)
	c.history.output = []byte("pinned output")[:0]
	msg, err := c.Read()
	var failure *HistoryFailure
	if msg != nil || !errors.As(err, &failure) || failure.Category != "history-stale" || string(c.history.output[:cap(c.history.output)]) != "pinned output" {
		t.Fatalf("stale result borrowed output or required an initial output grant: %v", err)
	}
}

func TestHistoryGeneratedIncomingDirectionalityAndDefaultIsolation(t *testing.T) {
	test := func(name, frame, method, wantID string, code int) {
		t.Run(name, func(t *testing.T) {
			var correlation HistoryCorrelation
			if err := correlation.EnableGeneratedIDs(); err != nil {
				t.Fatal(err)
			}
			if _, err := correlation.BeginGeneratedMethod("routing:subscribe"); err != nil {
				t.Fatal(err)
			}
			c := historyIncomingTestReader(t, frame+"\n", &correlation)
			msg, err := c.Read()
			if code != 0 {
				var failure *HistoryFailure
				if !errors.As(err, &failure) || failure.Refusal == nil || failure.FailPending || failure.ScheduleRefetch || failure.Disposition != "refuse-request" || failure.Refusal.Code != code || string(failure.Refusal.ReplyID) != wantID {
					t.Fatalf("request refusal: %+v, %v", failure, err)
				}
			} else if err != nil || msg == nil || !msg.IsRequest() || msg.Method != method {
				t.Fatalf("incoming request: %+v, %v", msg, err)
			}
			if !correlation.MatchesPending([]byte(`"jh:1"`)) || correlation.Quarantined() {
				t.Fatal("incoming request changed outbound pending")
			}
		})
	}
	test("ordinary collision", `{"jsonrpc":"2.0","id":"\u006ah:1","method":"echo","params":{}}`, "echo", "", 0)
	test("reserved unissued ordinary request", `{"jsonrpc":"2.0","id":"jh:999","method":"echo","params":{}}`, "echo", "", 0)
	test("public collision refusal", `{"jsonrpc":"2.0","id":"jh:1","method":"routing:subscribe","params":null}`, "", `"jh:1"`, -32062)
	test("private collision refusal", `{"jsonrpc":"2.0","id":"jh:1","method":"routing/request","params":{}}`, "", `"jh:1"`, -32601)
	test("ordinary over512 caller ID", `{"jsonrpc":"2.0","id":"jh:`+strings.Repeat("1", 600)+`","method":"echo","params":{}}`, "echo", "", 0)
	test("history over512 caller ID", `{"jsonrpc":"2.0","id":"jh:`+strings.Repeat("1", 600)+`","method":"routing:subscribe","params":{}}`, "", "null", -32062)
	msg, err := historyTestReader(t, `{"jsonrpc":"2.0","id":"jh:999","result":{"ordinary":true}}`+"\n", &HistoryCorrelation{}).Read()
	if err != nil || msg == nil || string(msg.Result) != `{"ordinary":true}` {
		t.Fatalf("default correlation reserved a namespace: %+v, %v", msg, err)
	}
}

func historyTestSchema(_ string, initial bool) (HistoryProjection, bool) {
	s := HistoryProjection{Type: reflect.TypeOf(historyTestValue{}), Bytes: HistoryRecordBytes, Depth: 16, Members: 2048, ArrayLimit: 256}
	if initial {
		s.Type = reflect.TypeOf(historyTestInitial{})
		s.Bytes = HistoryInitialBytes
		s.Depth = 18
		s.Members = HistoryInitialBytes
	}
	return s, true
}

func historyTestReader(t *testing.T, input string, correlation *HistoryCorrelation) *Codec {
	t.Helper()
	c, err := NewHistoryCodec(rw{strings.NewReader(input), io.Discard}, 1<<20, make([]byte, 0, HistoryFrameBytes), make([]byte, 0, HistoryFrameBytes), make([]uint32, HistoryInitialBytes/4), correlation, historyTestSchema, nil)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func TestHistoryRawGuardUnknownProjection(t *testing.T) {
	raw := []byte(`{"unknown":{"large":"` + strings.Repeat("x", 100000) + `"},"schema":1}`)
	out := make([]byte, 0, 16)
	s, _ := historyTestSchema("", false)
	projected, stripped, err := ProjectHistory(raw, out, make([]uint32, HistoryRecordBytes/4), s)
	if err != nil {
		t.Fatal(err)
	}
	if string(projected) != `{"schema":1}` || stripped != 1 || cap(projected) != 16 {
		t.Fatalf("projection bytes=%d stripped=%d capacity=%d", len(projected), stripped, cap(projected))
	}
}

func TestHistoryRawGuardGrammar(t *testing.T) {
	for _, raw := range []string{`{}`, `[]`, `[{},[],null,true,false,-0,1.5e-2,"\\\"\u0000"]`, ` { "a" : [1,2] } `, `"\ud800"`, `0e9999999999999999999999999999999999999`} {
		if !historyJSONValid([]byte(raw), 20) || !json.Valid([]byte(raw)) {
			t.Fatalf("legal grammar rejected: %q", raw)
		}
	}
	for _, raw := range []string{"", `{`, `{"x":}`, `{"x":1,}`, `[1,]`, `01`, `1.`, `1e`, `1e+`, `truefalse`, `{} {}`, `"\x20"`, "\"line\nfeed\"", `{"x" 1}`, `{"x":1 "y":2}`} {
		if historyJSONValid([]byte(raw), 20) || json.Valid([]byte(raw)) {
			t.Fatalf("malformed grammar accepted: %q", raw)
		}
	}
	if historyJSONValid([]byte(strings.Repeat("[", 21)+`0`+strings.Repeat("]", 21)), 20) {
		t.Fatal("grammar exceeded bounded stack")
	}
}

func TestHistoryRawGuardDuplicateKeys(t *testing.T) {
	test := func(name, raw string) {
		t.Run(name, func(t *testing.T) {
			s, _ := historyTestSchema("", false)
			if _, _, err := ProjectHistory([]byte(raw), make([]byte, 0, 64), make([]uint32, HistoryRecordBytes/4), s); !errors.Is(err, ErrHistoryInvalid) {
				t.Fatalf("error=%v", err)
			}
		})
	}
	test("known escaped then literal", `{"\u0073chema":1,"schema":1}`)
	test("known literal then escaped", `{"schema":1,"\u0073chema":1}`)
	test("unknown nested", `{"schema":1,"unknown":{"a":1,"\u0061":2}}`)
	test("unknown after known", `{"schema":1,"unknown":1,"unknown":2}`)
	test("unknown before known", `{"unknown":1,"unknown":2,"schema":1}`)
	test("surrogate spelling", `{"schema":1,"unknown":{"\ud800":1,"\ufffd":2}}`)
	test("unknown array subtree", `{"schema":1,"unknown":[{"x":1,"x":2}]}`)
}

func TestHistoryRawGuardShapeAndLimits(t *testing.T) {
	test := func(name, raw string, capacity int) {
		t.Run(name, func(t *testing.T) {
			s, _ := historyTestSchema("", false)
			if _, _, err := ProjectHistory([]byte(raw), make([]byte, 0, capacity), make([]uint32, HistoryRecordBytes/4), s); err == nil {
				t.Fatal("invalid value accepted")
			}
		})
	}
	test("null required", `{"schema":null}`, 64)
	test("wrong known type", `{"schema":"1"}`, 64)
	test("missing known field", `{"unknown":1}`, 64)
	test("output cannot grow", `{"schema":1}`, 4)
	test("unknown depth", `{"schema":1,"unknown":`+strings.Repeat("[", 16)+`0`+strings.Repeat("]", 16)+`}`, 64)
	test("unknown members", `{"schema":1,"unknown":{`+strings.Repeat(`"a":0,`, 2048)+`"last":0}}`, 64)
	test("invalid utf8", "{\"schema\":1,\"x\":\"\xff\"}", 64)
}

func TestHistoryRawGuardMemberBoundary(t *testing.T) {
	value := func(count int) []byte {
		var raw strings.Builder
		raw.WriteString(`{"schema":1,"unknown":{`)
		for i := 0; i < count; i++ {
			if i != 0 {
				raw.WriteByte(',')
			}
			raw.WriteString(`"k` + strconv.Itoa(i) + `":0`)
		}
		raw.WriteString(`}}`)
		return []byte(raw.String())
	}
	schema, _ := historyTestSchema("", false)
	if _, _, err := ProjectHistory(value(2046), make([]byte, 0, 64), make([]uint32, HistoryRecordBytes/4), schema); err != nil {
		t.Fatalf("2048 members rejected: %v", err)
	}
	if _, _, err := ProjectHistory(value(2047), make([]byte, 0, 64), make([]uint32, HistoryRecordBytes/4), schema); err == nil {
		t.Fatal("2049 members accepted")
	}
}

func TestHistoryFrameInPlaceProjection(t *testing.T) {
	var correlation HistoryCorrelation
	if err := correlation.BeginMethod("routing:subscribe", []byte(`"\u0069d"`)); err != nil {
		t.Fatal(err)
	}
	input := `{"result":{"requests":[{"schema":1e15}],"unknown":"discard"},"id":"id","jsonrpc":"2.0"}` + "\n"
	raw := make([]byte, 0, HistoryFrameBytes)
	c, err := NewHistoryCodec(rw{strings.NewReader(input), io.Discard}, WorkerFrameBytes, raw, nil, make([]uint32, HistoryInitialBytes/4), &correlation, historyTestSchema, nil)
	if err != nil {
		t.Fatal(err)
	}
	msg, err := c.Read()
	if err != nil {
		t.Fatal(err)
	}
	if string(msg.Result) != `{"requests":[{"schema":1e15}]}` || msg.ID == nil || string(*msg.ID) != `"id"` {
		t.Fatalf("in-place envelope=%+v", msg)
	}
	if cap(c.history.raw) != HistoryFrameBytes || cap(c.history.output) != HistoryFrameBytes || cap(c.history.offsets)*4 != HistoryInitialBytes {
		t.Fatal("reader grants changed")
	}
	if &c.history.raw[:cap(c.history.raw)][0] != &c.history.output[:cap(c.history.output)][0] {
		t.Fatal("in-place projection acquired a second frame")
	}
	correlation.Complete()
	if err := correlation.Begin([]byte(`"next"`)); err != nil {
		t.Fatal(err)
	}
}

func TestHistoryFrameMaximumBudgets(t *testing.T) {
	if HistoryRecordBytes != 131072 || HistoryInitialBytes != 33554432 || HistoryEnvelopeBytes != 4096 || HistoryRecordFrameBytes != 135168 || HistoryFrameBytes != 33558528 || HistoryIDBytes != 512 {
		t.Fatal("history grant changed")
	}
	t.Run("record at R and envelope at E", func(t *testing.T) {
		value := `{"schema":1,"unknown":"` + strings.Repeat("x", HistoryRecordBytes-len(`{"schema":1,"unknown":""}`)) + `"}`
		prefix := `{"jsonrpc":"2.0","method":"routing/request","params":`
		frame := prefix + value + strings.Repeat(" ", HistoryEnvelopeBytes-len(prefix)-2) + "}\n"
		if len(frame) != HistoryRecordFrameBytes {
			t.Fatalf("fixture bytes=%d", len(frame))
		}
		msg, err := historyTestReader(t, frame, &HistoryCorrelation{}).Read()
		if err != nil || msg == nil || string(msg.Params) != `{"schema":1}` {
			t.Fatalf("read=%v err=%v", msg, err)
		}
	})
	t.Run("initial at M raw id at 512 envelope at E", func(t *testing.T) {
		id := `"` + strings.Repeat(`\u0061`, 85) + `"`
		if len(id) != 512 {
			t.Fatal("wrong ID fixture")
		}
		var correlation HistoryCorrelation
		if err := correlation.Begin([]byte(id)); err != nil {
			t.Fatal(err)
		}
		value := `{"requests":[],"unknown":"` + strings.Repeat("x", HistoryInitialBytes-len(`{"requests":[],"unknown":""}`)) + `"}`
		prefix := `{"jsonrpc":"2.0","id":` + id + `,"result":`
		frame := prefix + value + strings.Repeat(" ", HistoryEnvelopeBytes-len(prefix)-2) + "}\n"
		if len(frame) != HistoryFrameBytes {
			t.Fatalf("fixture bytes=%d", len(frame))
		}
		msg, err := historyTestReader(t, frame, &correlation).Read()
		if err != nil || msg == nil || string(msg.Result) != `{"requests":[]}` || string(*msg.ID) != id {
			t.Fatalf("initial failed: %v", err)
		}
	})
}

func TestHistoryFrameLogicalBudgets(t *testing.T) {
	test := func(name, frame, disposition string) {
		t.Run(name, func(t *testing.T) {
			_, err := historyTestReader(t, frame+"\n", &HistoryCorrelation{}).Read()
			var failure *HistoryFailure
			if !errors.As(err, &failure) || failure.Category != "history-invalid" || failure.Disposition != disposition {
				t.Fatalf("error=%v", err)
			}
			if disposition == "refuse-request" && (failure.Refusal == nil || failure.Refusal.Code != -32062 || failure.Refusal.Message != "history-invalid" || string(failure.Refusal.ReplyID) != "null" || failure.FailPending || failure.ScheduleRefetch) {
				t.Fatalf("over-cap ID refusal=%+v", failure)
			}
		})
	}
	test("record R plus one", `{"jsonrpc":"2.0","method":"routing/request","params":{"schema":1,"x":"`+strings.Repeat("x", HistoryRecordBytes)+`"}}`, "discard")
	test("envelope E plus one", `{"jsonrpc":"2.0","method":"routing/request","params":{"schema":1},"x":"`+strings.Repeat("x", HistoryEnvelopeBytes)+`"}`, "discard")
	test("ID 513", `{"jsonrpc":"2.0","method":"routing:get-initial","id":"`+strings.Repeat("a", 511)+`","params":{}}`, "refuse-request")
}

func TestHistoryFramePhysicalFraming(t *testing.T) {
	test := func(name, input string) {
		t.Run(name, func(t *testing.T) {
			_, err := historyTestReader(t, input, &HistoryCorrelation{}).Read()
			if !errors.Is(err, ErrHistoryFraming) || err.Error() != "history-framing" {
				t.Fatalf("error=%v", err)
			}
		})
	}
	test("missing newline", `{"jsonrpc":"2.0","method":"ordinary"}`)
	test("H without newline", strings.Repeat(" ", HistoryFrameBytes))
	test("H plus one with newline", strings.Repeat(" ", HistoryFrameBytes)+"\n")
}

func TestHistoryFramePrivateReaderLeases(t *testing.T) {
	var correlation HistoryCorrelation
	input := `{"jsonrpc":"2.0","method":"routing/request","params":{"schema":1}}` + "\n" + strings.Repeat(" ", 2<<20) + "{}\n" + `{"jsonrpc":"2.0","method":"ordinary"}` + "\n"
	c, err := NewHistoryCodec(rw{strings.NewReader(input), io.Discard}, 1<<20, make([]byte, 0, 1<<20), make([]byte, 0, HistoryRecordFrameBytes), make([]uint32, HistoryRecordBytes/4), &correlation, historyTestSchema, nil)
	if err != nil {
		t.Fatal(err)
	}
	msg, err := c.Read()
	if err != nil || msg == nil || string(msg.Params) != `{"schema":1}` {
		t.Fatalf("private record=%+v err=%v", msg, err)
	}
	_, err = c.Read()
	var failure *HistoryFailure
	if !errors.As(err, &failure) || failure.Disposition != "quarantine" {
		t.Fatalf("in-H discarded frame=%v", err)
	}
	msg, err = c.Read()
	if err != nil || msg == nil || msg.Method != "ordinary" {
		t.Fatalf("ordinary traffic lost after history quarantine: %v", err)
	}
	if cap(c.history.raw) != 1<<20 || cap(c.history.output) != 135168 || cap(c.history.offsets)*4 != 131072 {
		t.Fatal("private reader acquired initial backing")
	}
}

func TestHistoryRawGuardCandidateCorrelation(t *testing.T) {
	test := func(name, input string) {
		t.Run(name, func(t *testing.T) {
			var correlation HistoryCorrelation
			if err := correlation.Begin([]byte(`"initial"`)); err != nil {
				t.Fatal(err)
			}
			_, err := historyTestReader(t, input+"\n", &correlation).Read()
			var failure *HistoryFailure
			if !errors.As(err, &failure) || failure.Disposition != "quarantine" || !failure.FailPending || !correlation.Quarantined() || failure.Error() != "history-invalid" || failure.Count != 1 {
				t.Fatalf("failure=%+v err=%v", failure, err)
			}
		})
	}
	test("history method before ordinary", `{"jsonrpc":"2.0","method":"routing/request","method":"ordinary","params":{"schema":1}}`)
	test("history method after ordinary", `{"jsonrpc":"2.0","method":"ordinary","\u006dethod":"routing/request","params":{"schema":1}}`)
	test("pending id before unrelated", `{"jsonrpc":"2.0","id":"initial","id":"ordinary","result":{"requests":[]}}`)
	test("pending id after unrelated", `{"jsonrpc":"2.0","id":"ordinary","\u0069d":"initial","result":{"requests":[]}}`)
	test("malformed unclassifiable", `{"unknown":"synthetic-private-sentinel`)
}

func TestHistoryRawGuardRetiredInitial(t *testing.T) {
	var correlation HistoryCorrelation
	if err := correlation.Begin([]byte(`"initial"`)); err != nil {
		t.Fatal(err)
	}
	correlation.Retire()
	if err := correlation.Begin([]byte(`"next"`)); !errors.Is(err, ErrHistoryBusy) {
		t.Fatalf("retired lease admitted next initial: %v", err)
	}
	input := `{"jsonrpc":"2.0","id":"\u0069nitial","result":{"requests":[],"unknown":"` + strings.Repeat("x", 2<<20) + `"}}` + "\n"
	_, err := historyTestReader(t, input, &correlation).Read()
	var failure *HistoryFailure
	if !errors.As(err, &failure) || failure.Disposition != "discard" || failure.Category != "history-stale" {
		t.Fatalf("late initial=%v", err)
	}
	correlation.ReleaseRetired()
	if err := correlation.Begin([]byte(`"next"`)); err != nil {
		t.Fatal(err)
	}
}

func TestHistoryRawGuardPendingFailureOnce(t *testing.T) {
	var correlation HistoryCorrelation
	if err := correlation.Begin([]byte(`1`)); err != nil {
		t.Fatal(err)
	}
	line := `{"jsonrpc":"2.0","id":1,"result":{"requests":null}}` + "\n"
	c := historyTestReader(t, line+line, &correlation)
	_, err := c.Read()
	var first *HistoryFailure
	if !errors.As(err, &first) || first.Disposition != "fail-call" || !first.FailPending {
		t.Fatalf("first failure=%v", err)
	}
	_, err = c.Read()
	var second *HistoryFailure
	if !errors.As(err, &second) || second.Disposition != "discard" || second.FailPending {
		t.Fatalf("second failure=%v", err)
	}
}

func TestHistoryRawGuardOverlongEquivalentPendingID(t *testing.T) {
	var correlation HistoryCorrelation
	if err := correlation.Begin([]byte(`"` + strings.Repeat("a", 100) + `"`)); err != nil {
		t.Fatal(err)
	}
	input := `{"jsonrpc":"2.0","id":"` + strings.Repeat(`\u0061`, 100) + `","result":{"requests":[]}}` + "\n"
	_, err := historyTestReader(t, input, &correlation).Read()
	var failure *HistoryFailure
	if !errors.As(err, &failure) || failure.Disposition != "fail-call" || !failure.FailPending {
		t.Fatalf("over-cap equivalent ID lost correlation: %v", err)
	}
}

func TestHistoryRawGuardBusyResult(t *testing.T) {
	if HistoryBusyCode != -32096 || HistoryBusyMessage != "routing history busy; retry" || ErrHistoryBusy.Error() != "routing history busy; retry" {
		t.Fatal("approved busy contract changed")
	}
	var correlation HistoryCorrelation
	if err := correlation.Begin([]byte(`1`)); err != nil {
		t.Fatal(err)
	}
	msg, err := historyTestReader(t, `{"jsonrpc":"2.0","id":1,"error":{"code":-32096,"message":"routing history busy; retry"}}`+"\n", &correlation).Read()
	if err != nil || msg == nil || msg.Error == nil || msg.Error.Code != -32096 || msg.Error.Message != "routing history busy; retry" {
		t.Fatalf("busy result=%+v err=%v", msg, err)
	}
}

func TestHistoryRawGuardOuterDecoderAliases(t *testing.T) {
	c := historyTestReader(t, "", &HistoryCorrelation{})
	test := func(name, input string, initial bool) {
		t.Run(name, func(t *testing.T) {
			// Establish independently that encoding/json accepts this spelling.
			var downstream Message
			if err := json.Unmarshal([]byte(input), &downstream); err != nil {
				t.Fatal(err)
			}
			if downstream.JSONRPC != "2.0" || (!initial && downstream.Method != "routing/request") || (initial && (downstream.ID == nil || string(*downstream.ID) != `"initial"`)) {
				t.Fatal("fixture is not a downstream envelope alias")
			}
			var correlation HistoryCorrelation
			if initial {
				if err := correlation.Begin([]byte(`"initial"`)); err != nil {
					t.Fatal(err)
				}
			}
			c.history.correlation = &correlation
			c.history.r.Reset(strings.NewReader(input + "\n"))
			msg, err := c.Read()
			if err != nil || msg == nil {
				t.Fatalf("alias read: %v", err)
			}
			if initial {
				if string(msg.Result) != `{"requests":[]}` {
					t.Fatal("initial alias bypassed projection")
				}
			} else if string(msg.Params) != `{"schema":1}` {
				t.Fatal("method alias bypassed projection")
			}
		})
	}
	test("capitalized method and params", `{"JSONRPC":"2.0","Method":"routing/request","Params":{"schema":1,"unknown":{"marker":"discard"}}}`, false)
	test("escaped capitalized method", `{"jsonrpc":"2.0","\u004dethod":"routing/request","params":{"schema":1,"unknown":0}}`, false)
	test("Unicode folded version and params", `{"jſonrpc":"2.0","method":"routing/request","paramſ":{"schema":1,"unknown":0}}`, false)
	test("escaped Unicode folded envelope", `{"j\u017fonrpc":"2.0","METHOD":"routing/request","PARAM\u017f":{"schema":1,"unknown":0}}`, false)
	test("capitalized pending ID", `{"jsonrpc":"2.0","ID":"initial","result":{"requests":[],"unknown":0}}`, true)
	test("escaped ID and Unicode folded result", `{"jsonrpc":"2.0","\u0049D":"initial","re\u017fult":{"requests":[],"unknown":0}}`, true)
	test("case variants inside unknown remain distinct", `{"jsonrpc":"2.0","Method":"routing/request","params":{"schema":1,"unknown":{"Method":0,"method":1}}}`, false)
}

func TestHistoryRawGuardOrdinaryOuterAliasesUnchanged(t *testing.T) {
	input := `{"JSONRPC":"2.0","method":"first","Method":"last","ID":"` + strings.Repeat("x", 600) + `"}` + "\n"
	msg, err := historyTestReader(t, input, &HistoryCorrelation{}).Read()
	if err != nil || msg == nil || msg.Method != "last" || msg.ID == nil || len(*msg.ID) != 602 {
		t.Fatalf("ordinary alias semantics changed: %v", err)
	}
}

func TestHistoryRawGuardBusyOuterAlias(t *testing.T) {
	var correlation HistoryCorrelation
	if err := correlation.Begin([]byte(`1`)); err != nil {
		t.Fatal(err)
	}
	input := `{"JSONRPC":"2.0","ID":1,"\u0045rror":{"code":-32096,"message":"routing history busy; retry"}}` + "\n"
	msg, err := historyTestReader(t, input, &correlation).Read()
	if err != nil || msg == nil || msg.Error == nil || msg.Error.Code != -32096 || msg.Error.Message != "routing history busy; retry" {
		t.Fatalf("busy alias read: %v", err)
	}
}

func TestHistoryRawGuardOuterAliasConflicts(t *testing.T) {
	c := historyTestReader(t, "", &HistoryCorrelation{})
	test := func(name, input, disposition string, pending bool) {
		t.Run(name, func(t *testing.T) {
			var correlation HistoryCorrelation
			if pending {
				if err := correlation.Begin([]byte(`"initial"`)); err != nil {
					t.Fatal(err)
				}
			}
			c.history.correlation = &correlation
			c.history.r.Reset(strings.NewReader(input + "\n"))
			msg, err := c.Read()
			var failure *HistoryFailure
			if msg != nil || !errors.As(err, &failure) || failure.Disposition != disposition || failure.Category != "history-invalid" || failure.Count != 1 || failure.FailPending != pending {
				t.Fatalf("alias failure=%+v err=%v", failure, err)
			}
		})
	}
	test("capitalized method cannot hide unknown duplicates", `{"jsonrpc":"2.0","Method":"routing/request","params":{"unknown":{"x":1,"x":2}}}`, "discard", false)
	test("history method before ordinary alias", `{"jsonrpc":"2.0","method":"routing/request","Method":"ordinary","params":{"schema":1}}`, "quarantine", false)
	test("ordinary alias before history method", `{"jsonrpc":"2.0","Method":"ordinary","method":"routing/request","params":{"schema":1}}`, "quarantine", false)
	test("history alias before ordinary method", `{"jsonrpc":"2.0","Method":"routing/request","method":"ordinary","params":{"schema":1}}`, "quarantine", false)
	test("ordinary method before escaped history alias", `{"jsonrpc":"2.0","method":"ordinary","\u004dethod":"routing/request","params":{"schema":1}}`, "quarantine", false)
	test("pending ID before unrelated alias", `{"jsonrpc":"2.0","id":"initial","ID":"ordinary","result":{"requests":[]}}`, "quarantine", true)
	test("unrelated alias before pending ID", `{"jsonrpc":"2.0","ID":"ordinary","id":"initial","result":{"requests":[]}}`, "quarantine", true)
	test("pending alias before unrelated ID", `{"jsonrpc":"2.0","ID":"initial","id":"ordinary","result":{"requests":[]}}`, "quarantine", true)
	test("unrelated ID before escaped pending alias", `{"jsonrpc":"2.0","id":"ordinary","\u0049D":"initial","result":{"requests":[]}}`, "quarantine", true)
	test("version alias after canonical", `{"jsonrpc":"2.0","j\u017fonrpc":"2.0","Method":"routing/request","params":{"schema":1}}`, "discard", false)
	test("version alias before canonical", `{"JSONRPC":"2.0","jsonrpc":"2.0","Method":"routing/request","params":{"schema":1}}`, "discard", false)
	test("params alias after canonical", `{"jsonrpc":"2.0","method":"routing/request","params":{"schema":1},"paramſ":{"schema":1}}`, "discard", false)
	test("params alias before canonical", `{"jsonrpc":"2.0","method":"routing/request","Params":{"schema":1},"params":{"schema":1}}`, "discard", false)
	test("result alias after canonical", `{"jsonrpc":"2.0","ID":"initial","result":{"requests":[]},"reſult":{"requests":[]}}`, "fail-call", true)
	test("result alias before canonical", `{"jsonrpc":"2.0","ID":"initial","RESULT":{"requests":[]},"result":{"requests":[]}}`, "fail-call", true)
	test("error alias after canonical", `{"jsonrpc":"2.0","ID":"initial","error":{"code":-32096,"message":"routing history busy; retry"},"Error":{"code":-32096,"message":"routing history busy; retry"}}`, "fail-call", true)
	test("error alias before canonical", `{"jsonrpc":"2.0","ID":"initial","ERROR":{"code":-32096,"message":"routing history busy; retry"},"error":{"code":-32096,"message":"routing history busy; retry"}}`, "fail-call", true)
	test("nested known field retains strict case", `{"jsonrpc":"2.0","Method":"routing/request","params":{"Schema":1}}`, "discard", false)
}

func TestHistoryRawGuardRepeatedQuarantineRefetch(t *testing.T) {
	var correlation HistoryCorrelation
	if err := correlation.Begin([]byte(`"initial"`)); err != nil {
		t.Fatal(err)
	}
	malformed := `{"synthetic-private-sentinel":` + "\n"
	ordinary := `{"jsonrpc":"2.0","id":"ordinary","result":{"ok":true}}` + "\n"
	recovery := `{"jsonrpc":"2.0","id":"recovery","result":{"requests":[]}}` + "\n"
	c := historyTestReader(t, malformed+malformed+ordinary+malformed+malformed+ordinary+recovery, &correlation)
	ordinaryCall := make(chan *Message, 1)
	ordinaryPending := map[string]chan *Message{"ordinary": ordinaryCall}
	scheduled, requests := false, 0
	readFailure := func(wantFail, wantSchedule bool) {
		t.Helper()
		msg, err := c.Read()
		var failure *HistoryFailure
		if msg != nil || !errors.As(err, &failure) || failure.Disposition != "quarantine" || failure.Category != "history-invalid" || failure.Count != 1 || failure.FailPending != wantFail || failure.ScheduleRefetch != wantSchedule {
			t.Fatalf("quarantine=%+v err=%v", failure, err)
		}
		if failure.ScheduleRefetch {
			if scheduled {
				t.Fatal("duplicate outstanding refetch request")
			}
			scheduled = true
			requests++
		}
		if ordinaryPending["ordinary"] != ordinaryCall || len(ordinaryCall) != 0 {
			t.Fatal("history failure changed unrelated pending call")
		}
	}
	readOrdinary := func() {
		t.Helper()
		msg, err := c.Read()
		if err != nil || msg == nil || msg.ID == nil || string(*msg.ID) != `"ordinary"` || string(msg.Result) != `{"ok":true}` {
			t.Fatalf("ordinary response lost: %v", err)
		}
		ordinaryPending["ordinary"] <- msg
		if <-ordinaryCall != msg {
			t.Fatal("ordinary caller received a different response")
		}
	}
	readFailure(true, true)
	readFailure(false, false)
	readOrdinary()
	correlation.ReleaseRetired()
	// Consume the one owner timer and start its authorized refetch. Quarantine
	// remains set, but this pending slot still needs its own failure-once signal.
	scheduled = false
	if err := correlation.Begin([]byte(`"refetch"`)); err != nil {
		t.Fatal(err)
	}
	if !correlation.Quarantined() {
		t.Fatal("starting refetch cleared quarantine")
	}
	readFailure(true, true)
	readFailure(false, false)
	readOrdinary()
	if requests != 2 || !scheduled || correlation.pendingN != 0 || correlation.retiredN == 0 {
		t.Fatal("refetch failure was unresolved or queued more than once")
	}
	correlation.ReleaseRetired()
	scheduled = false
	if err := correlation.Begin([]byte(`"recovery"`)); err != nil {
		t.Fatal(err)
	}
	msg, err := c.Read()
	if err != nil || msg == nil || string(msg.Result) != `{"requests":[]}` {
		t.Fatalf("guarded recovery failed: %v", err)
	}
	correlation.Complete()
	if correlation.Quarantined() || correlation.refetchQueued || scheduled {
		t.Fatal("validated recovery retained refetch work")
	}
}

func TestHistoryRawGuardRejectsUnapprovedBusyResult(t *testing.T) {
	c := historyTestReader(t, "", &HistoryCorrelation{})
	for _, input := range []string{
		`{"jsonrpc":"2.0","id":1,"error":{"code":-32061,"message":"history-busy"}}`,
		`{"jsonrpc":"2.0","id":1,"error":{"code":-32096,"message":"history-busy"}}`,
		`{"jsonrpc":"2.0","id":1,"error":{"code":-32061,"message":"routing history busy; retry"}}`,
	} {
		var correlation HistoryCorrelation
		if err := correlation.Begin([]byte(`1`)); err != nil {
			t.Fatal(err)
		}
		c.history.correlation = &correlation
		c.history.r.Reset(strings.NewReader(input + "\n"))
		msg, err := c.Read()
		var failure *HistoryFailure
		if msg != nil || !errors.As(err, &failure) || failure.Disposition != "fail-call" || !failure.FailPending || failure.Error() != "history-invalid" {
			t.Fatalf("unapproved busy accepted: %v", err)
		}
	}
}

func TestHistoryRawGuardErrorProjection(t *testing.T) {
	test := func(name, body string, wantBusy bool) {
		t.Run(name, func(t *testing.T) {
			var correlation HistoryCorrelation
			if err := correlation.Begin([]byte(`1`)); err != nil {
				t.Fatal(err)
			}
			c := historyTestReader(t, `{"jsonrpc":"2.0","id":1,"error":`+body+"}\n", &correlation)
			msg, err := c.Read()
			if wantBusy {
				if err != nil || msg == nil || msg.Error == nil || msg.Error.Code != -32096 || msg.Error.Message != "routing history busy; retry" || len(msg.Error.Data) != 0 {
					t.Fatalf("known busy metadata was not preserved: %v", err)
				}
				return
			}
			var failure *HistoryFailure
			if msg != nil || !errors.As(err, &failure) || failure.Disposition != "fail-call" || !failure.FailPending {
				t.Fatalf("invalid known error was rehabilitated by an unknown field: %v", err)
			}
		})
	}
	test("unknown alias cannot repair known message", `{"code":-32096,"message":"unapproved","Message":"routing history busy; retry"}`, false)
	test("unknown alias cannot replace known message", `{"code":-32096,"message":"routing history busy; retry","Message":"unapproved"}`, true)
	test("unknown fields stripped", `{"code":-32096,"message":"routing history busy; retry","unknown":{"private":"discard"}}`, true)
	test("unknown duplicates still rejected", `{"code":-32096,"message":"routing history busy; retry","unknown":{"key":0,"\u006bey":1}}`, false)
}

func TestHistoryPreencodedWriteShortWrites(t *testing.T) {
	var writer oneByteWriter
	c := writeCodec(&writer)
	frame := []byte("{\"jsonrpc\":\"2.0\",\"method\":\"routing/request\",\"params\":{}}\n")
	if err := c.WritePreencoded(nil, context.Background(), frame); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(writer.buf.Bytes(), frame) {
		t.Fatal("short write lost bytes")
	}
	if err := c.Notify("ordinary", nil); err != nil {
		t.Fatal(err)
	}
}

func TestHistoryPreencodedWriteInvalidFrame(t *testing.T) {
	var writer bytes.Buffer
	c := writeCodec(&writer)
	for _, frame := range []string{`{}`, "{}\n{}\n", "invalid\n"} {
		if err := c.WritePreencoded(nil, context.Background(), []byte(frame)); !errors.Is(err, ErrHistoryInvalid) {
			t.Fatalf("error=%v", err)
		}
	}
	if writer.Len() != 0 {
		t.Fatal("invalid frame reached writer")
	}
}

func TestHistoryPreencodedWriteBrokenTransport(t *testing.T) {
	interrupts := 0
	c := NewCodecMaxFrameWithWriteInterrupt(rw{bytes.NewReader(nil), &failingFrameWriter{err: io.ErrClosedPipe}}, 1<<20, func() { interrupts++ })
	if err := c.WritePreencoded(nil, context.Background(), []byte("{}\n")); !errors.Is(err, io.ErrClosedPipe) {
		t.Fatalf("broken write=%v", err)
	}
	if interrupts != 1 {
		t.Fatalf("interrupt calls=%d", interrupts)
	}
	if err := c.WritePreencoded(nil, context.Background(), []byte("{}\n")); !errors.Is(err, ErrPeerClosed) {
		t.Fatalf("broken transport reused: %v", err)
	}
}

// Every wait is a deterministic barrier. Parent cancellation interrupts active
// IO; acquisition cancellation never releases an active borrower's frame.
type historyBarrierWriter struct {
	entered chan struct{}
	release chan struct{}
	written chan []byte
}

func (w *historyBarrierWriter) Write(frame []byte) (int, error) {
	close(w.entered)
	<-w.release
	w.written <- append([]byte(nil), frame...)
	return len(frame), nil
}

func TestHistoryPreencodedWriteAcquisitionLifetime(t *testing.T) {
	w := &historyBarrierWriter{make(chan struct{}), make(chan struct{}), make(chan []byte, 1)}
	c := writeCodec(w)
	acquire, cancelAcquire := context.WithCancel(context.Background())
	defer cancelAcquire()
	frame := []byte("{}\n")
	done := make(chan error, 1)
	go func() { done <- c.WritePreencoded(acquire, context.Background(), frame) }()
	<-w.entered
	cancelAcquire()
	if err := c.WritePreencoded(nil, context.Background(), []byte("{}\n")); !errors.Is(err, ErrHistoryBusy) {
		t.Fatalf("gate not pinned: %v", err)
	}
	waiter, cancelWaiter := context.WithCancel(context.Background())
	cancelWaiter()
	if err := c.WritePreencoded(waiter, context.Background(), []byte("{}\n")); !errors.Is(err, context.Canceled) {
		t.Fatalf("waiter=%v", err)
	}
	select {
	case err := <-done:
		t.Fatalf("borrow released before Write completed: %v", err)
	default:
	}
	close(w.release)
	if err := <-done; err != nil {
		t.Fatalf("acquisition cancellation retired active transport: %v", err)
	}
	if got := <-w.written; !bytes.Equal(got, frame) {
		t.Fatal("borrow changed")
	}
	if c.retired {
		t.Fatal("history gate deadline retired parent")
	}
}

func TestHistoryPreencodedWriteParentCancellationJoinsInterrupt(t *testing.T) {
	w := &historyBarrierWriter{make(chan struct{}), make(chan struct{}), make(chan []byte, 1)}
	interruptEntered, interruptRelease := make(chan struct{}), make(chan struct{})
	c := NewCodecMaxFrameWithWriteInterrupt(rw{bytes.NewReader(nil), w}, 1<<20, func() { close(w.release); close(interruptEntered); <-interruptRelease })
	parent, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- c.WritePreencoded(nil, parent, []byte("{}\n")) }()
	<-w.entered
	cancel()
	<-interruptEntered
	<-w.written
	select {
	case err := <-done:
		t.Fatalf("returned before interrupt joined: %v", err)
	default:
	}
	if err := c.WritePreencoded(nil, context.Background(), []byte("{}\n")); !errors.Is(err, ErrHistoryBusy) {
		t.Fatalf("gate released before interruption: %v", err)
	}
	close(interruptRelease)
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("active result=%v", err)
	}
	if err := c.WritePreencoded(nil, context.Background(), []byte("{}\n")); !errors.Is(err, ErrPeerClosed) {
		t.Fatalf("parent transport not retired: %v", err)
	}
}

func TestHistoryRawGuardKnownDecode(t *testing.T) {
	s, _ := historyTestSchema("", false)
	projected, _, err := ProjectHistory([]byte(`{"schema":1,"extra":null}`), make([]byte, 0, 64), make([]uint32, 64), s)
	if err != nil {
		t.Fatal(err)
	}
	var value historyTestValue
	if err := json.Unmarshal(projected, &value); err != nil {
		t.Fatal(err)
	}
	if value.Schema != 1 {
		t.Fatalf("schema=%d", value.Schema)
	}
}

func historyIncomingTestReader(t *testing.T, input string, correlation *HistoryCorrelation) *Codec {
	t.Helper()
	c := historyTestReader(t, input, correlation)
	c.history.schema = func(method string, initial bool) (HistoryProjection, bool) {
		schema, ok := historyTestSchema(method, initial)
		if !initial && (method == "routing:get-initial" || method == "routing:subscribe" || method == "routing:unsubscribe") {
			schema.Type = reflect.TypeOf(struct{}{})
		}
		return schema, ok
	}
	return c
}

func TestHistoryIncomingRequestIDAdmission(t *testing.T) {
	c := historyIncomingTestReader(t, "", &HistoryCorrelation{})
	test := func(name, id, wantID string, wantCode int) {
		t.Run(name, func(t *testing.T) {
			c.history.r.Reset(strings.NewReader(`{"jsonrpc":"2.0","id":` + id + `,"method":"routing:get-initial","params":{}}` + "\n"))
			msg, err := c.Read()
			if wantCode == 0 {
				if err != nil || msg == nil || !msg.IsRequest() || string(*msg.ID) != wantID || string(msg.Params) != `{}` {
					t.Fatalf("legal ID read=%v err=%v", msg, err)
				}
				return
			}
			var failure *HistoryFailure
			if msg != nil || !errors.As(err, &failure) || failure.Disposition != "refuse-request" || failure.Refusal == nil || failure.FailPending || failure.ScheduleRefetch {
				t.Fatalf("refusal=%+v err=%v", failure, err)
			}
			if failure.Refusal.Code != wantCode || failure.Refusal.Message != "history-invalid" || string(failure.Refusal.ReplyID) != wantID || failure.Error() != "history-invalid" {
				t.Fatalf("fixed refusal=%+v", failure.Refusal)
			}
			if &failure.Refusal.ReplyID[0] != &c.history.idToken[0] || len(failure.Refusal.ReplyID) > 512 {
				t.Fatal("refusal ID does not borrow the bounded control slot")
			}
		})
	}
	id512 := `"` + strings.Repeat("a", 510) + `"`
	escaped512 := `"` + strings.Repeat(`\u0061`, 85) + `"`
	test("512 encoded bytes", id512, id512, 0)
	test("512 escaped encoded bytes", escaped512, escaped512, 0)
	test("existing numeric token policy", "-1.5e+2", "-1.5e+2", 0)
	test("513 encoded bytes", `"`+strings.Repeat("a", 511)+`"`, "null", -32062)
	test("513 escaped bytes but shorter decoded ID", `"`+strings.Repeat(`\u0061`, 85)+`x"`, "null", -32062)
	test("large encoded ID", `"`+strings.Repeat("synthetic-id-secret", 4096)+`"`, "null", -32062)
	test("over-cap numeric token", strings.Repeat("1", 513), "null", -32062)
}

func TestHistoryIncomingRequestFixedRefusals(t *testing.T) {
	c := historyIncomingTestReader(t, "", &HistoryCorrelation{})
	test := func(name, method, id, params, wantID string, wantCode int, wantMessage string) {
		t.Run(name, func(t *testing.T) {
			frame := `{"jsonrpc":"2.0","id":` + id + `,"method":"` + method + `"` + params + `}` + "\n"
			c.history.r.Reset(strings.NewReader(frame))
			var writes bytes.Buffer
			c.writer = &writes
			msg, err := c.Read()
			var failure *HistoryFailure
			if msg != nil || !errors.As(err, &failure) || failure.Refusal == nil || failure.Disposition != "refuse-request" || failure.FailPending || failure.ScheduleRefetch {
				t.Fatalf("refusal=%+v err=%v", failure, err)
			}
			refusal := failure.Refusal
			if string(refusal.ReplyID) != wantID || refusal.Code != wantCode || refusal.Message != wantMessage || failure.Error() != "history-invalid" || writes.Len() != 0 {
				t.Fatalf("fixed refusal=%+v writes=%d", refusal, writes.Len())
			}
			// The caller owns this independently funded control reply before
			// another Read. Raw ID spelling bypasses generic outer marshaling.
			errorRaw, err := json.Marshal(&RPCError{Code: refusal.Code, Message: refusal.Message})
			if err != nil {
				t.Fatal(err)
			}
			reply := append([]byte(`{"jsonrpc":"2.0","id":`), refusal.ReplyID...)
			reply = append(reply, `,"error":`...)
			reply = append(reply, errorRaw...)
			reply = append(reply, '}', '\n')
			if len(reply) > 4096 || bytes.Contains(reply, []byte("synthetic-secret")) {
				t.Fatal("refusal exceeded E or leaked rejected data")
			}
			var decoded struct {
				JSONRPC string          `json:"jsonrpc"`
				ID      json.RawMessage `json:"id"`
				Error   *RPCError       `json:"error"`
				Method  string          `json:"method"`
				Params  json.RawMessage `json:"params"`
				Result  json.RawMessage `json:"result"`
			}
			if err := json.Unmarshal(reply, &decoded); err != nil {
				t.Fatal(err)
			}
			if decoded.JSONRPC != "2.0" || string(decoded.ID) != wantID || decoded.Error == nil || decoded.Error.Code != wantCode || decoded.Error.Message != wantMessage || decoded.Method != "" || decoded.Params != nil || decoded.Result != nil {
				t.Fatalf("encoded refusal=%+v", decoded)
			}
		})
	}
	id := `"\u0061<>&"`
	test("null public params", "routing:get-initial", id, `,"params":null`, id, -32062, "history-invalid")
	test("scalar public params", "routing:subscribe", id, `,"params":"synthetic-secret"`, id, -32062, "history-invalid")
	test("array public params", "routing:unsubscribe", id, `,"params":[]`, id, -32062, "history-invalid")
	test("missing public params", "routing:get-initial", id, "", id, -32062, "history-invalid")
	test("public params depth admission", "routing:subscribe", id, `,"params":`+strings.Repeat("[", 20)+`0`+strings.Repeat("]", 20), id, -32062, "history-invalid")
	test("public params byte admission", "routing:subscribe", id, `,"params":{"secret":"`+strings.Repeat("x", HistoryRecordBytes)+`"}`, id, -32062, "history-invalid")
	test("public envelope admission", "routing:subscribe", id, `,"params":{},"secret":"`+strings.Repeat("x", HistoryEnvelopeBytes)+`"`, id, -32062, "history-invalid")
	test("maximum raw reply ID", "routing:subscribe", `"`+strings.Repeat(`\u0061`, 85)+`"`, `,"params":null`, `"`+strings.Repeat(`\u0061`, 85)+`"`, -32062, "history-invalid")
	test("private record call", "routing/request", id, `,"params":{"synthetic-secret":0}`, id, -32601, "method not found")
	test("private window call", "routing/history-window", id, `,"params":{}`, id, -32601, "method not found")
	test("private end call", "routing/history-window-end", id, `,"params":{}`, id, -32601, "method not found")
	test("private accepted call", "routing/history-accepted", id, `,"params":{}`, id, -32601, "method not found")
	test("private changed call", "routing:changed", id, `,"params":null`, id, -32601, "method not found")
	test("private call without params", "routing/request", id, "", id, -32601, "method not found")
	test("private over-cap ID", "routing/request", `"`+strings.Repeat("a", 511)+`"`, `,"params":{}`, "null", -32062, "history-invalid")
}

func TestHistoryIncomingRequestNoUnsafeReplyAttribution(t *testing.T) {
	c := historyIncomingTestReader(t, "", &HistoryCorrelation{})
	test := func(name, frame, disposition string) {
		t.Run(name, func(t *testing.T) {
			correlation := HistoryCorrelation{}
			c.history.correlation = &correlation
			c.history.r.Reset(strings.NewReader(frame + "\n"))
			msg, err := c.Read()
			var failure *HistoryFailure
			if msg != nil || !errors.As(err, &failure) || failure.Disposition != disposition || failure.Refusal != nil || failure.FailPending || failure.Error() != "history-invalid" {
				t.Fatalf("unsafe attribution=%+v err=%v", failure, err)
			}
		})
	}
	test("notification", `{"jsonrpc":"2.0","method":"routing:subscribe","params":null}`, "discard")
	test("private notification", `{"jsonrpc":"2.0","method":"routing/request","params":null}`, "discard")
	test("null ID", `{"jsonrpc":"2.0","id":null,"method":"routing:subscribe","params":null}`, "discard")
	test("object ID", `{"jsonrpc":"2.0","id":{},"method":"routing:subscribe","params":null}`, "discard")
	test("wrong version", `{"jsonrpc":"1.0","id":1,"method":"routing:subscribe","params":null}`, "discard")
	test("missing version", `{"id":1,"method":"routing:subscribe","params":null}`, "discard")
	test("request with result", `{"jsonrpc":"2.0","id":1,"method":"routing:subscribe","params":null,"result":{}}`, "discard")
	test("request with error", `{"jsonrpc":"2.0","id":1,"method":"routing:subscribe","params":null,"error":null}`, "discard")
	test("duplicate version alias", `{"jsonrpc":"2.0","JSONRPC":"2.0","id":1,"method":"routing:subscribe","params":null}`, "discard")
	test("duplicate params alias", `{"jsonrpc":"2.0","id":1,"method":"routing:subscribe","params":null,"Params":null}`, "discard")
	test("duplicate unknown outer", `{"jsonrpc":"2.0","id":1,"method":"routing:subscribe","params":null,"secret":0,"secret":1}`, "discard")
	test("duplicate unknown nested", `{"jsonrpc":"2.0","id":1,"method":"routing:subscribe","params":{"secret":{"a":0,"\u0061":1}}}`, "discard")
	test("duplicate beyond accepted params depth", `{"jsonrpc":"2.0","id":1,"method":"routing:subscribe","params":`+strings.Repeat("[", 20)+`{"a":0,"a":1}`+strings.Repeat("]", 20)+`}`, "discard")
	test("duplicate before over-cap refusal", `{"jsonrpc":"2.0","id":"`+strings.Repeat("a", 511)+`","method":"routing:subscribe","params":{"a":0,"a":1}}`, "discard")
	test("duplicate IDs", `{"jsonrpc":"2.0","id":1,"ID":2,"method":"routing:subscribe","params":null}`, "quarantine")
	test("duplicate IDs reversed", `{"jsonrpc":"2.0","ID":2,"id":1,"method":"routing:subscribe","params":null}`, "quarantine")
	test("history method first", `{"jsonrpc":"2.0","id":1,"method":"routing:subscribe","Method":"ordinary","params":null}`, "quarantine")
	test("history method last", `{"jsonrpc":"2.0","id":1,"Method":"ordinary","method":"routing:subscribe","params":null}`, "quarantine")
	test("unclassifiable syntax", `{"synthetic-secret":`, "quarantine")
}

func TestHistoryIncomingRequestCollisionDoesNotSettleOutbound(t *testing.T) {
	c := historyIncomingTestReader(t, "", &HistoryCorrelation{})
	test := func(name, frame, wantMethod, wantDisposition string) {
		t.Run(name, func(t *testing.T) {
			var correlation HistoryCorrelation
			if err := correlation.Begin([]byte(`"same"`)); err != nil {
				t.Fatal(err)
			}
			c.history.correlation = &correlation
			c.history.r.Reset(strings.NewReader(frame + "\n" + `{"jsonrpc":"2.0","id":"same","result":{"requests":[]}}` + "\n"))
			msg, err := c.Read()
			if wantDisposition == "" {
				if err != nil || msg == nil || !msg.IsRequest() || msg.Method != wantMethod {
					t.Fatalf("incoming request=%v err=%v", msg, err)
				}
			} else {
				var failure *HistoryFailure
				if msg != nil || !errors.As(err, &failure) || failure.Disposition != wantDisposition || failure.FailPending || failure.ScheduleRefetch {
					t.Fatalf("incoming refusal settled outbound: %+v err=%v", failure, err)
				}
			}
			if correlation.pendingN != len(`"same"`) || correlation.retiredN != 0 || correlation.Quarantined() {
				t.Fatal("incoming ID collision changed outbound ownership")
			}
			response, err := c.Read()
			if err != nil || response == nil || !response.IsResponse() || string(*response.ID) != `"same"` || string(response.Result) != `{"requests":[]}` {
				t.Fatalf("genuine outbound response=%v err=%v", response, err)
			}
		})
	}
	test("public request", `{"jsonrpc":"2.0","id":"same","method":"routing:get-initial","params":{}}`, "routing:get-initial", "")
	test("invalid public params", `{"jsonrpc":"2.0","id":"same","method":"routing:get-initial","params":null}`, "", "refuse-request")
	test("private request", `{"jsonrpc":"2.0","id":"same","method":"routing/request","params":{}}`, "", "refuse-request")
	test("ordinary request", `{"jsonrpc":"2.0","id":"same","method":"ordinary","params":{}}`, "ordinary", "")
	test("ordinary method with response fields", `{"jsonrpc":"2.0","id":"same","method":"ordinary","result":{"requests":[]}}`, "", "discard")
	test("null method cannot become response", `{"jsonrpc":"2.0","id":"same","method":null,"result":{"requests":[]}}`, "", "discard")
	test("empty method cannot become response", `{"jsonrpc":"2.0","id":"same","method":"","result":{"requests":[]}}`, "", "discard")
}

func TestHistoryIncomingRequestRefusalBorrowAndOrdinaryProgress(t *testing.T) {
	id := `"\u0061<>&"`
	ordinaryID := `"` + strings.Repeat("ordinary-id", 100) + `"`
	input := `{"jsonrpc":"2.0","id":` + id + `,"method":"routing:subscribe","params":null}` + "\n" +
		`{"jsonrpc":"2.0","id":` + ordinaryID + `,"method":"ordinary","params":{}}` + "\n" +
		`{"jsonrpc":"2.0","id":"next","method":"routing:unsubscribe","params":null}` + "\n"
	c := historyIncomingTestReader(t, input, &HistoryCorrelation{})
	_, err := c.Read()
	var failure *HistoryFailure
	if !errors.As(err, &failure) || failure.Refusal == nil {
		t.Fatalf("first refusal=%v", err)
	}
	ownedID := append([]byte(nil), failure.Refusal.ReplyID...)
	ordinary, err := c.Read()
	if err != nil || ordinary == nil || !ordinary.IsRequest() || ordinary.Method != "ordinary" || string(*ordinary.ID) != ordinaryID {
		t.Fatalf("ordinary traffic/ID cap changed: %v err=%v", ordinary, err)
	}
	_, err = c.Read()
	var next *HistoryFailure
	if !errors.As(err, &next) || next.Refusal == nil || string(next.Refusal.ReplyID) != `"next"` || string(ownedID) != id {
		t.Fatalf("funded ID copy changed: %v err=%v", ownedID, err)
	}
}

func TestHistoryIncomingRefusalDoesNotReplyToResponse(t *testing.T) {
	var correlation HistoryCorrelation
	if err := correlation.Begin([]byte("1")); err != nil {
		t.Fatal(err)
	}
	c := historyIncomingTestReader(t, `{"jsonrpc":"2.0","id":1,"result":{"requests":null}}`+"\n", &correlation)
	msg, err := c.Read()
	var failure *HistoryFailure
	if msg != nil || !errors.As(err, &failure) || failure.Disposition != "fail-call" || !failure.FailPending || failure.Refusal != nil {
		t.Fatalf("response became incoming refusal: %+v err=%v", failure, err)
	}
	if correlation.pendingN != 0 || correlation.retiredN != 1 {
		t.Fatal("ordinary pending failure ownership changed")
	}
}

func TestHistoryIncomingOverlongEquivalentIDKeepsOutbound(t *testing.T) {
	id := `"` + strings.Repeat("a", 100) + `"`
	encoded := `"` + strings.Repeat(`\u0061`, 100) + `"`
	var correlation HistoryCorrelation
	if err := correlation.Begin([]byte(id)); err != nil {
		t.Fatal(err)
	}
	input := `{"jsonrpc":"2.0","id":` + encoded + `,"method":"routing:get-initial","params":{}}` + "\n" +
		`{"jsonrpc":"2.0","id":` + id + `,"result":{"requests":[]}}` + "\n"
	c := historyIncomingTestReader(t, input, &correlation)
	msg, err := c.Read()
	var failure *HistoryFailure
	if msg != nil || !errors.As(err, &failure) || failure.Refusal == nil || string(failure.Refusal.ReplyID) != "null" || failure.Refusal.Code != -32062 || failure.FailPending || failure.ScheduleRefetch {
		t.Fatalf("incoming over-cap collision=%+v err=%v", failure, err)
	}
	if correlation.pendingN != len(id) || correlation.retiredN != 0 || correlation.Quarantined() {
		t.Fatal("over-cap incoming request retired outbound initial")
	}
	response, err := c.Read()
	if err != nil || response == nil || !response.IsResponse() || string(*response.ID) != id || string(response.Result) != `{"requests":[]}` {
		t.Fatalf("genuine outbound response=%v err=%v", response, err)
	}
}

func TestHistoryReadOwnershipDiscriminator(t *testing.T) {
	test := func(name, frame string, pending, borrowed bool) {
		t.Run(name, func(t *testing.T) {
			var correlation HistoryCorrelation
			if pending {
				if err := correlation.Begin([]byte("1")); err != nil {
					t.Fatal(err)
				}
			}
			c := historyIncomingTestReader(t, frame+"\n", &correlation)
			msg, err := c.Read()
			if err != nil || msg == nil || msg.borrowedRead != borrowed {
				t.Fatalf("read ownership = %v, want borrowed %v (err %v)", msg, borrowed, err)
			}
			encoded, err := json.Marshal(msg)
			if err != nil {
				t.Fatal(err)
			}
			var wire map[string]json.RawMessage
			if err := json.Unmarshal(encoded, &wire); err != nil {
				t.Fatal(err)
			}
			for key := range wire {
				switch key {
				case "jsonrpc", "id", "method", "params", "result", "error":
				default:
					t.Fatalf("internal ownership leaked onto wire as %s", key)
				}
			}
		})
	}
	test("ordinary owned", `{"jsonrpc":"2.0","id":9,"method":"ordinary","params":{}}`, false, false)
	test("projected request", `{"jsonrpc":"2.0","id":9,"method":"routing:subscribe","params":{}}`, false, true)
	test("projected notification", `{"jsonrpc":"2.0","method":"routing/request","params":{"schema":1}}`, false, true)
	test("projected result", `{"jsonrpc":"2.0","id":1,"result":{"requests":[]}}`, true, true)
	test("projected error ID", `{"jsonrpc":"2.0","id":1,"error":{"code":-32096,"message":"routing history busy; retry"}}`, true, true)
}
