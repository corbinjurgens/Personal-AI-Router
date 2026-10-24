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
)

type historyTestValue struct {
	Schema int64 `json:"schema"`
}
type historyTestInitial struct {
	Requests []historyTestValue `json:"requests"`
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
	test := func(name, frame string) {
		t.Run(name, func(t *testing.T) {
			_, err := historyTestReader(t, frame+"\n", &HistoryCorrelation{}).Read()
			var failure *HistoryFailure
			if !errors.As(err, &failure) || failure.Category != "history-invalid" || failure.Disposition != "discard" {
				t.Fatalf("error=%v", err)
			}
		})
	}
	test("record R plus one", `{"jsonrpc":"2.0","method":"routing/request","params":{"schema":1,"x":"`+strings.Repeat("x", HistoryRecordBytes)+`"}}`)
	test("envelope E plus one", `{"jsonrpc":"2.0","method":"routing/request","params":{"schema":1},"x":"`+strings.Repeat("x", HistoryEnvelopeBytes)+`"}`)
	test("ID 513", `{"jsonrpc":"2.0","method":"routing:get-initial","id":"`+strings.Repeat("a", 511)+`","params":{}}`)
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
