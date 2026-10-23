// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"errors"
	"io"
	"testing"
	"testing/synctest"
	"time"

	"nvpair-shared/nodepolicy"
)

type routingControlWriter struct {
	replies chan *Message
	fail    bool
}

func (w *routingControlWriter) Read([]byte) (int, error) { return 0, io.EOF }
func (w *routingControlWriter) Write(data []byte) (int, error) {
	if w.fail {
		return 0, errors.New("response write failed")
	}
	var msg Message
	if err := json.Unmarshal(data, &msg); err != nil {
		return 0, err
	}
	w.replies <- &msg
	return len(data), nil
}

func routingBrokerRequest(id, method, params string) *Message {
	raw := json.RawMessage(id)
	return &Message{ID: &raw, Method: method, Params: json.RawMessage(params)}
}

const routingBrokerCancelParams = `{"request":{"originNodeId":"self-node","proxyRunId":"run-a","requestId":"9223372036854775807"},"regenerate":true}`

func TestRoutingCancelBrokerRelay(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		w := &routingControlWriter{replies: make(chan *Message, 4)}
		b := &Broker{codec: NewCodec(w), nodeID: "self-node"}
		_, worker := attachFakeProxy(t, b)
		worker.setAnswer(func(method string, params json.RawMessage) (any, string) {
			switch method {
			case "routing/cancel":
				var in nodepolicy.RoutingCancelParams
				if err := json.Unmarshal(params, &in); err != nil {
					t.Error(err)
				}
				if in.Request.RequestID != "9223372036854775807" || !in.Regenerate {
					t.Errorf("params=%s", params)
				}
				return nodepolicy.RoutingCancelResult{Status: "conflict_committed"}, ""
			case "routing/active":
				return nodepolicy.RoutingActiveResult{Requests: []nodepolicy.RoutingRequestKey{{OriginNodeID: "self-node", ProxyRunID: "run-a", RequestID: "9"}}}, ""
			default:
				t.Errorf("unexpected method %s", method)
				return nil, "unexpected method"
			}
		})
		b.handleMessage(routingBrokerRequest("1", "routing:cancel", routingBrokerCancelParams))
		out := <-w.replies
		var cancelled nodepolicy.RoutingCancelResult
		if out.Error != nil || json.Unmarshal(out.Result, &cancelled) != nil || cancelled.Status != "conflict_committed" {
			t.Fatalf("cancel response=%+v", out)
		}
		synctest.Wait()
		b.handleMessage(routingBrokerRequest("2", "routing:active", `{}`))
		out = <-w.replies
		var active nodepolicy.RoutingActiveResult
		if out.Error != nil || json.Unmarshal(out.Result, &active) != nil || len(active.Requests) != 1 || active.Requests[0].RequestID != "9" {
			t.Fatalf("active response=%+v", out)
		}
		synctest.Wait()
		if b.routingForward.Load() || len(worker.snapshot()) != 2 {
			t.Fatal("relay retained gate or sent extra calls")
		}
	})
}

func TestRoutingCancelBrokerRejectsSourceParams(t *testing.T) {
	w := &routingControlWriter{replies: make(chan *Message, 1)}
	b := &Broker{codec: NewCodec(w), nodeID: "self-node"}
	test := func(name, method, params string) {
		t.Run(name, func(t *testing.T) {
			b.handleMessage(routingBrokerRequest("1", method, params))
			out := <-w.replies
			if out.Error == nil || out.Error.Code != -32602 || b.routingForward.Load() {
				t.Fatalf("source accepted: %+v", out)
			}
		})
	}
	test("other origin", "routing:cancel", `{"request":{"originNodeId":"peer","proxyRunId":"run-a","requestId":"1"},"regenerate":false}`)
	test("missing regenerate", "routing:cancel", `{"request":{"originNodeId":"self-node","proxyRunId":"run-a","requestId":"1"}}`)
	test("duplicate key", "routing:cancel", `{"request":{"originNodeId":"self-node","proxyRunId":"run-a","requestId":"1","requestId":"2"},"regenerate":false}`)
	test("active extra field", "routing:active", `{"originNodeId":"peer"}`)
}

func TestRoutingCancelBrokerSingleFlightAndBound(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		w := &routingControlWriter{replies: make(chan *Message, 3)}
		b := &Broker{codec: NewCodec(w), nodeID: "self-node"}
		_, worker := attachFakeProxy(t, b)
		worker.setAnswer(func(string, json.RawMessage) (any, string) { return nil, holdReply })
		start := time.Now()
		b.handleMessage(routingBrokerRequest("1", "routing:cancel", routingBrokerCancelParams))
		synctest.Wait()
		if !b.routingForward.Load() || len(worker.snapshot()) != 1 {
			t.Fatal("first forward not waiting at proxy")
		}
		b.handleMessage(routingBrokerRequest("2", "routing:active", `{}`))
		out := <-w.replies
		if out.Error == nil || out.Error.Code != -32000 || string(*out.ID) != "2" || len(worker.snapshot()) != 1 || time.Since(start) != 0 {
			t.Fatal("second forward was not promptly refused")
		}
		out = <-w.replies
		if out.Error == nil || out.Error.Code != -32000 || string(*out.ID) != "1" || time.Since(start) != 5*time.Second {
			t.Fatalf("bound response=%+v elapsed=%s", out, time.Since(start))
		}
		synctest.Wait()
		if b.routingForward.Load() {
			t.Fatal("timeout retained forwarding gate")
		}
	})
}

func TestRoutingCancelBrokerResultValidationAndRelease(t *testing.T) {
	test := func(name, method string, result any, workerError string) {
		t.Run(name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				w := &routingControlWriter{replies: make(chan *Message, 2)}
				b := &Broker{codec: NewCodec(w), nodeID: "self-node"}
				_, worker := attachFakeProxy(t, b)
				worker.setAnswer(func(string, json.RawMessage) (any, string) { return result, workerError })
				params := `{}`
				if method == "routing:cancel" {
					params = routingBrokerCancelParams
				}
				b.handleMessage(routingBrokerRequest("1", method, params))
				out := <-w.replies
				if out.Error == nil || out.Error.Code != -32000 {
					t.Fatalf("invalid result accepted: %+v", out)
				}
				synctest.Wait()
				if b.routingForward.Load() {
					t.Fatal("error retained forwarding gate")
				}
				worker.setAnswer(func(string, json.RawMessage) (any, string) {
					return nodepolicy.RoutingActiveResult{Requests: []nodepolicy.RoutingRequestKey{}}, ""
				})
				b.handleMessage(routingBrokerRequest("2", "routing:active", `{}`))
				out = <-w.replies
				if out.Error != nil {
					t.Fatalf("gate not reusable: %+v", out.Error)
				}
			})
		})
	}
	test("unknown status", "routing:cancel", json.RawMessage(`{"status":"ok"}`), "")
	test("duplicate status", "routing:cancel", json.RawMessage(`{"status":"cancelled","status":"not_found"}`), "")
	test("null active", "routing:active", json.RawMessage(`{"requests":null}`), "")
	test("foreign active key", "routing:active", json.RawMessage(`{"requests":[{"originNodeId":"peer","proxyRunId":"run","requestId":"1"}]}`), "")
	test("malformed active key", "routing:active", json.RawMessage(`{"requests":[{"originNodeId":"self-node","proxyRunId":"run","requestId":"01"}]}`), "")
	test("worker control error", "routing:active", nil, "busy")
}

func TestRoutingCancelBrokerUnavailableAndWriteErrorRelease(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		w := &routingControlWriter{replies: make(chan *Message, 1), fail: true}
		b := &Broker{codec: NewCodec(w), nodeID: "self-node"}
		b.handleMessage(routingBrokerRequest("1", "routing:active", `{}`))
		synctest.Wait()
		if b.routingForward.Load() {
			t.Fatal("unavailable proxy / response error retained gate")
		}
		w.fail = false
		b.handleMessage(routingBrokerRequest("2", "routing:active", `{}`))
		out := <-w.replies
		if out.Error == nil || out.Error.Code != -32000 {
			t.Fatal("unavailable proxy was not reported")
		}
	})
}

func TestRoutingCancelBrokerTransportErrorRelease(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		w := &routingControlWriter{replies: make(chan *Message, 1)}
		b := &Broker{codec: NewCodec(w), nodeID: "self-node"}
		publishTestProxy(b, &proxyProcess{})
		b.handleMessage(routingBrokerRequest("1", "routing:active", `{}`))
		out := <-w.replies
		if out.Error == nil || out.Error.Code != -32000 {
			t.Fatal("transport failure not reported")
		}
		synctest.Wait()
		if b.routingForward.Load() {
			t.Fatal("transport error retained gate")
		}
	})
}

func TestRoutingCancelBrokerSuccessWriteErrorRelease(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		w := &routingControlWriter{replies: make(chan *Message, 1), fail: true}
		b := &Broker{codec: NewCodec(w), nodeID: "self-node"}
		_, worker := attachFakeProxy(t, b)
		worker.setAnswer(func(string, json.RawMessage) (any, string) {
			return nodepolicy.RoutingActiveResult{Requests: []nodepolicy.RoutingRequestKey{}}, ""
		})
		b.handleMessage(routingBrokerRequest("1", "routing:active", `{}`))
		synctest.Wait()
		if b.routingForward.Load() || len(worker.snapshot()) != 1 {
			t.Fatal("successful response write failure retained gate")
		}
		w.fail = false
		b.handleMessage(routingBrokerRequest("2", "routing:active", `{}`))
		out := <-w.replies
		if out.Error != nil {
			t.Fatalf("retry failed: %+v", out.Error)
		}
	})
}
