// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"testing"

	settings "nvpair-shared/enginesettings"
)

func TestSwitchRelayCorrelatesCodedReply(t *testing.T) {
	for _, code := range []int{-32010, -32011, -32000} {
		t.Run(fmt.Sprint(code), func(t *testing.T) {
			r := &switchRelay{}
			r.send = func(method string, value any) error {
				request, ok := value.(switchRelayRequest)
				if !ok || method != "engine:switch-request" || request.Caller != "pinned-peer" || request.Method != settings.SwitchMethod {
					t.Fatalf("relay request: %s %+v", method, value)
				}
				raw, err := json.Marshal(switchRelayReply{ID: request.ID, Error: "refused", ErrorCode: code})
				if err != nil {
					return err
				}
				r.reply(raw)
				return nil
			}
			_, err := r.call(context.Background(), settings.SwitchMethod, "pinned-peer", json.RawMessage(`{}`))
			if rpcCode(err) != code {
				t.Fatalf("code = %d, want %d (%v)", rpcCode(err), code, err)
			}
			if len(r.pending) != 0 {
				t.Fatal("pending call leaked")
			}
		})
	}
}

func TestSwitchRelayCancellationRetiresOnlyWaiter(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	r := &switchRelay{}
	var sent []string
	r.send = func(method string, _ any) error {
		sent = append(sent, method)
		if method == "engine:switch-request" {
			cancel()
		}
		return nil
	}
	_, err := r.call(ctx, settings.SwitchMethod, "peer", json.RawMessage(`{}`))
	if !errors.Is(err, context.Canceled) || len(r.pending) != 0 {
		t.Fatalf("cancel = %v; pending = %d", err, len(r.pending))
	}
	if len(sent) != 2 || sent[0] != "engine:switch-request" || sent[1] != "engine:switch-cancel" {
		t.Fatalf("sent = %v", sent)
	}
}

func TestSwitchRelayBoundsPendingAndIgnoresUncorrelatedReplies(t *testing.T) {
	r := &switchRelay{pending: map[string]chan switchRelayReply{}, send: func(string, any) error { t.Fatal("full relay sent request"); return nil }}
	for i := 0; i < maxPendingSwitchCalls; i++ {
		r.pending[fmt.Sprint(i)] = make(chan switchRelayReply, 1)
	}
	_, err := r.call(context.Background(), settings.SwitchStatusMethod, "peer", json.RawMessage(`{}`))
	var busy *switchBusyError
	if !errors.As(err, &busy) {
		t.Fatalf("full relay = %v", err)
	}
	r.reply(json.RawMessage(`{"id":"missing","result":{}}`))
	r.reply(json.RawMessage(`not json`))
	if len(r.pending) != maxPendingSwitchCalls {
		t.Fatal("uncorrelated reply changed pending calls")
	}
}

func TestSwitchRelaySendFailureAndSuccessReleaseSlot(t *testing.T) {
	r := &switchRelay{send: func(string, any) error { return errors.New("pipe closed") }}
	if _, err := r.call(context.Background(), settings.SwitchStatusMethod, "peer", json.RawMessage(`{}`)); err == nil || len(r.pending) != 0 {
		t.Fatalf("send failure = %v, pending=%d", err, len(r.pending))
	}
	r.send = func(_ string, value any) error {
		request, ok := value.(switchRelayRequest)
		if !ok {
			return errors.New("unexpected relay payload")
		}
		raw, err := json.Marshal(switchRelayReply{ID: request.ID, Result: json.RawMessage(`{"activeRequestId":"test"}`)})
		if err != nil {
			return err
		}
		r.reply(raw)
		return nil
	}
	raw, err := r.call(context.Background(), settings.SwitchStatusMethod, "peer", json.RawMessage(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	var status settings.SwitchStatus
	if err := json.Unmarshal(raw, &status); err != nil || status.ActiveRequestID != "test" {
		t.Fatalf("status = %+v, %v", status, err)
	}
}
