// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

// TestJoinRequestRetryAcrossRenameKeepsRequesterName renames the requester
// between two attempts of the same outgoing request. The first endpoint
// records the request at the target and then answers 404, so the requester
// retries the second endpoint; the retry must carry the name captured when
// the request was created, or the target rejects it as conflicting-request.
func TestJoinRequestRetryAcrossRenameKeepsRequesterName(t *testing.T) {
	m := joinTestManager(t)
	target := joinTestManager(t)
	if _, _, err := target.foundCluster("admitting cluster"); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	m.joinContext = ctx

	var mu sync.Mutex
	var statuses, names []string
	apply := func(w http.ResponseWriter, req *http.Request) (joinRequestReply, int, string, bool) {
		var wire joinRequestWire
		if err := json.NewDecoder(req.Body).Decode(&wire); err != nil {
			t.Error(err)
			return joinRequestReply{}, 0, "", false
		}
		if wire.Operation != "request" {
			w.WriteHeader(http.StatusNotFound)
			return joinRequestReply{}, 0, "", false
		}
		reply, status, reason := target.applyJoinRequestWire("127.0.0.1", wire)
		mu.Lock()
		statuses = append(statuses, http.StatusText(status)+" "+reason)
		names = append(names, wire.Requester.Name)
		mu.Unlock()
		return reply, status, reason, true
	}
	retried := make(chan struct{})
	first := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if _, _, _, ok := apply(w, req); !ok {
			return
		}
		renamed := "Renamed requester"
		if _, err := m.setSelfName(&renamed); err != nil {
			t.Error(err)
		}
		w.WriteHeader(http.StatusNotFound)
	}))
	defer first.Close()
	second := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		reply, status, _, ok := apply(w, req)
		if !ok {
			return
		}
		w.WriteHeader(status)
		if err := json.NewEncoder(w).Encode(reply); err != nil {
			t.Error(err)
		}
		close(retried)
	}))
	defer second.Close()

	original := m.selfName()
	r := joinTestRecord(t, m, "outgoing")
	m.joinMu.Lock()
	r.requesterName = original
	m.joinMu.Unlock()
	done := make(chan struct{})
	go func() {
		m.driveJoinRequest(r.RequestID, []string{strings.TrimPrefix(first.URL, "http://"), strings.TrimPrefix(second.URL, "http://")})
		close(done)
	}()
	select {
	case <-retried:
	case <-time.After(5 * time.Second):
		t.Fatal("request was not retried on the second endpoint")
	}
	cancel()
	<-done

	mu.Lock()
	defer mu.Unlock()
	if len(statuses) != 2 || statuses[1] != "OK " {
		t.Fatalf("target outcomes = %q, want the retry accepted", statuses)
	}
	if names[0] != original || names[1] != original {
		t.Fatalf("requester names = %q, want %q on every attempt", names, original)
	}
	if m.selfName() == original {
		t.Fatal("the requester was not renamed between attempts")
	}
}
