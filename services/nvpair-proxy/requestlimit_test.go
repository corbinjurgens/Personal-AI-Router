// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

// TestOversizedRequestIsRejectedBeforeDispatch: the proxy buffers each request
// so failover can replay it, so an unbounded body is an unbounded allocation.
// A body over the cap must get 413 and never reach an engine; one under it must
// be forwarded intact.
func TestOversizedRequestIsRejectedBeforeDispatch(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		var hits atomic.Int32
		var lastLen atomic.Int64
		upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			hits.Add(1)
			lastLen.Store(r.ContentLength)
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"done":true}`))
		}))
		defer upstream.Close()

		disc := NewDiscovery()
		disc.AddManual(nodeForModel(t, "serving-node", upstream.URL, tc.advertisedModel))
		p := newTestProxy(tc.profile, NewCodec(rwNop{}), disc, tc.profile.FacadePort)
		p.maxRequestBytes = 256

		body := func(n int) string {
			return fmt.Sprintf(`{"model":%q,"prompt":%q}`, tc.requestedModel, strings.Repeat("x", n))
		}

		rec := httptest.NewRecorder()
		p.soleFacade().handleHTTP(rec, httptest.NewRequest(http.MethodPost, tc.inferencePath, strings.NewReader(body(1024))))
		if rec.Code != http.StatusRequestEntityTooLarge {
			t.Fatalf("oversized status = %d, want 413", rec.Code)
		}
		if hits.Load() != 0 {
			t.Fatal("an oversized request must not be dispatched to an engine")
		}

		small := body(16)
		rec = httptest.NewRecorder()
		p.soleFacade().handleHTTP(rec, httptest.NewRequest(http.MethodPost, tc.inferencePath, strings.NewReader(small)))
		if rec.Code != http.StatusOK {
			t.Fatalf("in-limit status = %d, want 200", rec.Code)
		}
		if hits.Load() != 1 || lastLen.Load() != int64(len(small)) {
			t.Fatalf("hits = %d, forwarded length = %d; want 1 request of %d bytes", hits.Load(), lastLen.Load(), len(small))
		}
	})
}

func TestRequestBodyLimitDefaultsWhenUnset(t *testing.T) {
	if got := (&Proxy{}).requestBodyLimit(); got != defaultMaxRequestBytes {
		t.Fatalf("requestBodyLimit() = %d, want default %d", got, defaultMaxRequestBytes)
	}
}
