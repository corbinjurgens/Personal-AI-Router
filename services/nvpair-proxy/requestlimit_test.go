// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestOwnedInferenceBuffersStartupAcceptsEffectiveBodyLimit(t *testing.T) {
	test := func(name string, configured, want int64) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			codec := NewCodec(rwNop{})
			p, err := initializeProxy(codec, configured)
			if err != nil {
				t.Fatalf("initializeProxy() error = %v", err)
			}
			if p == nil {
				t.Fatal("initializeProxy() returned no proxy")
			}
			if p.maxRequestBytes != configured {
				t.Fatalf("stored CLI limit = %d, want %d", p.maxRequestBytes, configured)
			}
			if got := p.requestBodyLimit(); got != want {
				t.Fatalf("effective body limit = %d, want %d", got, want)
			}
			if p.buffers == nil {
				t.Fatal("startup did not retain the configured buffer root")
			}
			if p.buffers.capacity != 256<<20 || p.buffers.wait != 5*time.Second {
				t.Fatalf("startup root capacity/wait = %d/%s, want %d/5s", p.buffers.capacity, p.buffers.wait, 256<<20)
			}
			if p.buffers.used != 0 || p.buffers.live != 0 || p.buffers.waiters.Len() != 0 {
				t.Fatalf("startup root used/live/waiters = %d/%d/%d, want 0/0/0", p.buffers.used, p.buffers.live, p.buffers.waiters.Len())
			}
		})
	}

	test("default CLI value", 64<<20, 64<<20)
	test("zero keeps default", 0, 64<<20)
	test("negative keeps default", -1, 64<<20)
	test("minimum integer keeps default", math.MinInt64, 64<<20)
	test("smaller positive limit", 1024, 1024)
	// The 256 MiB startup root must fit two bodies plus a separate probe byte.
	const safeStartupBodyLimit = (256<<20 - 1) / 2
	test("exact safe startup headroom", safeStartupBodyLimit, 128<<20-1)
}

func TestOwnedInferenceBuffersStartupRejectsUnsafeBodyLimit(t *testing.T) {
	test := func(name string, configured int64) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			p, err := initializeProxy(NewCodec(rwNop{}), configured)
			if err == nil {
				t.Fatalf("initializeProxy(%d) accepted an unsafe startup body limit", configured)
			}
			if p != nil {
				t.Fatal("invalid startup exposed a proxy for external setup")
			}
			if !strings.Contains(err.Error(), "invalid --max-request-bytes") ||
				!strings.Contains(err.Error(), "twice the individual body limit plus one byte") {
				t.Fatalf("startup error = %q, want CLI flag and checked headroom diagnostic", err)
			}
		})
	}

	test("one byte above safe startup headroom", 128<<20)
	test("larger CLI limit", 192<<20)
	test("overlap arithmetic would overflow", (math.MaxInt64-1)/2+1)
	test("maximum integer limit", math.MaxInt64)
}

func TestOwnedInferenceBuffersStartupValidatesBeforeExternalSetup(t *testing.T) {
	codec := NewCodec(rwNop{})
	p, err := initializeProxy(codec, 80<<20)
	if err != nil {
		t.Fatalf("initializeProxy() error = %v", err)
	}
	if p.requestBodyLimit() != 80<<20 || p.codec != codec {
		t.Fatal("startup did not install the effective CLI limit on its codec's proxy")
	}
	if p.accounts.storePath != "" || p.accounts.lockPath != "" || p.accounts.lock != nil || p.accounts.pathErr != nil {
		t.Fatal("startup validation configured or opened the account store")
	}
	if p.mesh != nil || p.baseConfigDir != "" {
		t.Fatal("startup validation opened cluster trust")
	}
	if len(p.enabledFacades()) != 0 || p.routing.ln != nil || p.routing.srv != nil {
		t.Fatal("startup validation exposed an inference listener")
	}
	if p.serveCtx != nil || p.cancel != nil {
		t.Fatal("startup validation started dispatch processing")
	}
}

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
