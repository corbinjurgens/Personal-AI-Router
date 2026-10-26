// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/nodepolicy"
	"nvpair-shared/schedulerwire"
)

func TestOwnedInferenceBuffersPlacement(t *testing.T) {
	assertNoPlacement := func(t *testing.T, fx *tierFixture) {
		t.Helper()
		fx.p.priorityMu.Lock()
		claims := len(fx.p.priorityReservations)
		fx.p.priorityMu.Unlock()
		if claims != 0 || len(notificationsOf(t, fx.w, workloadSubmittedMethod)) != 0 || len(notificationsOf(t, fx.w, workloadStartedMethod)) != 0 {
			t.Fatalf("preparation caused placement: claims=%d submitted=%d started=%d", claims, len(notificationsOf(t, fx.w, workloadSubmittedMethod)), len(notificationsOf(t, fx.w, workloadStartedMethod)))
		}
		if fx.eng["a"].hits() != 0 {
			t.Fatal("preparation reached the upstream")
		}
	}
	fixture := func(t *testing.T) *tierFixture {
		t.Helper()
		pol := ollamaTierPolicy(map[string][]nodepolicy.TierEntry{"weak": {{Kind: nodepolicy.EntryLocal, Engine: "ollama", Model: "small:1b"}}})
		return newTierFixture(t, ollamaCase(t), pol, map[string][]string{"a": {"small:1b"}}, nil)
	}
	t.Run("impossible transformation has no claim or published placement", func(t *testing.T) {
		fx := fixture(t)
		fx.p.buffers = bufferTestBudget(t, (128<<10)+registryNextTestBytes(fx.p), time.Hour)
		rec := fx.do(t, "/api/generate", `{"model":"weak"}`)
		if rec.Code != http.StatusRequestEntityTooLarge {
			t.Fatalf("impossible status=%d body=%s", rec.Code, rec.Body.String())
		}
		assertNoPlacement(t, fx)
		bufferAssertTotals(t, fx.p.buffers, 0, 0, 0)
	})
	t.Run("queued transformation has no claim or published placement", func(t *testing.T) {
		fx := fixture(t)
		preparationBytes := bufferFixtureRemotePreparationBytes(t, `{"model":"small:1b"}`)
		fx.p.buffers = bufferTestBudget(t, (256<<10)+preparationBytes+registryNextTestBytes(fx.p), time.Hour)
		// Offset the added preparation room until release, preserving the
		// original 128 KiB blocker against the transform's 96 KiB allowance.
		hold := bufferMustAcquire(t, fx.p.buffers, (128<<10)+preparationBytes)
		defer hold.close()
		rec := httptest.NewRecorder()
		done := make(chan struct{})
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		go func() {
			defer close(done)
			fx.p.soleFacade().handleHTTP(rec, httptest.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(`{"model":"weak"}`)).WithContext(ctx))
		}()
		waitForCond(t, 2*time.Second, "pre-send storage waiter", func() bool {
			fx.p.buffers.mu.Lock()
			defer fx.p.buffers.mu.Unlock()
			return fx.p.buffers.waiters.Len() == 1
		})
		assertNoPlacement(t, fx)
		hold.close()
		select {
		case <-done:
		case <-time.After(2 * time.Second):
			t.Fatal("prepared request did not complete")
		}
		if rec.Code != http.StatusOK || fx.eng["a"].hits() != 1 || fx.eng["a"].lastBody() != `{"model":"small:1b"}` {
			t.Fatalf("released preparation: status=%d hits=%d body=%q", rec.Code, fx.eng["a"].hits(), fx.eng["a"].lastBody())
		}
		fx.p.priorityMu.Lock()
		claims := len(fx.p.priorityReservations)
		fx.p.priorityMu.Unlock()
		if claims != 0 {
			t.Fatalf("completed request retained %d claims", claims)
		}
		bufferAssertTotals(t, fx.p.buffers, 0, 0, 0)
	})
	t.Run("retry clears prior placement and claim before storage wait", func(t *testing.T) {
		pol := ollamaTierPolicy(map[string][]nodepolicy.TierEntry{"weak": {{Kind: nodepolicy.EntryLocal, Engine: "ollama", Model: "small:1b"}}})
		fx := newTierFixture(t, ollamaCase(t), pol, map[string][]string{"a": {"small:1b"}, "b": {"small:1b"}}, nil)
		preparationBytes := bufferFixtureRemotePreparationBytes(t, `{"model":"small:1b"}`)
		capacity := (256 << 10) + preparationBytes + registryNextTestBytes(fx.p)
		// The first attempt needs preparation room in addition to the blocker
		// that will offset that same room while the retry waits.
		fx.p.buffers = bufferTestBudget(t, capacity+preparationBytes, time.Hour)
		type heldResult struct {
			lease *bufferLease
			err   error
		}
		held := make(chan heldResult, 1)
		var firstCalls atomic.Int32
		first := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if firstCalls.Add(1) == 1 {
				lease, err := fx.p.buffers.acquire(r.Context(), (128<<10)+preparationBytes)
				// Remove startup headroom before retry preparation. Once the
				// first attempt drains, effective pressure is again 256/128 KiB.
				if err == nil {
					err = fx.p.buffers.reconfigure(capacity, time.Hour)
					if err != nil {
						lease.close()
					}
				}
				held <- heldResult{lease, err}
			}
			w.WriteHeader(http.StatusServiceUnavailable)
		}))
		defer first.Close()
		f := fx.p.soleFacade()
		f.discovery.AddManual(nodeForModel(t, "a", first.URL, "small:1b"))
		rec := httptest.NewRecorder()
		done := make(chan struct{})
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		go func() {
			defer close(done)
			f.handleHTTP(rec, httptest.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(`{"model":"weak"}`)).WithContext(ctx))
		}()
		var hold *bufferLease
		select {
		case result := <-held:
			if result.err != nil {
				t.Fatal(result.err)
			}
			hold = result.lease
			defer hold.close()
		case <-time.After(2 * time.Second):
			t.Fatal("first attempt did not establish contention")
		}
		waitForCond(t, 2*time.Second, "retry storage waiter", func() bool {
			fx.p.buffers.mu.Lock()
			defer fx.p.buffers.mu.Unlock()
			return fx.p.buffers.waiters.Len() == 1
		})
		fx.p.priorityMu.Lock()
		claims := len(fx.p.priorityReservations)
		fx.p.priorityMu.Unlock()
		publications := notificationsOf(t, fx.w, workloadSubmittedMethod)
		if claims != 0 || len(publications) < 3 {
			t.Fatalf("waiting retry: claims=%d history=%d", claims, len(publications))
		}
		var priorParams, currentParams workloadParams
		if err := json.Unmarshal(publications[len(publications)-2], &priorParams); err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(publications[len(publications)-1], &currentParams); err != nil {
			t.Fatal(err)
		}
		prior, current := priorParams.WorkloadInfo, currentParams.WorkloadInfo
		if prior.ScheduledOn != "a" || current.ScheduledOn != "" || prior.ID != current.ID || fx.eng["b"].hits() != 0 {
			t.Fatalf("retry placement: prior=%+v current=%+v b hits=%d", prior, current, fx.eng["b"].hits())
		}
		hold.close()
		select {
		case <-done:
		case <-time.After(2 * time.Second):
			t.Fatal("retry did not complete after contention released")
		}
		if rec.Code != http.StatusOK || firstCalls.Load() != 1 || fx.eng["b"].hits() != 1 {
			t.Fatalf("retry result: status=%d a=%d b=%d", rec.Code, firstCalls.Load(), fx.eng["b"].hits())
		}
		bufferAssertTotals(t, fx.p.buffers, 0, 0, 0)
	})
}

func TestOwnedInferenceBuffersReservationRecheck(t *testing.T) {
	run := func(name string, change func(*Proxy), want []string) {
		t.Run(name, func(t *testing.T) {
			fx := newTierFixture(t, ollamaCase(t), nodepolicy.Default(), map[string][]string{"a": {"m"}, "b": {"m"}}, nil)
			f := fx.p.soleFacade()
			cands := f.resolveCandidates("m")
			entered, resume := make(chan struct{}), make(chan struct{})
			release := sync.OnceFunc(func() { close(resume) })
			defer release()
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			type result struct {
				out  []candidate
				held reservation
				err  error
			}
			done := make(chan result, 1)
			var seen []string
			go func() {
				out, held, err := f.reserveRoundPrepared(ctx, cands, false, "m", func(_ context.Context, c candidate, _ nodepolicy.Profile, _ bool, _ uint64) error {
					fx.p.priorityMu.Lock()
					claims := len(fx.p.priorityReservations)
					fx.p.priorityMu.Unlock()
					if claims != 0 {
						return fmt.Errorf("preparation observed %d premature claims", claims)
					}
					seen = append(seen, c.id)
					if len(seen) == 1 {
						close(entered)
						select {
						case <-resume:
						case <-ctx.Done():
							return ctx.Err()
						}
					}
					return nil
				})
				done <- result{out, held, err}
			}()
			select {
			case <-entered:
			case <-time.After(2 * time.Second):
				t.Fatal("preparation did not start")
			}
			changed := make(chan struct{})
			go func() { change(fx.p); close(changed) }()
			select {
			case <-changed:
			case <-time.After(2 * time.Second):
				t.Fatal("storage wait held a selection or admission lock")
			}
			release()
			var got result
			select {
			case got = <-done:
			case <-time.After(2 * time.Second):
				t.Fatal("selection did not finish")
			}
			defer fx.p.releaseReservation(got.held)
			if got.err != nil || !slices.Equal(seen, want) || len(got.out) == 0 || got.out[0].id != want[len(want)-1] || got.held.nodeID != want[len(want)-1] {
				t.Fatalf("selection: seen=%v held=%+v err=%v, want %v", seen, got.held, got.err, want)
			}
		})
	}
	run("changed least-load candidate is prepared before its claim", func(p *Proxy) { p.SetPriority([]string{"b", "a"}) }, []string{"a", "b"})
	run("changed policy rebuilds the same candidate before its claim", func(p *Proxy) {
		policy := p.admission.currentPolicy()
		p.admission.setPolicy(policy)
	}, []string{"a", "a"})
}

func TestParseRequestNeeds(t *testing.T) {
	caps := func(body, path string) []string {
		return parseRequestNeeds([]byte(body), path).names()
	}
	cases := []struct {
		name, body, path string
		want             []string
	}{
		{"plain chat", `{"model":"weak","messages":[{"role":"user","content":"hi"}]}`, "/v1/chat/completions", nil},
		{"openai tools", `{"model":"weak","tools":[{"type":"function"}]}`, "/v1/chat/completions", []string{"tools"}},
		{"empty tools", `{"model":"weak","tools":[]}`, "/v1/chat/completions", nil},
		{"legacy functions", `{"model":"weak","functions":[{"name":"f"}]}`, "/v1/chat/completions", []string{"tools"}},
		{"openai image part", `{"model":"weak","messages":[{"role":"user","content":[{"type":"text","text":"x"},{"type":"image_url","image_url":{"url":"data:"}}]}]}`, "/v1/chat/completions", []string{"vision"}},
		{"ollama message images", `{"model":"weak","messages":[{"role":"user","content":"x","images":["aGk="]}]}`, "/api/chat", []string{"vision"}},
		{"ollama generate images", `{"model":"weak","prompt":"x","images":["aGk="]}`, "/api/generate", []string{"vision"}},
		{"json schema", `{"model":"weak","response_format":{"type":"json_schema","json_schema":{}}}`, "/v1/chat/completions", []string{"structured"}},
		{"json object", `{"model":"weak","response_format":{"type":"json_object"}}`, "/v1/chat/completions", []string{"structured"}},
		{"text format", `{"model":"weak","response_format":{"type":"text"}}`, "/v1/chat/completions", nil},
		{"ollama format", `{"model":"weak","format":"json"}`, "/api/chat", []string{"structured"}},
		{"ollama schema", `{"model":"weak","format":{"type":"object"}}`, "/api/chat", []string{"structured"}},
		{"embeddings route", `{"model":"weak","input":"x"}`, "/v1/embeddings", []string{"embeddings"}},
		{"ollama embed", `{"model":"weak","input":"x"}`, "/api/embed", []string{"embeddings"}},
		{"everything", `{"model":"weak","tools":[{}],"format":"json","messages":[{"images":["x"]}]}`, "/api/chat", []string{"tools", "vision", "structured"}},
	}
	for _, c := range cases {
		if got := caps(c.body, c.path); !slices.Equal(got, c.want) {
			t.Errorf("%s: needs = %v, want %v", c.name, got, c.want)
		}
	}

	body := `{"model":"weak","max_tokens":1000}`
	if got := parseRequestNeeds([]byte(body), "/v1/completions").contextTokens; got != len(body)/4+1000 {
		t.Errorf("context with max_tokens = %d, want %d", got, len(body)/4+1000)
	}
	body = `{"model":"weak","max_completion_tokens":50}`
	if got := parseRequestNeeds([]byte(body), "/v1/chat/completions").contextTokens; got != len(body)/4+50 {
		t.Errorf("context with max_completion_tokens = %d", got)
	}
	body = `{"model":"weak","options":{"num_predict":300}}`
	if got := parseRequestNeeds([]byte(body), "/api/generate").contextTokens; got != len(body)/4+300 {
		t.Errorf("context with num_predict = %d", got)
	}
	body = `{"model":"weak","options":{"num_predict":-1}}`
	if got := parseRequestNeeds([]byte(body), "/api/generate").contextTokens; got != len(body)/4 {
		t.Errorf("an unbounded num_predict must not inflate the estimate: %d", got)
	}
}

func TestParseRequestNeedsDuplicateMessagesMatchesLegacyDecoder(t *testing.T) {
	legacyVision := func(t *testing.T, body []byte) bool {
		t.Helper()
		var probe struct {
			Messages []struct {
				Content json.RawMessage   `json:"content"`
				Images  []json.RawMessage `json:"images"`
			} `json:"messages"`
		}
		if err := json.Unmarshal(body, &probe); err != nil {
			var typeError *json.UnmarshalTypeError
			if !errors.As(err, &typeError) {
				t.Fatalf("legacy oracle syntax: %v", err)
			}
		}
		for _, message := range probe.Messages {
			if len(message.Images) > 0 {
				return true
			}
			var parts []struct {
				Type string `json:"type"`
			}
			if err := json.Unmarshal(message.Content, &parts); err == nil {
				for _, part := range parts {
					if part.Type == "image_url" || part.Type == "image" || part.Type == "input_image" {
						return true
					}
				}
			}
		}
		return false
	}
	for _, tc := range []struct{ name, body string }{
		{"retained images", `{"messages":[{"images":[1]}],"messages":[{}]}`},
		{"retained content", `{"messages":[{"content":[{"type":"image_url"}]}],"messages":[{}]}`},
		{"images null clears", `{"messages":[{"images":[1]}],"messages":[{"images":null}]}`},
		{"images empty clears", `{"messages":[{"images":[1]}],"messages":[{"images":[]}]}`},
		{"content null clears", `{"messages":[{"content":[{"type":"image"}]}],"messages":[{"content":null}]}`},
		{"content string replaces", `{"messages":[{"content":[{"type":"image"}]}],"messages":[{"content":"plain"}]}`},
		{"null element retains", `{"messages":[{"images":[1]}],"messages":[null]}`},
		{"mistyped element retains", `{"messages":[{"content":[{"type":"input_image"}]}],"messages":[17]}`},
		{"mistyped images retain", `{"messages":[{"images":[1]}],"messages":[{"images":false,"content":"plain"}]}`},
		{"partial object continues after image type error", `{"messages":[{}],"messages":[{"images":42,"content":[{"type":"image_url"}]}]}`},
		{"partial object replaces content before image type error", `{"messages":[{"content":[{"type":"image"}]}],"messages":[{"content":0,"images":42}]}`},
		{"mistyped outer slice retains", `{"messages":[{"images":[1]}],"messages":false}`},
		{"empty outer slice discards backing", `{"messages":[{"images":[1]}],"messages":[],"messages":[{}]}`},
		{"null outer slice discards backing", `{"messages":[{"images":[1]}],"messages":null,"messages":[{}]}`},
		{"shorter slice retains hidden slots", `{"messages":[{},{"images":[1]}],"messages":[{}],"messages":[{},{}]}`},
		{"folded field names retain", `{"MESSAGES":[{"IMAGES":[1]}],"messages":[{"images":42}]}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			body := []byte(tc.body)
			want := legacyVision(t, body)
			got := slices.Contains(parseRequestNeeds(body, "/v1/chat/completions").caps, nodepolicy.CapVision)
			if got != want {
				t.Fatalf("vision = %v, legacy decoder = %v, body = %s", got, want, body)
			}
		})
	}
}

func TestParseRequestNeedsOwnedMessagesWorkspace(t *testing.T) {
	for _, tc := range []struct {
		name     string
		capacity int64
		limit    bool
	}{
		{name: "refuses before state vector allocation", capacity: 295, limit: true},
		{name: "exact state vector capacity releases", capacity: 296},
	} {
		t.Run(tc.name, func(t *testing.T) {
			budget := bufferTestBudget(t, tc.capacity, 0)
			scope, err := budget.newRequest(tc.capacity)
			if err != nil {
				t.Fatal(err)
			}
			owner := &ownedInferenceBuffers{scope: scope}
			defer owner.close()
			ctx := context.WithValue(context.Background(), ownedInferenceContextKey{}, owner)
			needs, err := parseRequestNeedsOwned(ctx, []byte(`{"messages":[{"images":[1]}],"messages":[{}]}`), "/api/chat")
			if tc.limit {
				bufferAssertLimit(t, err)
			} else if err != nil || !slices.Contains(needs.caps, nodepolicy.CapVision) {
				t.Fatalf("needs = %+v, error = %v; want retained vision", needs, err)
			}
			bufferAssertTotals(t, budget, 0, 0, 0)
		})
	}
}

// This is the former token/raw-message decoder, independent of the borrowed
// scanner. Completed names survive a malformed later member in this oracle.
func legacyTierJSONModelNames(body []byte) []string {
	decoder := json.NewDecoder(bytes.NewReader(body))
	if token, err := decoder.Token(); err != nil || token != json.Delim('{') {
		return nil
	}
	var names []string
	for decoder.More() {
		token, err := decoder.Token()
		key, ok := token.(string)
		if err != nil || !ok {
			return names
		}
		var raw json.RawMessage
		if err := decoder.Decode(&raw); err != nil {
			return names
		}
		if strings.EqualFold(key, "model") || strings.EqualFold(key, "name") {
			var value string
			if err := json.Unmarshal(raw, &value); err == nil {
				names = append(names, value)
			}
		}
	}
	return names
}

func TestTierAliasJSONMatchesLegacyCompletedMembers(t *testing.T) {
	for _, tc := range []struct{ name, body string }{
		{"NBSP before object", "\u00a0{\"name\":\"weak-api\"}"},
		{"NBSP before key", "{\u00a0\"name\":\"weak-api\"}"},
		{"NBSP before colon", "{\"name\"\u00a0:\"weak-api\"}"},
		{"NBSP before value", "{\"name\":\u00a0\"weak-api\"}"},
		{"JSON whitespace before object and key", " \t\r\n{ \t\r\n\"name\" \t\r\n: \t\r\n\"weak-api\"}"},
		{"completed prefix survives malformed suffix", `{"model":"weak-api","later":`},
		{"completed prefix before NBSP", "{\"name\":\"weak-api\",\u00a0\"later\":0}"},
		{"escaped folded key and alias", `{"N\u0041ME":"\u0077eak-api"}`},
		{"duplicate member", `{"model":"plain","model":"strong-local"}`},
		{"wrong type ignored", `{"model":17,"name":"medium-api"}`},
		{"null member ignored", `{"model":null,"name":"medium-local"}`},
		{"uncompleted string ignored", `{"model":"weak-api`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			want := ""
			for _, name := range legacyTierJSONModelNames([]byte(tc.body)) {
				trimmed := strings.TrimSpace(name)
				if _, mode, ok := nodepolicy.ParseAlias(trimmed); ok && mode != nodepolicy.AliasBare {
					want = trimmed
					break
				}
			}
			if got := jsonModelAlias([]byte(tc.body)); got != want {
				t.Fatalf("alias = %q, legacy decoder = %q, body = %q", got, want, tc.body)
			}
		})
	}
}

func TestValidateRequestIdentifiersJSONBound(t *testing.T) {
	quoted := func(value string) string {
		t.Helper()
		raw, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		return string(raw)
	}
	for _, tc := range []struct {
		name, body string
		reject     bool
	}{
		{name: "raw exact 256", body: `{"model":` + quoted(strings.Repeat("a", 256)) + `}`},
		{name: "raw 257", body: `{"model":` + quoted(strings.Repeat("a", 257)) + `}`, reject: true},
		{name: "escaped exact 256", body: `{"model":"` + strings.Repeat(`\u0061`, 256) + `"}`},
		{name: "escaped 257", body: `{"model":"` + strings.Repeat(`\u0061`, 257) + `"}`, reject: true},
		{name: "multibyte exact 256", body: `{"name":` + quoted(strings.Repeat("あ", 85)+"a") + `}`},
		{name: "multibyte 257", body: `{"name":` + quoted(strings.Repeat("あ", 85)+"aa") + `}`, reject: true},
		{name: "escaped multibyte exact 256", body: `{"name":"` + strings.Repeat(`\u3042`, 85) + `a"}`},
		{name: "escaped multibyte 257", body: `{"name":"` + strings.Repeat(`\u3042`, 85) + `aa"}`, reject: true},
		{name: "padded alias exact 256", body: `{"model":` + quoted(strings.Repeat(" ", 248)+"weak-api") + `}`},
		{name: "padded alias 257 before trimming", body: `{"model":` + quoted(strings.Repeat(" ", 249)+"weak-api") + `}`, reject: true},
		{name: "folded name 257", body: `{"N\u0041ME":` + quoted(strings.Repeat("a", 257)) + `}`, reject: true},
		{name: "earlier duplicate 257", body: `{"model":` + quoted(strings.Repeat("a", 257)) + `,"model":"m"}`, reject: true},
		{name: "completed prefix 257", body: `{"name":` + quoted(strings.Repeat("a", 257)) + `,"later":`, reject: true},
		{name: "uncompleted member ignored", body: `{"name":"` + strings.Repeat("a", 257)},
		{name: "NBSP before object ignored", body: "\u00a0{\"name\":" + quoted(strings.Repeat("a", 257)) + `}`},
		{name: "NBSP before key ignored", body: "{\u00a0\"name\":" + quoted(strings.Repeat("a", 257)) + `}`},
		{name: "nested model is outside top-level guard", body: `{"nested":{"model":` + quoted(strings.Repeat("a", 257)) + `}}`},
		{name: "nonstring model ignored", body: `{"model":[` + quoted(strings.Repeat("a", 257)) + `]}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := httptest.NewRequest(http.MethodPost, "/api/show", nil)
			err := validateRequestIdentifiers(r, []byte(tc.body))
			var identifierError *requestIdentifierError
			if tc.reject {
				if !errors.As(err, &identifierError) {
					t.Fatalf("error = %v, want typed identifier refusal", err)
				}
				if identifierError.HTTPStatus() != http.StatusBadRequest {
					t.Fatalf("identifier status = %d, want 400", identifierError.HTTPStatus())
				}
			} else if err != nil {
				t.Fatalf("exact or irrelevant identifier refused: %v", err)
			}
		})
	}
}

func TestValidateRequestIdentifiersQueryAndPathBound(t *testing.T) {
	for _, tc := range []struct {
		name, target, query string
		reject              bool
	}{
		{name: "query exact 256", target: "/props", query: "model=" + strings.Repeat("a", 256)},
		{name: "query 257", target: "/props", query: "model=" + strings.Repeat("a", 257), reject: true},
		{name: "escaped query exact 256", target: "/props", query: "mo%64el=" + strings.Repeat("%61", 256)},
		{name: "escaped query 257", target: "/props", query: "mo%64el=" + strings.Repeat("%61", 257), reject: true},
		{name: "multibyte query exact 256", target: "/props", query: "model=" + url.QueryEscape(strings.Repeat("あ", 85)+"a")},
		{name: "multibyte query 257", target: "/props", query: "model=" + url.QueryEscape(strings.Repeat("あ", 85)+"aa"), reject: true},
		{name: "padded query exact 256", target: "/props", query: "model=" + strings.Repeat("+", 248) + "weak-api"},
		{name: "padded query 257 before trimming", target: "/props", query: "model=" + strings.Repeat("+", 249) + "weak-api", reject: true},
		{name: "earlier duplicate query 257", target: "/props", query: "model=" + strings.Repeat("a", 257) + "&model=m", reject: true},
		{name: "invalid percent field ignored", target: "/props", query: "model=" + strings.Repeat("a", 257) + "%zz"},
		{name: "semicolon field ignored", target: "/props", query: "model=" + strings.Repeat("a", 257) + ";ignored"},
		{name: "query key case remains exact", target: "/props", query: "Model=" + strings.Repeat("a", 257)},
		{name: "model path exact 256", target: "/v1/models/" + strings.Repeat("a", 256)},
		{name: "model path 257", target: "/v1/models/" + strings.Repeat("a", 257), reject: true},
		{name: "native model path exact 256", target: "/api/v0/models/" + url.PathEscape(strings.Repeat("あ", 85)+"a")},
		{name: "native model path 257", target: "/api/v0/models/" + url.PathEscape(strings.Repeat("あ", 85)+"aa"), reject: true},
		{name: "padded alias path exact 256", target: "/v1/models/" + strings.Repeat("%20", 248) + "weak-api"},
		{name: "padded alias path 257 before trimming", target: "/v1/models/" + strings.Repeat("%20", 249) + "weak-api", reject: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := httptest.NewRequest(http.MethodGet, tc.target, nil)
			r.URL.RawQuery = tc.query
			err := validateRequestIdentifiers(r, nil)
			var identifierError *requestIdentifierError
			if tc.reject {
				if !errors.As(err, &identifierError) {
					t.Fatalf("error = %v, want typed identifier refusal", err)
				}
				if identifierError.HTTPStatus() != http.StatusBadRequest {
					t.Fatalf("identifier status = %d, want 400", identifierError.HTTPStatus())
				}
			} else if err != nil {
				t.Fatalf("exact or irrelevant identifier refused: %v", err)
			}
		})
	}
}

func TestTierSearchOrder(t *testing.T) {
	cases := []struct {
		tier             string
		stronger, weaker bool
		want             []string
	}{
		{"weak", false, false, []string{"weak"}},
		{"weak", true, false, []string{"weak", "medium", "strong"}},
		{"medium", true, true, []string{"medium", "strong", "weak"}},
		{"strong", false, true, []string{"strong", "medium", "weak"}},
		{"strong", true, false, []string{"strong"}},
	}
	for _, c := range cases {
		got := tierSearchOrder(c.tier, nodepolicy.TierPolicy{AllowStronger: c.stronger, AllowWeaker: c.weaker})
		if !slices.Equal(got, c.want) {
			t.Errorf("%s stronger=%v weaker=%v: %v, want %v", c.tier, c.stronger, c.weaker, got, c.want)
		}
	}
}

func TestReplaceModelPreservesEveryOtherByte(t *testing.T) {
	body := `{ "stream" :true,"model" : "weak" ,"messages":[{"content":"model: \"x\" é"}],"n":1.50e2 }` + "\n"
	got, ok := replaceModel([]byte(body), "qwen3:8b")
	if !ok {
		t.Fatal("replaceModel refused a valid body")
	}
	want := `{ "stream" :true,"model" : "qwen3:8b" ,"messages":[{"content":"model: \"x\" é"}],"n":1.50e2 }` + "\n"
	if string(got) != want {
		t.Fatalf("got\n%s\nwant\n%s", got, want)
	}
	// Only the top-level model is replaced.
	nested := `{"meta":{"model":"weak"},"model":"weak"}`
	got, _ = replaceModel([]byte(nested), "m")
	if string(got) != `{"meta":{"model":"weak"},"model":"m"}` {
		t.Fatalf("nested model touched: %s", got)
	}
	for _, bad := range []string{``, `[]`, `{"model":"weak"} trailing`, `{"stream":true}`, `{"model":`} {
		if _, ok := replaceModel([]byte(bad), "m"); ok {
			t.Errorf("replaceModel accepted %q", bad)
		}
	}
}

func TestMergeDefaults(t *testing.T) {
	cases := []struct{ body, defaults, want string }{
		{`{"model":"m"}`, `{"keep_alive":"10m"}`, `{"model":"m","keep_alive":"10m"}`},
		{`{"model":"m","keep_alive":"1m"}`, `{"keep_alive":"10m"}`, `{"model":"m","keep_alive":"1m"}`},
		{`{"model":"m","options":{"temperature":0}}`, `{"options":{"num_ctx":4096,"temperature":1}}`, `{"model":"m","options":{"temperature":0,"num_ctx":4096}}`},
		{`{"model":"m","options":null}`, `{"options":{"num_ctx":4096}}`, `{"model":"m","options":null}`},
		{`{}`, `{"a":1,"b":[1,2]}`, `{"a":1,"b":[1,2]}`},
		{`{"model":"m"}`, ``, `{"model":"m"}`},
		{`not json`, `{"a":1}`, `not json`},
	}
	for _, c := range cases {
		if got := string(mergeDefaults([]byte(c.body), []byte(c.defaults))); got != c.want {
			t.Errorf("merge(%s, %s) = %s, want %s", c.body, c.defaults, got, c.want)
		}
	}
}

func TestRequestOptionsBodyModes(t *testing.T) {
	for _, tc := range []struct {
		name, body, options, want string
		mode                      nodepolicy.RequestOptionsMode
		rewrite                   bool
	}{
		{"caller temperature wins defaults", ` {"model":"m", "temperature": 0.8} `, `{"temperature":0.1}`, ` {"model":"m", "temperature": 0.8} `, nodepolicy.RequestOptionsDefaults, false},
		{"omitted mode preserves defaults", `{"a":null}`, `{"a":{"n":1}}`, `{"a":null}`, "", false},
		{"nested configured leaves and absent fields", ` {"model":"m","a":{"x":1,"untouched":1.50e2},"free":[1, 2]} `, `{"a":{"x":2,"missing":true},"new":null}`, ` {"model":"m","a":{"x":2,"untouched":1.50e2,"missing":true},"free":[1, 2],"new":null} `, nodepolicy.RequestOptionsEnforced, false},
		{"null replaced by object", `{"a":null}`, `{"a":{"x":1}}`, `{"a":{"x":1}}`, nodepolicy.RequestOptionsEnforced, false},
		{"array replaced by object", `{"a":[1,2]}`, `{"a":{"x":1}}`, `{"a":{"x":1}}`, nodepolicy.RequestOptionsEnforced, false},
		{"object replaced by array", `{"a":{"x":1}}`, `{"a":[3,4]}`, `{"a":[3,4]}`, nodepolicy.RequestOptionsEnforced, false},
		{"object replaced by scalar", `{"a":{"x":1}}`, `{"a":false}`, `{"a":false}`, nodepolicy.RequestOptionsEnforced, false},
		{"scalar replaced by null", `{"a":1}`, `{"a":null}`, `{"a":null}`, nodepolicy.RequestOptionsEnforced, false},
		{"array is one configured leaf", `{"a":[{"x":1,"free":2}]}`, `{"a":[{"x":3}]}`, `{"a":[{"x":3}]}`, nodepolicy.RequestOptionsEnforced, false},
		{"unconfigured controls preserve bytes", `{"temperature":0.8,"tools":[ {"type":"function"} ],"stream":true,"max_tokens":17}`, `{"temperature":0.1}`, `{"temperature":0.1,"tools":[ {"type":"function"} ],"stream":true,"max_tokens":17}`, nodepolicy.RequestOptionsEnforced, false},
		{"escaped canonical configured member", `{"\u0061":1}`, `{"a":2}`, `{"\u0061":2}`, nodepolicy.RequestOptionsEnforced, false},
		{"unconfigured duplicates preserved", `{"a":1,"free":{"x":1,"X":2},"n":1,"n":2}`, `{"a":2}`, `{"a":2,"free":{"x":1,"X":2},"n":1,"n":2}`, nodepolicy.RequestOptionsEnforced, false},
		{"unconfigured duplicate at configured object level", `{"a":{"x":1,"free":0,"free":1}}`, `{"a":{"x":2}}`, `{"a":{"x":2,"free":0,"free":1}}`, nodepolicy.RequestOptionsEnforced, false},
		{"configured array replaces ambiguous subtree", `{"a":{"x":1,"X":2}}`, `{"a":[]}`, `{"a":[]}`, nodepolicy.RequestOptionsEnforced, false},
		{"unchanged object retains whitespace", " {\"a\": { \"x\": 1 }, \"free\": 1.0}\n", `{"a":{"x":1}}`, " {\"a\": { \"x\": 1 }, \"free\": 1.0}\n", nodepolicy.RequestOptionsEnforced, false},
		{"model rewrite and enforcement", `{"model":"weak","a":0}`, `{"a":2}`, `{"model":"m","a":2}`, nodepolicy.RequestOptionsEnforced, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			body := []byte(tc.body)
			out, release, err := transformBody(nil, body, "m", tc.rewrite, []byte(tc.options), tc.mode)
			if err != nil {
				t.Fatalf("transform: %v", err)
			}
			defer release()
			if string(out) != tc.want {
				t.Fatalf("body=%q, want %q", out, tc.want)
			}
			if tc.body == tc.want {
				if &out[0] != &body[0] {
					t.Fatal("unchanged body was copied")
				}
			} else if cap(out) != len(out) {
				t.Fatalf("send capacity=%d length=%d", cap(out), len(out))
			}
		})
	}
}

func TestRequestOptionsConfiguredAmbiguity(t *testing.T) {
	for _, tc := range []struct{ name, body string }{
		{"exact duplicate", `{"a":1,"a":2}`},
		{"escaped duplicate", `{"a":1,"\u0061":2}`},
		{"folded duplicate", `{"a":1,"A":2}`},
		{"single folded variant", `{"A":2}`},
		{"nested exact duplicate", `{"a":{"x":1,"x":2}}`},
		{"nested escaped duplicate", `{"a":{"x":1,"\u0078":2}}`},
		{"nested folded variant", `{"a":{"X":2}}`},
		{"folded parent", `{"A":{"x":2}}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			out, release, err := transformBody(nil, []byte(tc.body), "", false, []byte(`{"a":{"x":3}}`), nodepolicy.RequestOptionsEnforced)
			if release != nil {
				defer release()
			}
			var invalid *ambiguousOptionsError
			if !errors.As(err, &invalid) || err.Error() != ambiguousOptionsMessage || invalid.HTTPStatus() != http.StatusBadRequest || out != nil {
				t.Fatalf("body=%q error=%v, want fixed typed bad request", out, err)
			}
		})
	}
}

func TestRequestOptionsModelRewriteAuthority(t *testing.T) {
	for _, key := range []string{`model`, `Model`, `mo\u0064el`} {
		t.Run(key, func(t *testing.T) {
			out, release, err := transformBody(nil, []byte(`{"model":"weak"}`), "m", true, []byte(`{"`+key+`":"forbidden"}`), nodepolicy.RequestOptionsEnforced)
			if release != nil {
				defer release()
			}
			if !errors.Is(err, errAmbiguousModel) || out != nil {
				t.Fatalf("configured model survived rewrite: body=%q error=%v", out, err)
			}
		})
	}
}

func TestRequestOptionsPreparedIdentity(t *testing.T) {
	body := []byte(`{"model":"m","temperature":0.8}`)
	options := []byte(`{"temperature":0.1}`)
	c := candidate{id: "self", self: true, model: "m"}
	prepared := preparedBodyIdentity{c, options, nodepolicy.RequestOptionsDefaults, 7}
	for _, tc := range []struct {
		name       string
		candidate  candidate
		mode       nodepolicy.RequestOptionsMode
		generation uint64
		reuse      bool
		want       string
	}{
		{"same body reused", c, nodepolicy.RequestOptionsDefaults, 7, true, string(body)},
		{"omitted mode reuses defaults", c, "", 7, true, string(body)},
		{"identical options different mode rebuilds", c, nodepolicy.RequestOptionsEnforced, 7, false, `{"model":"m","temperature":0.1}`},
		{"new generation rebuilds", c, nodepolicy.RequestOptionsDefaults, 8, false, string(body)},
		{"candidate failover rebuilds", candidate{id: "other", self: true, model: "m"}, nodepolicy.RequestOptionsDefaults, 7, false, string(body)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			reuse := prepared.matches(tc.candidate, append([]byte(nil), options...), tc.mode, tc.generation)
			if reuse != tc.reuse {
				t.Fatalf("reuse=%v, want %v", reuse, tc.reuse)
			}
			out := body
			if !reuse {
				var release func()
				var err error
				out, release, err = transformBody(nil, body, "", false, options, tc.mode)
				if err != nil {
					t.Fatalf("rebuild: %v", err)
				}
				defer release()
			}
			if string(out) != tc.want {
				t.Fatalf("prepared body=%s, want %s", out, tc.want)
			}
		})
	}
	if prepared.matches(c, []byte(`{"temperature":0.2}`), nodepolicy.RequestOptionsDefaults, 7) {
		t.Fatal("different option bytes reused")
	}
}

// tierFixture is one facade with manual peers and a policy.
type tierFixture struct {
	p   *Proxy
	w   *recordingWriter
	eng map[string]*recordingEngine
}

func newTierFixture(t *testing.T, tc engineCase, pol nodepolicy.Policy, nodes map[string][]string, loaded map[string][]string) *tierFixture {
	t.Helper()
	fx := &tierFixture{w: &recordingWriter{}, eng: map[string]*recordingEngine{}}
	disc := NewDiscovery()
	ids := make([]string, 0, len(nodes))
	for id, models := range nodes {
		e := newRecordingEngine(t, http.StatusOK)
		fx.eng[id] = e
		n := nodeFor(t, id, e.URL)
		n.Models = models
		n.Loaded = loaded[id]
		disc.AddManual(n)
		ids = append(ids, id)
	}
	slices.Sort(ids)
	fx.p = newTestProxy(tc.profile, NewCodec(fx.w), disc, tc.profile.FacadePort)
	fx.p.SetPriority(ids)
	if err := pol.Validate(); err != nil {
		t.Fatal(err)
	}
	fx.p.admission.setPolicy(pol)
	return fx
}

func (fx *tierFixture) do(t *testing.T, path, body string) *httptest.ResponseRecorder {
	t.Helper()
	rec := httptest.NewRecorder()
	fx.p.soleFacade().handleHTTP(rec, httptest.NewRequest(http.MethodPost, path, strings.NewReader(body)))
	return rec
}

func (fx *tierFixture) servedBy(t *testing.T) string {
	t.Helper()
	for id, e := range fx.eng {
		if e.hits() > 0 {
			return id
		}
	}
	return ""
}

func ollamaTierPolicy(tiers map[string][]nodepolicy.TierEntry) nodepolicy.Policy {
	pol := nodepolicy.Default()
	pol.Tiers = tiers
	return pol
}

func TestTier_CapabilityFilterAndBodyRewrite(t *testing.T) {
	tc := ollamaCase(t)
	pol := ollamaTierPolicy(map[string][]nodepolicy.TierEntry{
		"weak": {
			{Kind: nodepolicy.EntryLocal, Engine: "ollama", Model: "small:1b"},
			{Kind: nodepolicy.EntryLocal, Engine: "ollama", Model: "small-tools:1b", Capabilities: []nodepolicy.Capability{nodepolicy.CapTools}},
		},
	})
	fx := newTierFixture(t, tc, pol, map[string][]string{
		"a": {"small:1b"}, "b": {"small-tools:1b"},
	}, nil)

	body := `{"model":"weak","tools":[{"type":"function"}],"temperature":0.2}`
	rec := fx.do(t, "/v1/chat/completions", body)
	if rec.Code != http.StatusOK || fx.servedBy(t) != "b" {
		t.Fatalf("status %d, served by %q, want b (the only tools model)", rec.Code, fx.servedBy(t))
	}
	if got := fx.eng["b"].lastBody(); got != `{"model":"small-tools:1b","tools":[{"type":"function"}],"temperature":0.2}` {
		t.Fatalf("engine got %s", got)
	}
	// Headers name what answered.
	for h, want := range map[string]string{
		nodepolicy.ModelHeader: "small-tools:1b", nodepolicy.EngineHeader: "ollama",
		nodepolicy.NodeHeader: "b", nodepolicy.TierHeader: "weak",
	} {
		if got := rec.Header().Get(h); got != want {
			t.Errorf("%s = %q, want %q", h, got, want)
		}
	}
}

func TestTier_ContextFilterAndStrongerFallback(t *testing.T) {
	tc := ollamaCase(t)
	pol := ollamaTierPolicy(map[string][]nodepolicy.TierEntry{
		"weak":   {{Kind: nodepolicy.EntryLocal, Engine: "ollama", Model: "small:1b", ContextTokens: 100}},
		"medium": {{Kind: nodepolicy.EntryLocal, Engine: "ollama", Model: "mid:7b", ContextTokens: 32000}},
	})
	fx := newTierFixture(t, tc, pol, map[string][]string{"a": {"small:1b"}, "b": {"mid:7b"}}, nil)

	// Small enough for weak.
	if rec := fx.do(t, "/v1/chat/completions", `{"model":"weak"}`); rec.Code != http.StatusOK || fx.servedBy(t) != "a" {
		t.Fatalf("short request: status %d served by %q, want a", rec.Code, fx.servedBy(t))
	}
	// Too long for weak's context: falls to the stronger tier.
	rec := fx.do(t, "/v1/chat/completions", `{"model":"weak","max_tokens":4000}`)
	if rec.Code != http.StatusOK || fx.eng["b"].hits() != 1 {
		t.Fatalf("long request: status %d, b hits %d", rec.Code, fx.eng["b"].hits())
	}
	if rec.Header().Get(nodepolicy.TierHeader) != "weak" || rec.Header().Get(nodepolicy.ModelHeader) != "mid:7b" {
		t.Fatalf("headers %v", rec.Header())
	}

	// Without allowStronger nothing is eligible: 404 naming the tier.
	pol.TierPolicy.AllowStronger = false
	fx.p.admission.setPolicy(pol)
	rec = fx.do(t, "/v1/chat/completions", `{"model":"weak","max_tokens":4000,"tools":[{}]}`)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want 404", rec.Code)
	}
	var out struct {
		Error string   `json:"error"`
		Needs []string `json:"needs"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	if out.Error != "no eligible model for tier weak" || !slices.Equal(out.Needs, []string{"tools"}) {
		t.Fatalf("404 body = %s", rec.Body)
	}
}

func TestTier_NoEntriesIs404WithEmptyNeeds(t *testing.T) {
	tc := ollamaCase(t)
	fx := newTierFixture(t, tc, nodepolicy.Default(), map[string][]string{"a": {"small:1b"}}, nil)
	rec := fx.do(t, "/api/chat", `{"model":"strong"}`)
	if rec.Code != http.StatusNotFound || !strings.Contains(rec.Body.String(), `"needs":[]`) {
		t.Fatalf("status %d body %s", rec.Code, rec.Body)
	}
	if fx.servedBy(t) != "" {
		t.Fatal("an ineligible tier request reached an engine")
	}
}

func TestTier_WarmPreference(t *testing.T) {
	tc := ollamaCase(t)
	tiers := map[string][]nodepolicy.TierEntry{
		"weak":   {{Kind: nodepolicy.EntryLocal, Engine: "ollama", Model: "cold:1b"}, {Kind: nodepolicy.EntryLocal, Engine: "ollama", Model: "warm:1b"}},
		"medium": {{Kind: nodepolicy.EntryLocal, Engine: "ollama", Model: "mid:7b"}},
	}
	nodes := map[string][]string{"a": {"cold:1b"}, "b": {"warm:1b"}, "c": {"mid:7b"}}

	// Within the requested tier a loaded model leads, ahead of entry order.
	fx := newTierFixture(t, tc, ollamaTierPolicy(tiers), nodes, map[string][]string{"b": {"warm:1b"}, "c": {"mid:7b"}})
	if rec := fx.do(t, "/v1/chat/completions", `{"model":"weak"}`); rec.Code != http.StatusOK || fx.servedBy(t) != "b" {
		t.Fatalf("tier mode: served by %q, want warm b", fx.servedBy(t))
	}

	// "tier" keeps tier order first: a warm stronger model never beats a cold
	// model of the requested tier.
	fx = newTierFixture(t, tc, ollamaTierPolicy(tiers), nodes, map[string][]string{"c": {"mid:7b"}})
	if rec := fx.do(t, "/v1/chat/completions", `{"model":"weak"}`); rec.Code != http.StatusOK || fx.servedBy(t) != "a" {
		t.Fatalf("tier mode, only medium warm: served by %q, want a", fx.servedBy(t))
	}

	// "any" prefers any loaded eligible model.
	pol := ollamaTierPolicy(tiers)
	pol.TierPolicy.PreferLoaded = "any"
	fx = newTierFixture(t, tc, pol, nodes, map[string][]string{"c": {"mid:7b"}})
	rec := fx.do(t, "/v1/chat/completions", `{"model":"weak"}`)
	if rec.Code != http.StatusOK || fx.servedBy(t) != "c" {
		t.Fatalf("any mode: served by %q, want warm c", fx.servedBy(t))
	}
}

// Workload events carry the concrete model and the tier asked for.
func TestTier_WorkloadCarriesRequestedModel(t *testing.T) {
	tc := ollamaCase(t)
	pol := ollamaTierPolicy(map[string][]nodepolicy.TierEntry{"medium": {{Kind: nodepolicy.EntryLocal, Engine: "ollama", Model: "mid:7b"}}})
	fx := newTierFixture(t, tc, pol, map[string][]string{"a": {"mid:7b"}}, nil)
	if rec := fx.do(t, "/api/generate", `{"model":"medium","prompt":"x"}`); rec.Code != http.StatusOK {
		t.Fatalf("status %d", rec.Code)
	}
	events := notificationsOf(t, fx.w, workloadStartedMethod)
	events = append(events, notificationsOf(t, fx.w, workloadSubmittedMethod)...)
	events = append(events, notificationsOf(t, fx.w, workloadCompletedMethod)...)
	if len(events) < 3 {
		t.Fatalf("got %d workload events", len(events))
	}
	for _, raw := range events {
		var params workloadParams
		if err := json.Unmarshal(raw, &params); err != nil {
			t.Fatal(err)
		}
		if params.WorkloadInfo.Model != "mid:7b" || params.WorkloadInfo.RequestedModel != "medium" {
			t.Fatalf("workload model=%q requestedModel=%q, want mid:7b / medium", params.WorkloadInfo.Model, params.WorkloadInfo.RequestedModel)
		}
	}

	// A request that names its model carries no requestedModel at all.
	fx2 := newTierFixture(t, tc, pol, map[string][]string{"a": {"mid:7b"}}, nil)
	fx2.do(t, "/api/generate", `{"model":"mid:7b"}`)
	for _, line := range fx2.w.lines() {
		if strings.Contains(string(line), "requestedModel") {
			t.Fatalf("requestedModel on a non-tier workload: %s", line)
		}
	}
}

// Routing headers are present on an ordinary routed response too.
func TestHandleHTTP_RoutingHeadersOnPlainRequest(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		e := newRecordingEngine(t, http.StatusOK)
		disc := NewDiscovery()
		disc.AddManual(nodeForModel(t, "n1", e.URL, tc.advertisedModel))
		p := testProxy(tc.profile, disc, tc.profile.FacadePort)
		rec := httptest.NewRecorder()
		p.soleFacade().handleHTTP(rec, tc.inferenceRequest())
		if rec.Header().Get(nodepolicy.ModelHeader) != tc.requestedModel ||
			rec.Header().Get(nodepolicy.EngineHeader) != tc.profile.Name ||
			rec.Header().Get(nodepolicy.NodeHeader) != "n1" ||
			rec.Header().Get(nodepolicy.TierHeader) != "" {
			t.Fatalf("headers = %v", rec.Header())
		}
	})
}

// twoEngineProxy hosts an Ollama and an LM Studio facade in one process, the
// way production does, without binding either.
func twoEngineProxy(t *testing.T, w *recordingWriter) (*Proxy, *facade, *facade) {
	t.Helper()
	oll, lms := ollamaCase(t).profile, lmstudioCase(t).profile
	p := newTestProxy(oll, NewCodec(w), NewDiscovery(), oll.FacadePort)
	lm := newFacade(p, lms, NewDiscovery(), lms.FacadePort)
	p.facades[lms.Name] = lm
	return p, p.facades[oll.Name], lm
}

// An OpenAI-compatible tier request entering through one engine's facade may
// be served by another engine on this node: the self candidate targets that
// engine's local backend, and admission is keyed to that engine.
func TestTier_CrossEngineSelfRouting(t *testing.T) {
	w := &recordingWriter{}
	p, oll, lm := twoEngineProxy(t, w)
	lmEngine := newRecordingEngine(t, http.StatusOK)
	ollEngine := newRecordingEngine(t, http.StatusOK)
	lm.discovery.AddManual(selfNode("self", lm.profile.FacadePort, "qwen3-8b"))
	setBackend(t, lm, lmEngine.URL, true)
	oll.discovery.AddManual(selfNode("self", oll.profile.FacadePort, "llama3:latest"))
	setBackend(t, oll, ollEngine.URL, true)

	pol := nodepolicy.Default()
	pol.Tiers = map[string][]nodepolicy.TierEntry{"strong": {{Kind: nodepolicy.EntryLocal, Engine: "lmstudio", Model: "qwen3-8b"}}}
	p.admission.setPolicy(pol)

	rec := httptest.NewRecorder()
	oll.handleHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"strong","messages":[]}`)))
	if rec.Code != http.StatusOK || lmEngine.hits() != 1 || ollEngine.hits() != 0 {
		t.Fatalf("status %d, lmstudio hits %d, ollama hits %d", rec.Code, lmEngine.hits(), ollEngine.hits())
	}
	if got := lmEngine.lastBody(); got != `{"model":"qwen3-8b","messages":[]}` {
		t.Fatalf("lmstudio engine got %s", got)
	}
	if rec.Header().Get(nodepolicy.EngineHeader) != "lmstudio" {
		t.Fatalf("%s = %q, want lmstudio", nodepolicy.EngineHeader, rec.Header().Get(nodepolicy.EngineHeader))
	}

	// Admission is the other engine's: draining LM Studio refuses it.
	p.admission.setDrain("lmstudio", true)
	rec = httptest.NewRecorder()
	oll.handleHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"strong"}`)))
	assertAdmissionRejection(t, rec, nodepolicy.RejectDraining)

	// A native route never crosses engines.
	p.admission.setDrain("lmstudio", false)
	rec = httptest.NewRecorder()
	oll.handleHTTP(rec, httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(`{"model":"strong"}`)))
	if rec.Code != http.StatusNotFound {
		t.Fatalf("native route crossed engines: status %d", rec.Code)
	}
}

// A peer candidate on another engine targets that peer's facade for that
// engine: the port the other facade's discovery overlay carries.
func TestTier_CrossEnginePeerUsesThatEnginesPort(t *testing.T) {
	w := &recordingWriter{}
	p, oll, lm := twoEngineProxy(t, w)
	peerLM := newRecordingEngine(t, http.StatusOK)
	lm.discovery.AddManual(nodeForModel(t, "peer", peerLM.URL, "qwen3-8b"))
	pol := nodepolicy.Default()
	pol.Tiers = map[string][]nodepolicy.TierEntry{"weak": {{Kind: nodepolicy.EntryLocal, Engine: "lmstudio", Model: "qwen3-8b"}}}
	p.admission.setPolicy(pol)

	cands := oll.resolveTierCandidates("weak", requestNeeds{}, true)
	if len(cands) != 1 || cands[0].engine != "lmstudio" || cands[0].url.Host != strings.TrimPrefix(peerLM.URL, "http://") {
		t.Fatalf("candidates = %+v", cands)
	}
	if cands[0].fac != lm {
		t.Fatal("a cross-engine candidate must be owned by its engine's facade")
	}
}

func TestModelList_OffersTiers(t *testing.T) {
	tc := lmstudioCase(t)
	e := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"object":"list","data":[{"id":"qwen3-8b"}]}`))
	}))
	defer e.Close()
	disc := NewDiscovery()
	disc.AddManual(nodeFor(t, "n1", e.URL))
	p := testProxy(tc.profile, disc, tc.profile.FacadePort)
	pol := nodepolicy.Default()
	pol.Tiers = map[string][]nodepolicy.TierEntry{
		"weak":   {{Kind: nodepolicy.EntryLocal, Engine: "lmstudio", Model: "qwen3-8b"}},
		"strong": {{Kind: nodepolicy.EntryLocal, Engine: "lmstudio", Model: "qwen3-8b"}},
	}
	p.admission.setPolicy(pol)
	rec := httptest.NewRecorder()
	p.soleFacade().handleHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/models", nil))
	var out struct {
		Data []struct {
			ID      string `json:"id"`
			OwnedBy string `json:"owned_by"`
		} `json:"data"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	var ids []string
	for _, d := range out.Data {
		ids = append(ids, d.ID+"/"+d.OwnedBy)
	}
	want := []string{
		"qwen3-8b/",
		"weak/pair", "weak-api/pair", "weak-local/pair",
		"strong/pair", "strong-api/pair", "strong-local/pair",
	}
	if !slices.Equal(ids, want) {
		t.Fatalf("model list = %v", ids)
	}
}

func TestModelList_NativeShapeGetsNoTiers(t *testing.T) {
	tc := ollamaCase(t)
	e := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"models":[{"name":"qwen3:latest","model":"qwen3:latest"}]}`))
	}))
	defer e.Close()
	disc := NewDiscovery()
	disc.AddManual(nodeFor(t, "n1", e.URL))
	p := testProxy(tc.profile, disc, tc.profile.FacadePort)
	pol := nodepolicy.Default()
	pol.Tiers = map[string][]nodepolicy.TierEntry{"weak": {{Kind: nodepolicy.EntryLocal, Engine: "ollama", Model: "qwen3"}}}
	p.admission.setPolicy(pol)
	rec := httptest.NewRecorder()
	p.soleFacade().handleHTTP(rec, httptest.NewRequest(http.MethodGet, "/api/tags", nil))
	if strings.Contains(rec.Body.String(), "weak") {
		t.Fatalf("native model list gained a tier: %s", rec.Body)
	}
}

// A version 1 policy file migrated to version 2 resolves exactly the candidates
// it resolved before: its entries are local entries.
func TestTier_MigratedV1PolicyResolvesTheSameCandidates(t *testing.T) {
	pol, err := nodepolicy.Parse([]byte(`{
		"version": 1,
		"tiers": {
			"weak":   [{"engine":"ollama","model":"small:1b"}, {"engine":"ollama","model":"small-tools:1b","capabilities":["tools"]}],
			"medium": [{"engine":"ollama","model":"mid:7b"}]
		}
	}`))
	if err != nil {
		t.Fatalf("Parse(v1): %v", err)
	}
	fx := newTierFixture(t, ollamaCase(t), pol, map[string][]string{
		"a": {"small:1b"}, "b": {"small-tools:1b"}, "c": {"mid:7b"}, "d": {"unrelated:1b"},
	}, nil)
	f := fx.p.soleFacade()

	type resolved struct {
		node, model string
		rank        int
	}
	resolve := func(needs requestNeeds) []resolved {
		var out []resolved
		for _, c := range f.resolveTierCandidates("weak", needs, false) {
			out = append(out, resolved{c.id, c.model, c.tierRank})
		}
		return out
	}
	want := []resolved{{"a", "small:1b", 0}, {"b", "small-tools:1b", 0}, {"c", "mid:7b", 1}}
	if got := resolve(requestNeeds{}); !slices.Equal(got, want) {
		t.Errorf("candidates = %+v, want %+v", got, want)
	}
	if got, want := resolve(requestNeeds{caps: []nodepolicy.Capability{nodepolicy.CapTools}}), []resolved{{"b", "small-tools:1b", 0}}; !slices.Equal(got, want) {
		t.Errorf("tools candidates = %+v, want %+v", got, want)
	}
	if rec := fx.do(t, "/v1/chat/completions", `{"model":"weak"}`); rec.Code != http.StatusOK || fx.servedBy(t) != "a" {
		t.Errorf("status %d served by %q, want a", rec.Code, fx.servedBy(t))
	}
}

// A v1 tier may list one engine and model twice with different declared
// bounds; each entry made a candidate eligible on its own, and still does.
func TestTier_MigratedV1NearDuplicateEntriesBothResolve(t *testing.T) {
	pol, err := nodepolicy.Parse([]byte(`{"version":1,"tiers":{"weak":[
		{"engine":"ollama","model":"small:1b","contextTokens":100},
		{"engine":"ollama","model":"small:1b","contextTokens":32000,"capabilities":["tools"]}]}}`))
	if err != nil {
		t.Fatalf("Parse(v1): %v", err)
	}
	pol.TierPolicy.AllowStronger = false
	fx := newTierFixture(t, ollamaCase(t), pol, map[string][]string{"a": {"small:1b"}}, nil)
	f := fx.p.soleFacade()
	count := func(needs requestNeeds) int { return len(f.resolveTierCandidates("weak", needs, false)) }

	if got := count(requestNeeds{caps: []nodepolicy.Capability{nodepolicy.CapTools}}); got != 1 {
		t.Errorf("tools request candidates = %d, want 1 through the second entry", got)
	}
	if got := count(requestNeeds{contextTokens: 1000}); got != 1 {
		t.Errorf("1000-token request candidates = %d, want 1 through the second entry", got)
	}
	if got := count(requestNeeds{contextTokens: 50}); got != 1 {
		t.Errorf("50-token request candidates = %d, want 1 (deduplicated by node and model)", got)
	}
	if got := count(requestNeeds{contextTokens: 40000}); got != 0 {
		t.Errorf("40000-token request candidates = %d, want 0", got)
	}
}

const apiTierPolicy = `{
  "version": 2,
  "tiers": {
    "weak":   [{"kind":"api","deployment":"cloud-small"}],
    "medium": [{"kind":"api","deployment":"cloud-small"}, {"kind":"local","engine":"ollama","model":"mid:7b"}]
  },
  "routing": {
    "accounts": [{"id":"acct","credential":"NVPAIR_PROVIDER_SECRET_ACCT","billing":"free",
      "perRequestCeiling":"0","dailyCeiling":"0","monthlyCeiling":"0"}],
    "deployments": [{"id":"cloud-small","account":"acct","model":"small-1","family":"alpha",
      "contextTokens":8192,"maxOutputTokens":1024,"enabled":false}]
  }
}`

func TestTier_APIEntriesAreSkipped(t *testing.T) {
	pol, err := nodepolicy.Parse([]byte(apiTierPolicy))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	pol.TierPolicy.AllowStronger = false
	fx := newTierFixture(t, ollamaCase(t), pol, map[string][]string{
		"a": {"mid:7b"}, "b": {"cloud-small", "small-1"},
	}, nil)
	f := fx.p.soleFacade()
	resolve := func(tier string) []string {
		var out []string
		for _, c := range f.resolveTierCandidates(tier, requestNeeds{}, false) {
			out = append(out, c.id+"/"+c.model)
		}
		return out
	}

	if got := resolve("weak"); len(got) != 0 {
		t.Errorf("an API-only tier resolved candidates %v, want none", got)
	}
	if rec := fx.do(t, "/v1/chat/completions", `{"model":"weak"}`); rec.Code != http.StatusNotFound || fx.servedBy(t) != "" {
		t.Errorf("API-only tier: status %d served by %q, want 404 and no engine request", rec.Code, fx.servedBy(t))
	}
	if got, want := resolve("medium"), []string{"a/mid:7b"}; !slices.Equal(got, want) {
		t.Errorf("mixed tier candidates = %v, want only the local entry %v", got, want)
	}

	// The kind decides, not the other fields: an API entry that also carries a
	// local engine and model is still not a local candidate. Validation refuses
	// such an entry, so it is installed past Validate.
	odd := nodepolicy.Default()
	odd.Tiers = map[string][]nodepolicy.TierEntry{
		"weak": {{Kind: nodepolicy.EntryAPI, Engine: "ollama", Model: "mid:7b", Deployment: "cloud-small"}},
	}
	fx.p.admission.setPolicy(odd)
	if got := resolve("weak"); len(got) != 0 {
		t.Errorf("an API entry resolved as a local candidate: %v", got)
	}
}

func TestModelList_OffersEveryTierWithEntriesAndBothVariants(t *testing.T) {
	pol, err := nodepolicy.Parse([]byte(apiTierPolicy))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	fx := newTierFixture(t, ollamaCase(t), pol, map[string][]string{"a": {"mid:7b"}}, nil)
	var ids []string
	for _, raw := range fx.p.soleFacade().tierModelEntries(map[string]modelListItem{}) {
		var item struct {
			ID string `json:"id"`
		}
		if err := json.Unmarshal(raw, &item); err != nil {
			t.Fatalf("decode %s: %v", raw, err)
		}
		ids = append(ids, item.ID)
	}
	want := []string{"weak", "weak-api", "weak-local", "medium", "medium-api", "medium-local"}
	if !slices.Equal(ids, want) {
		t.Fatalf("tier model entries = %v, want %v (weak has only an API entry; strong has none)", ids, want)
	}
}

// declare gives a fixture node an advertised role declaration, the way the
// relay overlay carries a peer's policy advertisement.
func (fx *tierFixture) declare(t *testing.T, id string, roles map[string]nodepolicy.AdvertisedRole) {
	t.Helper()
	f := fx.p.soleFacade()
	for _, n := range f.discovery.Nodes() {
		if n.ID == id {
			n.Roles = roles
			n.PolicyRevision = "1"
			f.discovery.AddManual(n)
			return
		}
	}
	t.Fatalf("no fixture node %q", id)
}

func ollamaRole(model string) nodepolicy.AdvertisedRole {
	return nodepolicy.AdvertisedRole{Engine: "ollama", Model: model}
}

// tierCand is the part of a candidate that tier resolution decides.
type tierCand struct {
	node, model string
	rank        int
	declared    bool
}

func resolvedTier(f *facade, tier string, needs requestNeeds, crossEngine bool) []tierCand {
	var out []tierCand
	for _, c := range f.resolveTierCandidates(tier, needs, crossEngine) {
		out = append(out, tierCand{c.id, c.model, c.tierRank, c.declared})
	}
	return out
}

// A node that declares roles is authoritative for itself: its declared role
// replaces the router's entries on that node, a tier it omits offers nothing
// even when the router's entry names a model it has, and an undeclared node
// still receives the router's entries.
func TestTierRoles_DeclaringNodeIsAuthoritative(t *testing.T) {
	pol := ollamaTierPolicy(map[string][]nodepolicy.TierEntry{
		"weak":   {{Kind: nodepolicy.EntryLocal, Engine: "ollama", Model: "router-weak:1b"}},
		"medium": {{Kind: nodepolicy.EntryLocal, Engine: "ollama", Model: "router-mid:7b"}},
	})
	pol.TierPolicy.AllowStronger = false
	fx := newTierFixture(t, ollamaCase(t), pol, map[string][]string{
		"a": {"router-weak:1b", "router-mid:7b", "own-weak:1b"},
		"b": {"router-weak:1b", "router-mid:7b"},
	}, nil)
	f := fx.p.soleFacade()

	t.Run("no declaration keeps router entries", func(t *testing.T) {
		want := []tierCand{{"a", "router-weak:1b", 0, false}, {"b", "router-weak:1b", 0, false}}
		if got := resolvedTier(f, "weak", requestNeeds{}, false); !slices.Equal(got, want) {
			t.Fatalf("candidates = %+v, want %+v", got, want)
		}
	})

	fx.declare(t, "a", map[string]nodepolicy.AdvertisedRole{"weak": ollamaRole("own-weak:1b")})

	t.Run("declared role replaces router entries on that node", func(t *testing.T) {
		want := []tierCand{{"a", "own-weak:1b", 0, true}, {"b", "router-weak:1b", 0, false}}
		if got := resolvedTier(f, "weak", requestNeeds{}, false); !slices.Equal(got, want) {
			t.Fatalf("candidates = %+v, want %+v", got, want)
		}
	})
	t.Run("omitted tier offers nothing", func(t *testing.T) {
		want := []tierCand{{"b", "router-mid:7b", 0, false}}
		if got := resolvedTier(f, "medium", requestNeeds{}, false); !slices.Equal(got, want) {
			t.Fatalf("candidates = %+v, want %+v (a declares no medium role)", got, want)
		}
	})
	t.Run("request is sent the declared model", func(t *testing.T) {
		rec := fx.do(t, "/v1/chat/completions", `{"model":"weak"}`)
		if rec.Code != http.StatusOK || fx.servedBy(t) != "a" {
			t.Fatalf("status %d served by %q, want a", rec.Code, fx.servedBy(t))
		}
		if got := fx.eng["a"].lastBody(); got != `{"model":"own-weak:1b"}` {
			t.Fatalf("engine got %s", got)
		}
	})
}

// Two peers may declare different models for the same tier; each offers its
// own, and a request to either is sent that node's model.
func TestTierRoles_PeersDeclareDifferentModelsForOneTier(t *testing.T) {
	fx := newTierFixture(t, ollamaCase(t), ollamaTierPolicy(nil), map[string][]string{
		"a": {"alpha:1b", "beta:1b"}, "b": {"alpha:1b", "beta:1b"},
	}, nil)
	fx.declare(t, "a", map[string]nodepolicy.AdvertisedRole{"weak": ollamaRole("beta:1b")})
	fx.declare(t, "b", map[string]nodepolicy.AdvertisedRole{"weak": ollamaRole("alpha:1b")})
	want := []tierCand{{"b", "alpha:1b", 0, true}, {"a", "beta:1b", 0, true}}
	if got := resolvedTier(fx.p.soleFacade(), "weak", requestNeeds{}, false); !slices.Equal(got, want) {
		t.Fatalf("candidates = %+v, want %+v", got, want)
	}
	// Equal load: reservation takes the first scheduler-listed node, a.
	if rec := fx.do(t, "/v1/chat/completions", `{"model":"weak"}`); rec.Code != http.StatusOK || fx.servedBy(t) != "a" {
		t.Fatalf("status %d served by %q, want a", rec.Code, fx.servedBy(t))
	}
	if got := fx.eng["a"].lastBody(); got != `{"model":"beta:1b"}` {
		t.Fatalf("engine got %s, want a's declared model", got)
	}
}

// A declaration never bypasses inventory: a role whose model the node does
// not have is no candidate, and the node does not fall back to the router's
// entries for that tier either.
func TestTierRoles_DeclaredModelMissingFromInventory(t *testing.T) {
	pol := ollamaTierPolicy(map[string][]nodepolicy.TierEntry{
		"weak": {{Kind: nodepolicy.EntryLocal, Engine: "ollama", Model: "have:1b"}},
	})
	pol.TierPolicy.AllowStronger = false
	fx := newTierFixture(t, ollamaCase(t), pol, map[string][]string{"a": {"have:1b"}}, nil)
	fx.declare(t, "a", map[string]nodepolicy.AdvertisedRole{"weak": ollamaRole("missing:1b")})
	if got := resolvedTier(fx.p.soleFacade(), "weak", requestNeeds{}, false); len(got) != 0 {
		t.Fatalf("candidates = %+v, want none", got)
	}
	if rec := fx.do(t, "/v1/chat/completions", `{"model":"weak"}`); rec.Code != http.StatusNotFound || fx.servedBy(t) != "" {
		t.Fatalf("status %d served by %q, want 404 and no engine request", rec.Code, fx.servedBy(t))
	}
}

// A declared role carries its node's advertised bounds, filtered like an
// entry's.
func TestTierRoles_AdvertisedBoundsFilter(t *testing.T) {
	fx := newTierFixture(t, ollamaCase(t), ollamaTierPolicy(nil), map[string][]string{
		"a": {"small:1b"}, "b": {"tools:1b"},
	}, nil)
	fx.declare(t, "a", map[string]nodepolicy.AdvertisedRole{"weak": {Engine: "ollama", Model: "small:1b", ContextTokens: 100}})
	fx.declare(t, "b", map[string]nodepolicy.AdvertisedRole{"weak": {Engine: "ollama", Model: "tools:1b", Capabilities: []nodepolicy.Capability{nodepolicy.CapTools}}})
	f := fx.p.soleFacade()
	if got, want := resolvedTier(f, "weak", requestNeeds{caps: []nodepolicy.Capability{nodepolicy.CapTools}}, false), []tierCand{{"b", "tools:1b", 0, true}}; !slices.Equal(got, want) {
		t.Errorf("tools candidates = %+v, want %+v", got, want)
	}
	if got, want := resolvedTier(f, "weak", requestNeeds{contextTokens: 1000}, false), []tierCand{{"b", "tools:1b", 0, true}}; !slices.Equal(got, want) {
		t.Errorf("long-context candidates = %+v, want %+v", got, want)
	}
}

// allowStronger and allowWeaker are the router's, applied to declared roles
// as to entries.
func TestTierRoles_StrongerAndWeakerFallback(t *testing.T) {
	pol := ollamaTierPolicy(nil)
	fx := newTierFixture(t, ollamaCase(t), pol, map[string][]string{"a": {"big:70b"}, "b": {"small:1b"}}, nil)
	fx.declare(t, "a", map[string]nodepolicy.AdvertisedRole{"strong": ollamaRole("big:70b")})
	fx.declare(t, "b", map[string]nodepolicy.AdvertisedRole{"weak": ollamaRole("small:1b")})
	f := fx.p.soleFacade()
	set := func(stronger, weaker bool) {
		pol.TierPolicy.AllowStronger, pol.TierPolicy.AllowWeaker = stronger, weaker
		fx.p.admission.setPolicy(pol)
	}

	set(false, false)
	if got, want := resolvedTier(f, "weak", requestNeeds{}, false), []tierCand{{"b", "small:1b", 0, true}}; !slices.Equal(got, want) {
		t.Errorf("weak, no fallback = %+v, want %+v", got, want)
	}
	if got := resolvedTier(f, "medium", requestNeeds{}, false); len(got) != 0 {
		t.Errorf("medium, no fallback = %+v, want none", got)
	}
	set(true, false)
	if got, want := resolvedTier(f, "medium", requestNeeds{}, false), []tierCand{{"a", "big:70b", 1, true}}; !slices.Equal(got, want) {
		t.Errorf("medium, allowStronger = %+v, want %+v", got, want)
	}
	set(false, true)
	if got, want := resolvedTier(f, "medium", requestNeeds{}, false), []tierCand{{"b", "small:1b", 1, true}}; !slices.Equal(got, want) {
		t.Errorf("medium, allowWeaker = %+v, want %+v", got, want)
	}
}

// Warmth orders declared candidates exactly as it orders entries, and of equal
// rank and warmth a declared candidate leads a router entry.
func TestTierRoles_WarmOrdering(t *testing.T) {
	nodes := map[string][]string{"a": {"entry:1b"}, "d": {"declared:1b"}, "m": {"mid:7b"}}
	tiers := map[string][]nodepolicy.TierEntry{"weak": {{Kind: nodepolicy.EntryLocal, Engine: "ollama", Model: "entry:1b"}}}
	order := func(t *testing.T, preferLoaded string, loaded map[string][]string) []string {
		t.Helper()
		pol := ollamaTierPolicy(tiers)
		pol.TierPolicy.PreferLoaded = preferLoaded
		fx := newTierFixture(t, ollamaCase(t), pol, nodes, loaded)
		fx.declare(t, "d", map[string]nodepolicy.AdvertisedRole{"weak": ollamaRole("declared:1b")})
		fx.declare(t, "m", map[string]nodepolicy.AdvertisedRole{"medium": ollamaRole("mid:7b")})
		var out []string
		for _, c := range resolvedTier(fx.p.soleFacade(), "weak", requestNeeds{}, false) {
			out = append(out, c.node)
		}
		return out
	}

	if got, want := order(t, "tier", nil), []string{"d", "a", "m"}; !slices.Equal(got, want) {
		t.Errorf("all cold = %v, want %v (declared before entry, then the stronger tier)", got, want)
	}
	if got, want := order(t, "tier", map[string][]string{"a": {"entry:1b"}}), []string{"a", "d", "m"}; !slices.Equal(got, want) {
		t.Errorf("entry warm = %v, want %v", got, want)
	}
	if got, want := order(t, "tier", map[string][]string{"m": {"mid:7b"}}), []string{"d", "a", "m"}; !slices.Equal(got, want) {
		t.Errorf("tier mode, stronger warm = %v, want %v", got, want)
	}
	if got, want := order(t, "any", map[string][]string{"m": {"mid:7b"}}), []string{"m", "d", "a"}; !slices.Equal(got, want) {
		t.Errorf("any mode, stronger warm = %v, want %v", got, want)
	}
}

// Load balancing among declared candidates reserves only within the leading
// class, so the least-loaded declared node of the requested tier leads and an
// idle node of a fallback tier is never promoted.
func TestTierRoles_LoadBalancingKeepsLeadingClass(t *testing.T) {
	fx := newTierFixture(t, ollamaCase(t), ollamaTierPolicy(nil), map[string][]string{
		"a": {"x:1b"}, "b": {"y:1b"}, "c": {"z:7b"},
	}, nil)
	fx.declare(t, "a", map[string]nodepolicy.AdvertisedRole{"weak": ollamaRole("x:1b")})
	fx.declare(t, "b", map[string]nodepolicy.AdvertisedRole{"weak": ollamaRole("y:1b")})
	fx.declare(t, "c", map[string]nodepolicy.AdvertisedRole{"medium": ollamaRole("z:7b")})
	applySnapshot(fx.p, schedulerwire.Priority{
		Nodes: []string{"c", "a", "b"},
		Ranks: []schedulerwire.NodeRank{{ID: "c", Pending: 0}, {ID: "a", Pending: 3, Rank: 1}, {ID: "b", Pending: 2, Rank: 2}},
	})
	f := fx.p.soleFacade()
	ordered, held := f.reserveRound(f.resolveTierCandidates("weak", requestNeeds{}, false), true)
	defer fx.p.releaseReservation(held)
	var got []string
	for _, c := range ordered {
		got = append(got, c.id)
	}
	if want := []string{"b", "a", "c"}; !slices.Equal(got, want) {
		t.Fatalf("reserved order = %v, want %v", got, want)
	}
}

// The router's own node follows the same rule through its policy's roles: a
// self declaration replaces the router's entries for this node, crosses
// engines only on an OpenAI-compatible route, and stays on its engine on a
// native one.
func TestTierRoles_SelfDeclarationCrossEngineAndNative(t *testing.T) {
	w := &recordingWriter{}
	p, oll, lm := twoEngineProxy(t, w)
	lmEngine := newRecordingEngine(t, http.StatusOK)
	ollEngine := newRecordingEngine(t, http.StatusOK)
	lm.discovery.AddManual(selfNode("self", lm.profile.FacadePort, "qwen3-8b"))
	setBackend(t, lm, lmEngine.URL, true)
	oll.discovery.AddManual(selfNode("self", oll.profile.FacadePort, "llama3:latest"))
	setBackend(t, oll, ollEngine.URL, true)

	pol := nodepolicy.Default()
	pol.Tiers = map[string][]nodepolicy.TierEntry{"strong": {{Kind: nodepolicy.EntryLocal, Engine: "ollama", Model: "llama3"}}}
	p.admission.setPolicy(pol)

	// Undeclared: the router entry serves this node.
	if got, want := resolvedTier(oll, "strong", requestNeeds{}, true), []tierCand{{"self", "llama3", 0, false}}; !slices.Equal(got, want) {
		t.Fatalf("undeclared self = %+v, want %+v", got, want)
	}

	pol.Roles = map[string]nodepolicy.Role{"strong": {Engine: "lmstudio", Model: "qwen3-8b"}}
	if err := pol.Validate(); err != nil {
		t.Fatal(err)
	}
	p.admission.setPolicy(pol)

	cands := oll.resolveTierCandidates("strong", requestNeeds{}, true)
	if len(cands) != 1 || cands[0].engine != "lmstudio" || cands[0].fac != lm || !cands[0].declared || cands[0].model != "qwen3-8b" {
		t.Fatalf("declared self candidates = %+v, want only the lmstudio role", cands)
	}
	rec := httptest.NewRecorder()
	oll.handleHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"strong"}`)))
	if rec.Code != http.StatusOK || lmEngine.hits() != 1 || ollEngine.hits() != 0 {
		t.Fatalf("status %d, lmstudio hits %d, ollama hits %d", rec.Code, lmEngine.hits(), ollEngine.hits())
	}

	// A native route never crosses engines, and the router's ollama entry no
	// longer applies to this node.
	rec = httptest.NewRecorder()
	oll.handleHTTP(rec, httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(`{"model":"strong"}`)))
	if rec.Code != http.StatusNotFound || ollEngine.hits() != 0 {
		t.Fatalf("native route: status %d, ollama hits %d, want 404 and none", rec.Code, ollEngine.hits())
	}
}

// A peer's declared role on another engine is reached through that engine's
// facade on an OpenAI-compatible route, and not at all on a native one.
func TestTierRoles_PeerCrossEngineRole(t *testing.T) {
	w := &recordingWriter{}
	p, oll, lm := twoEngineProxy(t, w)
	peerLM := newRecordingEngine(t, http.StatusOK)
	n := nodeForModel(t, "peer", peerLM.URL, "qwen3-8b")
	n.Roles = map[string]nodepolicy.AdvertisedRole{"weak": {Engine: "lmstudio", Model: "qwen3-8b"}}
	lm.discovery.AddManual(n)
	p.admission.setPolicy(nodepolicy.Default())

	cands := oll.resolveTierCandidates("weak", requestNeeds{}, true)
	if len(cands) != 1 || cands[0].fac != lm || cands[0].engine != "lmstudio" || !cands[0].declared ||
		cands[0].url.Host != strings.TrimPrefix(peerLM.URL, "http://") {
		t.Fatalf("cross-engine candidates = %+v", cands)
	}
	if got := oll.resolveTierCandidates("weak", requestNeeds{}, false); len(got) != 0 {
		t.Fatalf("native candidates = %+v, want none", got)
	}
	if got := lm.resolveTierCandidates("weak", requestNeeds{}, false); len(got) != 1 || !got[0].declared {
		t.Fatalf("own-engine candidates = %+v, want the declared role", got)
	}
}

// qualifiedRole is an Ollama role advertising finite qualification.
func qualifiedRole(model string) nodepolicy.AdvertisedRole {
	return nodepolicy.AdvertisedRole{Engine: "ollama", Model: model, ContextTokens: 4096, MaxOutputTokens: 1024,
		Qualification: &nodepolicy.QualificationRef{ID: "q", Revision: "1"}}
}

// advertise gives a fixture node a role declaration and a strict capability,
// as its relayed policy advertisement would.
func (fx *tierFixture) advertise(t *testing.T, id string, roles map[string]nodepolicy.AdvertisedRole, capability *nodepolicy.StrictAdmissionCapability) {
	t.Helper()
	fx.declare(t, id, roles)
	f := fx.p.soleFacade()
	n, ok := discoveredNode(f, id)
	if !ok {
		t.Fatalf("no fixture node %q", id)
	}
	n.StrictAdmission = capability
	f.discovery.AddManual(n)
}

func strictCapabilityFor(engine string, models ...string) *nodepolicy.StrictAdmissionCapability {
	c := &nodepolicy.StrictAdmissionCapability{Version: nodepolicy.StrictAdmissionVersion}
	for _, m := range models {
		c.Models = append(c.Models, nodepolicy.StrictAdmissionModel{Engine: engine, Model: m, MaxConcurrent: 1, Identity: strictRequest().Identity})
	}
	return c
}

// Hybrid role eligibility admits only a declared role with finite
// qualification on a node that advertises a valid strict entry for it; legacy
// tier resolution of the same request is unchanged.
func TestHybridLocalCandidatesRequireQualifiedDeclaredRoles(t *testing.T) {
	pol := ollamaTierPolicy(map[string][]nodepolicy.TierEntry{"weak": {{Kind: nodepolicy.EntryLocal, Engine: "ollama", Model: "q:1b"}}})
	pol.TierPolicy.AllowStronger = false
	nodes := map[string][]string{}
	for _, id := range []string{"eligible", "unqualified", "no-capability", "unsupported", "other-model", "undeclared", "malformed"} {
		nodes[id] = []string{"q:1b"}
	}
	fx := newTierFixture(t, ollamaCase(t), pol, nodes, nil)
	f := fx.p.soleFacade()
	var unsupported nodepolicy.StrictAdmissionCapability
	if err := json.Unmarshal([]byte(`{"version":2,"models":[{"engine":"ollama","model":"q:1b"}]}`), &unsupported); err != nil {
		t.Fatal(err)
	}
	malformed := qualifiedRole("q:1b")
	malformed.MaxOutputTokens = 4096
	fx.advertise(t, "eligible", map[string]nodepolicy.AdvertisedRole{"weak": qualifiedRole("q:1b")}, strictCapabilityFor("ollama", "q:1b"))
	fx.advertise(t, "unqualified", map[string]nodepolicy.AdvertisedRole{"weak": ollamaRole("q:1b")}, strictCapabilityFor("ollama", "q:1b"))
	fx.advertise(t, "no-capability", map[string]nodepolicy.AdvertisedRole{"weak": qualifiedRole("q:1b")}, nil)
	fx.advertise(t, "unsupported", map[string]nodepolicy.AdvertisedRole{"weak": qualifiedRole("q:1b")}, &unsupported)
	fx.advertise(t, "other-model", map[string]nodepolicy.AdvertisedRole{"weak": qualifiedRole("q:1b")}, strictCapabilityFor("ollama", "r:1b"))
	fx.advertise(t, "malformed", map[string]nodepolicy.AdvertisedRole{"weak": malformed}, strictCapabilityFor("ollama", "q:1b"))

	got := f.hybridLocalCandidates("weak", requestNeeds{}, true)
	if len(got) != 1 || got[0].id != "eligible" || got[0].strict.Identity != strictRequest().Identity || !got[0].role.HybridQualified() {
		t.Fatalf("hybrid candidates = %+v, want only the qualified, capable declared role", got)
	}
	if legacy := resolvedTier(f, "weak", requestNeeds{}, true); len(legacy) != len(nodes) {
		t.Fatalf("legacy tier candidates = %+v, want every node unchanged", legacy)
	}
}

// For itself the router reads its own installed capability, so a withdrawn
// identity removes it at once.
func TestHybridLocalCandidatesSelfUsesItsOwnCapability(t *testing.T) {
	w := &recordingWriter{}
	p, oll, _ := twoEngineProxy(t, w)
	e := newRecordingEngine(t, http.StatusOK)
	oll.discovery.AddManual(selfNode("self", oll.profile.FacadePort, "llama3:latest"))
	setBackend(t, oll, e.URL, true)
	installQualified(t, p.admission, admKey{engine: "ollama", model: "llama3:latest"})
	pol := p.admission.currentPolicy()
	pol.Roles = map[string]nodepolicy.Role{"weak": {Engine: "ollama", Model: "llama3:latest"}}
	pol.Admission.MaxConcurrentPerModel = 1
	p.admission.setPolicy(pol)

	got := oll.hybridLocalCandidates("weak", requestNeeds{}, false)
	if len(got) != 1 || !got[0].self || got[0].strict.Identity != strictRequest().Identity {
		t.Fatalf("hybrid candidates = %+v, want this node with its installed identity", got)
	}
	p.admission.setModelIdentity(map[string]nodepolicy.EngineIdentity{})
	if got := oll.hybridLocalCandidates("weak", requestNeeds{}, false); len(got) != 0 {
		t.Fatalf("hybrid candidates after the identity was withdrawn = %+v, want none", got)
	}
}

// Hybrid selection is the legacy tier order filtered to eligible candidates,
// never a reordering of its own.
func TestHybridLocalCandidatesKeepTierOrder(t *testing.T) {
	pol := ollamaTierPolicy(map[string][]nodepolicy.TierEntry{"weak": {{Kind: nodepolicy.EntryLocal, Engine: "ollama", Model: "q:1b"}}})
	pol.TierPolicy.AllowStronger = false
	fx := newTierFixture(t, ollamaCase(t), pol, map[string][]string{"a": {"q:1b"}, "b": {"q:1b"}, "c": {"q:1b"}}, nil)
	fx.advertise(t, "a", map[string]nodepolicy.AdvertisedRole{"weak": qualifiedRole("q:1b")}, strictCapabilityFor("ollama", "q:1b"))
	fx.advertise(t, "b", map[string]nodepolicy.AdvertisedRole{"weak": ollamaRole("q:1b")}, strictCapabilityFor("ollama", "q:1b"))
	fx.advertise(t, "c", map[string]nodepolicy.AdvertisedRole{"weak": qualifiedRole("q:1b")}, strictCapabilityFor("ollama", "q:1b"))
	f := fx.p.soleFacade()

	var want []string
	for _, c := range f.resolveTierCandidates("weak", requestNeeds{}, true) {
		if c.id != "b" {
			want = append(want, c.id)
		}
	}
	var got []string
	for _, c := range f.hybridLocalCandidates("weak", requestNeeds{}, true) {
		got = append(got, c.id)
	}
	if len(want) != 2 || !slices.Equal(got, want) {
		t.Fatalf("hybrid order = %v, want the tier order without the unqualified node: %v", got, want)
	}
}
