// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"runtime"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/jsonrpc"
)

func TestModelPublisherCatalogCacheRetainsBasisAndExpires(t *testing.T) {
	digest := strings.Repeat("a", 64)
	c, calls := publisherTestCatalog(t, publisherTestBody(digest), http.StatusOK)
	p := publisherTestRecord(digest)
	first, acquired, basis, ok := c.publisherDigest(context.Background(), p)
	if !ok || first != digest || basis != "provider" || acquired == "" {
		t.Fatalf("first digest=%s time=%s basis=%s ok=%t", first, acquired, basis, ok)
	}
	second, cachedAt, basis, ok := c.publisherDigest(context.Background(), p)
	if !ok || second != first || cachedAt != acquired || basis != "cache" || calls.Load() != 1 {
		t.Fatalf("cached digest=%s time=%s basis=%s calls=%d", second, cachedAt, basis, calls.Load())
	}
	entry := c.publisherRevision(p.Repository + "@" + p.Revision)
	entry.mu.Lock()
	entry.fetched = time.Now().Add(-catalogCacheTTL - time.Second)
	entry.mu.Unlock()
	c.client.Transport = healthProbeTransport(func(*http.Request) (*http.Response, error) {
		calls.Add(1)
		return nil, errors.New("synthetic refresh failure")
	})
	if _, _, _, ok := c.publisherDigest(context.Background(), p); ok || calls.Load() != 2 {
		t.Fatal("expired statement silently reused after failed refresh")
	}
	if _, _, _, ok := c.publisherDigest(context.Background(), p); ok || calls.Load() != 2 {
		t.Fatal("expired statement bypassed provider failure backoff")
	}
}

func TestModelPublisherCatalogCacheCapacity(t *testing.T) {
	c := newCatalogService()
	for index := 0; index <= llamaCPPSearchCacheLimit; index++ {
		c.publisherRevision(fmt.Sprintf("owner/repo@%040x", index))
	}
	if len(c.publisherRevisions) != llamaCPPSearchCacheLimit || len(c.publisherOrder) != llamaCPPSearchCacheLimit {
		t.Fatalf("unbounded publisher cache: entries=%d order=%d", len(c.publisherRevisions), len(c.publisherOrder))
	}
	if _, exists := c.publisherRevisions[fmt.Sprintf("owner/repo@%040x", 0)]; exists {
		t.Fatal("oldest provider metadata was not evicted")
	}
}

func TestModelPublisherRejectsProviderRedirect(t *testing.T) {
	c := newCatalogService()
	calls := 0
	c.client.Transport = healthProbeTransport(func(r *http.Request) (*http.Response, error) {
		calls++
		if calls != 1 {
			t.Error("provider redirect reached a different origin")
		}
		return &http.Response{StatusCode: http.StatusFound, Header: http.Header{"Location": {"https://other.invalid/statement"}}, Body: io.NopCloser(strings.NewReader("")), Request: r}, nil
	})
	if _, _, _, ok := c.publisherDigest(context.Background(), publisherTestRecord(strings.Repeat("a", 64))); ok || calls != 1 {
		t.Fatalf("redirect became publisher authority: ok=%t requests=%d", ok, calls)
	}
}

// TestLlamaCPPCatalogFitsInAFrame is the guard for the failure that made this
// method unusable: the reply is one JSON-RPC line, and a line over the
// worker-path frame cap is a terminal read error, not a dropped message. The
// broker's peer then closes while the child keeps running, so nothing restarts
// and every later engine:* call hangs.
//
// The check is on the marshalled result of the largest browse list the
// catalogue can serve, every publisher returning a full page of wide rows,
// because it is bytes on the wire that matter and a row can grow by widening.
func TestLlamaCPPCatalogFitsInAFrame(t *testing.T) {
	srv := httptest.NewServer(fixtureRequestHandler(http.MethodGet, "/catalog", func(w http.ResponseWriter, r *http.Request) {
		author := r.URL.Query().Get("author")
		rows := make([]hfModelRow, llamaCPPLimit)
		for i := range rows {
			rows[i] = ggufRow(fmt.Sprintf("%s/%s-%d", author, strings.Repeat("n", 100), i), i)
			rows[i].Tags = append(rows[i].Tags, strings.Repeat("t", 100))
		}
		_ = json.NewEncoder(w).Encode(rows)
	}))
	defer srv.Close()
	c := newCatalogService()
	c.baseURL = srv.URL + "/catalog"
	res, err := c.Catalog(context.Background(), "llamacpp", "linux", "amd64", "")
	if err != nil {
		t.Fatalf("llama.cpp catalog: %v", err)
	}
	if want := len(llamaCPPPublishers) * llamaCPPLimit; len(res.Models) != want {
		t.Fatalf("got %d models, want %d", len(res.Models), want)
	}
	body, err := json.Marshal(res)
	if err != nil {
		t.Fatalf("marshal catalog: %v", err)
	}
	// The real frame carries a JSON-RPC envelope around this; leave room for it.
	const envelopeAllowance = 4096
	if len(body)+envelopeAllowance > jsonrpc.WorkerFrameBytes {
		t.Errorf("marshalled llama.cpp catalog is %d bytes, over the %d-byte frame cap; "+
			"filter or paginate engine:catalog rather than raising the cap again",
			len(body), jsonrpc.WorkerFrameBytes)
	}
}

// listingBody is a listing holding one llama.cpp download.
func listingBody(t *testing.T) []byte {
	t.Helper()
	body, err := json.Marshal([]hfModelRow{ggufRow("owner/Model-GGUF", 5)})
	if err != nil {
		t.Fatalf("marshal listing: %v", err)
	}
	return body
}

// TestListingCacheCoalescesConcurrentCallers is the guard for the request
// herd. Both front ends plus the warm-up can ask at once, and the point of the
// in-flight channel is that they share one upstream request.
//
// Run under -race this also covers the close-under-lock fix: the channel used to
// be closed after releasing the mutex, leaving a window where an arriving caller
// saw no request in flight and started a second one.
func TestListingCacheCoalescesConcurrentCallers(t *testing.T) {
	var requests atomic.Int32
	release := make(chan struct{})
	var releaseOnce sync.Once
	unblock := func() { releaseOnce.Do(func() { close(release) }) }
	srv := httptest.NewServer(fixtureRequestHandler(http.MethodGet, "/catalog", func(w http.ResponseWriter, _ *http.Request) {
		requests.Add(1)
		<-release // hold the request open so the callers genuinely overlap
		_, _ = w.Write(listingBody(t))
	}))
	defer srv.Close()
	defer unblock()

	c := newCatalogService()
	c.baseURL = srv.URL + "/catalog"
	cache := c.llamaCPPSearch("model")

	const callers = 8
	var wg sync.WaitGroup
	errs := make([]error, callers)
	counts := make([]int, callers)
	for i := range callers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			models, _, err := cache.get(context.Background())
			errs[i], counts[i] = err, len(models)
		}()
	}

	// Let the callers pile up behind the one in flight, then answer.
	time.Sleep(50 * time.Millisecond)
	assertFixtureRejectsProbe(t, srv.URL)
	unblock()
	wg.Wait()

	if got := requests.Load(); got != 1 {
		t.Errorf("%d callers produced %d upstream requests, want 1", callers, got)
	}
	for i := range callers {
		if errs[i] != nil {
			t.Errorf("caller %d: %v", i, errs[i])
		}
		if counts[i] != 1 {
			t.Errorf("caller %d got %d models, want 1", i, counts[i])
		}
	}
}

// TestListingCacheBacksOffAfterFailure is the guard for a stall: only a
// success stamped the cache, so against a dead upstream every later call retried
// and waited out the full timeout before handing back the same stale list.
func TestListingCacheBacksOffAfterFailure(t *testing.T) {
	var requests atomic.Int32
	fail := true
	srv := httptest.NewServer(fixtureRequestHandler(http.MethodGet, "/catalog", func(w http.ResponseWriter, _ *http.Request) {
		requests.Add(1)
		if fail {
			w.WriteHeader(http.StatusBadGateway)
			return
		}
		_, _ = w.Write(listingBody(t))
	}))
	defer srv.Close()

	c := newCatalogService()
	c.baseURL = srv.URL + "/catalog"
	cache := c.llamaCPPSearch("model")
	assertFixtureRejectsProbe(t, srv.URL)

	// Seed a good list, then start failing.
	fail = false
	if _, _, err := cache.get(context.Background()); err != nil {
		t.Fatalf("seed fetch: %v", err)
	}
	fail = true

	// Force a refresh by ageing the cache past its TTL.
	cache.mu.Lock()
	cache.fetched = time.Now().Add(-2 * catalogCacheTTL)
	cache.mu.Unlock()

	before := requests.Load()
	for range 3 {
		models, _, err := cache.get(context.Background())
		if err != nil {
			t.Fatalf("a failed refresh should still serve the stale list: %v", err)
		}
		if len(models) != 1 {
			t.Errorf("stale list lost its models: %d", len(models))
		}
	}
	if got := requests.Load() - before; got != 1 {
		t.Errorf("three calls after a failure made %d requests, want 1 then backoff", got)
	}
}

// TestListingCacheBacksOffWithNothingCached is the same guard for a machine
// that has never reached the upstream. The backoff was keyed on having a list
// to serve, so with an empty cache every call started a fresh fetch and waited
// out the full timeout — the warm-up, then each modal open, then the terminal
// browser, each in turn.
func TestListingCacheBacksOffWithNothingCached(t *testing.T) {
	var requests atomic.Int32
	srv := httptest.NewServer(fixtureRequestHandler(http.MethodGet, "/catalog", func(w http.ResponseWriter, _ *http.Request) {
		requests.Add(1)
		w.WriteHeader(http.StatusBadGateway)
	}))
	defer srv.Close()

	c := newCatalogService()
	c.baseURL = srv.URL + "/catalog"
	cache := c.llamaCPPSearch("model")
	assertFixtureRejectsProbe(t, srv.URL)

	for i := range 3 {
		if _, _, err := cache.get(context.Background()); err == nil {
			t.Fatalf("call %d: a failing upstream with nothing cached returned no error", i)
		}
	}
	if got := requests.Load(); got != 1 {
		t.Errorf("three calls against a failing upstream made %d requests, want 1 then backoff", got)
	}

	// Once the backoff has run out, the next call tries again.
	cache.mu.Lock()
	cache.failed = time.Now().Add(-2 * catalogRetryAfterFailure)
	cache.mu.Unlock()
	_, _, _ = cache.get(context.Background())
	if got := requests.Load(); got != 2 {
		t.Errorf("after the backoff expired: %d requests, want a second attempt", got)
	}
}

func TestNormalizeCatalogEngine(t *testing.T) {
	expectNormalized := func(in, want string) {
		t.Helper()
		if got := normalizeCatalogEngine(in); got != want {
			t.Errorf("normalizeCatalogEngine(%q) = %q, want %q", in, got, want)
		}
	}
	expectNormalized("LlamaCpp", "llamacpp")
	expectNormalized("  llamacpp ", "llamacpp")
	expectNormalized("llamacpp", "llamacpp")
	expectNormalized("llama-cpp", "llamacpp")
	expectNormalized("llama.cpp", "llamacpp")
	expectNormalized("vllm", "vllm")
}

// TestCatalogRejectsUnknownEngine checks an engine with no curated source errors
// rather than returning an empty list that reads as "nothing available".
func TestCatalogRejectsUnknownEngine(t *testing.T) {
	c := newCatalogService()
	if _, err := c.Catalog(context.Background(), "vllm", "linux", "amd64", ""); err == nil {
		t.Error("unknown engine returned a catalog")
	}
}

// TestFilterForTarget is the guard for two versions of one bug. The list was
// first filtered by whichever machine served it, so browsing for a Mac peer
// from a Linux box hid every model that peer could use; then by operating
// system alone, so an Intel Mac was offered MLX models it can never install.
func TestFilterForTarget(t *testing.T) {
	models := []CatalogModel{
		{ID: "plain/gguf", Name: "plain/gguf"},
		{ID: "apple/mlx", Name: "apple/mlx", AppleOnly: true},
	}

	if got := filterForTarget(models, "darwin", "arm64"); len(got) != 2 {
		t.Errorf("Apple Silicon target got %d models, want both", len(got))
	}
	for _, target := range []struct{ platform, arch string }{
		{"darwin", "amd64"},
		{"darwin", ""},
		{"linux", "arm64"},
		{"windows", "amd64"},
	} {
		got := filterForTarget(models, target.platform, target.arch)
		if len(got) != 1 || got[0].ID != "plain/gguf" {
			t.Errorf("%s/%s target got %v, want only the portable model",
				target.platform, target.arch, got)
		}
	}
}

// TestCatalogEchoesTarget checks the reply says which machine it was filtered
// for, so a client can tell the operator rather than presenting a filtered list
// as universal.
func TestCatalogEchoesTarget(t *testing.T) {
	srv, _ := llamaCPPServer(t, "")
	c := newCatalogService()
	c.baseURL = srv.URL + "/catalog"
	res, err := c.Catalog(context.Background(), "llamacpp", "darwin", "amd64", "")
	if err != nil {
		t.Fatalf("catalog: %v", err)
	}
	if res.Platform != "darwin" || res.Arch != "amd64" {
		t.Errorf("target = %s/%s, want the requested darwin/amd64", res.Platform, res.Arch)
	}

	// Omitting both means this host.
	res, err = c.Catalog(context.Background(), "llamacpp", "", "", "")
	if err != nil {
		t.Fatalf("catalog: %v", err)
	}
	if res.Platform != runtime.GOOS || res.Arch != runtime.GOARCH {
		t.Errorf("target = %s/%s, want the local %s/%s",
			res.Platform, res.Arch, runtime.GOOS, runtime.GOARCH)
	}

	// A named platform does not borrow this host's architecture: that would
	// describe a machine the caller did not ask about.
	res, err = c.Catalog(context.Background(), "llamacpp", "darwin", "", "")
	if err != nil {
		t.Fatalf("catalog: %v", err)
	}
	if res.Arch != "" {
		t.Errorf("Arch = %q for a platform named without one, want it left unknown", res.Arch)
	}
}

// ggufRow is a listing entry llama.cpp can download, for a test to break one
// field of.
func ggufRow(id string, downloads int, files ...string) hfModelRow {
	if len(files) == 0 {
		files = []string{"model-Q4_K_M.gguf"}
	}
	siblings := make([]hfSibling, len(files))
	for i, f := range files {
		siblings[i] = hfSibling{RFilename: f}
	}
	return hfModelRow{
		ID:           id,
		Downloads:    downloads,
		LastModified: "2026-09-01T12:00:00.000Z",
		Tags:         []string{"gguf", "conversational"},
		Gated:        json.RawMessage(`false`),
		PipelineTag:  "text-generation",
		Siblings:     siblings,
	}
}

// TestNormalizeLlamaCPPRowsKeepsOnlyDownloadable is the guard for offering a
// download that cannot succeed. The listing's GGUF filter also matches gated
// and private repos, embedding models, and repos with no file at the offered
// quantization, and every one of those fails only after the operator picked it.
func TestNormalizeLlamaCPPRowsKeepsOnlyDownloadable(t *testing.T) {
	gated := ggufRow("a/gated", 1)
	gated.Gated = json.RawMessage(`"auto"`)
	unstated := ggufRow("a/unstated", 1)
	unstated.Gated = nil
	private := ggufRow("a/private", 1)
	private.Private = true
	untagged := ggufRow("a/untagged", 1)
	untagged.Tags = []string{"conversational"}
	embedding := ggufRow("a/embedding", 1)
	embedding.PipelineTag = "feature-extraction"
	undated := ggufRow("a/undated", 1)
	undated.LastModified = "yesterday"

	rows := []hfModelRow{
		ggufRow("ggml-org/kept", 10),
		ggufRow("a/split", 5, "q4/model-Q4_K_M-00001-of-00002.gguf", "q4/model-Q4_K_M-00002-of-00002.gguf"),
		gated, unstated, private, untagged, embedding, undated,
		ggufRow("a/no-q4", 1, "model-Q8_0.gguf"),
		ggufRow("a/projector-only", 1, "mmproj-model-Q4_K_M.gguf"),
		ggufRow("a/eagle3-head-only", 1, "eagle3-model-Q4_K_M.gguf"),
		ggufRow("a/dflash-head-only", 1, "dflash-model-Q4_K_M.gguf"),
		ggufRow("a/dspark-head-only", 1, "dspark-model-Q4_K_M.gguf"),
		ggufRow("a/second-shard-only", 1, "model-Q4_K_M-00002-of-00002.gguf"),
		ggufRow("../escape", 1),
		ggufRow("a/has space", 1),
	}
	got := normalizeLlamaCPPRows(rows)
	ids := make([]string, len(got))
	for i, m := range got {
		ids[i] = m.ID
	}
	if strings.Join(ids, ",") != "ggml-org/kept:Q4_K_M,a/split:Q4_K_M" {
		t.Fatalf("kept %v, want only the two downloadable repos, most downloaded first", ids)
	}

	m := got[0]
	if m.Name != m.ID {
		t.Errorf("name %q and id %q disagree; both must be the download name", m.Name, m.ID)
	}
	if m.Author != "ggml-org" || m.URL != "https://huggingface.co/ggml-org/kept" {
		t.Errorf("author %q, url %q", m.Author, m.URL)
	}
	if !hasTag(m.Tags, "Q4_K_M") {
		t.Errorf("tags %v do not name the quantization offered", m.Tags)
	}
}

// TestNormalizeLlamaCPPRowsDedupesAcrossPublishers checks a repo listed twice,
// as it is when a search and a publisher page overlap, yields one row.
func TestNormalizeLlamaCPPRowsDedupesAcrossPublishers(t *testing.T) {
	got := normalizeLlamaCPPRows([]hfModelRow{ggufRow("a/model", 1), ggufRow("a/model", 7)})
	if len(got) != 1 || got[0].Downloads != 7 {
		t.Errorf("got %+v, want one row carrying the later entry", got)
	}
}

// llamaCPPServer answers the listing for each approved publisher with one repo
// of its own, and a search with a repo named after the query. It records every
// request's query string.
func llamaCPPServer(t *testing.T, failAuthor string) (*httptest.Server, func() []string) {
	t.Helper()
	var mu sync.Mutex
	var requests []string
	srv := httptest.NewServer(fixtureRequestHandler(http.MethodGet, "/catalog", func(w http.ResponseWriter, r *http.Request) {
		q := r.URL.Query()
		mu.Lock()
		requests = append(requests, r.URL.RawQuery)
		mu.Unlock()
		if q.Get("filter") != "gguf" || q.Get("full") != "true" {
			t.Errorf("listing asked for %q, want GGUF repos in full", r.URL.RawQuery)
		}
		rows := []hfModelRow{}
		switch author, search := q.Get("author"), q.Get("search"); {
		case author != "" && author == failAuthor:
			w.WriteHeader(http.StatusBadGateway)
			return
		case author != "":
			rows = []hfModelRow{ggufRow(author+"/model-GGUF", 1)}
		case search == "nothing":
		case search != "":
			rows = []hfModelRow{ggufRow("found/"+strings.ReplaceAll(search, " ", "-"), 1)}
		}
		_ = json.NewEncoder(w).Encode(rows)
	}))
	t.Cleanup(srv.Close)
	assertFixtureRejectsProbe(t, srv.URL)
	return srv, func() []string {
		mu.Lock()
		defer mu.Unlock()
		return append([]string(nil), requests...)
	}
}

// TestLlamaCPPCatalogReadsEveryPublisher checks the browse list is the union of
// the approved publishers, and that the reply says a search is possible.
func TestLlamaCPPCatalogReadsEveryPublisher(t *testing.T) {
	srv, _ := llamaCPPServer(t, "")
	c := newCatalogService()
	c.baseURL = srv.URL + "/catalog"

	res, err := c.Catalog(context.Background(), "llama.cpp", "linux", "arm64", "")
	if err != nil {
		t.Fatalf("llama.cpp catalog: %v", err)
	}
	if len(res.Models) != len(llamaCPPPublishers) {
		t.Fatalf("got %d models, want one from each of %v", len(res.Models), llamaCPPPublishers)
	}
	for _, publisher := range llamaCPPPublishers {
		if !slices.ContainsFunc(res.Models, func(m CatalogModel) bool { return m.Author == publisher }) {
			t.Errorf("nothing from %s", publisher)
		}
	}
	if !res.Searchable || res.Query != "" {
		t.Errorf("searchable %v, query %q; want a searchable browse list", res.Searchable, res.Query)
	}
}

// TestLlamaCPPCatalogFailsWhole checks one publisher failing fails the fetch,
// rather than caching for six hours a list that is silently missing it.
func TestLlamaCPPCatalogFailsWhole(t *testing.T) {
	srv, _ := llamaCPPServer(t, "bartowski")
	c := newCatalogService()
	c.baseURL = srv.URL + "/catalog"

	if res, err := c.Catalog(context.Background(), "llamacpp", "", "", ""); err == nil {
		t.Errorf("a failed publisher still produced a list of %d", len(res.Models))
	}
}

// TestLlamaCPPSearch checks a query reaches the upstream once, is cached
// regardless of case, and that a search matching nothing is an answer rather
// than a failure.
func TestLlamaCPPSearch(t *testing.T) {
	srv, requests := llamaCPPServer(t, "")
	c := newCatalogService()
	c.baseURL = srv.URL + "/catalog"

	res, err := c.Catalog(context.Background(), "llamacpp", "", "", "  Gemma 3  ")
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if res.Query != "Gemma 3" || len(res.Models) != 1 || res.Models[0].ID != "found/Gemma-3:Q4_K_M" {
		t.Fatalf("query %q answered with %+v", res.Query, res.Models)
	}
	if _, err := c.Catalog(context.Background(), "llamacpp", "", "", "gemma 3"); err != nil {
		t.Fatalf("repeat search: %v", err)
	}
	if got := len(requests()); got != 1 {
		t.Errorf("the same query twice made %d requests, want 1", got)
	}

	res, err = c.Catalog(context.Background(), "llamacpp", "", "", "nothing")
	if err != nil {
		t.Fatalf("a search matching nothing failed: %v", err)
	}
	if len(res.Models) != 0 {
		t.Errorf("got %d models for a query that matches none", len(res.Models))
	}
}

// TestFailedSearchDoesNotRevealTheQuery is the guard for search text leaking
// into logs. net/http puts the request URL in its error, and a search's URL
// carries what the operator typed; returned as-is, it was logged here and by
// every client that relays the error.
//
// The message cannot be compared whole: it ends in the operating system's
// connection error, which names a random port. What has to hold is that the
// error carrying the URL is gone.
func TestFailedSearchDoesNotRevealTheQuery(t *testing.T) {
	srv := httptest.NewServer(http.NotFoundHandler())
	srv.Close() // every request now fails to connect
	c := newCatalogService()
	c.baseURL = srv.URL + "/catalog"

	_, err := c.Catalog(context.Background(), "llamacpp", "", "", "private model name")
	if err == nil {
		t.Fatal("a search against an unreachable upstream succeeded")
	}
	var urlErr *url.Error
	if errors.As(err, &urlErr) {
		t.Errorf("the error still carries the request URL: %v", err)
	}
}

// TestOverLongSearchIsRefused checks a query past maxCatalogQuery is an error
// that never reaches the upstream, rather than being shortened into a search
// for something the operator did not type, and that the error does not repeat
// the query.
func TestOverLongSearchIsRefused(t *testing.T) {
	srv, requests := llamaCPPServer(t, "")
	c := newCatalogService()
	c.baseURL = srv.URL + "/catalog"

	query := strings.TrimSpace(strings.Repeat("private ", maxCatalogQuery/len("private ")+1))
	_, err := c.Catalog(context.Background(), "llamacpp", "", "", query)
	if err == nil {
		t.Fatalf("a %d-character query was answered", len(query))
	}
	if got, want := err.Error(), "a model search is limited to 100 characters"; got != want {
		t.Errorf("error %q, want %q", got, want)
	}
	if got := requests(); len(got) != 0 {
		t.Errorf("an over-long query reached the upstream: %v", got)
	}
}

// TestSearchAtTheLengthLimitIsAnswered checks the limit counts the query the
// operator meant: exactly maxCatalogQuery characters is searched in full, and
// surrounding space does not count against it.
func TestSearchAtTheLengthLimitIsAnswered(t *testing.T) {
	srv, requests := llamaCPPServer(t, "")
	c := newCatalogService()
	c.baseURL = srv.URL + "/catalog"

	query := strings.Repeat("a", maxCatalogQuery)
	res, err := c.Catalog(context.Background(), "llamacpp", "", "", "  "+query+"  ")
	if err != nil {
		t.Fatalf("a query at the limit failed: %v", err)
	}
	if res.Query != query {
		t.Errorf("searched for %d characters, want all %d", len(res.Query), len(query))
	}
	if got := len(requests()); got != 1 {
		t.Errorf("a query at the limit made %d requests, want 1", got)
	}
}

// TestLlamaCPPSearchesAreBounded checks the per-query caches are evicted least
// recently used first, since the set of possible queries is unbounded.
func TestLlamaCPPSearchesAreBounded(t *testing.T) {
	c := newCatalogService()
	first := c.llamaCPPSearch("first")
	for i := range llamaCPPSearchCacheLimit {
		c.llamaCPPSearch(fmt.Sprint("query ", i))
	}
	if len(c.llamaSearches) != llamaCPPSearchCacheLimit {
		t.Errorf("%d searches kept, want at most %d", len(c.llamaSearches), llamaCPPSearchCacheLimit)
	}
	if c.llamaCPPSearch("first") == first {
		t.Error("the least recently used search was not evicted")
	}
}
