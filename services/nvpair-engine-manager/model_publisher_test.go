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
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
)

const publisherTestRepo = "owner/tiny-GGUF"
const publisherTestRevision = "0123456789abcdef0123456789abcdef01234567"
const publisherTestFile = "tiny-Q4_K_M.gguf"

func publisherTestRecord(digest string) modelPublisherFile {
	return modelPublisherFile{Path: publisherTestFile, Status: "verified", Provider: "huggingface", Repository: publisherTestRepo, Revision: publisherTestRevision, File: publisherTestFile, Algorithm: "sha256", ExpectedDigest: digest, ActualDigest: digest, AcquisitionBasis: "provider", AcquiredAt: "2026-01-01T00:00:00Z"}
}

func publisherTestBody(digest string) string {
	return fmt.Sprintf(`{"id":%q,"sha":%q,"private":false,"gated":false,"siblings":[{"rfilename":%q,"lfs":{"sha256":%q}}]}`, publisherTestRepo, publisherTestRevision, publisherTestFile, digest)
}

// Inject the catalogue's existing transport boundary. This fixture never opens
// a listener, resolves a provider hostname or forwards to a native transport.
func publisherTestCatalog(t *testing.T, body string, status int) (*catalogService, *atomic.Int64) {
	t.Helper()
	c := newCatalogService()
	calls := &atomic.Int64{}
	c.client.Transport = healthProbeTransport(func(r *http.Request) (*http.Response, error) {
		calls.Add(1)
		want := hfModelsAPI + "/" + publisherTestRepo + "/revision/" + publisherTestRevision + "?blobs=true"
		if r.Method != http.MethodGet || r.URL.String() != want || r.Header.Get("Authorization") != "" || r.URL.User != nil {
			t.Errorf("unexpected public provider selector: method=%s url=%s", r.Method, r.URL)
		}
		return &http.Response{StatusCode: status, Status: http.StatusText(status), Header: make(http.Header), Body: io.NopCloser(strings.NewReader(body)), Request: r}, nil
	})
	return c, calls
}

func publisherCopySource(t *testing.T) *fakeModelSource {
	t.Helper()
	root := t.TempDir()
	writeLlamaCache(t, root, publisherTestRepo, publisherTestRevision, map[string][]byte{publisherTestFile: testBytes(80_000, 3)})
	e := copyTestExecutor(t, "llamacpp", root, true)
	copyTestNativeModels(t, e, "llamacpp", publisherTestRepo+":Q4_K_M")
	source := sourceFromExecutor(t, e, "llamacpp", publisherTestRepo+":Q4_K_M")
	for i := range source.list.Files {
		source.list.Files[i].Publisher = publisherSelector(source.list, source.list.Files[i])
	}
	return source
}

func TestModelPublisherDestinationAcquiresOwnStatement(t *testing.T) {
	plentyOfDisk(t)
	test := func(name, expectedStatus string, providerDigest func(string) string) {
		t.Run(name, func(t *testing.T) {
			src := publisherCopySource(t)
			actual := src.list.Files[0].SHA256
			// The peer asserts verified with its own timestamp. It is deliberately
			// inconsistent with the independently acquired mismatch case.
			peer := publisherTestRecord(actual)
			src.list.Files[0].Publisher = &peer
			c, calls := publisherTestCatalog(t, publisherTestBody(providerDigest(actual)), http.StatusOK)
			dst := copyTestExecutor(t, "llamacpp", t.TempDir(), true)
			res, err := dst.copyModelFromPublisher(context.Background(), src, "llamacpp", src.list.Model, "", nil, nil, nil, c)
			if err != nil {
				t.Fatal(err)
			}
			comparison := res.PublisherComparison
			if calls.Load() != 1 || comparison == nil || comparison.Status != expectedStatus || len(comparison.Files) != res.Files {
				t.Fatalf("calls=%d comparison=%+v", calls.Load(), comparison)
			}
			got := comparison.Files[0]
			if got.Status != expectedStatus || got.ExpectedDigest != providerDigest(actual) || got.ActualDigest != actual || got.Provider != "huggingface" || got.Repository != publisherTestRepo || got.Revision != publisherTestRevision || got.File != publisherTestFile || got.AcquiredAt == peer.AcquiredAt || got.AcquisitionBasis != "provider" {
				t.Fatalf("destination adopted peer observation or lost tuple: %+v", got)
			}
			_, store, err := dst.modelStoreFor("llamacpp")
			if err != nil {
				t.Fatal(err)
			}
			copied, err := os.ReadFile(filepath.Join(llamaRepoDir(store.root, publisherTestRepo), "snapshots", publisherTestRevision, publisherTestFile))
			if err != nil || sha256Hex(copied) != actual {
				t.Fatalf("authorized copy did not publish intact bytes: %v", err)
			}
		})
	}
	test("exact independently acquired agreement", "verified", func(actual string) string { return actual })
	test("publisher mismatch is visible and copy still publishes", "mismatch", func(string) string { return strings.Repeat("f", 64) })
}

func TestModelPublisherUnavailableNeverAdoptsPeerVerification(t *testing.T) {
	plentyOfDisk(t)
	src := publisherCopySource(t)
	peer := publisherTestRecord(src.list.Files[0].SHA256)
	src.list.Files[0].Publisher = &peer
	dst := copyTestExecutor(t, "llamacpp", t.TempDir(), true)
	// Executor-only callers have no provider owner, not an ambient HTTP fallback.
	res, err := dst.CopyModelFrom(context.Background(), src, "llamacpp", src.list.Model, nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.PublisherComparison == nil || res.PublisherComparison.Status != "unavailable" || res.PublisherComparison.Files[0].ExpectedDigest != "" || res.PublisherComparison.Files[0].AcquiredAt != "" {
		t.Fatalf("peer assertion upgraded: %+v", res.PublisherComparison)
	}
}

func TestModelPublisherCustomAliasDoesNotPersistPeerStatement(t *testing.T) {
	plentyOfDisk(t)
	src := sourceFromExecutor(t, newCustomCopyFixture(t).source, "llamacpp", reportedCopyAlias)
	for i := range src.list.Files {
		file := &src.list.Files[i]
		peer := publisherTestRecord(file.SHA256)
		peer.Path = file.Path
		file.Publisher = &peer
	}
	catalog, calls := publisherTestCatalog(t, publisherTestBody(src.list.Files[0].SHA256), http.StatusOK)
	dst, store := customDestination(t)
	res, err := dst.copyModelFromPublisher(context.Background(), src, "llamacpp", reportedCopyAlias, "", nil, nil, nil, catalog)
	if err != nil {
		t.Fatal(err)
	}
	if calls.Load() != 0 || res.PublisherComparison == nil || res.PublisherComparison.Status != "unavailable" {
		t.Fatalf("custom alias adopted provider mapping: calls=%d result=%+v", calls.Load(), res)
	}
	receipt, err := readCopyReceipt(store, reportedCopyAlias)
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range receipt.List.Files {
		if f.Publisher != nil {
			t.Fatalf("peer statement entered local receipt: %+v", f.Publisher)
		}
	}
	replayed, complete, err := dst.completedCopy(context.Background(), "llamacpp", reportedCopyAlias, res.ExportID, nil)
	if err != nil || !complete || replayed.PublisherComparison == nil || replayed.PublisherComparison.Status != "unavailable" || !replayed.PublisherComparison.Replayed {
		t.Fatalf("missing receipt observation upgraded: result=%+v complete=%t error=%v", replayed, complete, err)
	}
}

func TestModelPublisherMappingRefusesUnprovenSelectors(t *testing.T) {
	actual := strings.Repeat("a", 64)
	test := func(name, adapter string, change func(*modelFileList)) {
		t.Run(name, func(t *testing.T) {
			peer := publisherTestRecord(actual)
			list := modelFileList{Engine: "llamacpp", Model: publisherTestRepo + ":Q4_K_M", Revision: publisherTestRevision, Files: []modelFile{{Path: publisherTestFile, SHA256: actual, Publisher: &peer}}}
			change(&list)
			c, calls := publisherTestCatalog(t, publisherTestBody(actual), http.StatusOK)
			compareModelPublisher(context.Background(), &list, adapter, c)
			if calls.Load() != 0 || publisherResult(list, false).Status != "unavailable" {
				t.Fatalf("unproven mapping contacted provider: calls=%d file=%+v", calls.Load(), list.Files[0])
			}
		})
	}
	test("unsupported adapter", "lmstudio", func(*modelFileList) {})
	test("custom layout", "llamacpp", func(list *modelFileList) { list.Layout = llamaArtifactsLayout })
	test("custom alias", "llamacpp", func(list *modelFileList) { list.Model = "my-model" })
	test("mutable revision", "llamacpp", func(list *modelFileList) { list.Revision = "main" })
	test("missing identity", "llamacpp", func(list *modelFileList) { list.Files[0].Publisher = nil })
	test("changed repository", "llamacpp", func(list *modelFileList) { list.Files[0].Publisher.Repository = "other/tiny-GGUF" })
	test("changed revision", "llamacpp", func(list *modelFileList) { list.Files[0].Publisher.Revision = strings.Repeat("b", 40) })
	test("similar filename", "llamacpp", func(list *modelFileList) { list.Files[0].Publisher.File = "other-Q4_K_M.gguf" })
	test("peer URL", "llamacpp", func(list *modelFileList) { list.Files[0].Publisher.Repository = "https://example.invalid/model" })
	test("peer traversal", "llamacpp", func(list *modelFileList) { list.Files[0].Publisher.File = "../model.gguf" })
}

func TestModelPublisherProviderFailuresAreUnavailable(t *testing.T) {
	actual := strings.Repeat("a", 64)
	test := func(name, body string, status int) {
		t.Run(name, func(t *testing.T) {
			c, calls := publisherTestCatalog(t, body, status)
			p := publisherTestRecord(actual)
			for attempt := 0; attempt < 2; attempt++ {
				if _, _, _, ok := c.publisherDigest(context.Background(), p); ok {
					t.Fatal("invalid provider response became a comparison")
				}
			}
			if calls.Load() != 1 {
				t.Fatalf("failure was not cached or backed off: calls=%d", calls.Load())
			}
		})
	}
	body := publisherTestBody(actual)
	test("provider refused", body, http.StatusForbidden)
	test("private repository", strings.Replace(body, `"private":false`, `"private":true`, 1), http.StatusOK)
	test("missing public fact", strings.Replace(body, `"private":false,`, "", 1), http.StatusOK)
	test("gated repository", strings.Replace(body, `"gated":false`, `"gated":"auto"`, 1), http.StatusOK)
	test("wrong immutable revision", strings.Replace(body, publisherTestRevision, strings.Repeat("b", 40), 1), http.StatusOK)
	test("wrong repository", strings.Replace(body, publisherTestRepo, "other/tiny-GGUF", 1), http.StatusOK)
	test("missing exact file", strings.Replace(body, publisherTestFile, "similar-Q4_K_M.gguf", 1), http.StatusOK)
	test("non-LFS file", strings.Replace(body, `"lfs":{"sha256":"`+actual+`"}`, `"blobId":"`+actual[:40]+`"`, 1), http.StatusOK)
	test("truncated digest", strings.Replace(body, actual, actual[:63], 1), http.StatusOK)
	test("SHA1 is not full SHA256", strings.Replace(body, actual, actual[:40], 1), http.StatusOK)
	test("duplicate revision", strings.Replace(body, `"sha":`, `"sha":"wrong","sha":`, 1), http.StatusOK)
	test("duplicate file digest", strings.Replace(body, `"sha256":`, `"sha256":"wrong","sha256":`, 1), http.StatusOK)
	test("oversize sibling set", fmt.Sprintf(`{"id":%q,"sha":%q,"private":false,"gated":false,"siblings":[%s]}`, publisherTestRepo, publisherTestRevision,
		strings.TrimSuffix(strings.Repeat(`{"rfilename":"tiny.gguf"},`, maxCopyArtifacts+1), ",")), http.StatusOK)
	test("malformed JSON", body[:len(body)-1], http.StatusOK)
	test("trailing JSON", body+`{}`, http.StatusOK)
	test("oversize body", strings.Repeat(" ", maxCatalogBody+1), http.StatusOK)
}

func TestModelPublisherTransportFailureDoesNotClaimObservation(t *testing.T) {
	c := newCatalogService()
	c.client.Transport = healthProbeTransport(func(*http.Request) (*http.Response, error) {
		return nil, errors.New("synthetic offline transport")
	})
	if digest, at, basis, ok := c.publisherDigest(context.Background(), publisherTestRecord(strings.Repeat("a", 64))); ok || digest != "" || at != "" || basis != "" {
		t.Fatalf("offline response invented observation: digest=%s at=%s ok=%t", digest, at, ok)
	}
}

func TestModelPublisherRejectsInvalidUTF8BeforeExactMatch(t *testing.T) {
	p := publisherTestRecord(strings.Repeat("a", 64))
	p.File = "model-\ufffd.gguf"
	body := strings.Replace(publisherTestBody(p.ActualDigest), publisherTestFile, "model-\xff.gguf", 1)
	c, calls := publisherTestCatalog(t, body, http.StatusOK)
	if _, _, _, ok := c.publisherDigest(context.Background(), p); ok || calls.Load() != 1 {
		t.Fatalf("invalid UTF-8 matched a replacement character: ok=%t calls=%d", ok, calls.Load())
	}
}

func TestModelPublisherStrictObservationRecords(t *testing.T) {
	valid := publisherTestRecord(strings.Repeat("a", 64))
	encoded, err := json.Marshal(valid)
	if err != nil {
		t.Fatal(err)
	}
	var roundTrip modelPublisherFile
	if err := json.Unmarshal(encoded, &roundTrip); err != nil || !reflect.DeepEqual(valid, roundTrip) {
		t.Fatalf("round trip: %+v %v", roundTrip, err)
	}
	test := func(name string, change func(map[string]any)) {
		t.Run(name, func(t *testing.T) {
			var value map[string]any
			if err := json.Unmarshal(encoded, &value); err != nil {
				t.Fatal(err)
			}
			change(value)
			data, err := json.Marshal(value)
			if err != nil {
				t.Fatal(err)
			}
			var record modelPublisherFile
			if json.Unmarshal(data, &record) == nil {
				t.Fatalf("invalid record accepted: %s", data)
			}
		})
	}
	test("missing digest", func(value map[string]any) { delete(value, "actualDigest") })
	test("truncated digest", func(value map[string]any) { value["actualDigest"] = "abc" })
	test("verified disagreement", func(value map[string]any) { value["expectedDigest"] = strings.Repeat("b", 64) })
	test("mismatch agreement", func(value map[string]any) { value["status"] = "mismatch" })
	test("unrecognized status", func(value map[string]any) { value["status"] = "trusted" })
	test("URL field", func(value map[string]any) { value["url"] = "https://example.invalid" })
	test("mutable revision", func(value map[string]any) { value["revision"] = "main" })
	test("unsafe relative file", func(value map[string]any) { value["file"] = "../file" })
	test("oversize path", func(value map[string]any) { value["path"] = strings.Repeat("a", 1025) })
	test("invalid time", func(value map[string]any) { value["acquiredAt"] = "yesterday" })
	test("future time", func(value map[string]any) { value["acquiredAt"] = "9999-01-01T00:00:00Z" })
	test("null observation", func(value map[string]any) { value["acquiredAt"] = nil })
	duplicate := strings.Replace(string(encoded), `"status":`, `"status":"unavailable","status":`, 1)
	if json.Unmarshal([]byte(duplicate), &roundTrip) == nil {
		t.Fatal("duplicate status accepted")
	}
}

func TestModelPublisherAggregationNeverHidesMissingFiles(t *testing.T) {
	verified := publisherTestRecord(strings.Repeat("a", 64))
	unavailable := modelPublisherFile{Path: "companion.gguf", Status: "unavailable", Algorithm: "sha256", ActualDigest: strings.Repeat("b", 64), AcquisitionBasis: "unavailable"}
	mismatch := verified
	mismatch.Status, mismatch.ExpectedDigest = "mismatch", strings.Repeat("c", 64)
	test := func(name string, files []modelPublisherFile, want string) {
		t.Run(name, func(t *testing.T) {
			if got := publisherAggregate(files); got != want {
				t.Fatalf("aggregate=%s want=%s", got, want)
			}
		})
	}
	test("no observations", nil, "unavailable")
	test("all files agree", []modelPublisherFile{verified}, "verified")
	test("companion unavailable", []modelPublisherFile{verified, unavailable}, "unavailable")
	test("mismatch survives unavailable companion", []modelPublisherFile{mismatch, unavailable}, "mismatch")
	p := &modelPublisherComparison{Status: "verified", Files: []modelPublisherFile{verified}}
	if validatePublisherComparison(p, 2) == nil {
		t.Fatal("partial file coverage became verified")
	}
}
