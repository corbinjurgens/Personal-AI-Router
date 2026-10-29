// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"nvpair-shared/engines"
)

func copyRetrySafetyFixture(t *testing.T) (*Manager, *modelDownloads, copyRequest, modelCopyMarker) {
	t.Helper()
	dir := t.TempDir()
	store := newModelDownloadStore(dir)
	if _, _, err := store.save(nil, nil, true, false); err != nil {
		t.Fatalf("seed private copy baseline: %v", err)
	}
	d := newModelDownloads(newOpID(), dir, map[string]string{"llamacpp": engines.ModelNamingExact})
	registry := NewRegistry()
	registry.engines["llamacpp"] = &Manifest{Engine: "llamacpp", ModelNaming: engines.ModelNamingExact}
	m := &Manager{exec: &Executor{switchIncarnation: d.source}, copyJournal: d, copyRegistry: registry}
	t.Cleanup(m.stopCopies)
	p := copyRequest{OpID: newOpID(), Engine: "llamacpp", Model: "tiny-retained", SourceNodeID: "source", DestinationNodeID: "destination", ExpectedExportID: strings.Repeat("a", 64)}
	ctx, release, err := m.reserveCopy(context.Background(), p)
	if err != nil {
		t.Fatalf("reserve original copy: %v", err)
	}
	defer release()
	_, err = m.copyWithEvidence(ctx, p, "destination", func() (copyModelResult, error) {
		return copyModelResult{}, copyUnknown(io.EOF)
	})
	var unknown *copyOutcomeUnknownError
	if !errors.As(err, &unknown) || len(d.copies) != 1 {
		t.Fatalf("retain original copy: error = %v, markers = %+v", err, d.copies)
	}
	return m, d, p, d.copies[p.OpID]
}

func copyRetryVerifiedResult(p copyRequest) copyModelResult {
	return copyModelResult{Engine: p.Engine, Model: p.Model, Layout: llamaArtifactsLayout, ExportID: strings.Repeat("a", 64), RequiresConfiguration: true, Durability: copyDurabilityUncertain, Files: 1, FilesSkipped: 1, BytesTotal: 8}
}

func TestModelCopyRetryRefusesRecoveryAndPredecessor(t *testing.T) {
	test := func(name string, prepare func(*modelDownloads) *modelDownloads) {
		t.Run(name, func(t *testing.T) {
			m, d, original, marker := copyRetrySafetyFixture(t)
			d = prepare(d)
			marker = d.copies[original.OpID]
			m.copyJournal, m.exec.switchIncarnation = d, d.source
			before, err := os.ReadFile(d.store.path)
			if err != nil {
				t.Fatalf("read retained journal: %v", err)
			}
			p := original
			p.OpID = newOpID()
			ctx, release, err := m.reserveCopy(context.Background(), p)
			if err != nil {
				t.Fatalf("reserve recovery retry: %v", err)
			}
			defer release()
			effects := 0
			_, err = m.copyWithEvidence(ctx, p, "destination", func() (copyModelResult, error) { effects++; return copyRetryVerifiedResult(p), nil })
			if !errors.Is(err, errDownloadRecovery) || effects != 0 {
				t.Fatalf("unsafe provenance admitted: error = %v, effects = %d", err, effects)
			}
			after, err := os.ReadFile(d.store.path)
			if err != nil || !bytes.Equal(before, after) || d.copies[original.OpID] != marker {
				t.Fatalf("refusal changed original provenance: error = %v, copies = %+v", err, d.copies)
			}
			if r := m.ownerReservations(); r.Complete || !r.Busy.Known || r.Busy.Value == nil || !*r.Busy.Value {
				t.Fatalf("unsafe provenance stopped blocking build apply: %+v", r)
			}
		})
	}
	test("download recovery", func(d *modelDownloads) *modelDownloads { d.recoveryRequired = true; return d })
	test("copy recovery", func(d *modelDownloads) *modelDownloads { d.copyRecoveryRequired = true; return d })
	test("uncertain provenance", func(d *modelDownloads) *modelDownloads {
		for id, marker := range d.copies {
			marker.Phase = "uncertain"
			d.copies[id] = marker
		}
		d.copyRecoveryRequired = true
		return d
	})
	test("predecessor incarnation", func(d *modelDownloads) *modelDownloads {
		return newModelDownloads(newOpID(), filepath.Dir(d.store.path), map[string]string{"llamacpp": engines.ModelNamingExact})
	})
}

func TestModelCopyRetryKeepsOriginalOnUnprovenCompletion(t *testing.T) {
	test := func(name string, outcome func(*testing.T, *Manager, copyRequest) (copyModelResult, error)) {
		t.Run(name, func(t *testing.T) {
			m, d, p, marker := copyRetrySafetyFixture(t)
			before, err := os.ReadFile(d.store.path)
			if err != nil {
				t.Fatalf("read retained journal: %v", err)
			}
			p.OpID = newOpID()
			// A matching request identity still requires a matching positive
			// terminal before the original evidence can be cleared.
			ctx, release, err := m.reserveCopy(context.Background(), p)
			if err != nil {
				t.Fatalf("reserve retry: %v", err)
			}
			defer release()
			effects := 0
			_, err = m.copyWithEvidence(ctx, p, "destination", func() (copyModelResult, error) { effects++; return outcome(t, m, p) })
			if err == nil || effects != 1 {
				t.Fatalf("unproven completion = %v, effects = %d", err, effects)
			}
			after, err := os.ReadFile(d.store.path)
			if err != nil || !bytes.Equal(before, after) || len(d.copies) != 1 || d.copies[marker.OpID] != marker {
				t.Fatalf("retry replaced or cleared original provenance: error = %v, markers = %+v", err, d.copies)
			}
		})
	}
	test("failed effect", func(*testing.T, *Manager, copyRequest) (copyModelResult, error) {
		return copyModelResult{}, io.ErrUnexpectedEOF
	})
	test("lost acknowledgement", func(*testing.T, *Manager, copyRequest) (copyModelResult, error) {
		return copyModelResult{}, copyUnknown(io.EOF)
	})
	test("wrong model", func(_ *testing.T, _ *Manager, p copyRequest) (copyModelResult, error) {
		res := copyRetryVerifiedResult(p)
		res.Model = "other"
		return res, nil
	})
	test("wrong export", func(_ *testing.T, _ *Manager, p copyRequest) (copyModelResult, error) {
		res := copyRetryVerifiedResult(p)
		res.ExportID = strings.Repeat("b", 64)
		return res, nil
	})
	test("cancelled positive terminal", func(t *testing.T, m *Manager, p copyRequest) (copyModelResult, error) {
		if !m.cancelCopy(p.OpID) {
			t.Fatal("retry was not cancellable")
		}
		return copyRetryVerifiedResult(p), nil
	})
}

func TestModelCopyRetryRejectsDifferentRetainedExport(t *testing.T) {
	m, d, p, original := copyRetrySafetyFixture(t)
	p.OpID, p.ExpectedExportID = newOpID(), strings.Repeat("b", 64)
	ctx, release, err := m.reserveCopy(context.Background(), p)
	if err != nil {
		t.Fatalf("reserve changed-export retry: %v", err)
	}
	defer release()
	effects := 0
	_, err = m.copyWithEvidence(ctx, p, "destination", func() (copyModelResult, error) { effects++; return copyRetryVerifiedResult(p), nil })
	if rpcCode(err) != codePolicyConflict || effects != 0 || len(d.copies) != 1 || d.copies[original.OpID] != original {
		t.Fatalf("changed export superseded retained identity: error = %v, effects = %d, markers = %+v", err, effects, d.copies)
	}
}

func TestModelCopyRetryFailedRetirementPreservesOriginal(t *testing.T) {
	m, d, p, original := copyRetrySafetyFixture(t)
	p.OpID = newOpID()
	before, err := os.ReadFile(d.store.path)
	if err != nil {
		t.Fatalf("read retained journal: %v", err)
	}
	d.store.replace.Replace = func(string, string) error { return errors.New("injected retry retirement failure") }
	ctx, release, err := m.reserveCopy(context.Background(), p)
	if err != nil {
		t.Fatalf("reserve matching retry: %v", err)
	}
	defer release()
	_, err = m.copyWithEvidence(ctx, p, "destination", func() (copyModelResult, error) { return copyRetryVerifiedResult(p), nil })
	var unknown *copyOutcomeUnknownError
	if !errors.As(err, &unknown) || len(d.copies) != 1 || d.copies[original.OpID] != original || !d.copyRetirements[original.OpID] {
		t.Fatalf("failed retirement lost original evidence: error = %v, markers = %+v", err, d.copies)
	}
	after, err := os.ReadFile(d.store.path)
	if err != nil || !bytes.Equal(before, after) {
		t.Fatalf("failed retirement changed durable provenance: %v", err)
	}
}

func TestModelCopyRetryRejectsOperationIDBoundToOtherContent(t *testing.T) {
	m, d, p, original := copyRetrySafetyFixture(t)
	other := original
	other.OpID, other.Model, other.ContentKey = newOpID(), "unrelated", "unrelated"
	if err := d.changeCopyMarker(other, false); err != nil {
		t.Fatalf("retain unrelated copy: %v", err)
	}
	p.OpID = other.OpID
	ctx, release, err := m.reserveCopy(context.Background(), p)
	if err != nil {
		t.Fatalf("reserve operation ID collision: %v", err)
	}
	defer release()
	effects := 0
	_, err = m.copyWithEvidence(ctx, p, "destination", func() (copyModelResult, error) { effects++; return copyRetryVerifiedResult(p), nil })
	if !errors.Is(err, errDownloadBusy) || effects != 0 || len(d.copies) != 2 || d.copies[original.OpID] != original || d.copies[other.OpID] != other {
		t.Fatalf("retry reused an operation ID for different content: error = %v, effects = %d, markers = %+v", err, effects, d.copies)
	}
}

func TestModelCopyRetryMatchingCompletionRetiresOnlyCurrentTuple(t *testing.T) {
	m, d, p, original := copyRetrySafetyFixture(t)
	other := original
	other.OpID, other.Model, other.ContentKey = newOpID(), "unrelated", "unrelated"
	if err := d.changeCopyMarker(other, false); err != nil {
		t.Fatalf("retain unrelated copy: %v", err)
	}
	a := reserveModelDownload(t, d, p.Engine, p.Model, engines.ModelNamingExact)
	if _, err := d.changeMarker(a, false); err != nil {
		t.Fatalf("retain catalog attempt: %v", err)
	}
	if err := d.finish(a, "failed", "download_failed", ""); err != nil {
		t.Fatalf("fail catalog attempt: %v", err)
	}
	before, generation := d.snapshot(), d.authorityGeneration
	p.OpID = newOpID()
	ctx, release, err := m.reserveCopy(context.Background(), p)
	if err != nil {
		t.Fatalf("reserve matching retry: %v", err)
	}
	defer release()
	_, err = m.copyWithEvidence(ctx, p, "destination", func() (copyModelResult, error) {
		_, copies, _, _, loadErr := d.store.loadJournal()
		if loadErr != nil || len(copies) != 2 || copies[original.OpID] != original || copies[other.OpID] != other {
			t.Fatalf("retry changed evidence before verification: error = %v, markers = %+v", loadErr, copies)
		}
		return copyRetryVerifiedResult(p), nil
	})
	if err != nil {
		t.Fatalf("matching retry: %v", err)
	}
	_, copies, complete, recovery, err := d.store.loadJournal()
	if err != nil || !complete || recovery || len(copies) != 1 || copies[other.OpID] != other || d.authorityGeneration != generation+1 || !d.snapshot().Equal(before) || d.unsafe[a.key].ID != a.entry.ID {
		t.Fatalf("matching completion changed unrelated evidence: error = %v, copies = %+v, generation = %d, state = %+v", err, copies, d.authorityGeneration, d.snapshot())
	}
}

func TestModelCopyRetryExcludesConcurrentClaim(t *testing.T) {
	m, d, p, original := copyRetrySafetyFixture(t)
	p.OpID = newOpID()
	ctx, release, err := m.reserveCopy(context.Background(), p)
	if err != nil {
		t.Fatalf("reserve first retry: %v", err)
	}
	defer release()
	_, err = m.copyWithEvidence(ctx, p, "destination", func() (copyModelResult, error) {
		concurrent := p
		concurrent.OpID = newOpID()
		second, releaseSecond, err := m.reserveCopy(context.Background(), concurrent)
		if err != nil {
			t.Fatalf("reserve concurrent request: %v", err)
		}
		defer releaseSecond()
		effects := 0
		_, err = m.copyWithEvidence(second, concurrent, "destination", func() (copyModelResult, error) { effects++; return copyRetryVerifiedResult(concurrent), nil })
		if !errors.Is(err, errDownloadBusy) || effects != 0 || d.copies[original.OpID] != original {
			t.Fatalf("concurrent retry claimed original evidence: error = %v, effects = %d, markers = %+v", err, effects, d.copies)
		}
		return copyModelResult{}, copyUnknown(io.EOF)
	})
	var unknown *copyOutcomeUnknownError
	if !errors.As(err, &unknown) || len(d.copies) != 1 || d.copies[original.OpID] != original {
		t.Fatalf("concurrent refusal erased original evidence: error = %v, markers = %+v", err, d.copies)
	}
}

func TestModelCopyJournalUsesAuthorityAtDownloadRevisionLimit(t *testing.T) {
	_, d, _, original := copyRetrySafetyFixture(t)
	d.revision = math.MaxUint64
	generation := d.authorityGeneration
	marker := original
	marker.OpID, marker.Model, marker.ContentKey = newOpID(), "independent", "independent"
	if err := d.changeCopyMarker(marker, false); err != nil {
		t.Fatalf("download revision blocked independent copy admission: %v", err)
	}
	if err := d.changeCopyMarker(marker, true); err != nil {
		t.Fatalf("copy retirement lost reserved authority: %v", err)
	}
	if d.revision != math.MaxUint64 || d.authorityGeneration != generation+2 || len(d.copies) != 1 || d.copies[original.OpID] != original {
		t.Fatalf("copy changed download revision or lost retained authority: revision = %d, generation = %d, copies = %+v", d.revision, d.authorityGeneration, d.copies)
	}
}

func TestModelCopyRetryReconcilesOnlyKnownOriginalExport(t *testing.T) {
	plentyOfDisk(t)
	fastRetries(t, 1)
	const model = "tiny-original-export"
	const path = "model/tiny.gguf"
	dataA, dataB := []byte("export A bytes"), []byte("export B bytes")
	listing := func(data []byte) modelFileList {
		t.Helper()
		list := modelFileList{Engine: "llamacpp", Model: model, Layout: llamaArtifactsLayout, Files: []modelFile{{Role: "model", Path: path, Size: int64(len(data)), SHA256: sha256Hex(data)}}}
		id, err := artifactExportID(list)
		if err != nil {
			t.Fatalf("synthetic export identity: %v", err)
		}
		list.ExportID = id
		return list
	}
	listA, listB := listing(dataA), listing(dataB)
	for _, tc := range []struct {
		name             string
		originalExpected string
		retryList        modelFileList
		retryData        []byte
		retryExpected    string
		wantRetained     bool
		wantConflict     bool
	}{
		{name: "unknown original A survives successful B", retryList: listB, retryData: dataB, retryExpected: listB.ExportID, wantRetained: true},
		{name: "known original A clears after verified A", originalExpected: listA.ExportID, retryList: listA, retryData: dataA, retryExpected: listA.ExportID},
		{name: "known original A refuses B", originalExpected: listA.ExportID, retryList: listB, retryData: dataB, retryExpected: listB.ExportID, wantRetained: true, wantConflict: true},
		{name: "known original A refuses an identity-less retry", originalExpected: listA.ExportID, retryList: listB, retryData: dataB, wantRetained: true, wantConflict: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			exec, store := customDestination(t)
			authority := rmNewFixture(t, "destination-certificate")
			m := copyControlManager(t, exec, authority.dir)
			copyControlDirectory(m, copyControlNode(t, "destination", "destination-certificate", nil))
			// This control starts with known history so matching completion can
			// prove clear authority independently of the historical baseline gate.
			if _, _, err := m.copyJournal.store.save(nil, nil, true, false); err != nil {
				t.Fatalf("seed complete copy baseline: %v", err)
			}
			m.copyJournal.copiesComplete = true
			p := copyRequest{OpID: newOpID(), Engine: "llamacpp", Model: model, SourceNodeID: "offline-source", DestinationNodeID: "destination", ExpectedExportID: tc.originalExpected}
			ctx, release, err := m.reserveCopy(context.Background(), p)
			if err != nil {
				t.Fatalf("reserve original export A: %v", err)
			}
			t.Cleanup(release)
			prefix := dataA[:4]
			interrupted := &fakeModelSource{list: listA, data: map[string][]byte{path: prefix}}
			_, err = m.copyWithEvidence(ctx, p, "destination", func() (copyModelResult, error) {
				return exec.copyModelFrom(ctx, interrupted, p.Engine, p.Model, p.ExpectedExportID, nil, nil, nil)
			})
			release()
			if !errors.Is(err, io.ErrUnexpectedEOF) {
				t.Fatalf("original export A must interrupt after positive bytes: %v", err)
			}
			partA := filepath.Join(copyAliasDir(store.root, model), listA.ExportID, filepath.FromSlash(path)) + ".part"
			partial, err := os.ReadFile(partA)
			if err != nil || !bytes.Equal(partial, prefix) {
				t.Fatalf("original export A partial = %q, error = %v", partial, err)
			}
			d := m.copyJournal
			original := d.copies[p.OpID]
			if len(d.copies) != 1 || original.OpID != p.OpID || original.ExpectedExportID != tc.originalExpected {
				t.Fatalf("original export provenance = %+v", d.copies)
			}
			journal, err := os.ReadFile(d.store.path)
			if err != nil {
				t.Fatalf("read interrupted export journal: %v", err)
			}
			p.OpID, p.ExpectedExportID = newOpID(), tc.retryExpected
			ctx, releaseRetry, err := m.reserveCopy(context.Background(), p)
			if err != nil {
				t.Fatalf("reserve identity-bound retry: %v", err)
			}
			t.Cleanup(releaseRetry)
			source := &fakeModelSource{list: tc.retryList, data: map[string][]byte{path: tc.retryData}}
			effects := 0
			res, err := m.copyWithEvidence(ctx, p, "destination", func() (copyModelResult, error) {
				effects++
				return exec.copyModelFrom(ctx, source, p.Engine, p.Model, p.ExpectedExportID, nil, nil, nil)
			})
			releaseRetry()
			if tc.wantConflict {
				if rpcCode(err) != codePolicyConflict || effects != 0 {
					t.Fatalf("different known export must refuse before effects: error = %v, effects = %d", err, effects)
				}
			} else {
				if err != nil || effects != 1 || res.ExportID != tc.retryList.ExportID {
					t.Fatalf("verified retry = %+v, error = %v, effects = %d", res, err, effects)
				}
				// Rehash the committed result through the real no-source receipt
				// path too. It cannot identify a different interrupted export.
				p.OpID = newOpID()
				verified, err := m.executeCopy(context.Background(), p, false, nil, nil)
				if err != nil || verified.ExportID != tc.retryList.ExportID || verified.FilesSkipped != 1 || verified.BytesCopied != 0 {
					t.Fatalf("verified receipt retry = %+v, error = %v", verified, err)
				}
			}
			_, copies, _, recovery, err := d.store.loadJournal()
			if err != nil || recovery {
				t.Fatalf("load reconciled provenance: error = %v, recovery = %v", err, recovery)
			}
			if tc.wantRetained {
				if len(copies) != 1 || copies[original.OpID] != original || d.copies[original.OpID] != original {
					t.Fatalf("retry discarded original export provenance: disk = %+v, memory = %+v", copies, d.copies)
				}
				after, err := os.ReadFile(d.store.path)
				if err != nil || !bytes.Equal(journal, after) {
					t.Fatalf("retry rewrote original export provenance: %v", err)
				}
				partial, err := os.ReadFile(partA)
				if err != nil || !bytes.Equal(partial, prefix) {
					t.Fatalf("original export A partial changed: bytes = %q, error = %v", partial, err)
				}
				if r := m.ownerReservations(); r.Complete || r.Reason != "content_authority_unknown" || !r.Busy.Known || r.Busy.Value == nil || !*r.Busy.Value {
					t.Fatalf("unresolved original export stopped blocking build apply: %+v", r)
				}
			} else {
				if len(copies) != 0 || len(d.copies) != 0 {
					t.Fatalf("matching known export did not retire provenance: disk = %+v, memory = %+v", copies, d.copies)
				}
				if _, err := os.Stat(partA); !errors.Is(err, os.ErrNotExist) {
					t.Fatalf("matching known export left its original partial: %v", err)
				}
				if r := m.ownerReservations(); !r.Complete || !r.Busy.Known || r.Busy.Value == nil || *r.Busy.Value {
					t.Fatalf("matching known export did not restore clear authority: %+v", r)
				}
			}
		})
	}
}
