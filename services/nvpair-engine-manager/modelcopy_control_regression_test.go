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
	"os"
	"path/filepath"
	"strings"
	"testing"

	"nvpair-shared/engines"
	"nvpair-shared/modeldownload"
)

func TestModelCopyBlockedSourceRetryReachesDownloadFence(t *testing.T) {
	source := copyTestExecutor(t, "ollama", t.TempDir(), true)
	_, sourceStore, err := source.modelStoreFor("ollama")
	if err != nil {
		t.Fatalf("source store: %v", err)
	}
	writeOllamaModel(t, sourceStore.root, "tiny-fenced", 16, 3)
	nativeCalls := copyTestNativeModels(t, source, "ollama", "tiny-fenced")
	control := &controlServer{exec: source}
	listings, bodies := 0, 0
	// Exercise the real source listing fence and client decoder in memory.
	// There is no listener, peer process or native engine in this fixture.
	client := &remoteClient{base: "https://source.invalid", controlHTTP: &http.Client{Transport: copyEvidenceTransport(func(r *http.Request) (*http.Response, error) {
		response := httptest.NewRecorder()
		switch r.URL.Path {
		case controlModelFilesPath:
			listings++
			control.handleModelFiles(response, r)
		case controlModelFilePath:
			bodies++
			return nil, errors.New("blocked source must not serve artifact bodies")
		default:
			return nil, fmt.Errorf("unexpected source route %s", r.URL.Path)
		}
		return response.Result(), nil
	})}}
	if _, err := client.modelFiles(context.Background(), "ollama", "tiny-fenced", nil); err != nil {
		t.Fatalf("positive source listing control: %v", err)
	}
	if listings != 1 || bodies != 0 {
		t.Fatalf("positive source control: listings = %d, bodies = %d", listings, bodies)
	}

	destination := copyTestExecutor(t, "ollama", t.TempDir(), true)
	_, destinationStore, err := destination.modelStoreFor("ollama")
	if err != nil {
		t.Fatalf("destination store: %v", err)
	}
	if err := os.MkdirAll(destinationStore.root, 0o755); err != nil {
		t.Fatalf("create empty destination store: %v", err)
	}
	withdrawals := 0
	destination.modelMutationParent = func(context.Context, string) error { withdrawals++; return nil }
	d := destination.downloads
	if _, _, err := d.store.save(nil, nil, true, false); err != nil {
		t.Fatalf("seed destination baseline: %v", err)
	}
	d.copiesComplete = true
	m := &Manager{copyJournal: d, copyRegistry: destination.reg}
	t.Cleanup(m.stopCopies)
	attempt := reserveModelDownload(t, source.downloads, "ollama", "tiny-fenced", engines.ModelNamingExact)
	if _, err := source.downloads.changeMarker(attempt, false); err != nil {
		t.Fatalf("persist unsafe source download: %v", err)
	}
	if err := source.downloads.update(attempt, modeldownload.PhaseInProgress, nil); err != nil {
		t.Fatalf("start source download: %v", err)
	}
	copyBlocked := func(phase string) {
		t.Helper()
		before := source.downloads.snapshot()
		if len(before.Entries) != 1 || before.Entries[0].Phase != phase || !before.Entries[0].BlocksReadiness {
			t.Fatalf("source download fence = %+v", before)
		}
		p := copyRequest{OpID: newOpID(), Engine: "ollama", Model: "tiny-fenced", SourceNodeID: "source", DestinationNodeID: "destination"}
		ctx, release, err := m.reserveCopy(context.Background(), p)
		if err != nil {
			t.Fatalf("reserve blocked-source copy: %v", err)
		}
		defer release()
		beforeListings := listings
		beforeNative := *nativeCalls
		_, err = m.copyWithEvidence(ctx, p, "destination", func() (copyModelResult, error) {
			return destination.CopyModelFrom(ctx, client, p.Engine, p.Model, nil)
		})
		if err == nil || !strings.Contains(err.Error(), "model content unavailable") || listings != beforeListings+1 {
			t.Fatalf("%s source must refuse at listing lifecycle fence: error = %v, new listings = %d", phase, err, listings-beforeListings)
		}
		if bodies != 0 || withdrawals != 0 || *nativeCalls != beforeNative || !source.downloads.snapshot().Equal(before) {
			t.Fatalf("blocked source caused transfer effects or changed lifecycle: bodies = %d, withdrawals = %d, native queries = %d, state = %+v", bodies, withdrawals, *nativeCalls-beforeNative, source.downloads.snapshot())
		}
		artifacts, err := os.ReadDir(destinationStore.root)
		if err != nil {
			t.Fatalf("inspect destination artifacts: %v", err)
		}
		if len(artifacts) != 0 {
			t.Fatalf("blocked source created destination artifacts: %v", artifacts)
		}
	}
	copyBlocked(modeldownload.PhaseInProgress)
	if err := source.downloads.finish(attempt, modeldownload.PhaseFailed, modeldownload.ReasonDownloadFailed, ""); err != nil {
		t.Fatalf("fail source catalog download: %v", err)
	}
	copyBlocked(modeldownload.PhaseFailed)
}

func TestModelCopyControlVerifiedReceiptRetryPreservesFailedDownloadState(t *testing.T) {
	destination, store := customDestination(t)
	const model = "tiny-catalog-failure"
	data := []byte("synthetic bytes")
	list := modelFileList{Engine: "llamacpp", Model: model, Layout: llamaArtifactsLayout, Files: []modelFile{{Role: "model", Path: "model/tiny.gguf", Size: int64(len(data)), SHA256: sha256Hex(data)}}}
	exportID, err := artifactExportID(list)
	if err != nil {
		t.Fatalf("synthetic export identity: %v", err)
	}
	list.ExportID = exportID
	aliasDir := copyAliasDir(store.root, model)
	writeTestFile(t, filepath.Join(aliasDir, exportID, "model", "tiny.gguf"), data)
	receipt, err := json.Marshal(copyReceipt{Version: 1, List: list})
	if err != nil {
		t.Fatalf("encode synthetic receipt: %v", err)
	}
	receiptPath := filepath.Join(aliasDir, "copy.json")
	writeTestFile(t, receiptPath, receipt)
	authority := rmNewFixture(t, "destination-certificate")
	m := copyControlManager(t, destination, authority.dir)
	copyControlDirectory(m, copyControlNode(t, "destination", "destination-certificate", nil))
	withdrawals := 0
	destination.modelMutationParent = func(context.Context, string) error { withdrawals++; return nil }
	d := destination.downloads
	attempt := reserveModelDownload(t, d, "llamacpp", model, engines.ModelNamingExact)
	if _, err := d.changeMarker(attempt, false); err != nil {
		t.Fatalf("persist catalog marker: %v", err)
	}
	if err := d.finish(attempt, modeldownload.PhaseFailed, modeldownload.ReasonDownloadFailed, ""); err != nil {
		t.Fatalf("fail catalog attempt: %v", err)
	}
	before := d.snapshot()
	if len(before.Entries) != 1 || before.Entries[0].ID != attempt.entry.ID || before.Entries[0].Phase != modeldownload.PhaseFailed || !before.Entries[0].BlocksReadiness {
		t.Fatalf("failed catalog baseline = %+v", before)
	}
	journal, err := os.ReadFile(d.store.path)
	if err != nil {
		t.Fatalf("read failed catalog journal: %v", err)
	}
	// Prove the receipt itself qualifies without source lookup or a catalog
	// mutation. Its durability stays uncertain when it is read again.
	verified, complete, err := destination.completedCopy(context.Background(), "llamacpp", model, exportID, nil)
	if err != nil || !complete || verified.FilesSkipped != 1 || verified.BytesCopied != 0 || verified.Durability != copyDurabilityUncertain || !d.snapshot().Equal(before) {
		t.Fatalf("local receipt verification control = %+v, complete = %v, error = %v, state = %+v", verified, complete, err, d.snapshot())
	}
	p := copyRequest{OpID: newOpID(), Engine: "llamacpp", Model: model, SourceNodeID: "offline-source", DestinationNodeID: "destination", ExpectedExportID: exportID}
	res, err := m.executeCopy(context.Background(), p, false, nil, nil)
	if err != nil || res.ExportID != exportID || res.FilesSkipped != 1 || res.BytesCopied != 0 || res.Durability != copyDurabilityUncertain {
		t.Fatalf("verified COPY retry = %+v, error = %v", res, err)
	}
	if after := d.snapshot(); !after.Equal(before) {
		t.Errorf("verified receipt retry changed failed catalog lifecycle: before = %+v, after = %+v", before, after)
	}
	afterJournal, err := os.ReadFile(d.store.path)
	if err != nil {
		t.Fatalf("read journal after receipt retry: %v", err)
	}
	if !bytes.Equal(afterJournal, journal) {
		t.Error("verified receipt retry rewrote failed catalog journal content")
	}
	afterReceipt, err := os.ReadFile(receiptPath)
	if err != nil {
		t.Fatalf("read receipt after retry: %v", err)
	}
	if !bytes.Equal(afterReceipt, receipt) || !d.blocked(attempt.key) || withdrawals != 0 {
		t.Errorf("verified receipt retry changed receipt, readiness fence or mutation admission: withdrawals = %d", withdrawals)
	}
}

func TestModelCopyControlUnsupportedDestinationPreservesPrivateJournal(t *testing.T) {
	test := func(name string, existingJournal bool) {
		t.Run(name, func(t *testing.T) {
			// A synthetic family with no adapter exercises the unsupported-store
			// contract without loading a Strata manifest or running any engine.
			exec := copyTestExecutor(t, "synthetic-no-store", filepath.Join(t.TempDir(), "unused-store"), false)
			d := exec.downloads
			if existingJournal {
				if _, _, err := d.store.save(nil, nil, true, false); err != nil {
					t.Fatalf("seed existing private journal: %v", err)
				}
				d.copiesComplete = true
			}
			authority := rmNewFixture(t, "destination-certificate")
			output := &syncBuffer{}
			m := NewManager(NewCodec(output), exec, authority.mesh, authority.dir)
			exec.switchRecoveryPending.Store(false)
			exec.nodeProjection.set(nodeProjection{})
			t.Cleanup(func() { m.stopCopies(); m.controlHTTP.CloseIdle(); m.controlReadyHTTP.CloseIdle() })
			copyControlDirectory(m, copyControlNode(t, "destination", "destination-certificate", nil))
			before, err := os.ReadFile(d.store.path)
			if existingJournal && err != nil || !existingJournal && !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("private journal baseline: %v", err)
			}
			p := copyRequest{OpID: newOpID(), Engine: "synthetic-no-store", Model: "tiny", SourceNodeID: "offline-source", DestinationNodeID: "destination"}
			_, err = m.executeCopy(context.Background(), p, false, nil, nil)
			const diagnostic = "destination is not ready for model copy: Model copy is unavailable for this engine on this device."
			if err == nil || err.Error() != diagnostic {
				t.Fatalf("unsupported destination refusal = %v, want %q", err, diagnostic)
			}
			frames := output.frames()
			if len(frames) != 1 {
				t.Fatalf("unsupported destination progress = %+v", frames)
			}
			var progress copyProgress
			if err := json.Unmarshal(frames[0].Params, &progress); err != nil {
				t.Fatalf("decode terminal refusal: %v", err)
			}
			if progress.OpID != p.OpID || progress.Stage != "error" || progress.Message != diagnostic {
				t.Fatalf("unsupported destination terminal = %+v", progress)
			}
			after, err := os.ReadFile(d.store.path)
			if existingJournal {
				if err != nil || !bytes.Equal(before, after) {
					t.Errorf("unsupported destination changed private journal: error = %v, before = %s, after = %s", err, before, after)
				}
			} else if !errors.Is(err, os.ErrNotExist) {
				t.Errorf("unsupported destination created private journal: error = %v, content = %s", err, after)
			}
			if len(d.copies) != 0 || len(m.activeCopies()) != 0 {
				t.Errorf("unsupported destination retained copy provenance or live work: retained = %+v, active = %+v", d.copies, m.activeCopies())
			}
		})
	}
	test("missing journal stays absent", false)
	test("existing journal stays byte identical", true)
}
