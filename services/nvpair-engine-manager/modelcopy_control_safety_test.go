// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"

	"nvpair-shared/clustertrust"
	"nvpair-shared/durablefile"
	"nvpair-shared/engines"
)

func copyReceiptSafetyFixture(t *testing.T) (*Manager, copyRequest, engineModelStore) {
	t.Helper()
	exec, store := customDestination(t)
	const model = "tiny-verified-retry"
	data := []byte("unit bytes")
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
	writeTestFile(t, filepath.Join(aliasDir, "copy.json"), receipt)
	authority := rmNewFixture(t, "destination-certificate")
	m := copyControlManager(t, exec, authority.dir)
	copyControlDirectory(m, copyControlNode(t, "destination", "destination-certificate", nil))
	p := copyRequest{OpID: newOpID(), Engine: "llamacpp", Model: model, SourceNodeID: "offline-source", DestinationNodeID: "destination", ExpectedExportID: exportID}
	return m, p, store
}

func retainCopyReceiptAttempt(t *testing.T, m *Manager, p copyRequest) modelCopyMarker {
	t.Helper()
	ctx, release, err := m.reserveCopy(context.Background(), p)
	if err != nil {
		t.Fatalf("reserve original receipt attempt: %v", err)
	}
	defer release()
	_, err = m.copyWithEvidence(ctx, p, "destination", func() (copyModelResult, error) { return copyModelResult{}, copyUnknown(io.EOF) })
	var unknown *copyOutcomeUnknownError
	if !errors.As(err, &unknown) {
		t.Fatalf("retain original receipt attempt: %v", err)
	}
	return m.copyJournal.copies[p.OpID]
}

func TestModelCopyReceiptPreflightHasNoJournalEffects(t *testing.T) {
	test := func(name string, prepare func(*testing.T, *Manager, copyRequest, engineModelStore), wantSuccess bool) {
		t.Run(name, func(t *testing.T) {
			m, p, store := copyReceiptSafetyFixture(t)
			prepare(t, m, p, store)
			d := m.copyJournal
			before := d.snapshot()
			journal, err := os.ReadFile(d.store.path)
			if err != nil {
				t.Fatalf("read preflight baseline: %v", err)
			}
			publications, withdrawals := 0, 0
			d.store.replace.CreateTemp = func(string, string, os.FileMode) (durablefile.File, error) {
				publications++
				return nil, errors.New("read-only preflight must not publish")
			}
			m.exec.modelMutationParent = func(context.Context, string) error {
				withdrawals++
				return errors.New("read-only preflight must not withdraw identity")
			}
			res, err := m.executeCopy(context.Background(), p, false, nil, nil)
			if (err == nil) != wantSuccess || wantSuccess && (res.ExportID != p.ExpectedExportID || res.FilesSkipped != 1 || res.BytesCopied != 0 || res.Durability != copyDurabilityUncertain) {
				t.Fatalf("receipt preflight result = %+v, error = %v, want success = %v", res, err, wantSuccess)
			}
			after, readErr := os.ReadFile(d.store.path)
			if readErr != nil || publications != 0 || withdrawals != 0 || !bytes.Equal(journal, after) || !d.snapshot().Equal(before) || len(m.activeCopies()) != 0 {
				t.Fatalf("preflight had effects: read error = %v, publications = %d, withdrawals = %d, state = %+v", readErr, publications, withdrawals, d.snapshot())
			}
		})
	}
	test("verified receipt", func(*testing.T, *Manager, copyRequest, engineModelStore) {}, true)
	test("tampered artifact", func(t *testing.T, _ *Manager, p copyRequest, store engineModelStore) {
		writeTestFile(t, filepath.Join(copyAliasDir(store.root, p.Model), p.ExpectedExportID, "model", "tiny.gguf"), []byte("wrong data"))
	}, false)
	test("malformed receipt", func(t *testing.T, _ *Manager, p copyRequest, store engineModelStore) {
		writeTestFile(t, filepath.Join(copyAliasDir(store.root, p.Model), "copy.json"), []byte("{"))
	}, false)
	test("download recovery refuses verified receipt", func(_ *testing.T, m *Manager, _ copyRequest, _ engineModelStore) {
		m.copyJournal.recoveryRequired = true
	}, false)
	test("copy recovery refuses verified receipt", func(_ *testing.T, m *Manager, _ copyRequest, _ engineModelStore) {
		m.copyJournal.copyRecoveryRequired = true
	}, false)
	test("predecessor marker refuses verified receipt", func(t *testing.T, m *Manager, p copyRequest, _ engineModelStore) {
		retainCopyReceiptAttempt(t, m, p)
		d := newModelDownloads(newOpID(), filepath.Dir(m.copyJournal.store.path), map[string]string{"llamacpp": engines.ModelNamingExact})
		m.copyJournal, m.exec.downloads, m.exec.switchIncarnation = d, d, d.source
	}, false)
}

func TestModelCopyReceiptRetryRemovesOnlyMatchingRetainedMarker(t *testing.T) {
	m, p, _ := copyReceiptSafetyFixture(t)
	original := retainCopyReceiptAttempt(t, m, p)
	p.OpID = newOpID()
	d := m.copyJournal
	before, generation := d.snapshot(), d.authorityGeneration
	publications := 0
	d.store.replace.Replace = func(tmp, target string) error {
		publications++
		_, copies, _, _, err := d.store.loadJournal()
		if err != nil || len(copies) != 1 || copies[original.OpID] != original {
			t.Errorf("receipt verification replaced original provenance before removal: error = %v, markers = %+v", err, copies)
		}
		return os.Rename(tmp, target)
	}
	res, err := m.executeCopy(context.Background(), p, false, nil, nil)
	if err != nil || res.ExportID != p.ExpectedExportID || res.BytesCopied != 0 || publications != 1 || len(d.copies) != 0 || d.authorityGeneration != generation+1 || !d.snapshot().Equal(before) {
		t.Fatalf("verified receipt reconciliation = %+v, error = %v, publications = %d, retained = %+v, state = %+v", res, err, publications, d.copies, d.snapshot())
	}
}

func TestModelCopyVerifiedReceiptPreservesUnknownCopyBaseline(t *testing.T) {
	m, p, _ := copyReceiptSafetyFixture(t)
	d := m.copyJournal
	if _, _, err := d.store.save(nil, nil, false, false); err != nil {
		t.Fatalf("seed unknown historical copy baseline: %v", err)
	}
	d.copiesComplete = false
	if _, err := m.executeCopy(context.Background(), p, false, nil, nil); err != nil {
		t.Fatalf("read-only receipt verification: %v", err)
	}
	_, copies, complete, recovery, err := d.store.loadJournal()
	if err != nil || complete || recovery || len(copies) != 0 {
		t.Fatalf("receipt verification fabricated historical completeness: error = %v, complete = %v, recovery = %v, copies = %+v", err, complete, recovery, copies)
	}
	if r := m.ownerReservations(); r.Complete || r.Busy.Known {
		t.Fatalf("unknown history became safe build-apply authority: %+v", r)
	}
}

func TestModelCopyControllerCapabilityRefusalHasNoJournalEffects(t *testing.T) {
	authority := rmNewFixture(t, "controller-certificate", "destination-certificate")
	destinationDir := authority.peer("destination-certificate").dir
	grantRemoteManagement(t, destinationDir, "1", map[string]bool{"controller-certificate": true})
	destination := copyTestExecutor(t, "synthetic-no-store", t.TempDir(), false)
	destination.nodeProjection.set(nodeProjection{})
	mesh := clustertrust.Open(destinationDir)
	control := &controlServer{exec: destination, mesh: mesh}
	var posts atomic.Int32
	server := rmTLSServer(t, mesh, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			posts.Add(1)
		}
		control.mux().ServeHTTP(w, r)
	}))
	m := copyControlManager(t, copyTestExecutor(t, "synthetic-no-store", t.TempDir(), false), authority.dir)
	copyControlDirectory(m, copyControlNode(t, "controller", "controller-certificate", nil), copyControlNode(t, "destination", "destination-certificate", server))
	journal, err := os.ReadFile(m.copyJournal.store.path)
	if err != nil {
		t.Fatalf("read controller baseline: %v", err)
	}
	publications := 0
	m.copyJournal.store.replace.CreateTemp = func(string, string, os.FileMode) (durablefile.File, error) {
		publications++
		return nil, errors.New("unsupported destination must not publish")
	}
	p := copyRequest{OpID: newOpID(), Engine: "synthetic-no-store", Model: "tiny", SourceNodeID: "offline-source", DestinationNodeID: "destination"}
	_, err = m.executeCopy(remoteFixtureContext(t), p, false, nil, nil)
	const diagnostic = "destination is not ready for model copy: Model copy is unavailable for this engine on this device."
	if err == nil || err.Error() != diagnostic || publications != 0 || posts.Load() != 0 || len(m.copyJournal.copies) != 0 {
		t.Fatalf("unsupported remote destination caused effects: error = %v, publications = %d, POSTs = %d, retained = %+v", err, publications, posts.Load(), m.copyJournal.copies)
	}
	after, err := os.ReadFile(m.copyJournal.store.path)
	if err != nil || !bytes.Equal(journal, after) {
		t.Fatalf("unsupported remote destination changed controller journal: %v", err)
	}
}
