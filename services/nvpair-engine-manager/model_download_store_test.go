// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"

	"nvpair-shared/engines"
	"nvpair-shared/modeldownload"
)

func TestModelDownloadJournalPreReplacementFailure(t *testing.T) {
	d := testModelDownloads(t)
	a := reserveModelDownload(t, d, "llamacpp", "model", engines.ModelNamingExact)
	if _, err := d.changeMarker(a, false); err != nil {
		t.Fatal(err)
	}
	before, err := os.ReadFile(d.store.path)
	if err != nil {
		t.Fatal(err)
	}
	d.store.replace.Replace = func(string, string) error { return errors.New("injected") }
	if uncertain, err := d.changeMarker(a, true); err == nil || uncertain {
		t.Fatalf("removal = %v/%v", uncertain, err)
	}
	after, err := os.ReadFile(d.store.path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(before, after) || len(d.unsafe) != 1 {
		t.Fatal("failed removal changed disk or memory")
	}
	if err := d.finish(a, "unknown", "persistence_error", ""); err != nil {
		t.Fatal(err)
	}
	if !d.snapshot().Entries[0].BlocksReadiness {
		t.Fatal("failed removal promoted content")
	}
}

func TestModelDownloadJournalLandedRemovalQualification(t *testing.T) {
	d := testModelDownloads(t)
	a := reserveModelDownload(t, d, "llamacpp", "model", engines.ModelNamingExact)
	if _, err := d.changeMarker(a, false); err != nil {
		t.Fatal(err)
	}
	d.store.replace.SyncDir = func(string) error { return errors.New("injected") }
	uncertain, err := d.changeMarker(a, true)
	if err == nil || !uncertain || len(d.unsafe) != 0 {
		t.Fatalf("landed removal = %v/%v/%d", uncertain, err, len(d.unsafe))
	}
	if err := d.finish(a, "complete", "", "uncertain"); err != nil {
		t.Fatal(err)
	}
	s := d.snapshot()
	if s.Entries[0].Persistence != "uncertain" || s.Entries[0].BlocksReadiness {
		t.Fatalf("qualification = %+v", s)
	}
	recovered := newModelDownloads(newOpID(), filepath.Dir(d.store.path), map[string]string{"llamacpp": engines.ModelNamingExact})
	if recovered.recoveryRequired || len(recovered.unsafe) != 0 {
		t.Fatal("memory disagreed with landed disk")
	}
}

func TestModelDownloadJournalLandedAdditionRefusesEffect(t *testing.T) {
	d := testModelDownloads(t)
	a := reserveModelDownload(t, d, "llamacpp", "model", engines.ModelNamingExact)
	d.store.replace.SyncDir = func(string) error { return errors.New("injected") }
	uncertain, err := d.changeMarker(a, false)
	if err == nil || !uncertain || len(d.unsafe) != 1 {
		t.Fatalf("addition = %v/%v/%d", uncertain, err, len(d.unsafe))
	}
}

func TestModelDownloadJournalRecovery(t *testing.T) {
	d := testModelDownloads(t)
	a := reserveModelDownload(t, d, "ollama", "demo", engines.ModelNamingImpliedLatestTag)
	if _, err := d.changeMarker(a, false); err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(d.store.path)
	if err != nil {
		t.Fatal(err)
	}
	if runtime.GOOS != "windows" && info.Mode().Perm() != 0600 {
		t.Fatalf("mode = %o", info.Mode().Perm())
	}
	recovered := newModelDownloads(newOpID(), filepath.Dir(d.store.path), map[string]string{"ollama": engines.ModelNamingImpliedLatestTag})
	s := recovered.snapshot()
	if s.SourceID == d.source || len(s.Entries) != 1 || s.Entries[0].Phase != "unknown" || !s.Entries[0].BlocksReadiness {
		t.Fatalf("recovery = %+v", s)
	}
	changed := newModelDownloads(newOpID(), filepath.Dir(d.store.path), map[string]string{"ollama": engines.ModelNamingExact})
	if !changed.snapshot().RecoveryRequired {
		t.Fatal("changed naming reinterpreted marker")
	}
}

func TestModelDownloadJournalMalformedRecovery(t *testing.T) {
	test := func(name, input string) {
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, modelDownloadFile)
			if err := os.WriteFile(path, []byte(input), 0600); err != nil {
				t.Fatal(err)
			}
			d := newModelDownloads(newOpID(), dir, nil)
			if !d.snapshot().RecoveryRequired {
				t.Fatal("malformed state accepted")
			}
			if _, err := d.reserve("llamacpp", "model", engines.ModelNamingExact); !errors.Is(err, errDownloadRecovery) {
				t.Fatalf("reservation = %v", err)
			}
			got, err := os.ReadFile(path)
			if err != nil || string(got) != input {
				t.Fatal("journal was rewritten")
			}
		})
	}
	test("invalid JSON", "{")
	test("null entries", `{"version":1,"entries":null,"copies":[],"copiesComplete":true}`)
	test("duplicate entries", `{"version":1,"entries":[],"entries":[],"copies":[],"copiesComplete":true}`)
	test("case variant", `{"version":1,"Entries":[],"copies":[],"copiesComplete":true}`)
	test("unknown field", `{"version":1,"entries":[],"copies":[],"copiesComplete":true,"unexpected":true}`)
	test("duplicate marker model", `{"version":1,"entries":[{"id":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","engine":"llamacpp","model":"one","model":"two","contentKey":"two","naming":"exact"}],"copies":[],"copiesComplete":true}`)
	test("oversize journal", strings.Repeat(" ", modeldownload.MaxJournalBytes+1))
}

func TestModelDownloadJournalEncodedByteBound(t *testing.T) {
	d := testModelDownloads(t)
	for i := 0; i < modeldownload.MaxUnsafeEntries; i++ {
		model := strings.Repeat("<", 500) + string(rune('a'+i))
		a := reserveModelDownload(t, d, "llamacpp", model, engines.ModelNamingExact)
		_, err := d.changeMarker(a, false)
		if err != nil {
			if len(d.unsafe) == 0 || len(d.unsafe) >= modeldownload.MaxUnsafeEntries {
				t.Fatal("byte bound did not precede slot bound")
			}
			return
		}
	}
	t.Fatal("escaped journal exceeded byte limit")
}

func TestModelDownloadJournalSerializesCrossEnginePublication(t *testing.T) {
	d := testModelDownloads(t)
	a := reserveModelDownload(t, d, "llamacpp", "one", engines.ModelNamingExact)
	b := reserveModelDownload(t, d, "ollama", "two", engines.ModelNamingImpliedLatestTag)
	entered, release := make(chan struct{}), make(chan struct{})
	var firstSync sync.Once
	d.store.replace.SyncDir = func(string) error { firstSync.Do(func() { close(entered); <-release }); return nil }
	first := make(chan error, 1)
	go func() { _, err := d.changeMarker(a, false); first <- err }()
	<-entered
	second := make(chan error, 1)
	go func() { _, err := d.changeMarker(b, false); second <- err }()
	// The short gate stays usable while the journal writer is held.
	if len(d.snapshot().Entries) != 2 {
		t.Fatal("snapshot blocked or lost reservation")
	}
	close(release)
	if err := <-first; err != nil {
		t.Fatal(err)
	}
	if err := <-second; err != nil {
		t.Fatal(err)
	}
	markers, err := d.store.load()
	if err != nil || len(markers) != 2 {
		t.Fatalf("cross-engine markers = %d/%v", len(markers), err)
	}
}

// All effects below are callbacks over private journal files. No executor,
// native engine, model store, listener or transfer is started by these cases.
func TestOwnerEvidenceCopyRetention(t *testing.T) {
	type fixture struct {
		manager   *Manager
		downloads *modelDownloads
		request   copyRequest
		self      string
	}
	setup := func(t *testing.T, self string) fixture {
		t.Helper()
		dir := t.TempDir()
		store := newModelDownloadStore(dir)
		if _, _, err := store.save(nil, nil, true); err != nil {
			t.Fatalf("seed explicit private baseline: %v", err)
		}
		d := newModelDownloads(newOpID(), dir, map[string]string{"llamacpp": engines.ModelNamingExact})
		registry := NewRegistry()
		registry.engines["llamacpp"] = &Manifest{Engine: "llamacpp", ModelNaming: engines.ModelNamingExact}
		m := &Manager{copyJournal: d, copyRegistry: registry}
		return fixture{manager: m, downloads: d, request: copyRequest{OpID: newOpID(), Engine: "llamacpp", Model: "private-model", SourceNodeID: "source", DestinationNodeID: "destination"}, self: self}
	}
	reserve := func(t *testing.T, f fixture) (context.Context, func()) {
		t.Helper()
		ctx, release, err := f.manager.reserveCopy(context.Background(), f.request)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(release)
		return ctx, release
	}
	result := func(p copyRequest) copyModelResult {
		return copyModelResult{Engine: p.Engine, Model: p.Model, Files: 1, BytesTotal: 10, BytesCopied: 10}
	}
	read := func(t *testing.T, d *modelDownloads) (map[downloadContentKey]modelDownloadMarker, map[string]modelCopyMarker, bool) {
		t.Helper()
		downloads, copies, complete, err := d.store.loadJournal()
		if err != nil {
			t.Fatalf("load journal: %v", err)
		}
		return downloads, copies, complete
	}
	beforeEffect := func(name, self, role string) {
		t.Run(name, func(t *testing.T) {
			f := setup(t, self)
			ctx, release := reserve(t, f)
			entered, resume := make(chan struct{}), make(chan struct{})
			var unblock sync.Once
			t.Cleanup(func() { unblock.Do(func() { close(resume) }) })
			f.downloads.store.replace.SyncDir = func(string) error { close(entered); <-resume; return nil }
			done := make(chan error, 1)
			go func() {
				_, err := f.manager.copyWithEvidence(ctx, f.request, f.self, func() (copyModelResult, error) {
					_, copies, complete, loadErr := f.downloads.store.loadJournal()
					if loadErr != nil {
						return copyModelResult{}, loadErr
					}
					marker := copies[f.request.OpID]
					if !complete || len(copies) != 1 || marker.Role != role || marker.OpID != f.request.OpID || marker.Engine != f.request.Engine || marker.Model != f.request.Model || marker.ContentKey != f.request.Model || marker.SourceNodeID != f.request.SourceNodeID || marker.DestinationNodeID != f.request.DestinationNodeID || marker.OwnerNodeID != self || marker.ManagerIncarnation != f.downloads.source || marker.Phase != "unsafe" {
						t.Errorf("before effect marker = %+v, complete = %v", marker, complete)
					}
					return copyModelResult{}, copyUnknown(io.ErrUnexpectedEOF)
				})
				done <- err
			}()
			<-entered
			live := f.manager.activeCopies()
			if len(live) != 1 || !live[0].pendingPublication || live[0].Progress != nil {
				t.Fatalf("pending publication not visible: %+v", live)
			}
			// Both short owner locks remain available while durable I/O waits.
			if f.downloads.snapshot().RecoveryRequired {
				t.Fatal("pending publication corrupted download evidence")
			}
			unblock.Do(func() { close(resume) })
			if err := <-done; err == nil {
				t.Fatal("unknown outcome succeeded")
			}
			release()
			_, copies, _ := read(t, f.downloads)
			if len(copies) != 1 || len(f.manager.activeCopies()) != 0 {
				t.Fatal("release erased retained evidence or kept live ownership")
			}
		})
	}
	beforeEffect("controller persists before forwarding", "controller", "controller")
	beforeEffect("destination persists before export import or file effects", "destination", "destination")

	admittedBeforeEffect := func(name, self, role string, inbound bool) {
		t.Run(name, func(t *testing.T) {
			f := setup(t, self)
			authority := rmNewFixture(t, "self-certificate")
			m := f.manager
			m.mesh, m.peers, m.codec = authority.mesh, newPeerDirectory(), NewCodec(&syncBuffer{})
			copyControlDirectory(m, copyControlNode(t, self, "self-certificate", nil))
			entered, resume := make(chan struct{}), make(chan struct{})
			var unblock sync.Once
			t.Cleanup(func() { unblock.Do(func() { close(resume) }); m.stopCopies() })
			f.downloads.store.replace.SyncDir = func(string) error { close(entered); <-resume; return nil }
			effects := make(chan struct{}, 1)
			stopped := errors.New("synthetic copy effect stopped")
			m.copyEffect = func(ctx context.Context, p copyRequest, owner string, _ func(copyProgress)) (copyModelResult, error) {
				effects <- struct{}{}
				_, copies, complete, err := f.downloads.store.loadJournal()
				if err != nil {
					return copyModelResult{}, err
				}
				marker := copies[p.OpID]
				if !complete || len(copies) != 1 || marker.OpID != f.request.OpID || marker.Role != role || marker.Model != f.request.Model || marker.ContentKey != f.request.Model || marker.Engine != f.request.Engine || marker.OwnerNodeID != self || marker.SourceNodeID != f.request.SourceNodeID || marker.DestinationNodeID != f.request.DestinationNodeID || marker.ManagerIncarnation != f.downloads.source || marker.Phase != "unsafe" {
					t.Errorf("actual admission dispatched before exact durable marker: %+v / %v", marker, complete)
				}
				live := m.activeCopies()
				if ctx.Err() != nil || owner != self || p != f.request || len(live) != 1 || live[0].pendingPublication {
					t.Errorf("effect admission = %+v / %+v / %v", p, live, ctx.Err())
				}
				return copyModelResult{}, stopped
			}
			done := make(chan error, 1)
			go func() {
				var err error
				if inbound {
					p := f.request
					p.DestinationNodeID = ""
					_, err = m.copyFromPeer(context.Background(), p, nil, nil)
				} else {
					_, err = m.executeCopy(context.Background(), f.request, false, nil, nil)
				}
				done <- err
			}()
			select {
			case <-entered:
			case err := <-done:
				t.Fatalf("actual admission ended before durable publication: %v", err)
			}
			live := m.activeCopies()
			if len(live) != 1 || live[0].copyRequest != f.request || !live[0].pendingPublication || len(effects) != 0 {
				t.Fatalf("actual pending publication = %+v, effects = %d", live, len(effects))
			}
			_, copies, complete := read(t, f.downloads)
			if len(copies) != 1 || !complete || copies[f.request.OpID].Role != role {
				t.Fatalf("pending durable record = %+v / %v", copies, complete)
			}
			unblock.Do(func() { close(resume) })
			if err := <-done; !errors.Is(err, stopped) || len(effects) != 1 {
				t.Fatalf("actual effect = %v, invocations = %d", err, len(effects))
			}
			_, copies, _ = read(t, f.downloads)
			if len(copies) != 1 || len(m.activeCopies()) != 0 {
				t.Fatal("actual admission retirement lost unsafe evidence or retained live ownership")
			}
		})
	}
	admittedBeforeEffect("executeCopy controller admission persists before first effect", "controller", "controller", false)
	admittedBeforeEffect("inbound destination admission persists before first effect", "destination", "destination", true)

	testSaveFailure := func(name string, landed bool) {
		t.Run(name, func(t *testing.T) {
			f := setup(t, "destination")
			ctx, release := reserve(t, f)
			if landed {
				f.downloads.store.replace.SyncDir = func(string) error { return errors.New("injected sync failure") }
			} else {
				f.downloads.store.replace.Replace = func(string, string) error { return errors.New("injected replace failure") }
			}
			effects := 0
			_, err := f.manager.copyWithEvidence(ctx, f.request, f.self, func() (copyModelResult, error) { effects++; return result(f.request), nil })
			release()
			if err == nil || effects != 0 || len(f.manager.activeCopies()) != 0 {
				t.Fatalf("save failure: error = %v, effects = %d", err, effects)
			}
			_, copies, _ := read(t, f.downloads)
			if (len(copies) == 1) != landed || f.downloads.copyRecoveryRequired != landed {
				t.Fatalf("retained = %+v, recovery = %v", copies, f.downloads.copyRecoveryRequired)
			}
			if landed && f.downloads.copies[f.request.OpID].Phase != "uncertain" {
				t.Fatal("landed uncertainty not retained in current owner")
			}
			successor := newModelDownloads(newOpID(), filepath.Dir(f.downloads.store.path), map[string]string{"llamacpp": engines.ModelNamingExact})
			if (len(successor.copies) == 1) != landed {
				t.Fatal("successor lost landed provenance")
			}
		})
	}
	testSaveFailure("pre replacement failure refuses and retires no effect work", false)
	testSaveFailure("landed uncertainty refuses and retains unsafe provenance", true)

	retainOutcome := func(name string, outcome func(fixture, context.Context) (copyModelResult, error)) {
		t.Run(name, func(t *testing.T) {
			f := setup(t, "destination")
			ctx, release := reserve(t, f)
			_, err := f.manager.copyWithEvidence(ctx, f.request, f.self, func() (copyModelResult, error) { return outcome(f, ctx) })
			if err == nil {
				t.Fatal("unproven completion succeeded")
			}
			release()
			_, copies, _ := read(t, f.downloads)
			if len(copies) != 1 || copies[f.request.OpID].ManagerIncarnation != f.downloads.source || len(f.manager.activeCopies()) != 0 {
				t.Fatalf("retained outcome = %+v", copies)
			}
		})
	}
	retainOutcome("unknown terminal", func(f fixture, _ context.Context) (copyModelResult, error) {
		return result(f.request), copyUnknown(io.ErrUnexpectedEOF)
	})
	retainOutcome("timeout", func(f fixture, _ context.Context) (copyModelResult, error) {
		return result(f.request), context.DeadlineExceeded
	})
	retainOutcome("cancelled terminal", func(f fixture, _ context.Context) (copyModelResult, error) {
		return result(f.request), context.Canceled
	})
	retainOutcome("cancellation request despite positive result", func(f fixture, _ context.Context) (copyModelResult, error) {
		f.manager.cancelCopy(f.request.OpID)
		return result(f.request), nil
	})
	retainOutcome("shutdown despite positive result", func(f fixture, _ context.Context) (copyModelResult, error) {
		f.manager.stopCopies()
		return result(f.request), nil
	})
	retainOutcome("mismatched model", func(f fixture, _ context.Context) (copyModelResult, error) {
		res := result(f.request)
		res.Model = "other-model"
		return res, nil
	})
	retainOutcome("mismatched engine", func(f fixture, _ context.Context) (copyModelResult, error) {
		res := result(f.request)
		res.Engine = "other-engine"
		return res, nil
	})

	t.Run("verified completion with qualified receipt durability clears marker", func(t *testing.T) {
		f := setup(t, "destination")
		ctx, release := reserve(t, f)
		_, err := f.manager.copyWithEvidence(ctx, f.request, f.self, func() (copyModelResult, error) {
			res := result(f.request)
			res.Layout, res.ExportID, res.RequiresConfiguration, res.Durability = llamaArtifactsLayout, strings.Repeat("a", 64), true, copyDurabilityUncertain
			return res, nil
		})
		if err != nil {
			t.Fatal(err)
		}
		release()
		_, copies, complete := read(t, f.downloads)
		if len(copies) != 0 || !complete {
			t.Fatal("positive local verification did not clear marker")
		}
	})

	verifiedCompletion := func(name, self string) {
		t.Run(name, func(t *testing.T) {
			f := setup(t, self)
			ctx, release := reserve(t, f)
			res, err := f.manager.copyWithEvidence(ctx, f.request, f.self, func() (copyModelResult, error) { return result(f.request), nil })
			if err != nil || !validCopyResult(res, f.request.Engine, f.request.Model) {
				t.Fatalf("completion = %+v / %v", res, err)
			}
			release()
			_, copies, complete := read(t, f.downloads)
			if len(copies) != 0 || !complete || len(f.downloads.copies) != 0 {
				t.Fatalf("verified completion not cleared: %+v / %v", copies, complete)
			}
		})
	}
	verifiedCompletion("verified controller completion", "controller")
	verifiedCompletion("verified destination completion", "destination")

	t.Run("completion publication failure keeps unsafe marker", func(t *testing.T) {
		f := setup(t, "destination")
		ctx, release := reserve(t, f)
		_, err := f.manager.copyWithEvidence(ctx, f.request, f.self, func() (copyModelResult, error) {
			f.downloads.store.replace.Replace = func(string, string) error { return errors.New("injected completion publication failure") }
			return result(f.request), nil
		})
		var unknown *copyOutcomeUnknownError
		if !errors.As(err, &unknown) {
			t.Fatalf("completion save failure = %v", err)
		}
		release()
		_, copies, _ := read(t, f.downloads)
		if len(copies) != 1 || len(f.downloads.copies) != 1 {
			t.Fatal("failed completion publication erased unsafe evidence")
		}
	})

	t.Run("expected export mismatch retains exact reservation", func(t *testing.T) {
		f := setup(t, "destination")
		f.request.ExpectedExportID = strings.Repeat("a", 64)
		ctx, release := reserve(t, f)
		_, err := f.manager.copyWithEvidence(ctx, f.request, f.self, func() (copyModelResult, error) {
			res := result(f.request)
			res.Layout, res.ExportID, res.RequiresConfiguration, res.Durability = llamaArtifactsLayout, strings.Repeat("b", 64), true, copyDurabilityConfirmed
			return res, nil
		})
		if err == nil {
			t.Fatal("mismatched export succeeded")
		}
		release()
		_, copies, _ := read(t, f.downloads)
		if len(copies) != 1 || copies[f.request.OpID].ExpectedExportID != f.request.ExpectedExportID {
			t.Fatalf("lost exact export reservation: %+v", copies)
		}
	})

	t.Run("removal requires exact retained identity", func(t *testing.T) {
		f := setup(t, "destination")
		ctx, _ := reserve(t, f)
		if _, err := f.manager.copyWithEvidence(ctx, f.request, f.self, func() (copyModelResult, error) { return copyModelResult{}, copyUnknown(io.EOF) }); err == nil {
			t.Fatal("unknown outcome succeeded")
		}
		marker := f.downloads.copies[f.request.OpID]
		marker.Model, marker.ContentKey = "different-model", "different-model"
		if err := f.downloads.changeCopyMarker(marker, true); !errors.Is(err, errDownloadBusy) {
			t.Fatalf("mismatched removal = %v", err)
		}
		_, copies, _ := read(t, f.downloads)
		if len(copies) != 1 || copies[f.request.OpID].Model != f.request.Model {
			t.Fatalf("mismatch erased marker: %+v", copies)
		}
	})

	t.Run("ambiguous or mismatched terminal response retains controller evidence", func(t *testing.T) {
		test := func(name string, change func(string) string) {
			t.Run(name, func(t *testing.T) {
				f := setup(t, "controller")
				ctx, release := reserve(t, f)
				payload, err := json.Marshal(result(f.request))
				if err != nil {
					t.Fatal(err)
				}
				frame, err := json.Marshal(streamFrame{Type: "result", OpID: f.request.OpID, Engine: f.request.Engine, Op: "copy", Result: payload})
				if err != nil {
					t.Fatal(err)
				}
				body := change(string(frame)) + "\n"
				client := &remoteClient{base: "https://private.invalid", controlHTTP: &http.Client{Transport: copyEvidenceTransport(func(*http.Request) (*http.Response, error) {
					return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(body))}, nil
				})}}
				_, err = f.manager.copyWithEvidence(ctx, f.request, f.self, func() (copyModelResult, error) { return client.copyModel(ctx, f.request, nil) })
				var unknown *copyOutcomeUnknownError
				if !errors.As(err, &unknown) {
					t.Fatalf("terminal response = %v", err)
				}
				release()
				_, copies, _ := read(t, f.downloads)
				if len(copies) != 1 {
					t.Fatal("invalid terminal response erased controller evidence")
				}
			})
		}
		test("operation mismatch", func(body string) string { return strings.Replace(body, `"opId":"`, `"opId":"0`, 1) })
		test("duplicate operation identity", func(body string) string {
			return strings.Replace(body, `"opId":`, `"opId":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","opId":`, 1)
		})
		test("duplicate terminal model", func(body string) string { return strings.Replace(body, `"model":`, `"model":"other","model":`, 1) })
		test("case folded terminal identity", func(body string) string { return strings.Replace(body, `"model":`, `"Model":`, 1) })
		test("duplicate export identity ending empty", func(body string) string {
			return strings.Replace(body, `"files":`, `"exportId":"`+strings.Repeat("a", 64)+`","exportId":"","files":`, 1)
		})
		test("case variant export identity ending empty", func(body string) string {
			return strings.Replace(body, `"files":`, `"exportId":"`+strings.Repeat("a", 64)+`","ExportId":"","files":`, 1)
		})
		test("case variant before empty exact export identity", func(body string) string {
			return strings.Replace(body, `"files":`, `"ExportId":"`+strings.Repeat("a", 64)+`","exportId":"","files":`, 1)
		})
		test("present empty export identity", func(body string) string { return strings.Replace(body, `"files":`, `"exportId":"","files":`, 1) })
		test("null optional export identity", func(body string) string { return strings.Replace(body, `"files":`, `"exportId":null,"files":`, 1) })
	})

	t.Run("shutdown successor keeps original incarnation and refuses predecessor completion", func(t *testing.T) {
		f := setup(t, "destination")
		ctx, release := reserve(t, f)
		if _, err := f.manager.copyWithEvidence(ctx, f.request, f.self, func() (copyModelResult, error) { return copyModelResult{}, copyUnknown(io.EOF) }); err == nil {
			t.Fatal("unknown outcome succeeded")
		}
		f.manager.stopCopies()
		release()
		successor := newModelDownloads(newOpID(), filepath.Dir(f.downloads.store.path), map[string]string{"llamacpp": engines.ModelNamingExact})
		marker := successor.copies[f.request.OpID]
		if len(successor.copies) != 1 || marker.ManagerIncarnation != f.downloads.source || marker.ManagerIncarnation == successor.source {
			t.Fatalf("successor provenance = %+v", marker)
		}
		if err := successor.changeCopyMarker(marker, true); !errors.Is(err, errDownloadRecovery) {
			t.Fatalf("predecessor completion = %v", err)
		}
	})

	t.Run("missing historical baseline stays unknown after download writes", func(t *testing.T) {
		dir := t.TempDir()
		d := newModelDownloads(newOpID(), dir, map[string]string{"llamacpp": engines.ModelNamingExact})
		if !d.copyRecoveryRequired {
			t.Fatal("missing state fabricated copy safety")
		}
		a := reserveModelDownload(t, d, "llamacpp", "download", engines.ModelNamingExact)
		if _, err := d.changeMarker(a, false); err != nil {
			t.Fatal(err)
		}
		_, _, complete := read(t, d)
		successor := newModelDownloads(newOpID(), dir, map[string]string{"llamacpp": engines.ModelNamingExact})
		if complete || !successor.copyRecoveryRequired {
			t.Fatal("download write fabricated historical completeness")
		}
		registry := NewRegistry()
		registry.engines["llamacpp"] = &Manifest{Engine: "llamacpp"}
		m := &Manager{copyJournal: successor, copyRegistry: registry}
		p := copyRequest{OpID: newOpID(), Engine: "llamacpp", Model: "private-model", SourceNodeID: "source", DestinationNodeID: "destination"}
		ctx, release, err := m.reserveCopy(context.Background(), p)
		if err != nil {
			t.Fatal(err)
		}
		defer release()
		effects := 0
		if _, err := m.copyWithEvidence(ctx, p, "destination", func() (copyModelResult, error) { effects++; return result(p), nil }); !errors.Is(err, errDownloadRecovery) || effects != 0 {
			t.Fatalf("historical admission = %v, effects = %d", err, effects)
		}
	})

	t.Run("malformed and old shapes fail closed without rewriting", func(t *testing.T) {
		f := setup(t, "destination")
		ctx, _ := reserve(t, f)
		if _, err := f.manager.copyWithEvidence(ctx, f.request, f.self, func() (copyModelResult, error) { return copyModelResult{}, copyUnknown(io.EOF) }); err == nil {
			t.Fatal("unknown outcome succeeded")
		}
		data, err := os.ReadFile(f.downloads.store.path)
		if err != nil {
			t.Fatal(err)
		}
		var journal modelDownloadJournal
		if err := json.Unmarshal(data, &journal); err != nil {
			t.Fatal(err)
		}
		malformed := func(name string, input []byte) {
			t.Run(name, func(t *testing.T) {
				dir := t.TempDir()
				path := filepath.Join(dir, modelDownloadFile)
				if err := os.WriteFile(path, input, 0600); err != nil {
					t.Fatal(err)
				}
				d := newModelDownloads(newOpID(), dir, map[string]string{"llamacpp": engines.ModelNamingExact})
				if !d.recoveryRequired || !d.copyRecoveryRequired {
					t.Fatal("malformed recovery claimed complete")
				}
				got, err := os.ReadFile(path)
				if err != nil || !bytes.Equal(got, input) {
					t.Fatalf("recovery rewrote journal: %v", err)
				}
			})
		}
		malformed("old shape", []byte(`{"version":1,"entries":[]}`))
		malformed("missing completeness", bytes.Replace(data, []byte(`,"copiesComplete":true`), nil, 1))
		malformed("null copies", bytes.Replace(data, []byte(`"copies":[`), []byte(`"copies":null,"ignored":[`), 1))
		malformed("duplicate role", bytes.Replace(data, []byte(`"role":"destination"`), []byte(`"role":"controller","role":"destination"`), 1))
		malformed("case variant role", bytes.Replace(data, []byte(`"role":`), []byte(`"Role":`), 1))
		malformed("unknown credential field", bytes.Replace(data, []byte(`"role":`), []byte(`"token":"private","role":`), 1))
		malformed("unpaired surrogate", bytes.Replace(data, []byte(`"private-model"`), []byte(`"\ud800"`), 1))
		invalid := func(name string, mutate func(*modelDownloadJournal)) {
			var next modelDownloadJournal
			if err := json.Unmarshal(data, &next); err != nil {
				t.Fatal(err)
			}
			mutate(&next)
			encoded, err := json.Marshal(next)
			if err != nil {
				t.Fatal(err)
			}
			malformed(name, encoded)
		}
		invalid("duplicate operation identity", func(j *modelDownloadJournal) { j.Copies = append(j.Copies, j.Copies[0]) })
		invalid("duplicate content endpoints", func(j *modelDownloadJournal) {
			other := j.Copies[0]
			other.OpID = newOpID()
			j.Copies = append(j.Copies, other)
		})
		invalid("role endpoint mismatch", func(j *modelDownloadJournal) { j.Copies[0].OwnerNodeID = "controller" })
		invalid("equal endpoints", func(j *modelDownloadJournal) { j.Copies[0].SourceNodeID = j.Copies[0].DestinationNodeID })
		invalid("invalid content identity", func(j *modelDownloadJournal) { j.Copies[0].ContentKey = "other-model" })
		invalid("invalid incarnation", func(j *modelDownloadJournal) { j.Copies[0].ManagerIncarnation = "unknown" })
		invalid("invalid phase", func(j *modelDownloadJournal) { j.Copies[0].Phase = "done" })
		invalid("aggregate bound", func(j *modelDownloadJournal) {
			for i := 0; i < modeldownload.MaxUnsafeEntries; i++ {
				j.Entries = append(j.Entries, modelDownloadMarker{ID: fmt.Sprintf("%032x", i+1), Engine: "llamacpp", Model: fmt.Sprintf("download-%d", i), ContentKey: fmt.Sprintf("download-%d", i), Naming: engines.ModelNamingExact})
			}
		})
	})

	t.Run("interleaved download publication preserves copy addition and removal", func(t *testing.T) {
		f := setup(t, "destination")
		ctx, release := reserve(t, f)
		a := reserveModelDownload(t, f.downloads, "llamacpp", "download", engines.ModelNamingExact)
		entered, resume := make(chan struct{}), make(chan struct{})
		var first sync.Once
		f.downloads.store.replace.SyncDir = func(string) error { first.Do(func() { close(entered); <-resume }); return nil }
		downloadDone := make(chan error, 1)
		go func() { _, err := f.downloads.changeMarker(a, false); downloadDone <- err }()
		<-entered
		copyEntered, copyResume := make(chan struct{}), make(chan struct{})
		var unblockDownload, unblockCopy sync.Once
		t.Cleanup(func() { unblockDownload.Do(func() { close(resume) }); unblockCopy.Do(func() { close(copyResume) }) })
		copyDone := make(chan error, 1)
		go func() {
			_, err := f.manager.copyWithEvidence(ctx, f.request, f.self, func() (copyModelResult, error) {
				close(copyEntered)
				<-copyResume
				return result(f.request), nil
			})
			copyDone <- err
		}()
		unblockDownload.Do(func() { close(resume) })
		if err := <-downloadDone; err != nil {
			t.Fatal(err)
		}
		<-copyEntered
		downloads, copies, complete := read(t, f.downloads)
		if len(downloads) != 1 || len(copies) != 1 || !complete {
			t.Fatalf("addition lost collection: %+v / %+v / %v", downloads, copies, complete)
		}
		// Now hold the download writer with its stale-before-copy view while
		// completion waits to remove only the matching copy record.
		removeEntered, removeResume := make(chan struct{}), make(chan struct{})
		var second, unblockRemove sync.Once
		t.Cleanup(func() { unblockRemove.Do(func() { close(removeResume) }) })
		f.downloads.store.replace.SyncDir = func(string) error { second.Do(func() { close(removeEntered); <-removeResume }); return nil }
		go func() { _, err := f.downloads.changeMarker(a, false); downloadDone <- err }()
		<-removeEntered
		downloads, copies, complete = read(t, f.downloads)
		if len(downloads) != 1 || len(copies) != 1 || !complete {
			t.Fatalf("download rewrite lost live copy collection: %+v / %+v / %v", downloads, copies, complete)
		}
		unblockCopy.Do(func() { close(copyResume) })
		unblockRemove.Do(func() { close(removeResume) })
		if err := <-downloadDone; err != nil {
			t.Fatal(err)
		}
		if err := <-copyDone; err != nil {
			t.Fatal(err)
		}
		release()
		downloads, copies, complete = read(t, f.downloads)
		if len(downloads) != 1 || downloads[a.key].ID != a.entry.ID || len(copies) != 0 || !complete {
			t.Fatalf("removal lost collection: %+v / %+v / %v", downloads, copies, complete)
		}
	})
}

type copyEvidenceTransport func(*http.Request) (*http.Response, error)

func (f copyEvidenceTransport) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }
