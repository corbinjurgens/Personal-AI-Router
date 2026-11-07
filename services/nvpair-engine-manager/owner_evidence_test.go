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
	"math"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"testing"

	"nvpair-shared/buildapply"
	"nvpair-shared/engines"
	"nvpair-shared/modeldownload"
)

// Private journal fixtures exercise the actual reservation owners, without
// peers, listeners, content files, native engines or transfer execution.
func TestOwnerEvidenceReservations(t *testing.T) {
	setup := func(t *testing.T) (*Manager, *modelDownloads, copyRequest) {
		t.Helper()
		dir := t.TempDir()
		store := newModelDownloadStore(dir)
		if _, _, err := store.save(nil, nil, true, false); err != nil {
			t.Fatalf("seed complete current-format baseline: %v", err)
		}
		d := newModelDownloads(newOpID(), dir, map[string]string{"llamacpp": engines.ModelNamingExact})
		reg := &Registry{engines: map[string]*Manifest{"llamacpp": {Engine: "llamacpp", ModelNaming: engines.ModelNamingExact}}}
		m := &Manager{exec: &Executor{managerIncarnation: d.source}, copyJournal: d, copyRegistry: reg}
		return m, d, copyRequest{OpID: newOpID(), Engine: "llamacpp", Model: "model", SourceNodeID: "source", DestinationNodeID: "destination"}
	}
	reserve := func(t *testing.T, m *Manager, p copyRequest) (context.Context, func()) {
		t.Helper()
		ctx, release, err := m.reserveCopy(context.Background(), p)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(release)
		return ctx, release
	}
	result := func(p copyRequest) copyModelResult {
		return copyModelResult{Engine: p.Engine, Model: p.Model, Files: 1, BytesTotal: 10, BytesCopied: 10}
	}
	clear := func(t *testing.T, r ownerReservationEvidence) {
		t.Helper()
		if !r.Complete {
			t.Fatalf("expected complete evidence: %+v", r)
		}
		requireOwnerBool(t, r.Busy, false)
	}

	t.Run("clean current-format baseline is complete and read-only", func(t *testing.T) {
		m, d, _ := setup(t)
		before, err := os.ReadFile(d.store.path)
		if err != nil {
			t.Fatal(err)
		}
		d.store.replace.SyncDir = func(string) error { t.Error("evidence wrote journal"); return nil }
		first, second := m.ownerReservations(), m.ownerReservations()
		clear(t, first)
		if !reflect.DeepEqual(first, second) || first.Token.Incarnation != d.source || first.Token.DownloadSource != d.source {
			t.Fatalf("unstable idle capture: %+v / %+v", first, second)
		}
		after, err := os.ReadFile(d.store.path)
		if err != nil || !bytes.Equal(before, after) {
			t.Fatalf("evidence changed journal: %v", err)
		}
	})

	t.Run("overlapping copy and download have one capture and token", func(t *testing.T) {
		m, d, p := setup(t)
		ctx, release := reserve(t, m, p)
		entered, proceed := make(chan struct{}), make(chan struct{})
		var once sync.Once
		defer once.Do(func() { close(proceed) })
		done := make(chan error, 1)
		go func() {
			_, err := m.copyWithEvidence(ctx, p, "destination", func() (copyModelResult, error) {
				close(entered)
				<-proceed
				return result(p), nil
			})
			done <- err
		}()
		select {
		case <-entered:
		case err := <-done:
			t.Fatalf("copy ended before completion seam: %v", err)
		}
		a := reserveModelDownload(t, d, "llamacpp", "other", engines.ModelNamingExact)
		r := m.ownerReservations()
		requireOwnerBool(t, r.Busy, true)
		if !r.Complete || len(r.ActiveDownloads) != 1 || len(r.Copies) != 1 || len(r.RetainedCopies) != 1 || r.Copies[0].Role != "destination" || r.Copies[0].PendingPublication || r.ActiveDownloads[0].Entry.ID != a.entry.ID || r.Token.DownloadRevision != d.revision || r.Token.CopyRevision != m.copyRevision || r.Token.ContentGeneration != d.authorityGeneration {
			t.Fatalf("joint capture = %+v", r)
		}
		if err := d.finish(a, "failed", "interrupted", ""); err != nil {
			t.Fatal(err)
		}
		once.Do(func() { close(proceed) })
		if err := <-done; err != nil {
			t.Fatal(err)
		}
		release()
		clear(t, m.ownerReservations())
	})

	t.Run("unknown copy retirement preserves controller evidence", func(t *testing.T) {
		m, _, p := setup(t)
		ctx, release := reserve(t, m, p)
		_, err := m.copyWithEvidence(ctx, p, "controller", func() (copyModelResult, error) {
			r := m.ownerReservations()
			if len(r.Copies) != 1 || r.Copies[0].Role != "controller" {
				t.Errorf("controller role = %+v", r)
			}
			return copyModelResult{}, copyUnknown(io.ErrUnexpectedEOF)
		})
		if err == nil {
			t.Fatal("unknown completion accepted")
		}
		release()
		r := m.ownerReservations()
		requireOwnerBool(t, r.Busy, true)
		if r.Complete || len(r.Copies) != 0 || len(r.RetainedCopies) != 1 || r.RetainedCopies[0].OpID != p.OpID {
			t.Fatalf("unknown retirement = %+v", r)
		}
	})

	t.Run("active and retained unsafe download share a key without hiding either", func(t *testing.T) {
		m, d, _ := setup(t)
		old := reserveModelDownload(t, d, "llamacpp", "model", engines.ModelNamingExact)
		if _, err := d.changeMarker(old, false); err != nil {
			t.Fatal(err)
		}
		if err := d.finish(old, "unknown", "interrupted", ""); err != nil {
			t.Fatal(err)
		}
		current := reserveModelDownload(t, d, "llamacpp", "model", engines.ModelNamingExact)
		r := m.ownerReservations()
		requireOwnerBool(t, r.Busy, true)
		if r.Complete || len(r.ActiveDownloads) != 1 || len(r.UnsafeDownloads) != 1 || r.ActiveDownloads[0].Entry.ID != current.entry.ID || r.UnsafeDownloads[0].Marker.ID != old.entry.ID || r.UnsafeDownloads[0].Incarnation != d.source {
			t.Fatalf("same-key reservations = %+v", r)
		}
	})

	t.Run("queued cancellation retires only after completion owner runs", func(t *testing.T) {
		m, d, _ := setup(t)
		a := reserveModelDownload(t, d, "llamacpp", "queued", engines.ModelNamingExact)
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		observed, proceed := make(chan struct{}), make(chan struct{})
		var once sync.Once
		defer once.Do(func() { close(proceed) })
		done := make(chan error, 1)
		go func() { <-ctx.Done(); close(observed); <-proceed; done <- d.finish(a, "failed", "interrupted", "") }()
		cancel()
		<-observed
		before := m.ownerReservations()
		requireOwnerBool(t, before.Busy, true)
		if len(before.ActiveDownloads) != 1 || before.ActiveDownloads[0].Entry.Phase != modeldownload.PhaseQueued {
			t.Fatalf("cancel prematurely retired queue: %+v", before)
		}
		once.Do(func() { close(proceed) })
		if err := <-done; err != nil {
			t.Fatal(err)
		}
		after := m.ownerReservations()
		clear(t, after)
		if before.Token.ContentGeneration == after.Token.ContentGeneration {
			t.Fatal("queued retirement did not advance authority")
		}
	})

	uncertain := func(name string, remove bool) {
		t.Run(name, func(t *testing.T) {
			m, d, _ := setup(t)
			a := reserveModelDownload(t, d, "llamacpp", "model", engines.ModelNamingExact)
			if remove {
				if _, err := d.changeMarker(a, false); err != nil {
					t.Fatal(err)
				}
			}
			before := m.ownerReservations()
			entered, proceed := make(chan struct{}), make(chan struct{})
			var once sync.Once
			defer once.Do(func() { close(proceed) })
			d.store.replace.SyncDir = func(string) error {
				close(entered)
				<-proceed
				return errors.New("uncertain directory sync")
			}
			done := make(chan error, 1)
			go func() { _, err := d.changeMarker(a, remove); done <- err }()
			select {
			case <-entered:
			case err := <-done:
				t.Fatalf("marker publication ended before sync seam: %v", err)
			}
			during := m.ownerReservations()
			requireOwnerBool(t, during.Busy, true)
			once.Do(func() { close(proceed) })
			if err := <-done; err == nil {
				t.Fatal("uncertain publication reported success")
			}
			if err := d.finish(a, "unknown", "interrupted", "uncertain"); err != nil {
				t.Fatal(err)
			}
			after := m.ownerReservations()
			if after.Complete || !after.Uncertain || after.Token.ContentGeneration == before.Token.ContentGeneration || (len(after.UnsafeDownloads) == 0) != remove {
				t.Fatalf("uncertain publication = %+v", after)
			}
			if remove {
				requireOwnerUnknown(t, after.Busy)
			} else {
				requireOwnerBool(t, after.Busy, true)
			}
		})
	}
	uncertain("uncertain download marker addition is retained and incomplete", false)
	uncertain("uncertain download marker removal cannot establish clear", true)

	uncertainCopy := func(name string, publication int) {
		t.Run(name, func(t *testing.T) {
			m, d, p := setup(t)
			ctx, release := reserve(t, m, p)
			before := m.ownerReservations()
			entered, proceed := make(chan struct{}), make(chan struct{})
			var once sync.Once
			defer once.Do(func() { close(proceed) })
			writes := 0
			d.store.replace.SyncDir = func(string) error {
				writes++
				if writes != publication {
					return nil
				}
				close(entered)
				<-proceed
				return errors.New("uncertain copy publication")
			}
			done := make(chan error, 1)
			go func() {
				_, err := m.copyWithEvidence(ctx, p, "controller", func() (copyModelResult, error) { return result(p), nil })
				done <- err
			}()
			select {
			case <-entered:
			case err := <-done:
				t.Fatalf("copy ended before publication seam: %v", err)
			}
			requireOwnerBool(t, m.ownerReservations().Busy, true)
			once.Do(func() { close(proceed) })
			if err := <-done; err == nil {
				t.Fatal("uncertain copy publication succeeded")
			}
			release()
			after := m.ownerReservations()
			if after.Complete || !after.CopyRecoveryRequired || !after.Uncertain || after.Token.ContentGeneration <= before.Token.ContentGeneration || (len(after.RetainedCopies) == 1) != (publication == 1) {
				t.Fatalf("uncertain copy evidence = %+v", after)
			}
			if publication == 1 {
				requireOwnerBool(t, after.Busy, true)
			} else {
				requireOwnerUnknown(t, after.Busy)
			}
		})
	}
	uncertainCopy("uncertain copy marker addition advances recovery authority", 1)
	uncertainCopy("uncertain copy marker removal advances recovery authority", 2)

	historical := func(name, document string, recovery bool) {
		t.Run(name, func(t *testing.T) {
			m, d, _ := setup(t)
			if document == "" {
				if err := os.Remove(d.store.path); err != nil {
					t.Fatal(err)
				}
			} else if err := os.WriteFile(d.store.path, []byte(document), 0600); err != nil {
				t.Fatal(err)
			}
			m.copyJournal = newModelDownloads(d.source, filepath.Dir(d.store.path), map[string]string{"llamacpp": engines.ModelNamingExact})
			r := m.ownerReservations()
			if r.Complete || m.copyJournal.copiesComplete || r.CopyRecoveryRequired != recovery {
				t.Fatalf("historical state declared complete: %+v", r)
			}
			if r.RecoveryRequired != recovery {
				t.Fatal("unreadable content baseline omitted recovery-required evidence")
			}
			requireOwnerUnknown(t, r.Busy)
			_, release := reserve(t, m, copyRequest{OpID: newOpID(), Engine: "llamacpp"})
			r = m.ownerReservations()
			requireOwnerBool(t, r.Busy, true)
			if r.Complete {
				t.Fatal("known busy hid unknown historical baseline")
			}
			release()
		})
	}
	historical("missing historical baseline stays unknown", "", false)
	historical("old journal shape requires recovery", `{"version":1,"entries":[]}`, true)
	historical("unreadable journal requires recovery", `invalid journal`, true)

	t.Run("reader and writer ABA changes authority without notifications", func(t *testing.T) {
		m, d, _ := setup(t)
		before := m.ownerReservations()
		key := downloadContentKey{engine: "llamacpp", model: "model"}
		read, err := d.acquireRead(key, false)
		if err != nil {
			t.Fatal(err)
		}
		during := m.ownerReservations()
		requireOwnerBool(t, during.Busy, true)
		if during.Readers["llamacpp"] != 1 {
			t.Fatalf("reader missing: %+v", during)
		}
		read()
		read()
		afterRead := m.ownerReservations()
		clear(t, afterRead)
		write, err := d.acquireWrite("llamacpp")
		if err != nil {
			t.Fatal(err)
		}
		during = m.ownerReservations()
		requireOwnerBool(t, during.Busy, true)
		if !reflect.DeepEqual(during.Writers, []string{"llamacpp"}) || len(during.ActiveDownloads) != 0 {
			t.Fatalf("writer without progress missing: %+v", during)
		}
		write()
		write()
		afterWrite := m.ownerReservations()
		clear(t, afterWrite)
		if before.Token.ContentGeneration+2 != afterRead.Token.ContentGeneration || afterRead.Token.ContentGeneration+2 != afterWrite.Token.ContentGeneration || before.Token.DownloadRevision != afterWrite.Token.DownloadRevision {
			t.Fatalf("ABA token = %+v / %+v / %+v", before.Token, afterRead.Token, afterWrite.Token)
		}
	})

	t.Run("copy identity role and shutdown change authority", func(t *testing.T) {
		m, _, p := setup(t)
		p.DestinationNodeID = ""
		ctx, release := reserve(t, m, p)
		before := m.ownerReservations()
		p.DestinationNodeID = "destination"
		m.copyMu.Lock()
		err := m.bindCopyAuthorityLocked(m.copies[p.OpID], p, "destination")
		m.copyMu.Unlock()
		if err != nil {
			t.Fatal(err)
		}
		after := m.ownerReservations()
		if before.Token.CopyRevision == after.Token.CopyRevision || after.Copies[0].DestinationNodeID != "destination" || after.Copies[0].Role != "destination" {
			t.Fatalf("role binding = %+v", after)
		}
		m.stopCopies()
		release()
		closed := m.ownerReservations()
		requireOwnerBool(t, closed.Busy, true)
		if !closed.Shutdown || len(closed.Copies) != 0 || ctx.Err() != context.Canceled || closed.Token.CopyRevision <= after.Token.CopyRevision {
			t.Fatalf("shutdown = %+v, context = %v", closed, ctx.Err())
		}
		m.stopCopies()
		if m.ownerReservations().Token != closed.Token {
			t.Fatal("repeated shutdown consumed capacity")
		}
	})

	t.Run("download percentages and nested copy progress are detached", func(t *testing.T) {
		m, d, p := setup(t)
		_, _ = reserve(t, m, p)
		a := reserveModelDownload(t, d, "llamacpp", "download", engines.ModelNamingExact)
		percent := 10
		if err := d.update(a, modeldownload.PhaseInProgress, &percent); err != nil {
			t.Fatal(err)
		}
		state := d.snapshot()
		m.copyMu.Lock()
		m.copies[p.OpID].Progress = &copyProgress{remoteProgress: remoteProgress{Percent: 20, DownloadState: &state, Message: "transport diagnostic"}}
		m.copyMu.Unlock()
		before := m.ownerReservations()
		public := m.activeCopies()
		if public[0].cancel != nil {
			t.Fatal("active copies leaked cancellation authority")
		}
		public[0].Progress.DownloadState.Entries[0].Model = "changed-public"
		*public[0].Progress.DownloadState.Entries[0].Percent = 99
		before.Copies[0].Progress.Percent = 90
		*before.Copies[0].Progress.DownloadState.Entries[0].Percent = 98
		*before.ActiveDownloads[0].Entry.Percent = 97
		after := m.ownerReservations()
		if after.Copies[0].Progress.Percent != 20 || *after.Copies[0].Progress.DownloadState.Entries[0].Percent != 10 || *after.ActiveDownloads[0].Entry.Percent != 10 || after.Copies[0].Progress.Message != "" {
			t.Fatalf("capture aliases owner or exposes diagnostics: %+v", after)
		}
		percent = 30
		if err := d.update(a, modeldownload.PhaseInProgress, &percent); err != nil {
			t.Fatal(err)
		}
		progress := m.ownerReservations()
		if progress.Token.ContentGeneration != after.Token.ContentGeneration || progress.Token.DownloadRevision == after.Token.DownloadRevision {
			t.Fatal("percentage changed authority or lost notification revision")
		}
	})

	t.Run("successor retains predecessor copy and unknown download origin", func(t *testing.T) {
		m, d, p := setup(t)
		ctx, release := reserve(t, m, p)
		if _, err := m.copyWithEvidence(ctx, p, "controller", func() (copyModelResult, error) { return copyModelResult{}, copyUnknown(io.ErrUnexpectedEOF) }); err == nil {
			t.Fatal("unknown copy accepted")
		}
		release()
		a := reserveModelDownload(t, d, "llamacpp", "old-download", engines.ModelNamingExact)
		if _, err := d.changeMarker(a, false); err != nil {
			t.Fatal(err)
		}
		newID := newOpID()
		successor := newModelDownloads(newID, filepath.Dir(d.store.path), map[string]string{"llamacpp": engines.ModelNamingExact})
		fresh := &Manager{exec: &Executor{managerIncarnation: newID}, copyJournal: successor}
		r := fresh.ownerReservations()
		requireOwnerBool(t, r.Busy, true)
		if r.Complete || len(r.Copies) != 0 || len(r.RetainedCopies) != 1 || r.RetainedCopies[0].ManagerIncarnation != d.source || len(r.UnsafeDownloads) != 1 || r.UnsafeDownloads[0].Incarnation != "" || r.UnsafeDownloads[0].Marker.ID != a.entry.ID || r.Token.Incarnation != newID {
			t.Fatalf("predecessor provenance = %+v", r)
		}
	})

	t.Run("contended subsection keeps known busy independently of completeness", func(t *testing.T) {
		m, d, p := setup(t)
		_, _ = reserve(t, m, p)
		d.mu.Lock()
		r := m.ownerReservations()
		d.mu.Unlock()
		requireOwnerBool(t, r.Busy, true)
		if r.Complete {
			t.Fatal("contended content owner declared complete")
		}
		m.copyMu.Lock()
		r = m.ownerReservations()
		m.copyMu.Unlock()
		requireOwnerUnknown(t, r.Busy)
	})

	t.Run("content saturation refuses admissions and preserves download retirement", func(t *testing.T) {
		m, d, _ := setup(t)
		d.authorityGeneration = math.MaxUint64 - 5
		a := reserveModelDownload(t, d, "llamacpp", "model", engines.ModelNamingExact)
		if _, err := d.reserve("llamacpp", "other", engines.ModelNamingExact); !errors.Is(err, errDownloadCapacity) {
			t.Fatalf("saturated download admission = %v", err)
		}
		if _, err := d.acquireWrite("other"); !errors.Is(err, errDownloadCapacity) {
			t.Fatalf("writer consumed retirement capacity: %v", err)
		}
		if _, err := d.acquireRead(downloadContentKey{engine: "other"}, false); !errors.Is(err, errDownloadCapacity) {
			t.Fatalf("reader consumed retirement capacity: %v", err)
		}
		if err := d.update(a, modeldownload.PhaseInProgress, nil); err != nil {
			t.Fatal(err)
		}
		if _, err := d.changeMarker(a, false); err != nil {
			t.Fatal(err)
		}
		if _, err := d.changeMarker(a, true); err != nil {
			t.Fatal(err)
		}
		if err := d.finish(a, "complete", "", ""); err != nil {
			t.Fatal(err)
		}
		r := m.ownerReservations()
		clear(t, r)
		if r.Token.ContentGeneration != math.MaxUint64 || d.authorityHeld != 0 {
			t.Fatalf("authority wrapped or leaked capacity: %+v, held %d", r.Token, d.authorityHeld)
		}
	})

	gateSaturation := func(name string, writer bool) {
		t.Run(name, func(t *testing.T) {
			m, d, _ := setup(t)
			d.authorityGeneration = math.MaxUint64 - 2
			var release func()
			var err error
			if writer {
				release, err = d.acquireWrite("llamacpp")
			} else {
				release, err = d.acquireRead(downloadContentKey{engine: "llamacpp"}, false)
			}
			if err != nil {
				t.Fatal(err)
			}
			if _, err := d.acquireRead(downloadContentKey{engine: "other"}, false); !errors.Is(err, errDownloadCapacity) {
				t.Fatalf("new reader consumed release capacity: %v", err)
			}
			release()
			release()
			r := m.ownerReservations()
			clear(t, r)
			if r.Token.ContentGeneration != math.MaxUint64 || d.authorityHeld != 0 {
				t.Fatalf("gate release wrapped: %+v", r)
			}
		})
	}
	gateSaturation("reader release survives saturation", false)
	gateSaturation("writer release survives saturation", true)

	t.Run("copy marker reserves authority retirement independently of download revision", func(t *testing.T) {
		m, d, p := setup(t)
		d.authorityGeneration, d.revision = math.MaxUint64-2, math.MaxUint64-2
		marker := modelCopyMarker{OpID: p.OpID, Engine: p.Engine, Model: p.Model, ContentKey: p.Model, Naming: engines.ModelNamingExact, Role: "controller", OwnerNodeID: "controller", SourceNodeID: p.SourceNodeID, DestinationNodeID: p.DestinationNodeID, ManagerIncarnation: d.source, Phase: "unsafe"}
		if err := d.changeCopyMarker(marker, false); err != nil {
			t.Fatal(err)
		}
		if _, err := d.reserve("llamacpp", "other", engines.ModelNamingExact); !errors.Is(err, errDownloadCapacity) {
			t.Fatalf("download consumed marker retirement: %v", err)
		}
		if err := d.changeCopyMarker(marker, true); err != nil {
			t.Fatalf("accepted marker lost retirement capacity: %v", err)
		}
		r := m.ownerReservations()
		clear(t, r)
		if d.authorityHeld != 0 || d.heldRevisions != 0 || r.Token.ContentGeneration != math.MaxUint64 || r.Token.DownloadRevision != math.MaxUint64-2 {
			t.Fatalf("marker retirement wrapped or leaked: %+v", r.Token)
		}
	})

	t.Run("copy saturation preserves identity publication retirement and shutdown", func(t *testing.T) {
		m, d, p := setup(t)
		m.copyRevision = math.MaxUint64 - 6
		p.DestinationNodeID = ""
		ctx, release := reserve(t, m, p)
		if _, _, err := m.reserveCopy(context.Background(), copyRequest{OpID: newOpID()}); !errors.Is(err, errOwnerAdmissionCapacity) {
			t.Fatalf("copy admission consumed retirement: %v", err)
		}
		p.DestinationNodeID = "destination"
		m.copyMu.Lock()
		err := m.bindCopyAuthorityLocked(m.copies[p.OpID], p, "")
		m.copyMu.Unlock()
		if err != nil {
			t.Fatal(err)
		}
		// Journal callbacks and completion effects must run outside both locks.
		unlocked := func() {
			if !m.copyMu.TryLock() {
				t.Error("copy lock held across callback")
			} else {
				m.copyMu.Unlock()
			}
			if !d.mu.TryLock() {
				t.Error("content lock held across callback")
			} else {
				d.mu.Unlock()
			}
		}
		d.store.replace.SyncDir = func(string) error { unlocked(); return nil }
		if _, err := m.copyWithEvidence(ctx, p, "controller", func() (copyModelResult, error) { unlocked(); return result(p), nil }); err != nil {
			t.Fatal(err)
		}
		release()
		clear(t, m.ownerReservations())
		m.stopCopies()
		r := m.ownerReservations()
		requireOwnerBool(t, r.Busy, true)
		if r.Token.CopyRevision != math.MaxUint64 || m.copyHeld != 0 {
			t.Fatalf("copy shutdown wrapped or leaked: %+v, held %d", r.Token, m.copyHeld)
		}
	})
}

func TestOwnerEvidenceUniverseMembership(t *testing.T) {
	e := &Executor{reg: &Registry{engines: map[string]*Manifest{"registered": {}}}, engines: map[string]*engineState{}, stopOnly: map[string]*engineState{"cleanup": {}}}
	saved := desiredOwnerEvidence{State: desiredStateFile{AllocatedPorts: map[string]int{"port": 0}}}
	got, ok := e.ownerUniverseMembership(saved)
	want := []ownerUniverseEntry{{Name: "cleanup", Cleanup: true}, {Name: "port", Cleanup: true}, {Name: "registered", Registered: true}}
	if !ok || !reflect.DeepEqual(got, want) {
		t.Fatalf("membership = %+v, %v; want %+v", got, ok, want)
	}
	got[0].Name = "changed"
	again, ok := e.ownerUniverseMembership(saved)
	if !ok || !reflect.DeepEqual(again, want) || len(e.engines) != 0 {
		t.Fatalf("membership changed owner state: %+v, %v", again, ok)
	}
	e.mu.Lock()
	_, ok = e.ownerUniverseMembership(saved)
	e.mu.Unlock()
	if ok {
		t.Fatal("contended membership must be unknown")
	}
}

// No constructor, logs, content gate, watcher, engine, file or native owner is
// initialized by these fixtures. The saved reader is the Task1 detached seam.
func ownerEvidenceFixture(t *testing.T) (*Executor, *engineState, desiredOwnerEvidence) {
	t.Helper()
	root := filepath.Join(os.TempDir(), "pair-owner-evidence-inert")
	platform := Platform{ModelsDir: filepath.Join(root, "models"), Runtime: Runtime{Port: 0}}
	manifest := &Manifest{Engine: "fake", Platforms: map[string]Platform{runtime.GOOS + "/" + runtime.GOARCH: platform}}
	st := &engineState{manifest: manifest, plat: &platform, installDir: filepath.Join(root, "fake"), modelsDir: platform.ModelsDir}
	e := &Executor{managerIncarnation: "manager-fixture", reg: &Registry{engines: map[string]*Manifest{"fake": manifest}}, engines: map[string]*engineState{"fake": st}, stopOnly: map[string]*engineState{},
		desired: newDesiredStateStore(t.TempDir()),
		client:  &http.Client{Transport: ownerEvidenceRejectHTTP{}}, emit: func(string, any) { panic("evidence notified") },
		inventory: map[string][]string{"fake": {"must-not-be-read"}},
	}
	return e, st, desiredOwnerEvidence{State: desiredStateFile{}}
}

type ownerEvidenceRejectHTTP struct{}

func (ownerEvidenceRejectHTTP) RoundTrip(*http.Request) (*http.Response, error) {
	panic("evidence contacted a native engine")
}

func readOwnerFixture(e *Executor, saved desiredOwnerEvidence) ownerUniverseEvidence {
	return e.captureOwnerUniverse(func() (desiredOwnerEvidence, error) { return saved, nil }, func() ([]string, bool) { return nil, true })
}

func ownerFixtureEntry(t *testing.T, got ownerUniverseEvidence, name string) ownerEngineEvidence {
	t.Helper()
	if !got.Complete || got.Incarnation != "manager-fixture" {
		t.Fatalf("incomplete fixture evidence: %+v", got)
	}
	for _, entry := range got.Entries {
		if entry.Name == name {
			return entry
		}
	}
	t.Fatalf("engine %q missing from %+v", name, got)
	return ownerEngineEvidence{}
}

func requireOwnerUnknown(t *testing.T, observation buildapply.Observation[bool]) {
	t.Helper()
	if observation.Known || observation.Value != nil {
		t.Fatalf("wanted unknown, got %+v", observation)
	}
}

func requireOwnerBool(t *testing.T, observation buildapply.Observation[bool], want bool) {
	t.Helper()
	if !observation.Known || observation.Value == nil || *observation.Value != want {
		t.Fatalf("observation = %+v, want known %v", observation, want)
	}
}

// Inert owners only: no Manager/Executor constructor, engine, listener, native
// detection, recovery or content mutation is needed to exercise the RPC.
func ownerEvidenceRPCFixture(t *testing.T) (*Manager, *bytes.Buffer) {
	t.Helper()
	e, _, _ := ownerEvidenceFixture(t)
	d := &modelDownloads{source: e.managerIncarnation, copiesComplete: true}
	e.downloads = d
	var out bytes.Buffer
	return &Manager{exec: e, copyJournal: d, codec: NewCodec(&out)}, &out
}

func callOwnerEvidenceRPC(t *testing.T, m *Manager, out *bytes.Buffer, params string) Message {
	t.Helper()
	out.Reset()
	id := json.RawMessage(`199`)
	m.handleMessage(context.Background(), &Message{JSONRPC: "2.0", ID: &id, Method: buildapply.EngineOwnerEvidenceMethod, Params: json.RawMessage(params)})
	var reply Message
	if err := json.Unmarshal(out.Bytes(), &reply); err != nil {
		t.Fatalf("decode owner RPC: %v", err)
	}
	return reply
}

func decodeOwnerEvidenceRPC(t *testing.T, reply Message) buildapply.EngineOwnerEvidence {
	t.Helper()
	if reply.Error != nil {
		t.Fatalf("owner RPC failed: %+v", reply.Error)
	}
	var got buildapply.EngineOwnerEvidence
	if err := json.Unmarshal(reply.Result, &got); err != nil {
		t.Fatalf("decode owner evidence: %v", err)
	}
	return got
}

func TestOwnerEvidenceRPC(t *testing.T) {
	t.Run("idle native identity without initializing engines", func(t *testing.T) {
		m, out := ownerEvidenceRPCFixture(t)
		m.exec.engines = map[string]*engineState{}
		m.exec.reg.engines = map[string]*Manifest{}
		before, err := os.ReadDir(filepath.Dir(m.exec.desired.path))
		if err != nil {
			t.Fatal(err)
		}
		first := decodeOwnerEvidenceRPC(t, callOwnerEvidenceRPC(t, m, out, `{}`))
		second := decodeOwnerEvidenceRPC(t, callOwnerEvidenceRPC(t, m, out, `{}`))
		if first.ManagerIncarnation != m.exec.managerIncarnation || first.CaptureToken == "" || !first.Complete || !reflect.DeepEqual(first, second) {
			t.Fatalf("unstable idle evidence: %+v / %+v", first, second)
		}
		if len(m.exec.engines) != 0 || !first.Universe.Known || first.Universe.Value == nil || *first.Universe.Value == nil || len(*first.Universe.Value) != 0 {
			t.Fatalf("idle read initialized engines or lost known empty: %+v", first)
		}
		after, err := os.ReadDir(filepath.Dir(m.exec.desired.path))
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(before, after) {
			t.Fatal("evidence wrote the owner directory")
		}
	})

	t.Run("response has exact private section names", func(t *testing.T) {
		m, out := ownerEvidenceRPCFixture(t)
		reply := callOwnerEvidenceRPC(t, m, out, `{}`)
		got := decodeOwnerEvidenceRPC(t, reply)
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(reply.Result, &fields); err != nil {
			t.Fatal(err)
		}
		want := []string{"managerIncarnation", "captureToken", "universe", "stopAuthority", "reservations", "complete", "reasons"}
		if len(fields) != len(want) {
			t.Fatalf("response fields: %v", fields)
		}
		for _, field := range want {
			if _, exists := fields[field]; !exists {
				t.Fatalf("missing %s", field)
			}
		}
		entry := (*got.Universe.Value)[0]
		if got.Complete || !entry.Registered || !entry.Supported.Known || entry.Installed.Known || entry.Running.Known || entry.TerminationVerified.Known || entry.RootsComplete {
			t.Fatalf("unobserved engine gained knowledge: %+v", got)
		}
	})

	t.Run("owned lifetime is bounded and external authority refuses", func(t *testing.T) {
		m, out := ownerEvidenceRPCFixture(t)
		st := m.exec.engines["fake"]
		st.proc = &managedProc{cmd: &exec.Cmd{Process: &os.Process{Pid: 4321}}, done: make(chan struct{})}
		st.runID, st.running, st.installed = "run-owned", true, true
		st.observeOwnerInstalledLocked()
		st.observeOwnerPresenceLocked(false)
		owned := decodeOwnerEvidenceRPC(t, callOwnerEvidenceRPC(t, m, out, `{}`))
		stop := (*owned.StopAuthority.Value)[0]
		requireOwnerBool(t, stop.Scoped, true)
		if !stop.ProcessLifetime.Known || stop.ProcessLifetime.Value.EngineRunID != "run-owned" || !stop.ProcessLifetime.Value.PID.Known || *stop.ProcessLifetime.Value.PID.Value != 4321 {
			t.Fatalf("lost owner lifetime: %+v", stop)
		}
		st.proc, st.adopted = nil, true
		st.observeOwnerPresenceLocked(false)
		external := decodeOwnerEvidenceRPC(t, callOwnerEvidenceRPC(t, m, out, `{}`))
		stop = (*external.StopAuthority.Value)[0]
		requireOwnerBool(t, stop.Scoped, false)
		if stop.ProcessLifetime.Known || stop.Reason != "external_management" || owned.CaptureToken == external.CaptureToken {
			t.Fatalf("external engine gained owned authority: %+v", stop)
		}
	})

	t.Run("download source mismatch withdraws reservation evidence", func(t *testing.T) {
		m, out := ownerEvidenceRPCFixture(t)
		m.copyJournal.source = "different-manager"
		got := decodeOwnerEvidenceRPC(t, callOwnerEvidenceRPC(t, m, out, `{}`))
		if got.ManagerIncarnation != m.exec.managerIncarnation || got.Reservations.Known || got.Complete {
			t.Fatalf("source mismatch trusted: %+v", got)
		}
	})

	t.Run("content and admission revisions fence otherwise equal captures", func(t *testing.T) {
		m, out := ownerEvidenceRPCFixture(t)
		first := decodeOwnerEvidenceRPC(t, callOwnerEvidenceRPC(t, m, out, `{}`))
		m.copyJournal.authorityGeneration++
		content := decodeOwnerEvidenceRPC(t, callOwnerEvidenceRPC(t, m, out, `{}`))
		m.exec.ownerAdmissionGeneration += 2
		admission := decodeOwnerEvidenceRPC(t, callOwnerEvidenceRPC(t, m, out, `{}`))
		if first.CaptureToken == content.CaptureToken || content.CaptureToken == admission.CaptureToken || first.Reservations.Value.Revision == content.Reservations.Value.Revision || content.Reservations.Value.Revision == admission.Reservations.Value.Revision {
			t.Fatal("authority generation changes were omitted from capture identity")
		}
	})

	t.Run("contended owner withdraws facts without waiting", func(t *testing.T) {
		m, out := ownerEvidenceRPCFixture(t)
		m.copyMu.Lock()
		defer m.copyMu.Unlock()
		got := decodeOwnerEvidenceRPC(t, callOwnerEvidenceRPC(t, m, out, `{}`))
		if got.Universe.Known || got.StopAuthority.Known || got.Reservations.Known || got.Complete {
			t.Fatalf("contended capture fabricated facts: %+v", got)
		}
	})

	t.Run("closed owned lifetime withdraws authority", func(t *testing.T) {
		m, out := ownerEvidenceRPCFixture(t)
		st := m.exec.engines["fake"]
		st.proc = &managedProc{cmd: &exec.Cmd{Process: &os.Process{Pid: 4321}}, done: make(chan struct{})}
		st.runID, st.running = "owned", true
		st.observeOwnerPresenceLocked(false)
		close(st.proc.done)
		got := decodeOwnerEvidenceRPC(t, callOwnerEvidenceRPC(t, m, out, `{}`))
		stop := (*got.StopAuthority.Value)[0]
		if stop.Scoped.Known || stop.ProcessLifetime.Known || stop.Reason != "process_lifetime_closed" {
			t.Fatalf("closed lifetime retained authority: %+v", stop)
		}
	})

	t.Run("retained download origin stays explicitly unknown", func(t *testing.T) {
		m, out := ownerEvidenceRPCFixture(t)
		key := downloadContentKey{engine: "fake", model: "private-model"}
		m.copyJournal.unsafe = map[downloadContentKey]modelDownloadMarker{key: {ID: "prior-op", Engine: "fake", Model: "private-model"}}
		got := decodeOwnerEvidenceRPC(t, callOwnerEvidenceRPC(t, m, out, `{}`))
		if !got.Reservations.Known || got.Reservations.Value.UnsafeCopiesComplete || len(got.Reservations.Value.Operations) != 1 {
			t.Fatalf("retained operation missing: %+v", got)
		}
		op := got.Reservations.Value.Operations[0]
		if !op.Retained || !op.Unsafe || op.SourceIncarnation.Known {
			t.Fatalf("invented predecessor origin: %+v", op)
		}
		requireOwnerBool(t, got.Reservations.Value.BlocksApply, true)
	})

	unknown := func(name string, mutate func(*Manager), section string) {
		t.Run(name, func(t *testing.T) {
			m, out := ownerEvidenceRPCFixture(t)
			mutate(m)
			got := decodeOwnerEvidenceRPC(t, callOwnerEvidenceRPC(t, m, out, `{}`))
			if got.Complete || got.ManagerIncarnation != m.exec.managerIncarnation {
				t.Fatalf("unknown state became complete: %+v", got)
			}
			found := false
			for _, reason := range got.Reasons {
				if reason.Section == section {
					found = true
				}
			}
			if !found {
				t.Fatalf("unknown section %s lacks bounded reason: %+v", section, got)
			}
		})
	}
	unknown("unavailable desired owner", func(m *Manager) { m.exec.desired = &desiredStateStore{} }, "universe")
	unknown("historical copy baseline", func(m *Manager) { m.copyJournal.copiesComplete = false }, "reservations")
	unknown("download recovery pending", func(m *Manager) { m.copyJournal.recoveryRequired = true }, "reservations")
	unknown("exhausted content authority", func(m *Manager) { m.copyJournal.authorityUncertain = true }, "reservations")
	unknown("admitted mutation", func(m *Manager) { m.exec.ownerAdmissionHeld = 1 }, "stopAuthority")
	unknown("unrepresented data roots", func(*Manager) {}, "universe")

	testRequest := func(name, params string) {
		t.Run(name, func(t *testing.T) {
			m, out := ownerEvidenceRPCFixture(t)
			reply := callOwnerEvidenceRPC(t, m, out, params)
			if reply.Error == nil || reply.Error.Code != -32602 || len(reply.Result) != 0 {
				t.Fatalf("accepted invalid request: %+v", reply)
			}
		})
	}
	testRequest("reject null", `null`)
	testRequest("reject omitted params", ``)
	testRequest("reject engine", `{"engine":"fake"}`)
	testRequest("reject node", `{"nodeId":"local"}`)
	testRequest("reject caller", `{"caller":"parent"}`)
	testRequest("reject operation", `{"operationId":"op"}`)
	testRequest("reject mutation", `{"stop":true}`)
	testRequest("reject case variant", `{"Engine":"fake"}`)
	testRequest("reject duplicate", `{"engine":"fake","engine":"fake"}`)
	testRequest("reject request overflow", `{}`+strings.Repeat(" ", buildapply.MaxRequestBytes))

	t.Run("response overflow is unavailable without partial authority", func(t *testing.T) {
		m, out := ownerEvidenceRPCFixture(t)
		for i := 0; i < buildapply.MaxStatusBytes/64; i++ {
			name := fmt.Sprintf("retained-%06d", i)
			m.exec.reg.engines[name] = &Manifest{}
		}
		reply := callOwnerEvidenceRPC(t, m, out, `{}`)
		if reply.Error == nil || reply.Error.Code != -32000 || len(reply.Result) != 0 {
			t.Fatalf("overflow emitted partial evidence: %+v", reply)
		}
	})

	t.Run("native restart replaces identity even at idle", func(t *testing.T) {
		m, out := ownerEvidenceRPCFixture(t)
		first := decodeOwnerEvidenceRPC(t, callOwnerEvidenceRPC(t, m, out, `{}`))
		m.exec.managerIncarnation, m.copyJournal.source = "successor-manager", "successor-manager"
		second := decodeOwnerEvidenceRPC(t, callOwnerEvidenceRPC(t, m, out, `{}`))
		if second.ManagerIncarnation == first.ManagerIncarnation || second.CaptureToken == first.CaptureToken {
			t.Fatal("successor retained predecessor identity")
		}
	})
}

func TestOwnerEvidenceUniverse(t *testing.T) {
	t.Run("full union retains unsupported saved and reservation names", func(t *testing.T) {
		e, st, saved := ownerEvidenceFixture(t)
		e.reg.engines["unsupported"] = &Manifest{}
		e.reg.engines["cold"] = e.reg.engines["fake"]
		e.stopOnly["removed-host"] = &engineState{plat: &Platform{}}
		saved.State.ProcessStops = map[string]processStopEvidence{"unsupported": {PID: 12}, "unregistered-cleanup": {PID: 13}}
		saved.State.AllocatedPorts = map[string]int{"port-zero": 0}
		e.downloads = &modelDownloads{active: map[downloadContentKey]*modelDownloadAttempt{{engine: "download"}: {}}, unsafe: map[downloadContentKey]modelDownloadMarker{{engine: "unsafe"}: {}}, writers: map[string]bool{"writer": true}, readers: map[string]int{"reader": 1}}
		got := e.captureOwnerUniverse(func() (desiredOwnerEvidence, error) { return saved, nil }, func() ([]string, bool) { return []string{"copy"}, true })
		want := []string{"cold", "copy", "download", "fake", "port-zero", "reader", "removed-host", "unregistered-cleanup", "unsafe", "unsupported", "writer"}
		var names []string
		for _, entry := range got.Entries {
			names = append(names, entry.Name)
			requireOwnerUnknown(t, entry.Running)
			requireOwnerUnknown(t, entry.PositiveStop)
			requireOwnerUnknown(t, entry.StopAuthority)
			if entry.RootsComplete || entry.RootsReason != "unmodeled_data_roots" {
				t.Fatalf("invented root coverage: %+v", entry)
			}
		}
		if !got.Complete || !reflect.DeepEqual(names, want) {
			t.Fatalf("universe = %+v; names want %v", got, want)
		}
		unsupported := ownerFixtureEntry(t, got, "unsupported")
		if !unsupported.Registered || unsupported.Supported || !unsupported.Cleanup {
			t.Fatalf("unsupported cleanup lost: %+v", unsupported)
		}
		if entry := ownerFixtureEntry(t, got, "unregistered-cleanup"); entry.Registered || !entry.Cleanup {
			t.Fatalf("retained cleanup lost: %+v", entry)
		}
		if len(e.engines) != 1 || st.logs != nil || st.contentMu != nil || st.relaunch != nil || st.healthStop != nil {
			t.Fatal("evidence initialized runtime owners")
		}
		if !reflect.DeepEqual(e.inventory, map[string][]string{"fake": {"must-not-be-read"}}) {
			t.Fatal("evidence touched model inventory")
		}
	})

	t.Run("installed and stopped publications are detached", func(t *testing.T) {
		e, st, saved := ownerEvidenceFixture(t)
		st.installed = true
		st.observeOwnerInstalledLocked()
		st.observeOwnerPresenceLocked(true)
		got := readOwnerFixture(e, saved)
		entry := ownerFixtureEntry(t, got, "fake")
		requireOwnerBool(t, entry.Installed, true)
		requireOwnerBool(t, entry.Running, false)
		requireOwnerBool(t, entry.PositiveStop, true)
		requireOwnerBool(t, entry.StopAuthority, true)
		if entry.AuthorityReason != "no_live_process" || entry.InstallRoot != st.installDir || entry.ModelRoot != st.modelsDir {
			t.Fatalf("stopped roots/authority = %+v", entry)
		}
		*entry.Installed.Value = false
		got.Entries[0].ModelRoot = "changed"
		again := ownerFixtureEntry(t, readOwnerFixture(e, saved), "fake")
		requireOwnerBool(t, again.Installed, true)
		if again.ModelRoot != st.modelsDir {
			t.Fatal("response aliases roots")
		}
	})

	t.Run("configuration change invalidates old observations", func(t *testing.T) {
		e, st, saved := ownerEvidenceFixture(t)
		st.installed = true
		st.observeOwnerInstalledLocked()
		st.observeOwnerPresenceLocked(true)
		st.modelsDir = filepath.Join(st.installDir, "new-models")
		entry := ownerFixtureEntry(t, readOwnerFixture(e, saved), "fake")
		requireOwnerUnknown(t, entry.Installed)
		requireOwnerUnknown(t, entry.StopAuthority)
		if entry.ModelRoot != st.modelsDir {
			t.Fatal("live configuration roots not captured")
		}
	})

	t.Run("unresolved roots never become known empty coverage", func(t *testing.T) {
		e, st, saved := ownerEvidenceFixture(t)
		st.modelsDir = "${UNRESOLVED}/models"
		entry := ownerFixtureEntry(t, readOwnerFixture(e, saved), "fake")
		if entry.ModelRoot != "" || entry.RootsComplete || entry.RootsReason == "" {
			t.Fatalf("unresolved root = %+v", entry)
		}
	})

	t.Run("owner contention returns without waiting", func(t *testing.T) {
		e, st, saved := ownerEvidenceFixture(t)
		check := func(t *testing.T) {
			t.Helper()
			if got := readOwnerFixture(e, saved); got.Complete || got.Reason != "busy" {
				t.Fatalf("contended capture = %+v", got)
			}
		}
		e.mu.Lock()
		check(t)
		e.mu.Unlock()
		st.mu.Lock()
		check(t)
		st.mu.Unlock()
		e.mutationAdmission.Lock()
		check(t)
		e.mutationAdmission.Unlock()
		e.downloads = &modelDownloads{}
		e.downloads.mu.Lock()
		check(t)
		e.downloads.mu.Unlock()
		e.desired = &desiredStateStore{}
		e.desired.mu.Lock()
		if _, err := e.desired.tryOwnerEvidence(); err == nil {
			t.Fatal("contended desired state reported known")
		}
		e.desired.mu.Unlock()
		manager := &Manager{exec: e}
		manager.copyMu.Lock()
		if _, ok := manager.ownerCopyNames(); ok {
			t.Fatal("contended copy owner reported known")
		}
		manager.copyMu.Unlock()
	})

	t.Run("copy owner contributes names without reading progress or native state", func(t *testing.T) {
		e, _, _ := ownerEvidenceFixture(t)
		manager := &Manager{exec: e, copies: map[string]*activeCopy{"one": {copyRequest: copyRequest{Engine: "retained-copy"}}}}
		names, ok := manager.ownerCopyNames()
		if !ok || !reflect.DeepEqual(names, []string{"retained-copy"}) {
			t.Fatalf("copy names = %v, %v", names, ok)
		}
		names[0] = "changed"
		if manager.copies["one"].Engine != "retained-copy" {
			t.Fatal("copy projection aliases owner")
		}
	})

	t.Run("unavailable saved owner does not claim a complete union", func(t *testing.T) {
		e, _, _ := ownerEvidenceFixture(t)
		got := e.captureOwnerUniverse(func() (desiredOwnerEvidence, error) { return desiredOwnerEvidence{}, errors.New("unavailable") }, func() ([]string, bool) { return nil, true })
		if got.Complete || got.Reason != "saved_state_unknown" || got.Incarnation != e.managerIncarnation {
			t.Fatalf("saved failure = %+v", got)
		}
	})

	t.Run("changes between captures invalidate the section", func(t *testing.T) {
		check := func(name string, change func(*Executor, *engineState, *desiredOwnerEvidence)) {
			t.Run(name, func(t *testing.T) {
				e, st, saved := ownerEvidenceFixture(t)
				reads := 0
				got := e.captureOwnerUniverse(func() (desiredOwnerEvidence, error) {
					reads++
					if reads == 2 {
						change(e, st, &saved)
					}
					return saved, nil
				}, func() ([]string, bool) { return nil, true })
				if got.Complete || got.Reason != "busy" {
					t.Fatalf("changed owner accepted: %+v", got)
				}
			})
		}
		check("roots", func(_ *Executor, st *engineState, _ *desiredOwnerEvidence) { st.modelsDir += "-replacement" })
		check("state membership", func(e *Executor, _ *engineState, _ *desiredOwnerEvidence) {
			e.engines["late"] = &engineState{plat: &Platform{}}
		})
		check("run", func(_ *Executor, st *engineState, _ *desiredOwnerEvidence) { st.runID = "next-run" })
		check("cached state", func(_ *Executor, st *engineState, _ *desiredOwnerEvidence) { st.installed = true })
		check("saved cleanup", func(_ *Executor, _ *engineState, saved *desiredOwnerEvidence) {
			saved.State = desiredStateFile{AllocatedPorts: map[string]int{"late": 12345}}
		})
	})
}

func TestOwnerEvidenceStopAuthority(t *testing.T) {
	owned := func(t *testing.T) (*Executor, *engineState, desiredOwnerEvidence) {
		t.Helper()
		e, st, saved := ownerEvidenceFixture(t)
		st.proc = &managedProc{cmd: &exec.Cmd{Process: &os.Process{Pid: 4321}}, done: make(chan struct{})}
		st.runID, st.running, st.installed = "run-one", true, true
		st.observeOwnerInstalledLocked()
		st.observeOwnerPresenceLocked(false)
		return e, st, saved
	}
	t.Run("owned process retains scoped authority without OS proof", func(t *testing.T) {
		e, _, saved := owned(t)
		entry := ownerFixtureEntry(t, readOwnerFixture(e, saved), "fake")
		requireOwnerBool(t, entry.StopAuthority, true)
		requireOwnerBool(t, entry.Running, true)
		requireOwnerBool(t, entry.PositiveStop, false)
		if entry.AuthorityReason != "owned_process_lifetime" {
			t.Fatalf("authority = %+v", entry)
		}
	})
	unknown := func(name, reason string, change func(*engineState)) {
		t.Run(name, func(t *testing.T) {
			e, st, saved := owned(t)
			change(st)
			entry := ownerFixtureEntry(t, readOwnerFixture(e, saved), "fake")
			requireOwnerUnknown(t, entry.StopAuthority)
			if entry.AuthorityReason != reason {
				t.Fatalf("authority = %+v; want %s", entry, reason)
			}
		})
	}
	unknown("closed process lifetime", "process_lifetime_closed", func(st *engineState) { close(st.proc.done) })
	unknown("process without lifetime", "process_lifetime_closed", func(st *engineState) { st.proc.done = nil })
	unknown("same PID replacement process", "unobserved", func(st *engineState) {
		st.proc = &managedProc{cmd: &exec.Cmd{Process: &os.Process{Pid: 4321}}, done: make(chan struct{})}
	})
	unknown("same PID replacement run", "unobserved", func(st *engineState) { st.runID = "run-two" })
	unknown("empty run", "unobserved", func(st *engineState) { st.runID = "" })
	unknown("reclaim failure", "reclaim_unknown", func(st *engineState) { st.reclaimFailed = true })
	unknown("PID only locator", "reclaim_unknown", func(st *engineState) { st.reclaimPID = 4321 })
	unknown("running without scoped lifetime", "scoped_lifetime_unknown", func(st *engineState) {
		st.proc = nil
		st.observeOwnerPresenceLocked(false)
	})
	unknown("endpoint stopped without physical process proof", "scoped_lifetime_unknown", func(st *engineState) {
		st.proc, st.running, st.runID = nil, false, ""
		st.observeOwnerPresenceLocked(false)
	})
	t.Run("adopted engine is known false even with Stop configured", func(t *testing.T) {
		e, st, saved := ownerEvidenceFixture(t)
		st.plat.Runtime.Stop = &StopSpec{Signal: "term"}
		st.running, st.adopted, st.runID = true, true, "external-run"
		st.observeOwnerPresenceLocked(false)
		entry := ownerFixtureEntry(t, readOwnerFixture(e, saved), "fake")
		requireOwnerBool(t, entry.StopAuthority, false)
		if entry.AuthorityReason != "external_management" {
			t.Fatalf("adopted authority = %+v", entry)
		}
	})
	t.Run("default false is never positive stopped evidence", func(t *testing.T) {
		e, _, saved := ownerEvidenceFixture(t)
		entry := ownerFixtureEntry(t, readOwnerFixture(e, saved), "fake")
		requireOwnerUnknown(t, entry.Running)
		requireOwnerUnknown(t, entry.PositiveStop)
		requireOwnerUnknown(t, entry.StopAuthority)
	})
	t.Run("retirement does not republish withdrawn observations", func(t *testing.T) {
		e, st, saved := owned(t)
		done, err := e.admitMutation("fake")
		if err != nil {
			t.Fatalf("mutation admission: %v", err)
		}
		done()
		entry := ownerFixtureEntry(t, readOwnerFixture(e, saved), "fake")
		requireOwnerUnknown(t, entry.Installed)
		requireOwnerUnknown(t, entry.Running)
		requireOwnerUnknown(t, entry.PositiveStop)
		requireOwnerUnknown(t, entry.StopAuthority)
		if !st.running || st.proc == nil || st.runID != "run-one" {
			t.Fatal("withdrawal changed lifecycle state")
		}
	})

	// Hold the actual admission owner while opMu and cached state stay unchanged.
	// Channels delimit the interval; evidence cannot rely on a lucky scheduling
	// delay or on seeing a native operation's intermediate state.
	admitted := func(name string, enter func(*Executor) (func(), error)) {
		t.Run(name, func(t *testing.T) {
			e, st, saved := owned(t)
			entered, release, finished := make(chan struct{}), make(chan struct{}), make(chan struct{})
			var entryErr error
			go func() {
				defer close(finished)
				done, err := enter(e)
				entryErr = err
				close(entered)
				if err != nil {
					return
				}
				<-release
				done()
			}()
			<-entered
			defer func() { close(release); <-finished }()
			if entryErr != nil {
				t.Fatalf("admission: %v", entryErr)
			}
			// Republish the exact old cache while still admitted. It stays busy.
			st.mu.Lock()
			st.observeOwnerInstalledLocked()
			st.observeOwnerPresenceLocked(false)
			st.mu.Unlock()
			entry := ownerFixtureEntry(t, readOwnerFixture(e, saved), "fake")
			requireOwnerUnknown(t, entry.Installed)
			requireOwnerUnknown(t, entry.Running)
			requireOwnerUnknown(t, entry.StopAuthority)
			if entry.AuthorityReason != "busy" || !st.running || st.runID != "run-one" {
				t.Fatalf("admitted interval = %+v", entry)
			}
		})
	}
	track := func(e *Executor) (func(), error) { return e.admitMutation("fake") }
	admitted("install admitted before native work", track)
	admitted("start waiting for lifecycle lock", track)
	admitted("ordinary stop admission", func(e *Executor) (func(), error) { return e.admitStop("fake") })
	admitted("global mutation", func(e *Executor) (func(), error) {
		e.mutationAdmission.Lock()
		defer e.mutationAdmission.Unlock()
		return e.trackMutationLocked("")
	})
	admitted("observation work", func(e *Executor) (func(), error) { return e.admitOwnerCheckpoint() })

	t.Run("admission enter exit ABA invalidates otherwise identical captures", func(t *testing.T) {
		e, _, saved := owned(t)
		reads := 0
		got := e.captureOwnerUniverse(func() (desiredOwnerEvidence, error) {
			reads++
			if reads == 2 {
				done, err := e.admitOwnerCheckpoint()
				if err != nil {
					t.Fatalf("observation admission: %v", err)
				}
				done()
			}
			return saved, nil
		}, func() ([]string, bool) { return nil, true })
		if got.Complete || e.ownerAdmissionGeneration != 2 || e.ownerAdmissionHeld != 0 {
			t.Fatalf("ABA accepted: %+v, generation=%d held=%d", got, e.ownerAdmissionGeneration, e.ownerAdmissionHeld)
		}
	})
	t.Run("admission reserves retirement without wrap or duplicate retirement", func(t *testing.T) {
		e, _, _ := owned(t)
		e.ownerAdmissionGeneration = math.MaxUint64 - 3
		first, err := e.admitOwnerCheckpoint()
		if err != nil {
			t.Fatalf("first admission: %v", err)
		}
		if done, err := e.admitOwnerCheckpoint(); err == nil || done != nil {
			t.Fatal("accepted work without retirement capacity")
		}
		first()
		first()
		if e.ownerAdmissionGeneration != math.MaxUint64-1 || e.ownerAdmissionHeld != 0 {
			t.Fatalf("retirement = %d, held=%d", e.ownerAdmissionGeneration, e.ownerAdmissionHeld)
		}
		if done, err := e.admitStop("fake"); err == nil || done != nil || e.activeMutations["fake"] != 0 {
			t.Fatal("stop admitted effects without retirement capacity")
		}
	})
}
