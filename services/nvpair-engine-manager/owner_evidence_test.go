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
	"time"

	"nvpair-shared/buildapply"
	"nvpair-shared/durablefile"
	"nvpair-shared/engines"
	settings "nvpair-shared/enginesettings"
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
		m := &Manager{exec: &Executor{switchIncarnation: d.source}, copyJournal: d, copyRegistry: reg}
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
		fresh := &Manager{exec: &Executor{switchIncarnation: newID}, copyJournal: successor}
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
	saved := desiredOwnerEvidence{State: desiredStateFile{Engines: map[string]bool{"retained": false}, AllocatedPorts: map[string]int{"port": 0}}}
	got, ok := e.ownerUniverseMembership(saved)
	want := []ownerUniverseEntry{{Name: "cleanup", Cleanup: true}, {Name: "port", Cleanup: true}, {Name: "registered", Registered: true}, {Name: "retained"}}
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
	e := &Executor{switchIncarnation: "manager-fixture", reg: &Registry{engines: map[string]*Manifest{"fake": manifest}}, engines: map[string]*engineState{"fake": st}, stopOnly: map[string]*engineState{},
		desired:   newDesiredStateStore(t.TempDir()),
		processes: func(context.Context) ([]string, error) { panic("evidence invoked process lookup") },
		client:    &http.Client{Transport: ownerEvidenceRejectHTTP{}}, emit: func(string, any) { panic("evidence notified") },
		inventory: map[string][]string{"fake": {"must-not-be-read"}},
	}
	return e, st, desiredOwnerEvidence{State: desiredStateFile{Engines: map[string]bool{}, OffBy: map[string]intentOrigin{}, Locked: map[string]bool{}}}
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
	d := &modelDownloads{source: e.switchIncarnation, copiesComplete: true}
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
		if first.ManagerIncarnation != m.exec.switchIncarnation || first.CaptureToken == "" || !first.Complete || !reflect.DeepEqual(first, second) {
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
		want := []string{"managerIncarnation", "captureToken", "intent", "locks", "universe", "stopAuthority", "reservations", "complete", "reasons"}
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

	t.Run("saved key presence and effective local Off survive", func(t *testing.T) {
		m, out := ownerEvidenceRPCFixture(t)
		document := []byte(`{"engines":{"retained":false},"locked":{"retained":false}}`)
		if err := os.WriteFile(m.exec.desired.path, document, 0o600); err != nil {
			t.Fatal(err)
		}
		got := decodeOwnerEvidenceRPC(t, callOwnerEvidenceRPC(t, m, out, `{}`))
		if !got.Intent.Known || !reflect.DeepEqual(got.Intent.Value.EnabledByEngine, map[string]bool{"retained": false}) || !reflect.DeepEqual(got.Intent.Value.OffByEngine, map[string]string{"retained": "local"}) || !reflect.DeepEqual(got.Locks.Value.LockedByEngine, map[string]bool{"retained": false}) {
			t.Fatalf("saved key semantics changed: %+v", got)
		}
		entries := *got.Universe.Value
		if len(entries) != 2 || entries[1].Engine != "retained" || entries[1].Registered || entries[1].Supported.Known {
			t.Fatalf("retained key lost: %+v", entries)
		}
		after, err := os.ReadFile(m.exec.desired.path)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(document, after) {
			t.Fatal("effective provenance was persisted by the read")
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
		if got.ManagerIncarnation != m.exec.switchIncarnation || got.Reservations.Known || got.Complete {
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
		if got.Universe.Known || got.StopAuthority.Known || got.Reservations.Known || got.Complete || !got.Intent.Known || !got.Locks.Known {
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
			if got.Complete || got.ManagerIncarnation != m.exec.switchIncarnation {
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
	unknown("unavailable desired owner", func(m *Manager) { m.exec.desired = &desiredStateStore{} }, "intent")
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
		m.exec.switchIncarnation, m.copyJournal.source = "successor-manager", "successor-manager"
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
		saved.State.Engines["saved-off"] = false
		saved.State.OffBy["origin-only"] = intentPeer
		saved.State.Locked["saved-unlocked"] = false
		saved.State.ProcessStops = map[string]processStopEvidence{"unsupported": {PID: 12}, "unregistered-cleanup": {PID: 13}}
		saved.State.AllocatedPorts = map[string]int{"port-zero": 0}
		saved.State.PendingSwitch = &settings.SwitchCheckpoint{SwitchPrepareRequest: settings.SwitchPrepareRequest{SourceEngine: "checkpoint-source", TargetEngine: "checkpoint-target"}, Phase: settings.SwitchCheckpointReleased}
		e.downloads = &modelDownloads{active: map[downloadContentKey]*modelDownloadAttempt{{engine: "download"}: {}}, unsafe: map[downloadContentKey]modelDownloadMarker{{engine: "unsafe"}: {}}, writers: map[string]bool{"writer": true}, readers: map[string]int{"reader": 1}}
		got := e.captureOwnerUniverse(func() (desiredOwnerEvidence, error) { return saved, nil }, func() ([]string, bool) { return []string{"copy"}, true })
		want := []string{"checkpoint-source", "checkpoint-target", "cold", "copy", "download", "fake", "origin-only", "port-zero", "reader", "removed-host", "saved-off", "saved-unlocked", "unregistered-cleanup", "unsafe", "unsupported", "writer"}
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
		e.switchAdmission.Lock()
		check(t)
		e.switchAdmission.Unlock()
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
		if got.Complete || got.Reason != "saved_state_unknown" || got.Incarnation != e.switchIncarnation {
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
		check("saved intent", func(_ *Executor, _ *engineState, saved *desiredOwnerEvidence) {
			saved.State = desiredStateFile{Engines: map[string]bool{"late": true}}
		})
		check("checkpoint", func(_ *Executor, _ *engineState, saved *desiredOwnerEvidence) {
			saved.State.PendingSwitch = &settings.SwitchCheckpoint{Phase: settings.SwitchCheckpointReserved}
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
	unknown("command without scoped lifetime", "scoped_lifetime_unknown", func(st *engineState) {
		st.proc = nil
		st.plat.Runtime.Mode = "command"
		st.observeOwnerPresenceLocked(false)
	})
	unknown("command endpoint stopped without physical process proof", "scoped_lifetime_unknown", func(st *engineState) {
		st.proc, st.running, st.runID = nil, false, ""
		st.plat.Runtime.Mode = "command"
		st.observeOwnerPresenceLocked(false)
	})
	t.Run("adopted command is known false even with Stop configured", func(t *testing.T) {
		e, st, saved := ownerEvidenceFixture(t)
		st.plat.Runtime.Mode, st.plat.Runtime.Stop = "command", &StopSpec{Cmd: []string{"must-not-run"}}
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
	admitted("ordinary stop admission", func(e *Executor) (func(), error) { return e.admitStop("fake", intentPeer, "") })
	admitted("global mutation", func(e *Executor) (func(), error) {
		e.switchAdmission.Lock()
		defer e.switchAdmission.Unlock()
		return e.trackMutationLocked("")
	})
	admitted("cold activation", func(e *Executor) (func(), error) {
		// An absent saved document is the normal cold admission baseline; no
		// directory/file is created by its existing read owner.
		return e.admitBoundEngineMutation("fake", "cold-operation")
	})
	admitted("checkpoint work", func(e *Executor) (func(), error) { return e.admitOwnerCheckpoint() })

	t.Run("unreleased checkpoint blocks unchanged lifecycle cache", func(t *testing.T) {
		e, _, saved := owned(t)
		saved.State.PendingSwitch = &settings.SwitchCheckpoint{SwitchPrepareRequest: settings.SwitchPrepareRequest{SourceEngine: "fake"}, Phase: settings.SwitchCheckpointReserved}
		entry := ownerFixtureEntry(t, readOwnerFixture(e, saved), "fake")
		requireOwnerUnknown(t, entry.StopAuthority)
		if entry.AuthorityReason != "busy" {
			t.Fatalf("checkpoint authority = %+v", entry)
		}
	})
	t.Run("admission enter exit ABA invalidates otherwise identical captures", func(t *testing.T) {
		e, _, saved := owned(t)
		reads := 0
		got := e.captureOwnerUniverse(func() (desiredOwnerEvidence, error) {
			reads++
			if reads == 2 {
				done, err := e.admitOwnerCheckpoint()
				if err != nil {
					t.Fatalf("checkpoint admission: %v", err)
				}
				done()
			}
			return saved, nil
		}, func() ([]string, bool) { return nil, true })
		if got.Complete || e.ownerAdmissionGeneration != 2 || e.ownerAdmissionHeld != 0 {
			t.Fatalf("ABA accepted: %+v, generation=%d held=%d", got, e.ownerAdmissionGeneration, e.ownerAdmissionHeld)
		}
	})
	t.Run("cold activation retires inside the same generation fence", func(t *testing.T) {
		e, _, _ := owned(t)
		done, err := e.admitBoundEngineMutation("fake", "cold-operation")
		if err != nil {
			t.Fatalf("cold admission: %v", err)
		}
		if e.boundActivation == nil || e.ownerAdmissionGeneration != 1 || e.ownerAdmissionHeld != 1 || e.switchMutations[""] != 1 {
			t.Fatal("cold entry lacks its reserved generation")
		}
		done()
		if e.boundActivation != nil || e.ownerAdmissionGeneration != 2 || e.ownerAdmissionHeld != 0 || e.switchMutations[""] != 0 || e.switchMutations["fake"] != 0 {
			t.Fatal("cold retirement escaped generation fence")
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
		if done, err := e.admitBoundEngineMutation("fake", "cold"); err == nil || done != nil || e.boundActivation != nil {
			t.Fatal("cold activation accepted exhausted generation")
		}
		if done, err := e.admitStop("fake", intentLocal, ""); err == nil || done != nil || e.switchMutations["fake"] != 0 {
			t.Fatal("stop admitted effects without retirement capacity")
		}
	})
}

func TestOwnerEvidenceDesiredState(t *testing.T) {
	newStore := func(t *testing.T, document string) *desiredStateStore {
		t.Helper()
		store := newDesiredStateStore(t.TempDir())
		if document != "" {
			if err := os.WriteFile(store.path, []byte(document), 0o600); err != nil {
				t.Fatalf("write desired state fixture: %v", err)
			}
		}
		return store
	}
	read := func(t *testing.T, store *desiredStateStore) desiredOwnerEvidence {
		t.Helper()
		got, err := store.ownerEvidence()
		if err != nil {
			t.Fatalf("read owner evidence: %v", err)
		}
		if got.State.Engines == nil || got.State.OffBy == nil || got.State.Locked == nil {
			t.Fatalf("complete saved state lacks maps: %+v", got.State)
		}
		if len(got.IntentRevision) != len("sha256:")+64 || !strings.HasPrefix(got.IntentRevision, "sha256:") ||
			len(got.LockRevision) != len("sha256:")+64 || !strings.HasPrefix(got.LockRevision, "sha256:") {
			t.Fatalf("missing semantic revisions: %+v", got)
		}
		return got
	}
	encode := func(t *testing.T, state desiredStateFile) string {
		t.Helper()
		data, err := json.Marshal(state)
		if err != nil {
			t.Fatalf("encode desired state fixture: %v", err)
		}
		return string(data)
	}
	unknown := func(t *testing.T, store *desiredStateStore) {
		t.Helper()
		got, err := store.ownerEvidence()
		if err == nil {
			t.Fatalf("invalid or unavailable document reported known: %+v", got)
		}
		if !reflect.DeepEqual(got, desiredOwnerEvidence{}) {
			t.Fatalf("unknown document leaked a partial baseline: %+v", got)
		}
	}
	// Apply to accepted and rejected reads. Count attempted writes, including
	// writes that would otherwise leave no changed bytes.
	unchanged := func(t *testing.T, store *desiredStateStore, action func()) {
		t.Helper()
		before, statErr := os.Stat(store.path)
		if statErr != nil && !os.IsNotExist(statErr) {
			t.Fatalf("stat before read: %v", statErr)
		}
		var original []byte
		if statErr == nil {
			var err error
			original, err = os.ReadFile(store.path)
			if err != nil {
				t.Fatalf("read original bytes: %v", err)
			}
		}
		writes := 0
		useEngineFileReplacer(t, durablefile.Replacer{CreateTemp: func(string, string, os.FileMode) (durablefile.File, error) {
			writes++
			return nil, errors.New("evidence must not write")
		}})
		action()
		if writes != 0 {
			t.Fatalf("evidence attempted %d writes", writes)
		}
		after, err := os.Stat(store.path)
		if os.IsNotExist(statErr) {
			if !os.IsNotExist(err) {
				t.Fatalf("missing file changed: stat error %v", err)
			}
		} else {
			if err != nil {
				t.Fatalf("stat after read: %v", err)
			}
			data, err := os.ReadFile(store.path)
			if err != nil {
				t.Fatalf("read bytes after evidence: %v", err)
			}
			if !bytes.Equal(data, original) || !before.ModTime().Equal(after.ModTime()) || before.Mode() != after.Mode() {
				t.Fatalf("evidence changed saved document: before=%q after=%q", original, data)
			}
		}
		entries, err := os.ReadDir(filepath.Dir(store.path))
		if err != nil {
			t.Fatalf("read owner directory: %v", err)
		}
		for _, entry := range entries {
			if entry.Name() != desiredStateFileName {
				t.Fatalf("evidence created an extra file: %s", entry.Name())
			}
		}
	}

	t.Run("explicit intent and effective Off origins", func(t *testing.T) {
		store := newStore(t, `{
			"engines":{"local-off":false,"peer-off":false,"legacy-off":false,"on":true},
			"offBy":{"local-off":"local","peer-off":"peer"}
		}`)
		unchanged(t, store, func() {
			got := read(t, store)
			wantEngines := map[string]bool{"local-off": false, "peer-off": false, "legacy-off": false, "on": true}
			wantOrigins := map[string]intentOrigin{"local-off": intentLocal, "peer-off": intentPeer, "legacy-off": intentLocal}
			if !reflect.DeepEqual(got.State.Engines, wantEngines) || !reflect.DeepEqual(got.State.OffBy, wantOrigins) {
				t.Fatalf("saved intent = %+v, want engines=%v origins=%v", got.State, wantEngines, wantOrigins)
			}
			if _, present := got.State.Engines["unset"]; present {
				t.Fatal("unset intent became explicit Off")
			}
			if got.State.PendingSwitch != nil || got.State.PendingSwitchPorts != nil {
				t.Fatal("absent checkpoint became present")
			}
		})
	})
	t.Run("retains unregistered intent locks and cleanup keys", func(t *testing.T) {
		stop := processStopEvidence{Bin: filepath.Join(t.TempDir(), "inert-image"), Port: 23456, Address: "127.0.0.1:23456", PID: 123}
		state := desiredStateFile{
			Engines: map[string]bool{"removed-on": true, "removed-locked": false}, OffBy: map[string]intentOrigin{"removed-locked": intentLocal},
			Locked:       map[string]bool{"removed-locked": true, "removed-unlocked": false},
			ProcessStops: map[string]processStopEvidence{"removed-stop": stop}, AllocatedPorts: map[string]int{"removed-port": 23457},
		}
		store := newStore(t, encode(t, state))
		unchanged(t, store, func() {
			got := read(t, store)
			if !reflect.DeepEqual(got.State, state) {
				t.Fatalf("retained document = %+v, want %+v", got.State, state)
			}
			if _, present := got.State.Engines["removed-unlocked"]; present {
				t.Fatal("false lock key fabricated intent")
			}
		})
	})
	t.Run("missing file is complete empty and stays missing", func(t *testing.T) {
		store := newStore(t, "")
		unchanged(t, store, func() {
			got := read(t, store)
			if len(got.State.Engines) != 0 || len(got.State.OffBy) != 0 || len(got.State.Locked) != 0 ||
				len(got.State.ProcessStops) != 0 || len(got.State.AllocatedPorts) != 0 || got.State.PendingSwitch != nil || got.State.PendingSwitchPorts != nil {
				t.Fatalf("missing saved document became nonempty: %+v", got.State)
			}
		})
	})
	t.Run("unresolved base directory is unknown", func(t *testing.T) {
		unknown(t, newDesiredStateStore(""))
	})
	t.Run("unreadable document is unknown", func(t *testing.T) {
		store := newStore(t, "")
		if err := os.Mkdir(store.path, 0o700); err != nil {
			t.Fatalf("create non-file fixture: %v", err)
		}
		unknown(t, store)
		info, err := os.Stat(store.path)
		if err != nil || !info.IsDir() {
			t.Fatalf("read changed non-file fixture: info=%v err=%v", info, err)
		}
	})

	reject := func(name, document string) {
		t.Run(name, func(t *testing.T) {
			store := newStore(t, document)
			unchanged(t, store, func() { unknown(t, store) })
		})
	}
	reject("malformed JSON is unknown without repair", `{"engines":`)
	reject("null document is unknown", `null`)
	reject("non-object document is unknown", `[]`)
	reject("null engines are unknown", `{"engines":null}`)
	reject("non-object engines are unknown", `{"engines":[]}`)
	reject("null intent is unknown", `{"engines":{"fake":null}}`)
	reject("non-boolean intent is unknown", `{"engines":{"fake":"false"}}`)
	reject("unknown origin is unknown", `{"engines":{"fake":false},"offBy":{"fake":"external"}}`)
	reject("null origin is unknown", `{"engines":{"fake":false},"offBy":{"fake":null}}`)
	reject("empty origin is unknown", `{"engines":{"fake":false},"offBy":{"fake":""}}`)
	reject("null origin map is unknown", `{"engines":{"fake":false},"offBy":null}`)
	reject("origin without intent is unknown", `{"engines":{},"offBy":{"removed":"peer"}}`)
	reject("origin for On is unknown", `{"engines":{"fake":true},"offBy":{"fake":"local"}}`)
	reject("null locks are unknown", `{"engines":{},"locked":null}`)
	reject("null lock is unknown", `{"engines":{},"locked":{"fake":null}}`)
	reject("non-boolean lock is unknown", `{"engines":{},"locked":{"fake":1}}`)
	reject("locked On is unknown", `{"engines":{"fake":true},"locked":{"fake":true}}`)
	reject("lock without saved intent is unknown", `{"engines":{},"locked":{"fake":true}}`)
	reject("locked peer Off is inconsistent", `{"engines":{"fake":false},"offBy":{"fake":"peer"},"locked":{"fake":true}}`)
	reject("invalid allocated port is unknown", `{"allocatedPorts":{"removed":65536}}`)
	reject("null allocated port is unknown", `{"allocatedPorts":{"removed":null}}`)
	reject("null cleanup map is unknown", `{"processStops":null}`)
	reject("invalid cleanup record is unknown", `{"processStops":{"removed":{"bin":"relative-image","port":12345,"address":"127.0.0.1:12345","pid":0}}}`)
	reject("orphan checkpoint ports are unknown", `{"pendingSwitchPorts":{"source":0,"target":0}}`)
	reject("invalid checkpoint is unknown", `{"pendingSwitch":{"phase":"reserved"}}`)
	reject("unrecognized saved field is unknown", `{"Engines":{"fake":false}}`)

	compareRevisions := func(name, first, second string, sameIntent, sameLocks bool) {
		t.Run(name, func(t *testing.T) {
			a := read(t, newStore(t, first))
			b := read(t, newStore(t, second))
			if (a.IntentRevision == b.IntentRevision) != sameIntent || (a.LockRevision == b.LockRevision) != sameLocks {
				t.Fatalf("revision equality intent=%v locks=%v, want intent=%v locks=%v; first=%+v second=%+v",
					a.IntentRevision == b.IntentRevision, a.LockRevision == b.LockRevision, sameIntent, sameLocks, a, b)
			}
		})
	}
	compareRevisions("canonical map order and whitespace across owner instances",
		`{"engines":{"a":false,"b":true},"offBy":{"a":"peer"},"locked":{"a":false,"b":false}}`,
		`{ "locked": {"b":false,"a":false}, "offBy":{"a":"peer"}, "engines":{"b":true,"a":false} }`, true, true)
	compareRevisions("legacy and explicit local Off have equal semantics", `{"engines":{"a":false}}`,
		`{"engines":{"a":false},"offBy":{"a":"local"}}`, true, true)
	compareRevisions("missing and explicit empty document have equal semantics", "", `{"engines":{},"offBy":{},"locked":{}}`, true, true)
	compareRevisions("unset and explicit Off have distinct intent revisions", `{}`, `{"engines":{"a":false}}`, false, true)
	compareRevisions("On and Off have distinct intent revisions", `{"engines":{"a":true}}`, `{"engines":{"a":false}}`, false, true)
	compareRevisions("local and peer Off have distinct intent revisions", `{"engines":{"a":false},"offBy":{"a":"local"}}`,
		`{"engines":{"a":false},"offBy":{"a":"peer"}}`, false, true)
	compareRevisions("false lock key is retained in lock revision", `{}`, `{"locked":{"removed":false}}`, true, false)
	compareRevisions("true and false locks have distinct lock revisions", `{"engines":{"a":false},"locked":{"a":false}}`,
		`{"engines":{"a":false},"locked":{"a":true}}`, true, false)
	compareRevisions("execution evidence does not change intent or lock revisions", `{"engines":{"a":false}}`,
		`{"engines":{"a":false},"allocatedPorts":{"removed":23456}}`, true, true)

	lockTransition := func(name, document string) {
		t.Run(name, func(t *testing.T) {
			store := newStore(t, document)
			before := read(t, store)
			changed, err := store.setLocked("fake", true)
			if err != nil || !changed {
				t.Fatalf("save lock and local Off: changed=%v err=%v", changed, err)
			}
			locked := read(t, store)
			enabled, present := locked.State.Engines["fake"]
			if !present || enabled || locked.State.OffBy["fake"] != intentLocal || !locked.State.Locked["fake"] {
				t.Fatalf("lock failed to capture local Off: %+v", locked.State)
			}
			if before.IntentRevision == locked.IntentRevision || before.LockRevision == locked.LockRevision {
				t.Fatal("lock-induced intent and lock revisions did not change")
			}
			changed, err = store.setLocked("fake", false)
			if err != nil || changed {
				t.Fatalf("unlock: changed=%v err=%v", changed, err)
			}
			unlocked := read(t, store)
			if !reflect.DeepEqual(unlocked.State.Engines, locked.State.Engines) || !reflect.DeepEqual(unlocked.State.OffBy, locked.State.OffBy) ||
				unlocked.IntentRevision != locked.IntentRevision || unlocked.LockRevision != before.LockRevision {
				t.Fatalf("unlock changed saved Off or failed to remove lock: locked=%+v unlocked=%+v", locked, unlocked)
			}
			if _, present := unlocked.State.Locked["fake"]; present {
				t.Fatal("unlock retained lock key")
			}
		})
	}
	lockTransition("locking On records local Off and unlock preserves Off", `{"engines":{"fake":true}}`)
	lockTransition("locking unset records local Off and unlock preserves Off", `{}`)
	lockTransition("locking peer Off records local Off and unlock preserves Off", `{"engines":{"fake":false},"offBy":{"fake":"peer"}}`)

	checkpointState := func() desiredStateFile {
		return desiredStateFile{
			Engines: map[string]bool{"source": true, "target": false}, OffBy: map[string]intentOrigin{"target": intentPeer}, Locked: map[string]bool{"target": false},
			PendingSwitch: &settings.SwitchCheckpoint{
				SwitchPrepareRequest: settings.SwitchPrepareRequest{
					OperationID: "owner-evidence", Caller: "fixture", Origin: settings.SwitchOriginLocal,
					SourceEngine: "source", TargetEngine: "target", PayloadHash: "sha256:" + strings.Repeat("a", 64),
					ConfigurationFence: "e3v1:" + strings.Repeat("b", 64), ManagerIncarnation: "predecessor",
					Deadline: time.Unix(1700000000, 0).UTC(),
				},
				Phase:     settings.SwitchCheckpointReserved,
				PreSource: settings.SwitchEngineState{Intent: settings.SwitchIntentOn},
				PreTarget: settings.SwitchEngineState{Intent: settings.SwitchIntentOff}, TargetProvenance: string(intentPeer),
				Source: settings.SwitchEngineState{Intent: settings.SwitchIntentOn}, Target: settings.SwitchEngineState{Intent: settings.SwitchIntentOff},
			},
			PendingSwitchPorts: &switchCheckpointPorts{Source: 23456, Target: 23457, SourcePID: 123, SourceRunID: "old-run"},
		}
	}
	t.Run("checkpoint status preserves unknown runtime observations", func(t *testing.T) {
		state := checkpointState()
		store := newStore(t, encode(t, state))
		unchanged(t, store, func() {
			got := read(t, store)
			if !reflect.DeepEqual(got.State, state) {
				t.Fatalf("checkpoint changed: got=%+v want=%+v", got.State, state)
			}
			for _, observed := range []settings.SwitchEngineState{got.State.PendingSwitch.PreSource, got.State.PendingSwitch.PreTarget, got.State.PendingSwitch.Source, got.State.PendingSwitch.Target} {
				if observed.Running != nil || observed.Healthy != nil || observed.ObservedAt != nil || observed.TerminationVerified {
					t.Fatalf("unknown checkpoint observation fabricated runtime evidence: %+v", observed)
				}
			}
		})
	})
	badCheckpoint := func(name string, change func(*settings.SwitchCheckpoint)) {
		t.Run(name, func(t *testing.T) {
			state := checkpointState()
			change(state.PendingSwitch)
			store := newStore(t, encode(t, state))
			unchanged(t, store, func() { unknown(t, store) })
		})
	}
	badCheckpoint("invalid checkpoint phase is unknown", func(cp *settings.SwitchCheckpoint) { cp.Phase = "invalid" })
	badCheckpoint("invalid checkpoint origin is unknown", func(cp *settings.SwitchCheckpoint) { cp.Origin = "peer" })
	badCheckpoint("invalid checkpoint Off origin is unknown", func(cp *settings.SwitchCheckpoint) { cp.TargetProvenance = "external" })
	badCheckpoint("checkpoint Off origin for On is unknown", func(cp *settings.SwitchCheckpoint) { cp.SourceProvenance = "local" })
	badCheckpoint("invalid checkpoint observation is unknown", func(cp *settings.SwitchCheckpoint) { cp.Source.Intent = "invalid" })
	badCheckpoint("checkpoint cannot fabricate termination from unknown running", func(cp *settings.SwitchCheckpoint) { cp.Source.TerminationVerified = true })
	for _, document := range []string{
		`{"engines":{"hidden":null},"engines":{}}`,
		`{"engines":{"hidden":true,"hidden":false}}`,
	} {
		t.Run("duplicate members refuse "+document, func(t *testing.T) {
			store := newStore(t, document)
			unchanged(t, store, func() { unknown(t, store) })
		})
	}
	t.Run("case variant nested null refuses", func(t *testing.T) {
		state := checkpointState()
		state.ProcessStops = map[string]processStopEvidence{"removed": {Bin: filepath.Join(t.TempDir(), "inert"), Port: 23456, Address: "127.0.0.1:23456", PID: 123}}
		document := strings.Replace(encode(t, state), `"pid":123`, `"PID":null`, 1)
		store := newStore(t, document)
		unchanged(t, store, func() { unknown(t, store) })
	})
	for _, path := range [][]string{
		{"pendingSwitch", "targetCommitted"},
		{"pendingSwitch", "targetProvenance"},
		{"pendingSwitch", "source", "terminationVerified"},
		{"pendingSwitchPorts", "source"},
		{"processStops", "removed", "pid"},
	} {
		t.Run("null nested scalar "+strings.Join(path, "."), func(t *testing.T) {
			state := checkpointState()
			state.ProcessStops = map[string]processStopEvidence{"removed": {Bin: filepath.Join(t.TempDir(), "inert"), Port: 23456, Address: "127.0.0.1:23456", PID: 123}}
			var fields map[string]json.RawMessage
			if err := json.Unmarshal([]byte(encode(t, state)), &fields); err != nil {
				t.Fatal(err)
			}
			var setNull func(map[string]json.RawMessage, []string)
			setNull = func(object map[string]json.RawMessage, keys []string) {
				if len(keys) == 1 {
					object[keys[0]] = json.RawMessage("null")
					return
				}
				var nested map[string]json.RawMessage
				if err := json.Unmarshal(object[keys[0]], &nested); err != nil {
					t.Fatal(err)
				}
				setNull(nested, keys[1:])
				data, err := json.Marshal(nested)
				if err != nil {
					t.Fatal(err)
				}
				object[keys[0]] = data
			}
			setNull(fields, path)
			data, err := json.Marshal(fields)
			if err != nil {
				t.Fatal(err)
			}
			store := newStore(t, string(data))
			unchanged(t, store, func() { unknown(t, store) })
		})
	}
	t.Run("checkpoint endpoint validation covers complete document", func(t *testing.T) {
		state := checkpointState()
		state.PendingSwitchPorts.Target = -1
		store := newStore(t, encode(t, state))
		unchanged(t, store, func() { unknown(t, store) })
	})
	t.Run("checkpoint does not change intent or lock semantic digests", func(t *testing.T) {
		state := checkpointState()
		before := read(t, newStore(t, encode(t, state)))
		state.PendingSwitch = nil
		state.PendingSwitchPorts = nil
		after := read(t, newStore(t, encode(t, state)))
		if before.IntentRevision != after.IntentRevision || before.LockRevision != after.LockRevision {
			t.Fatal("checkpoint changed protected intent or lock semantics")
		}
	})

	t.Run("all returned maps and nested checkpoint pointers are detached", func(t *testing.T) {
		state := checkpointState()
		stopped, healthy := false, false
		observedAt := time.Unix(1700000001, 0).UTC()
		state.PendingSwitch.Source.Running = &stopped
		state.PendingSwitch.Source.Healthy = &healthy
		state.PendingSwitch.Source.ObservedAt = &observedAt
		state.PendingSwitch.Source.TerminationVerified = true
		state.PendingSwitch.Error = &settings.SwitchError{Code: settings.SwitchOperationErrorCode, Kind: settings.SwitchErrorInterrupted, Message: "fixture"}
		stop := processStopEvidence{Bin: filepath.Join(t.TempDir(), "inert-image"), Port: 23456, Address: "127.0.0.1:23456", PID: 123}
		state.ProcessStops = map[string]processStopEvidence{"removed": stop}
		state.AllocatedPorts = map[string]int{"removed": 23456}
		store := newStore(t, encode(t, state))
		unchanged(t, store, func() {
			first, second := read(t, store), read(t, store)
			first.State.Engines["source"] = false
			first.State.OffBy["target"] = intentLocal
			first.State.Locked["target"] = true
			first.State.ProcessStops["removed"] = processStopEvidence{}
			first.State.AllocatedPorts["removed"] = 1
			first.State.PendingSwitch.Phase = settings.SwitchCheckpointReleased
			first.State.PendingSwitchPorts.Source = 1
			*first.State.PendingSwitch.Source.Running = true
			*first.State.PendingSwitch.Source.Healthy = true
			*first.State.PendingSwitch.Source.ObservedAt = time.Time{}
			first.State.PendingSwitch.Error.Message = "mutated"
			third := read(t, store)
			if !reflect.DeepEqual(second.State, state) || !reflect.DeepEqual(third, second) {
				t.Fatalf("returned evidence aliases another capture: second=%+v third=%+v", second, third)
			}
		})
	})

	t.Run("setLocked and evidence capture one joint intent lock baseline", func(t *testing.T) {
		store := newStore(t, `{"engines":{"fake":true}}`)
		before := read(t, store)
		saving := make(chan struct{})
		releaseSave := make(chan struct{})
		writerDone := make(chan struct{})
		var writerChanged bool
		var writerErr error
		useEngineFileReplacer(t, durablefile.Replacer{CreateTemp: func(dir, base string, mode os.FileMode) (durablefile.File, error) {
			close(saving)
			<-releaseSave
			return os.CreateTemp(dir, base+".*.tmp")
		}})
		go func() {
			defer close(writerDone)
			writerChanged, writerErr = store.setLocked("fake", true)
		}()
		select {
		case <-saving: // setLocked changed both fields and still owns desired.mu.
		case <-writerDone:
			t.Fatalf("setLocked did not reach the save boundary: %v", writerErr)
		}
		readerStarted := make(chan struct{})
		readerDone := make(chan struct{})
		captureBoundary := make(chan struct{})
		releaseCapture := make(chan struct{})
		var captured desiredOwnerEvidence
		var captureErr error
		go func() {
			defer close(readerDone)
			close(readerStarted)
			captured, captureErr = store.captureOwnerEvidence(func() ([]byte, error) {
				close(captureBoundary)
				<-releaseCapture
				if store.mu.TryLock() {
					store.mu.Unlock()
					return nil, errors.New("capture read does not hold desired-state owner lock")
				}
				return store.readDocument()
			})
		}()
		<-readerStarted
		var releaseOnce sync.Once
		release := func() { releaseOnce.Do(func() { close(releaseSave) }) }
		// Release and join both owners before cleanup, including on failure.
		defer func() {
			release()
			select {
			case <-releaseCapture:
			default:
				close(releaseCapture)
			}
			<-writerDone
			<-readerDone
		}()
		if store.mu.TryLock() {
			store.mu.Unlock()
			t.Fatal("setLocked released the owner boundary before saving")
		}
		select {
		case <-readerDone:
			t.Fatalf("evidence escaped the held desired-state owner: %+v, %v", captured, captureErr)
		default:
		}
		release()
		<-writerDone
		<-captureBoundary
		close(releaseCapture)
		<-readerDone
		if writerErr != nil || !writerChanged || captureErr != nil {
			t.Fatalf("joint capture: changed=%v write=%v read=%v", writerChanged, writerErr, captureErr)
		}
		enabled, present := captured.State.Engines["fake"]
		if !present || enabled || captured.State.OffBy["fake"] != intentLocal || !captured.State.Locked["fake"] ||
			captured.IntentRevision == before.IntentRevision || captured.LockRevision == before.LockRevision {
			t.Fatalf("capture combined different intent/lock baselines: before=%+v after=%+v", before, captured)
		}
		if after := read(t, store); !reflect.DeepEqual(after, captured) {
			t.Fatalf("joint capture revisions do not describe saved document: captured=%+v after=%+v", captured, after)
		}
	})
}
