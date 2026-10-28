// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"io/fs"
	"net"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"runtime/debug"
	"strings"
	"sync"
	"testing"
	"time"

	"nvpair-shared/buildapply"
	"nvpair-shared/jsonrpc"
)

type buildApplyOwnerFrameWriter struct{ frames chan []byte }

func (w buildApplyOwnerFrameWriter) Write(frame []byte) (int, error) {
	w.frames <- append([]byte(nil), frame...)
	return len(frame), nil
}
func (buildApplyOwnerFrameWriter) Close() error { return nil }

func TestBuildApplyOwnerEvidenceIdleReadUsesCurrentBroker(t *testing.T) {
	profile := t.TempDir()
	for _, key := range []string{"HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "XDG_CONFIG_HOME"} {
		t.Setenv(key, profile)
	}
	s := newService(config{})
	frames := make(chan []byte, 1)
	run := &brokerRun{proc: &brokerProc{stdin: buildApplyOwnerFrameWriter{frames: frames}}, exited: make(chan struct{})}
	s.broker = run
	evidence := buildApplyPreviewEvidence(t)
	envelope := buildapply.ImpactEnvelope{Impact: buildapply.Impact{Fences: buildapply.Fences{ManagerIncarnation: buildapply.Known("manager")}}, ManagerEvidence: buildapply.Known(evidence)}
	done := make(chan struct {
		evidence buildApplyEvidence
		err      error
	}, 1)
	go func() {
		got, err := s.cfg.buildApplyOwners.Read(context.Background())
		done <- struct {
			evidence buildApplyEvidence
			err      error
		}{got, err}
	}()
	var request jsonrpc.Message
	if err := json.Unmarshal(bytes.TrimSpace(<-frames), &request); err != nil {
		t.Fatal(err)
	}
	if request.Method != "build:apply-impact" || request.ID == nil || !bytes.Equal(request.Params, []byte(`{}`)) {
		t.Fatalf("idle owner request = %+v", request)
	}
	result, err := json.Marshal(buildapply.Result[buildapply.ImpactEnvelope]{Available: true, Value: &envelope})
	if err != nil {
		t.Fatal(err)
	}
	s.routeResponse(&jsonrpc.Message{ID: request.ID, Result: result})
	read := <-done
	if read.err != nil {
		t.Fatal(read.err)
	}
	if read.evidence.ManagerIncarnation != evidence.ManagerIncarnation || read.evidence.ManagerCaptureToken != evidence.CaptureToken ||
		!read.evidence.Facts.ManagerIdleRunID || !read.evidence.Facts.ModelDataRoots ||
		!read.evidence.Impact.Fences.ServiceInstance.Known || !read.evidence.Impact.Fences.BrokerGeneration.Known || !read.evidence.Impact.Fences.RuntimeGeneration.Known ||
		read.evidence.Saved || read.evidence.Enforced || read.evidence.PhysicalStop || read.evidence.OwnedWorkers != nil {
		t.Fatalf("idle owner projection = %+v", read.evidence)
	}
	if s.buildPreview != nil || len(s.pending) != 0 {
		t.Fatal("idle owner read retained a preview or pending broker request")
	}
}

func TestBuildApplyOwnerEvidenceRetiredBrokerReplyRefuses(t *testing.T) {
	s := newService(config{})
	frames := make(chan []byte, 1)
	run := &brokerRun{proc: &brokerProc{stdin: buildApplyOwnerFrameWriter{frames: frames}}, exited: make(chan struct{})}
	s.broker = run
	done := make(chan error, 1)
	go func() { _, err := s.readBuildApplyImpact(context.Background(), run); done <- err }()
	var request jsonrpc.Message
	if err := json.Unmarshal(bytes.TrimSpace(<-frames), &request); err != nil {
		t.Fatal(err)
	}
	if request.ID == nil || request.Method != "build:apply-impact" {
		t.Fatalf("retired broker request = %+v", request)
	}
	evidence := buildApplyPreviewEvidence(t)
	envelope := buildapply.ImpactEnvelope{Impact: buildapply.Impact{Fences: buildapply.Fences{ManagerIncarnation: buildapply.Known("manager")}}, ManagerEvidence: buildapply.Known(evidence)}
	result, err := json.Marshal(buildapply.Result[buildapply.ImpactEnvelope]{Available: true, Value: &envelope})
	if err != nil {
		t.Fatal(err)
	}
	s.mu.Lock()
	s.broker = &brokerRun{}
	s.mu.Unlock()
	s.routeResponse(&jsonrpc.Message{ID: request.ID, Result: result})
	if err := <-done; err == nil {
		t.Fatal("retired broker reply supplied current manager evidence")
	}
	if len(s.pending) != 0 {
		t.Fatal("retired broker read retained pending request")
	}
}

func TestBuildApplyPreviewLifetime(t *testing.T) {
	newFixture := func(t *testing.T) (*Service, *client, *brokerRun, time.Time) {
		t.Helper()
		server, peer := net.Pipe()
		t.Cleanup(func() { _ = peer.Close() })
		c := newClient(1, server)
		t.Cleanup(func() { c.closeNow(); <-c.written })
		run := &brokerRun{}
		now := time.Now()
		s := newService(config{})
		s.clients[c] = struct{}{}
		s.broker = run
		s.buildPreview = &buildApplyPreviewState{client: c, broker: run, expires: now.Add(buildapply.PreviewLifetimeSeconds * time.Second)}
		return s, c, run, now
	}
	t.Run("disconnect clears the slot", func(t *testing.T) {
		s, c, _, _ := newFixture(t)
		s.detach(c, false)
		if s.buildPreview != nil {
			t.Fatal("preview survived detach")
		}
	})
	t.Run("changed broker refuses the slot", func(t *testing.T) {
		s, c, _, now := newFixture(t)
		s.broker = nil
		if s.currentBuildApplyPreviewLocked(c, now) != nil || s.buildPreview != nil {
			t.Fatal("preview survived broker change")
		}
	})
	t.Run("actual broker loss clears the slot", func(t *testing.T) {
		s, _, run, _ := newFixture(t)
		s.brokerExited(run)
		if s.buildPreview != nil || s.broker != nil {
			t.Fatal("preview survived brokerExited")
		}
	})
	t.Run("expiry refuses the slot", func(t *testing.T) {
		s, c, _, now := newFixture(t)
		if s.currentBuildApplyPreviewLocked(c, now.Add((buildapply.PreviewLifetimeSeconds+1)*time.Second)) != nil || s.buildPreview != nil {
			t.Fatal("preview survived expiry")
		}
	})
}

// These boundaries return synthetic package/owner evidence; the real bounded
// parser and hostile trees are exercised in shared/buildapply. No fixture ever
// starts a GUI, broker, engine or candidate executable.
func buildApplyPreviewFixture(t *testing.T) (buildapply.PreviewRequest, buildApplyPreviewReads) {
	t.Helper()
	root, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	profile := filepath.Join(root, "private-profile")
	if err := os.MkdirAll(profile, 0700); err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"HOME", "USERPROFILE", "LOCALAPPDATA", "APPDATA", "XDG_CONFIG_HOME"} {
		t.Setenv(key, profile)
	}
	stage := filepath.Join(root, "stage")
	target := filepath.Join(root, "target")
	layout, ok := buildapply.LayoutFor(runtime.GOOS)
	if !ok {
		t.Fatal("unsupported fixture platform")
	}
	if layout.RootName != "" {
		stage = filepath.Join(stage, layout.RootName)
		target = filepath.Join(target, layout.RootName)
	}
	gui := filepath.Join(target, filepath.FromSlash(layout.Executable))
	service := filepath.Join(target, filepath.FromSlash(layout.ServiceDir), layout.ServiceExecutable)
	brokerExecutable := brokerName
	if runtime.GOOS == "windows" {
		brokerExecutable += ".exe"
	}
	broker := filepath.Join(target, filepath.FromSlash(layout.ServiceDir), brokerExecutable)
	for _, path := range []string{gui, service, broker, filepath.Join(stage, "inert")} {
		if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte("inert bytes"), 0700); err != nil {
			t.Fatal(err)
		}
	}
	const stamp = "git:0123456789abcdef0123456789abcdef01234567"
	digest := sha256.Sum256([]byte("inert bytes"))
	hash := hex.EncodeToString(digest[:])
	identity := identityWithBuildID(buildapply.Identity{SHA256: buildapply.Known(hash)}, stamp)
	request := buildapply.PreviewRequest{CandidatePath: stage, GUI: buildapply.GUIFacts{PID: os.Getpid() + 1, Executable: gui, BuildID: stamp}}
	evidence := buildApplyPreviewEvidence(t)
	reads := buildApplyPreviewReads{
		inspect: func(_ context.Context, path, _, _ string) (*buildapply.Package, error) {
			r, err := buildapply.OpenRealRoot(path)
			if err != nil {
				return nil, err
			}
			return &buildapply.Package{Root: r, Layout: layout, Service: identity, GUI: identity, Bytes: 11, SHA256: hash}, nil
		},
		process: func(pid int) (buildapply.ProcessIdentity, error) {
			if pid == request.GUI.PID {
				return buildapply.ProcessIdentity{Executable: gui, Created: "GUI lifetime"}, nil
			}
			return buildapply.ProcessIdentity{Executable: service, Created: "service lifetime"}, nil
		},
		brokerProcess: func() (buildapply.ProcessIdentity, error) {
			return buildapply.ProcessIdentity{Executable: broker, Created: "broker lifetime"}, nil
		},
		executable: func() (string, error) { return service, nil },
		protected:  func() ([]string, error) { return []string{profile}, nil },
		impact: func(context.Context) (buildapply.ImpactEnvelope, error) {
			return buildapply.ImpactEnvelope{Impact: buildapply.Impact{Complete: true, Fences: buildapply.Fences{PolicyRevision: buildapply.Known("7"), ManagerIncarnation: buildapply.Known("manager")}}, ManagerEvidence: buildapply.Known(evidence)}, nil
		},
		revalidate: func(_ context.Context, p *buildapply.Package) error { return p.Root.Revalidate() },
		parent:     func(r *buildapply.RealRoot, _ int64) error { return r.Revalidate() },
		host:       func() buildapply.Observation[string] { return buildapply.Known("00000000-0000-4000-8000-000000000000") },
		running:    func() buildapply.Identity { return identity },
	}
	return request, reads
}

func buildApplyPreviewEvidence(t *testing.T) buildapply.EngineOwnerEvidence {
	t.Helper()
	revision := "sha256:" + strings.Repeat("a", 64)
	e := buildapply.EngineOwnerEvidence{
		ManagerIncarnation: "manager",
		Intent:             buildapply.Known(buildapply.EngineOwnerIntent{Revision: revision, EnabledByEngine: map[string]bool{}, OffByEngine: map[string]string{}}),
		Locks:              buildapply.Known(buildapply.EngineOwnerLocks{Revision: revision, LockedByEngine: map[string]bool{}}),
		Universe:           buildapply.Known([]buildapply.EngineOwnerUniverseEntry{}),
		StopAuthority:      buildapply.Known([]buildapply.EngineOwnerStopAuthority{}),
		Reservations:       buildapply.Known(buildapply.EngineOwnerReservations{Revision: revision, BlocksApply: buildapply.Known(false), UnsafeCopiesComplete: true, Operations: []buildapply.EngineOwnerOperation{}}),
		Reasons:            []buildapply.EngineOwnerReason{},
	}
	if err := e.Finalize(); err != nil {
		t.Fatalf("finalize preview owner fixture: %v", err)
	}
	return e
}

func buildApplyPreviewEvidenceWithRoot(t *testing.T, root string) buildapply.EngineOwnerEvidence {
	t.Helper()
	e := buildApplyPreviewEvidence(t)
	e.Universe = buildapply.Known([]buildapply.EngineOwnerUniverseEntry{{
		Engine: "fake", Registered: true, Supported: buildapply.Known(true), Installed: buildapply.Known(false),
		Running: buildapply.Known(false), TerminationVerified: buildapply.Known(true),
		ModelRoots: buildapply.Known([]string{root}), DataRoots: buildapply.Known([]string{}), RootsComplete: true,
	}})
	e.StopAuthority = buildapply.Known([]buildapply.EngineOwnerStopAuthority{{
		Engine: "fake", Scoped: buildapply.Known(true), Adopted: buildapply.Known(false),
		ReclaimPending: buildapply.Known(false), Reason: "no_live_process",
	}})
	if err := e.Finalize(); err != nil {
		t.Fatalf("finalize configured root: %v", err)
	}
	return e
}

func TestBuildApplyOwnerEvidenceProtectedRootsPrecedeInspection(t *testing.T) {
	test := func(name string, root func(*testing.T, buildapply.PreviewRequest, buildApplyPreviewReads) string) {
		t.Run(name, func(t *testing.T) {
			request, reads := buildApplyPreviewFixture(t)
			configured := root(t, request, reads)
			evidence := buildApplyPreviewEvidenceWithRoot(t, configured)
			reads.impact = func(context.Context) (buildapply.ImpactEnvelope, error) {
				return buildapply.ImpactEnvelope{Impact: buildapply.Impact{Complete: true, Fences: buildapply.Fences{ManagerIncarnation: buildapply.Known("manager")}}, ManagerEvidence: buildapply.Known(evidence)}, nil
			}
			inspectCalls := 0
			inspect := reads.inspect
			reads.inspect = func(ctx context.Context, path, platform, arch string) (*buildapply.Package, error) {
				inspectCalls++
				return inspect(ctx, path, platform, arch)
			}
			_, _, _, _, err := captureBuildApplyPreview(context.Background(), request, reads)
			if err == nil || inspectCalls != 0 {
				t.Fatalf("overlapping %s root inspected package: calls=%d err=%v", name, inspectCalls, err)
			}
		})
	}
	test("candidate in configured model root", func(_ *testing.T, request buildapply.PreviewRequest, _ buildApplyPreviewReads) string {
		return request.CandidatePath
	})
	test("target in configured model root", func(t *testing.T, request buildapply.PreviewRequest, reads buildApplyPreviewReads) string {
		t.Helper()
		service, err := reads.executable()
		if err != nil {
			t.Fatal(err)
		}
		root, err := buildapply.TargetFromExecutables(request.GUI.Executable, service, runtime.GOOS)
		if err != nil {
			t.Fatal(err)
		}
		return root
	})
}

func TestBuildApplyOwnerEvidenceCoverageAndIdentityRefuseBeforeInspection(t *testing.T) {
	test := func(name string, change func(*testing.T, *buildapply.ImpactEnvelope)) {
		t.Run(name, func(t *testing.T) {
			request, reads := buildApplyPreviewFixture(t)
			read := reads.impact
			reads.impact = func(ctx context.Context) (buildapply.ImpactEnvelope, error) {
				envelope, err := read(ctx)
				if err == nil {
					change(t, &envelope)
				}
				return envelope, err
			}
			inspectCalls := 0
			reads.inspect = func(context.Context, string, string, string) (*buildapply.Package, error) {
				inspectCalls++
				return nil, nil
			}
			_, _, _, _, err := captureBuildApplyPreview(context.Background(), request, reads)
			if err == nil || inspectCalls != 0 {
				t.Fatalf("%s allowed package inspection: calls=%d err=%v", name, inspectCalls, err)
			}
		})
	}
	test("unknown manager document", func(_ *testing.T, v *buildapply.ImpactEnvelope) {
		v.ManagerEvidence = buildapply.Observation[buildapply.EngineOwnerEvidence]{}
	})
	test("different manager lifetime", func(_ *testing.T, v *buildapply.ImpactEnvelope) {
		v.Impact.Fences.ManagerIncarnation = buildapply.Known("successor")
	})
	test("incomplete root coverage", func(t *testing.T, v *buildapply.ImpactEnvelope) {
		e := buildApplyPreviewEvidenceWithRoot(t, filepath.Join(t.TempDir(), "configured-models"))
		(*e.Universe.Value)[0].RootsComplete = false
		if err := e.Finalize(); err != nil {
			t.Fatal(err)
		}
		v.ManagerEvidence = buildapply.Known(e)
	})
}

func TestBuildApplyOwnerEvidenceRootReplacementRefusesPreview(t *testing.T) {
	request, reads := buildApplyPreviewFixture(t)
	first := buildApplyPreviewEvidenceWithRoot(t, filepath.Join(t.TempDir(), "first-models"))
	last := buildApplyPreviewEvidenceWithRoot(t, filepath.Join(t.TempDir(), "replacement-models"))
	calls := 0
	reads.impact = func(context.Context) (buildapply.ImpactEnvelope, error) {
		calls++
		evidence := first
		if calls == 2 {
			evidence = last
		}
		return buildapply.ImpactEnvelope{Impact: buildapply.Impact{Complete: true, Fences: buildapply.Fences{ManagerIncarnation: buildapply.Known("manager")}}, ManagerEvidence: buildapply.Known(evidence)}, nil
	}
	preview, candidate, target, _, err := captureBuildApplyPreview(context.Background(), request, reads)
	if err != nil {
		t.Fatal(err)
	}
	defer candidate.Close()
	defer target.Close()
	if calls != 2 || preview.ObservationsComplete || preview.Impact.Complete {
		t.Fatalf("replacement root accepted: calls=%d preview=%+v", calls, preview)
	}
}

func TestBuildApplyOwnerEvidencePrivateChangeRefusesPreview(t *testing.T) {
	test := func(name string, change func(*buildapply.EngineOwnerEvidence)) {
		t.Run(name, func(t *testing.T) {
			request, reads := buildApplyPreviewFixture(t)
			root := filepath.Join(t.TempDir(), "configured-models")
			first := buildApplyPreviewEvidenceWithRoot(t, root)
			last := buildApplyPreviewEvidenceWithRoot(t, root)
			change(&last)
			if err := last.Finalize(); err != nil {
				t.Fatalf("finalize changed owner: %v", err)
			}
			calls := 0
			reads.impact = func(context.Context) (buildapply.ImpactEnvelope, error) {
				calls++
				evidence := first
				if calls == 2 {
					evidence = last
				}
				return buildapply.ImpactEnvelope{Impact: buildapply.Impact{Complete: true, Fences: buildapply.Fences{ManagerIncarnation: buildapply.Known("manager")}}, ManagerEvidence: buildapply.Known(evidence)}, nil
			}
			preview, candidate, target, _, err := captureBuildApplyPreview(context.Background(), request, reads)
			if err != nil {
				t.Fatal(err)
			}
			defer candidate.Close()
			defer target.Close()
			if calls != 2 || preview.Impact.Complete || preview.ObservationsComplete {
				t.Fatalf("private %s change accepted: calls=%d preview=%+v", name, calls, preview)
			}
		})
	}
	test("intent revision", func(e *buildapply.EngineOwnerEvidence) { e.Intent.Value.Revision = "sha256:" + strings.Repeat("b", 64) })
	test("lock revision", func(e *buildapply.EngineOwnerEvidence) { e.Locks.Value.Revision = "sha256:" + strings.Repeat("b", 64) })
	test("reservation revision", func(e *buildapply.EngineOwnerEvidence) {
		e.Reservations.Value.Revision = "sha256:" + strings.Repeat("b", 64)
	})
	test("root coverage loss", func(e *buildapply.EngineOwnerEvidence) { (*e.Universe.Value)[0].RootsComplete = false })
}

func TestBuildApplyOwnerEvidencePrivateRootsStayOutOfPreview(t *testing.T) {
	request, reads := buildApplyPreviewFixture(t)
	root := filepath.Join(t.TempDir(), "private-manager-models")
	evidence := buildApplyPreviewEvidenceWithRoot(t, root)
	reads.impact = func(context.Context) (buildapply.ImpactEnvelope, error) {
		return buildapply.ImpactEnvelope{Impact: buildapply.Impact{Complete: true, Fences: buildapply.Fences{ManagerIncarnation: buildapply.Known("manager")}}, ManagerEvidence: buildapply.Known(evidence)}, nil
	}
	preview, candidate, target, _, err := captureBuildApplyPreview(context.Background(), request, reads)
	if err != nil {
		t.Fatal(err)
	}
	defer candidate.Close()
	defer target.Close()
	data, err := json.Marshal(preview)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(data, []byte(root)) || bytes.Contains(data, []byte("managerEvidence")) || bytes.Contains(data, []byte(evidence.CaptureToken)) {
		t.Fatalf("private manager data escaped public preview: %s", data)
	}
}

func TestBuildApplyPreviewRepeatedCaptureDoesNotMutate(t *testing.T) {
	request, reads := buildApplyPreviewFixture(t)
	s := newService(config{})
	c := &client{out: make(chan []byte, 4)}
	run := &brokerRun{}
	s.clients[c] = struct{}{}
	s.broker = run
	defer func() { s.mu.Lock(); s.invalidateBuildApplyPreviewLocked(nil); s.mu.Unlock() }()
	before, err := os.ReadFile(request.GUI.Executable)
	if err != nil {
		t.Fatal(err)
	}
	beforeInfo, err := os.Stat(request.GUI.Executable)
	if err != nil {
		t.Fatal(err)
	}
	protected, err := reads.protected()
	if err != nil {
		t.Fatal(err)
	}
	fixtureRoot := filepath.Dir(protected[0])
	// Freeze setup-time directory metadata after every setup handle is closed.
	// A create/delete during preview must change this witness too.
	past := time.Date(2000, 1, 1, 0, 0, 0, 0, time.UTC)
	if err := filepath.WalkDir(fixtureRoot, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() {
			return os.Chtimes(path, past, past)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	beforeTree := buildApplyFixtureTree(t, fixtureRoot)
	if !reflect.DeepEqual(beforeTree, buildApplyFixtureTree(t, fixtureRoot)) {
		t.Fatal("fixture directory baseline is not stable")
	}
	var prior string
	for i := 0; i < 2; i++ {
		result := s.previewBuildApplyUsing(c, request, &reads)
		if !result.Available || result.Value == nil || !result.Value.ObservationsComplete {
			t.Fatalf("preview = %+v", result)
		}
		if result.Value.PreviewID == "" || result.Value.PreviewID == prior {
			t.Fatal("selection did not replace its memory-only preview identity")
		}
		prior = result.Value.PreviewID
		if s.broker != run || len(s.pending) != 0 || s.stopping || s.restarts != 0 || len(c.out) != 0 {
			t.Fatal("preview entered lifecycle or emitted extra notifications")
		}
		if s.buildPreview == nil || s.buildPreview.client != c || s.buildPreview.broker != run {
			t.Fatal("preview lost its lifetime owner")
		}
	}
	after, err := os.ReadFile(request.GUI.Executable)
	if err != nil {
		t.Fatal(err)
	}
	afterInfo, err := os.Stat(request.GUI.Executable)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(before, after) || !beforeInfo.ModTime().Equal(afterInfo.ModTime()) || !os.SameFile(beforeInfo, afterInfo) {
		t.Fatal("preview wrote fixture bytes")
	}
	afterTree := buildApplyFixtureTree(t, fixtureRoot)
	if !reflect.DeepEqual(beforeTree, afterTree) {
		for path, before := range beforeTree {
			if after := afterTree[path]; before != after {
				t.Errorf("preview changed %s: before=%q after=%q", path, before, after)
			}
		}
		for path := range afterTree {
			if _, ok := beforeTree[path]; !ok {
				t.Errorf("preview added %s", path)
			}
		}
		t.Fatal("preview wrote fixture/profile/operation files")
	}
	if _, err := os.Stat(filepath.Join(filepath.Dir(request.CandidatePath), "build-apply")); !os.IsNotExist(err) {
		t.Fatalf("preview created operation state: %v", err)
	}
}

func buildApplyFixtureTree(t *testing.T, root string) map[string]string {
	t.Helper()
	result := map[string]string{}
	err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		info, err := entry.Info()
		if err != nil {
			return err
		}
		value := info.Mode().String() + " " + info.ModTime().UTC().Format(time.RFC3339Nano)
		if info.Mode().IsRegular() {
			data, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			sum := sha256.Sum256(data)
			value += " " + hex.EncodeToString(sum[:])
		}
		result[path] = value
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return result
}

func TestBuildApplyPreviewMissingRunningVCSRemainsUnknown(t *testing.T) {
	request, reads := buildApplyPreviewFixture(t)
	reads.running = func() buildapply.Identity { return buildapply.Identity{} }
	preview, candidate, target, _, err := captureBuildApplyPreview(context.Background(), request, reads)
	if err != nil {
		t.Fatal(err)
	}
	defer candidate.Close()
	defer target.Close()
	if preview.ObservationsComplete || preview.RunningService.BuildID.Known || preview.RunningService.SourceRevision.Known || preview.RunningService.Dirty.Known {
		t.Fatalf("missing running identity = %+v", preview.RunningService)
	}
	if !preview.RunningService.SHA256.Known || !preview.RequestedService.BuildID.Known {
		t.Fatal("missing running VCS erased independent artifact/bundle facts")
	}
}

func TestBuildApplyPreviewRunningMetadataRetainsIndependentUnknowns(t *testing.T) {
	const revision = "0123456789abcdef0123456789abcdef01234567"
	id := buildApplyRuntimeIdentity([]debug.BuildSetting{{Key: "vcs", Value: "git"}, {Key: "vcs.revision", Value: revision}})
	if !id.SourceRevision.Known || *id.SourceRevision.Value != revision || id.Dirty.Known || id.BuildID.Known {
		t.Fatalf("partial running metadata = %+v", id)
	}
	missing := buildApplyRuntimeIdentity(nil)
	if missing.SourceRevision.Known || missing.Dirty.Known || missing.BuildID.Known {
		t.Fatalf("invented running identity = %+v", missing)
	}
}

func TestBuildApplyPreviewMissingRequestedOrBundledVCSRemainsUnknown(t *testing.T) {
	for _, name := range []string{"requested", "bundled"} {
		t.Run(name, func(t *testing.T) {
			request, reads := buildApplyPreviewFixture(t)
			inspect := reads.inspect
			reads.inspect = func(ctx context.Context, path, platform, arch string) (*buildapply.Package, error) {
				p, err := inspect(ctx, path, platform, arch)
				if err != nil {
					return nil, err
				}
				if (name == "requested") == (path == request.CandidatePath) {
					p.Service = buildapply.Identity{SHA256: p.Service.SHA256}
				}
				return p, nil
			}
			preview, candidate, target, _, err := captureBuildApplyPreview(context.Background(), request, reads)
			if err != nil {
				t.Fatal(err)
			}
			defer candidate.Close()
			defer target.Close()
			id := preview.RequestedService
			if name == "bundled" {
				id = preview.BundledService
			}
			if preview.ObservationsComplete || id.BuildID.Known || id.SourceRevision.Known || id.Dirty.Known || !id.SHA256.Known {
				t.Fatalf("missing source facts = %+v", id)
			}
		})
	}
}

func TestBuildApplyPreviewIncompleteOwnersCannotStart(t *testing.T) {
	request, reads := buildApplyPreviewFixture(t)
	reads.impact = func(context.Context) (buildapply.ImpactEnvelope, error) {
		return buildapply.ImpactEnvelope{Impact: buildapply.Impact{Refusals: []buildapply.Refusal{{Code: "owner_evidence_unavailable", Detail: "Required owner evidence is missing."}}}}, nil
	}
	s := newService(config{})
	c := &client{out: make(chan []byte, 2)}
	s.clients[c] = struct{}{}
	s.broker = &brokerRun{}
	defer func() { s.mu.Lock(); s.invalidateBuildApplyPreviewLocked(nil); s.mu.Unlock() }()
	preview := s.previewBuildApplyUsing(c, request, &reads)
	if preview.Available || preview.Refusal == nil || preview.Refusal.Code != "unsafe_package" {
		t.Fatalf("missing owners did not remain incomplete: %+v", preview)
	}
	if s.buildPreview != nil {
		t.Fatal("unknown manager roots left an available preview")
	}
	params, err := json.Marshal(buildapply.StartRequest{PreviewID: "unavailable-preview", GUI: request.GUI})
	if err != nil {
		t.Fatal(err)
	}
	id := json.RawMessage(`"start"`)
	s.handleBuildApply(c, &jsonrpc.Message{ID: &id, Method: "service/build-apply-start", Params: params})
	select {
	case frame := <-c.out:
		var response jsonrpc.Message
		if err := json.Unmarshal(frame, &response); err != nil {
			t.Fatal(err)
		}
		var result buildapply.Result[buildapply.BuildApplyOperation]
		if err := json.Unmarshal(response.Result, &result); err != nil {
			t.Fatal(err)
		}
		if response.Error != nil || result.Available || result.Refusal == nil {
			t.Fatalf("incomplete start = %+v", response)
		}
	default:
		t.Fatal("start did not refuse")
	}
	if s.stopping || s.restarts != 0 || len(s.pending) != 0 {
		t.Fatal("incomplete preview entered lifecycle")
	}
}

func TestBuildApplyPreviewChangedOwnerFenceIsIncomplete(t *testing.T) {
	request, reads := buildApplyPreviewFixture(t)
	read := reads.impact
	calls := 0
	reads.impact = func(ctx context.Context) (buildapply.ImpactEnvelope, error) {
		envelope, err := read(ctx)
		if err != nil {
			return envelope, err
		}
		calls++
		revision := "7"
		if calls > 1 {
			revision = "8"
		}
		envelope.Impact.Fences.PolicyRevision = buildapply.Known(revision)
		return envelope, nil
	}
	preview, candidate, target, _, err := captureBuildApplyPreview(context.Background(), request, reads)
	if err != nil {
		t.Fatal(err)
	}
	defer candidate.Close()
	defer target.Close()
	if calls != 2 || preview.ObservationsComplete || preview.Impact.Complete || preview.Impact.Fences.PolicyRevision.Known {
		t.Fatalf("mixed owner fences accepted: %+v", preview)
	}
}

func TestBuildApplyPreviewWorkBeforePauseCanChange(t *testing.T) {
	request, reads := buildApplyPreviewFixture(t)
	read := reads.impact
	calls := 0
	reads.impact = func(ctx context.Context) (buildapply.ImpactEnvelope, error) {
		envelope, err := read(ctx)
		calls++
		envelope.Impact.ActiveWork = buildapply.Known(calls)
		envelope.Impact.InterruptionPolicy.IncludesWorkBeforeFence = true
		return envelope, err
	}
	preview, candidate, target, _, err := captureBuildApplyPreview(context.Background(), request, reads)
	if err != nil {
		t.Fatal(err)
	}
	defer candidate.Close()
	defer target.Close()
	if !preview.ObservationsComplete || !preview.Impact.ActiveWork.Known || *preview.Impact.ActiveWork.Value != 2 || !preview.Impact.InterruptionPolicy.IncludesWorkBeforeFence {
		t.Fatalf("moving work observation = %+v", preview.Impact)
	}
}

func TestBuildApplyPreviewCompletedSlotExpiresAfterSixtySeconds(t *testing.T) {
	request, reads := buildApplyPreviewFixture(t)
	s := newService(config{})
	c := &client{out: make(chan []byte, 2)}
	s.clients[c] = struct{}{}
	s.broker = &brokerRun{}
	result := s.previewBuildApplyUsing(c, request, &reads)
	if !result.Available || s.buildPreview == nil {
		t.Fatalf("preview = %+v", result)
	}
	s.mu.Lock()
	selected := s.buildPreview
	candidate := selected.candidate
	got := s.currentBuildApplyPreviewLocked(c, selected.expires.Add(time.Second))
	s.mu.Unlock()
	if got != nil || s.buildPreview != nil {
		t.Fatal("completed preview survived more than 60 seconds")
	}
	if err := candidate.Root.Revalidate(); err == nil {
		t.Fatal("expired candidate root remained open")
	}
}

type buildApplyBlockedWriter struct {
	entered  chan struct{}
	release  chan struct{}
	finished chan struct{}
	once     sync.Once
}

func (w *buildApplyBlockedWriter) Write([]byte) (int, error) {
	close(w.entered)
	<-w.release
	close(w.finished)
	return 0, io.ErrClosedPipe
}
func (w *buildApplyBlockedWriter) Close() error { w.once.Do(func() { close(w.release) }); return nil }

func TestBuildApplyPreviewReadTimeoutDoesNotAccumulateBlockedWriters(t *testing.T) {
	s := newService(config{})
	c := &client{out: make(chan []byte, 2)}
	s.clients[c] = struct{}{}
	w := &buildApplyBlockedWriter{entered: make(chan struct{}), release: make(chan struct{}), finished: make(chan struct{})}
	t.Cleanup(func() { w.Close(); <-w.finished })
	run := &brokerRun{proc: &brokerProc{stdin: w}, exited: make(chan struct{})}
	s.broker = run
	s.buildPreview = &buildApplyPreviewState{client: c, broker: run}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() { _, err := s.readBuildApplyImpact(ctx, run); done <- err }()
	<-w.entered
	cancel()
	if err := <-done; err == nil {
		t.Fatal("blocked impact read outlived its cancellation")
	}
	s.mu.Lock()
	s.invalidateBuildApplyPreviewLocked(c)
	pending := len(s.pending)
	s.mu.Unlock()
	if pending != 0 {
		t.Fatal("cancelled impact retained a pending RPC")
	}
	if _, err := s.readBuildApplyImpact(context.Background(), run); err == nil {
		t.Fatal("retired blocked writer admitted another read")
	}
	w.Close()
	<-w.finished
}

func TestBuildApplyPreviewInvalidatesDuringActualClientDetach(t *testing.T) {
	request, reads := buildApplyPreviewFixture(t)
	server, peer := net.Pipe()
	defer peer.Close()
	c := newClient(1, server)
	defer func() { c.closeNow(); <-c.written }()
	s := newService(config{})
	s.clients[c] = struct{}{}
	s.broker = &brokerRun{}
	read := reads.impact
	calls := 0
	reads.impact = func(ctx context.Context) (buildapply.ImpactEnvelope, error) {
		calls++
		if calls == 2 {
			s.detach(c, false)
		}
		return read(ctx)
	}
	result := s.previewBuildApplyUsing(c, request, &reads)
	if result.Available || result.Refusal == nil || result.Refusal.Code != "changed" || s.buildPreview != nil {
		t.Fatalf("detached capture = %+v", result)
	}
}

func TestBuildApplyPreviewActualBrokerLossRetiresCapture(t *testing.T) {
	request, reads := buildApplyPreviewFixture(t)
	s := newService(config{})
	c := &client{out: make(chan []byte, 4)}
	s.clients[c] = struct{}{}
	run := &brokerRun{}
	s.broker = run
	read := reads.impact
	calls := 0
	reads.impact = func(ctx context.Context) (buildapply.ImpactEnvelope, error) {
		calls++
		if calls == 2 {
			s.brokerExited(run)
		}
		return read(ctx)
	}
	result := s.previewBuildApplyUsing(c, request, &reads)
	if result.Available || result.Refusal == nil || result.Refusal.Code != "changed" || s.buildPreview != nil {
		t.Fatalf("retired generation capture = %+v", result)
	}
}
