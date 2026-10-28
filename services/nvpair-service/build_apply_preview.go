// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"runtime/debug"
	"strings"
	"time"

	"nvpair-shared/appdir"
	"nvpair-shared/buildapply"
	"nvpair-shared/buildinfo"
	"nvpair-shared/jsonrpc"
)

const buildApplyPreviewReadBudget = 10 * time.Minute
const buildApplyImpactReadBudget = 5 * time.Second

// buildApplyPreviewState is guarded by Service.mu and belongs to one client
// and broker lifetime. It is never persisted.
type buildApplyPreviewState struct {
	client    *client
	broker    *brokerRun
	expires   time.Time
	preview   buildapply.BuildApplyPreview
	candidate *buildapply.Package
	target    *buildapply.Package
	gui       buildapply.ProcessIdentity
	cancel    context.CancelFunc
	timer     *time.Timer
	writing   bool
	retired   bool
}

// invalidateBuildApplyPreviewLocked requires Service.mu. A nil client retires
// the slot regardless of its owner, for use by the broker lifetime owner.
func (s *Service) invalidateBuildApplyPreviewLocked(c *client) {
	if s.buildPreview != nil && (c == nil || s.buildPreview.client == c) {
		p := s.buildPreview
		p.retired = true
		if !p.writing {
			s.buildPreview = nil
		}
		if p.cancel != nil {
			p.cancel()
		}
		if p.timer != nil {
			p.timer.Stop()
		}
		if p.candidate != nil {
			p.candidate.Close()
			p.candidate = nil
		}
		if p.target != nil {
			p.target.Close()
			p.target = nil
		}
	}
}

// currentBuildApplyPreviewLocked refuses a disconnected client, a changed
// broker, or an expired preview even before the corresponding cleanup runs.
func (s *Service) currentBuildApplyPreviewLocked(c *client, now time.Time) *buildApplyPreviewState {
	p := s.buildPreview
	if p == nil {
		return nil
	}
	_, attached := s.clients[p.client]
	exited := false
	if p.broker != nil {
		select {
		case <-p.broker.exited:
			exited = true
		default:
		}
	}
	if p.retired || !attached || p.broker == nil || exited || p.broker != s.broker || !p.expires.IsZero() && !now.Before(p.expires) {
		s.invalidateBuildApplyPreviewLocked(nil)
		return nil
	}
	if p.client != c || p.expires.IsZero() {
		return nil
	}
	return p
}

// All capture effects are expressible as read boundaries. No maintenance,
// journal, lifecycle, peer, candidate execution or policy writer is reachable.
type buildApplyPreviewReads struct {
	inspect       func(context.Context, string, string, string) (*buildapply.Package, error)
	process       func(int) (buildapply.ProcessIdentity, error)
	brokerProcess func() (buildapply.ProcessIdentity, error)
	executable    func() (string, error)
	protected     func() ([]string, error)
	impact        func(context.Context) (buildapply.ImpactEnvelope, error)
	revalidate    func(context.Context, *buildapply.Package) error
	parent        func(*buildapply.RealRoot, int64) error
	host          func() buildapply.Observation[string]
	running       func() buildapply.Identity
}

func (s *Service) respondBuildApplyPreview(c *client, id *json.RawMessage, request buildapply.PreviewRequest) {
	state, ctx := s.beginBuildApplyPreview(c)
	if state == nil {
		c.respond(id, buildapply.Unavailable[buildapply.BuildApplyPreview]())
		return
	}
	var replyID *json.RawMessage
	if id != nil {
		copied := append(json.RawMessage(nil), (*id)...)
		replyID = &copied
	}
	// Keep the ordinary reader free to observe EOF/shutdown while hashing.
	// Registration happens before dispatch, preserving request order.
	go func() { c.respond(replyID, s.completeBuildApplyPreview(c, request, nil, state, ctx)) }()
}

func (s *Service) previewBuildApplyUsing(c *client, request buildapply.PreviewRequest, supplied *buildApplyPreviewReads) buildapply.Result[buildapply.BuildApplyPreview] {
	state, ctx := s.beginBuildApplyPreview(c)
	if state == nil {
		return buildapply.Unavailable[buildapply.BuildApplyPreview]()
	}
	return s.completeBuildApplyPreview(c, request, supplied, state, ctx)
}

func (s *Service) beginBuildApplyPreview(c *client) (*buildApplyPreviewState, context.Context) {
	s.mu.Lock()
	run := s.broker
	_, attached := s.clients[c]
	if run == nil || !attached || s.stopping {
		s.mu.Unlock()
		return nil, nil
	}
	// Keep at most one potentially blocked read writer per Service. A timed
	// out write is retired, but cannot be interrupted by stopping the broker
	// from a read-only preview. It retains only this bounded transport owner.
	if s.buildImpactWriting || s.buildPreview != nil && s.buildPreview.writing {
		s.mu.Unlock()
		return nil, nil
	}
	// Retire a prior capture as well as a completed preview. Late results may
	// neither replace the newer selection nor transfer it to another client.
	s.invalidateBuildApplyPreviewLocked(nil)
	ctx, cancel := context.WithTimeout(context.Background(), buildApplyPreviewReadBudget)
	state := &buildApplyPreviewState{client: c, broker: run, cancel: cancel}
	s.buildPreview = state
	s.mu.Unlock()
	return state, ctx
}

func (s *Service) completeBuildApplyPreview(c *client, request buildapply.PreviewRequest, supplied *buildApplyPreviewReads, state *buildApplyPreviewState, ctx context.Context) buildapply.Result[buildapply.BuildApplyPreview] {
	run := state.broker
	cancel := state.cancel
	reads := buildApplyPreviewReads{inspect: buildapply.InspectPackage, process: buildapply.InspectProcess, executable: os.Executable, protected: buildApplyProtectedRoots, running: currentBuildApplyIdentity}
	reads.brokerProcess = func() (buildapply.ProcessIdentity, error) {
		if run.proc == nil || run.proc.pid <= 0 {
			return buildapply.ProcessIdentity{}, errors.New("broker process identity unavailable")
		}
		return buildapply.InspectProcess(run.proc.pid)
	}
	reads.revalidate = func(ctx context.Context, p *buildapply.Package) error { return p.Revalidate(ctx) }
	reads.parent = buildapply.CheckTargetParent
	reads.host = readBuildApplyHostID
	reads.impact = func(ctx context.Context) (buildapply.ImpactEnvelope, error) { return s.readBuildApplyImpact(ctx, run) }
	if supplied != nil {
		reads = *supplied
	}
	preview, candidate, target, gui, err := captureBuildApplyPreview(ctx, request, reads)
	cancel()
	s.mu.Lock()
	defer s.mu.Unlock()
	_, attached := s.clients[c]
	if state.retired || s.buildPreview != state || !attached || s.broker != run || s.stopping {
		if candidate != nil {
			candidate.Close()
		}
		if target != nil {
			target.Close()
		}
		return refusedPreview("changed", "The client or broker lifetime changed during preview.")
	}
	if err != nil {
		s.invalidateBuildApplyPreviewLocked(nil)
		if errors.Is(err, buildapply.ErrPackageTooLarge) {
			return refusedPreview("oversized_build", "The staged build exceeds the 8 GiB total, 2 GiB file, 100,000 entry or 4,096 UTF-8 byte relative-path limit.")
		}
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return refusedPreview("preview_unavailable", "The bounded preview read was cancelled or timed out.")
		}
		return refusedPreview("unsafe_package", "The package, installation or native process could not be safely validated.")
	}
	var id [16]byte
	if _, err := rand.Read(id[:]); err != nil {
		candidate.Close()
		target.Close()
		s.invalidateBuildApplyPreviewLocked(nil)
		return buildapply.Unavailable[buildapply.BuildApplyPreview]()
	}
	state.expires = time.Now().Add(buildapply.PreviewLifetimeSeconds * time.Second)
	preview.PreviewID = hex.EncodeToString(id[:])
	preview.ExpiresAt = state.expires.UTC().Format(time.RFC3339Nano)
	bindBuildApplyImpact(&preview.Impact, s.startedAt, s.runs)
	state.preview = preview
	state.candidate = candidate
	state.target = target
	state.gui = gui
	state.timer = time.AfterFunc(time.Until(state.expires), func() {
		s.mu.Lock()
		defer s.mu.Unlock()
		if s.buildPreview == state {
			s.invalidateBuildApplyPreviewLocked(nil)
		}
	})
	return buildapply.Result[buildapply.BuildApplyPreview]{Available: true, Value: &preview}
}

func bindBuildApplyImpact(impact *buildapply.Impact, startedAt time.Time, runs int) {
	impact.Fences.ServiceInstance = buildapply.Known(fmt.Sprintf("%d-%d", os.Getpid(), startedAt.UnixNano()))
	impact.Fences.BrokerGeneration = buildapply.Known(fmt.Sprintf("%d", runs))
	runtimeFence := fmt.Sprintf("%d-%d-%d", os.Getpid(), startedAt.UnixNano(), runs)
	if fence := impact.Fences.RuntimeGeneration; fence.Known && fence.Value != nil {
		runtimeFence += "-" + *fence.Value
	}
	impact.Fences.RuntimeGeneration = buildapply.Known(runtimeFence)
}

func refusedPreview(code, detail string) buildapply.Result[buildapply.BuildApplyPreview] {
	// Operational diagnostics only; never surface raw owner payloads or paths
	// from a parse error. Bound this final public detail independently.
	if len(detail) > buildapply.MaxDetailBytes {
		detail = "The selected build could not be safely inspected."
	}
	return buildapply.Result[buildapply.BuildApplyPreview]{Refusal: &buildapply.Refusal{Code: code, Detail: detail}}
}

func (s *Service) readBuildApplyImpact(ctx context.Context, run *brokerRun) (buildapply.ImpactEnvelope, error) {
	ctx, cancel := context.WithTimeout(ctx, buildApplyImpactReadBudget)
	defer cancel()
	// Unlike callBroker, this owner sends the strict empty request object and
	// retires its pending entry when its read budget ends.
	ch := make(chan *jsonrpc.Message, 1)
	written := make(chan struct{})
	s.mu.Lock()
	if run == nil || s.broker != run || s.stopping {
		s.mu.Unlock()
		return buildapply.ImpactEnvelope{}, errors.New("broker changed")
	}
	state := s.buildPreview
	if s.buildImpactWriting || ctx.Err() != nil || state != nil && state.broker == run && state.retired {
		s.mu.Unlock()
		return buildapply.ImpactEnvelope{}, errors.New("owner read is unavailable")
	}
	s.buildImpactWriting = true
	if state != nil && state.broker == run {
		state.writing = true
	}
	s.nextID++
	id := s.nextID
	s.pending[id] = &pendingRequest{internal: func(m *jsonrpc.Message) {
		select {
		case ch <- m:
		default:
		}
	}, method: "build:apply-impact", run: run}
	s.mu.Unlock()
	defer func() { s.mu.Lock(); delete(s.pending, id); s.mu.Unlock() }()
	// The ordinary transport writer owns its existing write bound. A blocked
	// writer must not keep the preview read waiting beyond its budget.
	go func() {
		defer func() {
			s.mu.Lock()
			s.buildImpactWriting = false
			if state != nil && state.broker == run {
				state.writing = false
				if state.retired && s.buildPreview == state {
					s.buildPreview = nil
				}
			}
			s.mu.Unlock()
			close(written)
		}()
		if ctx.Err() != nil {
			return
		}
		if err := run.writeMessageContext(ctx, &jsonrpc.Message{JSONRPC: "2.0", ID: rawID(id), Method: "build:apply-impact", Params: json.RawMessage(`{}`)}); err != nil {
			select {
			case ch <- nil:
			default:
			}
		}
	}()
	select {
	case <-written:
	case <-ctx.Done():
		return buildapply.ImpactEnvelope{}, ctx.Err()
	}
	select {
	case <-ctx.Done():
		return buildapply.ImpactEnvelope{}, ctx.Err()
	case response := <-ch:
		var result buildapply.Result[buildapply.ImpactEnvelope]
		if response == nil || response.Error != nil || len(response.Result) > buildapply.MaxStatusBytes || json.Unmarshal(response.Result, &result) != nil || !result.Available || result.Value == nil {
			return buildapply.ImpactEnvelope{}, errors.New("local owner impact unavailable")
		}
		s.mu.Lock()
		current := s.broker == run && !s.stopping
		s.mu.Unlock()
		if !current {
			return buildapply.ImpactEnvelope{}, errors.New("broker changed during owner read")
		}
		return *result.Value, nil
	}
}

// Manager roots include owner-known install directories in DataRoots. A
// missing universe entry or unknown coverage cannot authorize package I/O.
func buildApplyManagerRoots(envelope buildapply.ImpactEnvelope) ([]string, error) {
	if !envelope.ManagerEvidence.Known || envelope.ManagerEvidence.Value == nil ||
		!envelope.Impact.Fences.ManagerIncarnation.Known || envelope.Impact.Fences.ManagerIncarnation.Value == nil {
		return nil, errors.New("manager root identity is unavailable")
	}
	evidence := *envelope.ManagerEvidence.Value
	if evidence.ManagerIncarnation == "" || evidence.CaptureToken == "" ||
		evidence.ManagerIncarnation != *envelope.Impact.Fences.ManagerIncarnation.Value ||
		!evidence.Universe.Known || evidence.Universe.Value == nil {
		return nil, errors.New("manager root universe is unavailable")
	}
	roots := []string{}
	for _, entry := range *evidence.Universe.Value {
		if !entry.RootsComplete || !entry.ModelRoots.Known || entry.ModelRoots.Value == nil ||
			!entry.DataRoots.Known || entry.DataRoots.Value == nil {
			return nil, errors.New("manager root coverage is unavailable")
		}
		roots = append(roots, (*entry.ModelRoots.Value)...)
		roots = append(roots, (*entry.DataRoots.Value)...)
	}
	return roots, nil
}

func captureBuildApplyPreview(ctx context.Context, request buildapply.PreviewRequest, reads buildApplyPreviewReads) (preview buildapply.BuildApplyPreview, candidate, target *buildapply.Package, gui buildapply.ProcessIdentity, err error) {
	defer func() {
		if err != nil {
			if candidate != nil {
				candidate.Close()
				candidate = nil
			}
			if target != nil {
				target.Close()
				target = nil
			}
		}
	}()
	preview.Platform = runtime.GOOS
	preview.Architecture = runtime.GOARCH
	gui, err = reads.process(request.GUI.PID)
	if err != nil {
		return
	}
	if gui.Created == "" || !sameBuildApplyImage(gui.Executable, request.GUI.Executable) {
		err = errors.New("native GUI process identity does not match")
		return
	}
	service, readErr := reads.executable()
	if readErr != nil {
		err = readErr
		return
	}
	serviceProcess, readErr := reads.process(os.Getpid())
	if readErr != nil || serviceProcess.Created == "" || !sameBuildApplyImage(serviceProcess.Executable, service) {
		err = errors.New("attached service process identity is unavailable")
		return
	}
	root, readErr := buildapply.TargetFromExecutables(gui.Executable, service, runtime.GOOS)
	if readErr != nil {
		err = readErr
		return
	}
	layout, _ := buildapply.LayoutFor(runtime.GOOS)
	brokerExecutable := brokerName
	if runtime.GOOS == "windows" {
		brokerExecutable += ".exe"
	}
	expectedBroker := filepath.Join(root, filepath.FromSlash(layout.ServiceDir), brokerExecutable)
	brokerProcess, readErr := reads.brokerProcess()
	if readErr != nil || brokerProcess.Created == "" || !sameBuildApplyImage(brokerProcess.Executable, expectedBroker) {
		err = errors.New("broker does not belong to the verified current installation")
		return
	}
	excluded, readErr := reads.protected()
	if readErr != nil {
		err = readErr
		return
	}
	preview.HostID = reads.host()
	first, impactErr := reads.impact(ctx)
	if impactErr != nil {
		err = impactErr
		return
	}
	managerRoots, rootErr := buildApplyManagerRoots(first)
	if rootErr != nil {
		err = rootErr
		return
	}
	excluded = append(excluded, managerRoots...)
	if err = buildapply.ValidateLocations(request.CandidatePath, root, excluded); err != nil {
		return
	}
	candidate, err = reads.inspect(ctx, request.CandidatePath, runtime.GOOS, runtime.GOARCH)
	if err != nil {
		return
	}
	target, err = reads.inspect(ctx, root, runtime.GOOS, runtime.GOARCH)
	if err != nil {
		return
	}
	if err = reads.parent(target.Root, candidate.Bytes); err != nil {
		return
	}
	preview.RequestedGUI = candidate.GUI
	preview.RequestedService = candidate.Service
	preview.BundledService = target.Service
	preview.CurrentGUI = target.GUI
	preview.CurrentGUI.BuildID = buildapply.Observation[string]{}
	preview.CurrentGUI.SourceRevision = buildapply.Observation[string]{}
	preview.CurrentGUI.Dirty = buildapply.Observation[bool]{}
	if validBuildApplyID(request.GUI.BuildID) {
		preview.CurrentGUI = identityWithBuildID(preview.CurrentGUI, request.GUI.BuildID)
	}
	// The attached process's own embedded identity is authoritative for the
	// running observation. A bundled service stamp cannot fill missing VCS.
	preview.RunningService = reads.running()
	preview.RunningService.SHA256 = target.Service.SHA256
	preview.PackageSHA256 = buildapply.Known(candidate.SHA256)
	preview.TargetLayout = buildapply.Known(target.Layout)
	if err = reads.revalidate(ctx, candidate); err != nil {
		return
	}
	if err = reads.revalidate(ctx, target); err != nil {
		return
	}
	last, lastErr := reads.impact(ctx)
	preview.Impact = last.Impact
	lastRoots, lastRootsErr := buildApplyManagerRoots(last)
	if lastErr != nil || lastRootsErr != nil || !reflect.DeepEqual(managerRoots, lastRoots) ||
		first.ManagerEvidence.Value.CaptureToken != last.ManagerEvidence.Value.CaptureToken ||
		first.ManagerEvidence.Value.ManagerIncarnation != last.ManagerEvidence.Value.ManagerIncarnation ||
		!sameBuildApplyImpact(first.Impact, last.Impact) {
		preview.Impact = buildapply.Impact{InterruptionPolicy: buildapply.InterruptionPolicy{StopsEngines: true, UnloadsModels: true, CommittedStreamsMayFail: true, IncludesWorkBeforeFence: true}, RestorePolicy: buildapply.RestorePolicy{RemainPaused: true, RequireLocalResume: true, PreserveIntentAndLocks: true, PreserveNewerPause: true}, Refusals: []buildapply.Refusal{{Code: "changed_or_unavailable", Detail: "Local owner evidence changed or became unavailable during capture."}}}
	}
	if err = buildapply.ValidateLocations(request.CandidatePath, root, excluded); err != nil {
		return
	}
	after, readErr := reads.process(request.GUI.PID)
	if readErr != nil || after != gui {
		err = errors.New("GUI process changed during preview")
		return
	}
	serviceAfter, readErr := reads.process(os.Getpid())
	if readErr != nil || serviceAfter != serviceProcess {
		err = errors.New("attached service process changed during preview")
		return
	}
	brokerAfter, readErr := reads.brokerProcess()
	if readErr != nil || brokerAfter != brokerProcess || !sameBuildApplyImage(brokerAfter.Executable, expectedBroker) {
		err = errors.New("broker process changed during preview")
		return
	}
	if !sameBuildApplyImage(service, filepath.Join(root, filepath.FromSlash(target.Layout.ServiceDir), target.Layout.ServiceExecutable)) {
		err = errors.New("running service installation changed")
		return
	}
	identitiesAgree := target.GUI.BuildID.Known && preview.CurrentGUI.BuildID.Known && *target.GUI.BuildID.Value == *preview.CurrentGUI.BuildID.Value && target.Service.BuildID.Known && preview.RunningService.BuildID.Known && *target.Service.BuildID.Value == *preview.RunningService.BuildID.Value
	if !identitiesAgree {
		preview.Impact.Refusals = append(preview.Impact.Refusals, buildapply.Refusal{Code: "running_bundle_identity", Detail: "Running GUI/service identity cannot be matched to the current installation's embedded identities."})
	}
	preview.ObservationsComplete = identitiesAgree && preview.Impact.Complete && preview.HostID.Known && reflect.DeepEqual(preview.HostID, reads.host()) && preview.CurrentGUI.BuildID.Known && preview.RequestedGUI.BuildID.Known && fullServiceIdentity(preview.RequestedService) && fullServiceIdentity(preview.BundledService) && fullServiceIdentity(preview.RunningService)
	if !preview.ObservationsComplete {
		preview.Impact.Refusals = append(preview.Impact.Refusals, buildapply.Refusal{Code: "incomplete_identity", Detail: "Full source/build identity or cached owner evidence is unavailable; artifact hashes do not replace it."})
	}
	err = ctx.Err()
	return
}

// Active counts are observations, not an admission reservation. Use the last
// read while retaining the explicit promise that work admitted before the
// eventual pause fence is included. Configuration/process/model fences must
// still match; never require a quiet inference interval just to preview drain.
func sameBuildApplyImpact(first, last buildapply.Impact) bool {
	first.ActiveWork = buildapply.Observation[int]{}
	last.ActiveWork = buildapply.Observation[int]{}
	withoutActive := func(observation buildapply.Observation[[]buildapply.EngineFact]) buildapply.Observation[[]buildapply.EngineFact] {
		if !observation.Known || observation.Value == nil {
			return observation
		}
		facts := append([]buildapply.EngineFact{}, (*observation.Value)...)
		for i := range facts {
			facts[i].Active = buildapply.Observation[bool]{}
		}
		return buildapply.Known(facts)
	}
	first.Engines = withoutActive(first.Engines)
	last.Engines = withoutActive(last.Engines)
	return reflect.DeepEqual(first, last)
}

func fullServiceIdentity(id buildapply.Identity) bool {
	return id.BuildID.Known && id.SourceRevision.Known && id.Dirty.Known && id.SHA256.Known
}

func sameBuildApplyImage(a, b string) bool {
	// A supplied UNC path or alias must not trigger a filesystem read. Native
	// process paths and bundle paths must have the same canonical spelling.
	if runtime.GOOS == "windows" {
		if !strings.EqualFold(filepath.Clean(a), filepath.Clean(b)) {
			return false
		}
	} else if filepath.Clean(a) != filepath.Clean(b) {
		return false
	}
	if !filepath.IsAbs(a) || strings.HasPrefix(a, `\\`) || strings.HasPrefix(a, "//") {
		return false
	}
	parent, err := buildapply.OpenRealRoot(filepath.Dir(a))
	if err != nil {
		return false
	}
	defer parent.Close()
	x, err := os.Lstat(a)
	if err != nil || !x.Mode().IsRegular() {
		return false
	}
	y, err := os.Lstat(b)
	return err == nil && y.Mode().IsRegular() && os.SameFile(x, y) && parent.Revalidate() == nil
}

func validBuildApplyID(id string) bool {
	var value buildinfo.ID
	data, err := json.Marshal(id)
	return err == nil && json.Unmarshal(data, &value) == nil && value != ""
}

func identityWithBuildID(id buildapply.Identity, value string) buildapply.Identity {
	if !validBuildApplyID(value) {
		return id
	}
	id.BuildID = buildapply.Known(value)
	id.SourceRevision = buildapply.Known(strings.TrimSuffix(strings.TrimPrefix(value, "git:"), "+dirty"))
	id.Dirty = buildapply.Known(strings.HasSuffix(value, "+dirty"))
	return id
}

func currentBuildApplyIdentity() buildapply.Identity {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return buildapply.Identity{}
	}
	return buildApplyRuntimeIdentity(info.Settings)
}

func buildApplyRuntimeIdentity(settings []debug.BuildSetting) buildapply.Identity {
	var vcs, revision, modified string
	for _, setting := range settings {
		switch setting.Key {
		case "vcs":
			vcs = setting.Value
		case "vcs.revision":
			revision = setting.Value
		case "vcs.modified":
			modified = setting.Value
		}
	}
	id := buildapply.Identity{}
	if vcs != "git" {
		return id
	}
	if (len(revision) == 40 || len(revision) == 64) && validBuildApplyID("git:"+revision) {
		id.SourceRevision = buildapply.Known(revision)
	}
	if modified == "true" || modified == "false" {
		id.Dirty = buildapply.Known(modified == "true")
	}
	if id.SourceRevision.Known && id.Dirty.Known {
		value := "git:" + revision
		if *id.Dirty.Value {
			value += "+dirty"
		}
		id.BuildID = buildapply.Known(value)
	}
	return id
}

func buildApplyProtectedRoots() ([]string, error) {
	_, brokerArgs := splitArgs(os.Args[1:])
	if hasFlag(brokerArgs, "cluster-dir") {
		return nil, errors.New("custom broker profile ownership is unavailable to this service preview")
	}
	data, err := appdir.Dir()
	if err != nil {
		return nil, err
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return nil, err
	}
	// Model stores outside appdir. Configured paths still require the manager
	// owner evidence and keep impact incomplete until that owner supplies it.
	roots := []string{data, filepath.Join(home, ".ollama"), filepath.Join(home, ".lmstudio"), filepath.Join(home, ".cache", "lm-studio")}
	for _, key := range []string{"OLLAMA_MODELS", "LMSTUDIO_MODELS"} {
		if path := os.Getenv(key); path != "" {
			if !filepath.IsAbs(path) {
				return nil, errors.New("model root is unresolved")
			}
			roots = append(roots, path)
		}
	}
	return roots, nil
}

// Read only the public UUID from existing owner files. nodeid.Resolve is not
// suitable here: it can mint and persist an identity on a missing-file read.
func readBuildApplyHostID() buildapply.Observation[string] {
	base, err := appdir.Dir()
	if err != nil {
		return buildapply.Observation[string]{}
	}
	for _, name := range []string{filepath.Join("cluster", "identity.json"), "node-id.json"} {
		path := filepath.Join(base, name)
		data, err := buildapply.ReadLocalFile(path, 64<<10)
		if err != nil {
			if os.IsNotExist(err) {
				continue
			}
			return buildapply.Observation[string]{}
		}
		var value struct {
			UUID string `json:"node_uuid"`
		}
		if json.Unmarshal(data, &value) != nil || len(value.UUID) != 36 {
			return buildapply.Observation[string]{}
		}
		valid := true
		for i, c := range value.UUID {
			if i == 8 || i == 13 || i == 18 || i == 23 {
				valid = valid && c == '-'
			} else {
				valid = valid && (c >= 'a' && c <= 'f' || c >= '0' && c <= '9')
			}
		}
		if !valid {
			return buildapply.Observation[string]{}
		}
		return buildapply.Known(value.UUID)
	}
	return buildapply.Observation[string]{}
}
