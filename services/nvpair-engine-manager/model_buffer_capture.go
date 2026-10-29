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
	"log/slog"
	"maps"
	"math"
	"math/big"
	"os"
	"reflect"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"nvpair-shared/modelconfig"
	"nvpair-shared/noderec"
)

const modelBufferCaptureBytes = 4 << 20
const modelBufferChildCharge = 768 << 10

// Preserve the existing serving decoder's behavior. Observation evidence has
// its own exact, duplicate-free allowlist; unrelated props never enter it.
func modelBufferReadProps(data []byte) modelBufferProps {
	fields := modelBufferPropsFields(data, []string{"build_info", "default_generation_settings", "total_slots"})
	var value modelBufferProps
	if len(fields["build_info"]) > 258 || json.Unmarshal(fields["build_info"], &value.Build) != nil || len(value.Build) > 256 {
		return modelBufferProps{}
	}
	if raw, ok := fields["total_slots"]; ok && json.Unmarshal(raw, &value.Slots) != nil {
		return modelBufferProps{}
	}
	if raw, ok := fields["default_generation_settings"]; ok {
		defaults := modelBufferPropsFields(raw, []string{"n_ctx"})
		if json.Unmarshal(defaults["n_ctx"], &value.Context) != nil {
			return modelBufferProps{}
		}
	}
	return value
}

func modelBufferPropsFields(data []byte, wanted []string) map[string]json.RawMessage {
	d := json.NewDecoder(bytes.NewReader(data))
	if token, err := d.Token(); err != nil || token != json.Delim('{') {
		return nil
	}
	fields := map[string]json.RawMessage{}
	for count := 0; d.More(); count++ {
		if count >= 64 {
			return nil
		}
		token, err := d.Token()
		key, ok := token.(string)
		if err != nil || !ok || len(key) > 256 {
			return nil
		}
		var raw json.RawMessage
		if d.Decode(&raw) != nil {
			return nil
		}
		for _, name := range wanted {
			if strings.EqualFold(name, key) {
				if name != key || fields[name] != nil {
					return nil
				}
				fields[name] = raw
			}
		}
	}
	if _, err := d.Token(); err != nil {
		return nil
	}
	if _, err := d.Token(); err != io.EOF {
		return nil
	}
	return fields
}

// These are the pinned SRV_INF envelopes, including the truncated __func__.
var modelBufferSpawn = regexp.MustCompile(`^srv +load: spawning server instance with name=(.{1,256}) on port ([0-9]{1,5})$`)
var modelBufferExit = regexp.MustCompile(`^srv +run: instance name=(.{1,256}) exited with status -?[0-9]+$`)
var modelBufferLoading = regexp.MustCompile(`^srv +load_model: loading model '.+'$`)
var modelBufferReady = regexp.MustCompile(`^srv +main: model loaded$`)
var modelBufferSleep = regexp.MustCompile(`^srv +handle_sleep: server is (entering|exiting) sleeping state$`)

type modelBufferFrozen struct {
	input                                                 modelEstimateInput
	options                                               modelconfig.RequestedLoad
	requested                                             modelconfig.RequestedLoad
	optionUnknowns                                        []string
	artifacts                                             []modelconfig.ArtifactSummary
	optionsID, inputID                                    string
	configurationKey                                      string
	context, slots                                        int32
	baseline                                              []modelconfig.BufferBaseline
	baselineDone                                          bool
	captureBudget                                         bool
	topologyID, hostPoolID, devicePoolID, deviceBindingID string
	hostValidUntil, deviceValidUntil                      time.Time
}

type modelBufferChild struct {
	model                              string // exact prepared name, never serialized
	port                               int
	generation                         uint64
	phase                              string
	parser                             modelBufferParser
	loading, ready, sleeping, poisoned bool
	stream                             string
	sequence                           [2]uint64
	sample                             modelconfig.BufferObservation
	checkpoint                         *modelBufferCheckpoint
	startedAt                          time.Time
}

// Independent observational currentness; no ServingConfig is required.
type modelBufferCapture struct {
	mu                                   sync.Mutex
	owner                                *modelBufferOwner
	token, engineID, managerID, sourceID string
	prepared                             *llamaServingLaunch
	proc                                 *managedProc
	run                                  string
	retired                              bool
	retiredFlag                          atomic.Bool
	released                             atomic.Bool
	jobs                                 atomic.Int64
	childCharge                          int
	sequence                             [2]uint64
	lastHash                             [2]string
	children                             map[int]*modelBufferChild
	usedPorts                            [1024]uint64
	frozen                               map[string]*modelBufferFrozen
	charge                               int
	lastSweep                            uint64
	boundary                             uint64
	confirmedAt                          time.Time
	wallHigh                             int64
	clockUncertain                       bool
	budgetOmitted                        bool
	unsupportedLogged                    bool // guarded by mu; one warning per capture
}

type modelBufferSourceDiagnostic struct {
	stage string
	err   error
}

func (d modelBufferSourceDiagnostic) warn() {
	slog.Warn("model buffer capture unavailable", "stage", d.stage, "reason", modelBufferDiagnosticError(d.err))
}

// These errors come only from file/source validation, never native output.
// Structured filesystem errors can contain relative paths without separators.
func modelBufferDiagnosticError(err error) string {
	var pathErr *os.PathError
	var linkErr *os.LinkError
	text := err.Error()
	var fieldErr *ServingFieldError
	if errors.As(err, &fieldErr) && fieldErr.Field == "serving" && fieldErr.ModelID == "" {
		text = fieldErr.Reason
	}
	// Fixed validator messages use slashes as punctuation, not paths.
	if text == "native library snapshot bound/cancellation" || text == "unsupported native injection/MIG/MPS environment" {
		return text
	}
	if errors.As(err, &pathErr) || errors.As(err, &linkErr) || strings.ContainsAny(text, "/\\:\"'\r\n\x00") {
		return fmt.Sprintf("%T", err)
	}
	return text
}

func modelBufferDiagnosticBuild(build string) string {
	if len(build) > 256 || strings.ContainsAny(build, "/\\:\"'\r\n\x00") {
		return "redacted"
	}
	return build
}

// The worker owns all estimation and disk publication. One coalescing pending
// checkpoint bounds pressure without ever waiting for disk in an output pump.
type modelBufferOwner struct {
	mu              sync.Mutex
	store           *modelBufferStore
	pending         *modelBufferCheckpoint
	running         bool
	done            chan struct{}
	closed          bool
	bytes, children int
}
type modelBufferCheckpoint struct {
	capture *modelBufferCapture
	port    int
	model   string
	frozen  *modelBufferFrozen
	sample  modelBufferStoredSample
	omitted atomic.Bool
}

func modelBufferHash(value any) string {
	data, err := json.Marshal(value)
	if err != nil {
		return ""
	}
	return modelLoadDigest(data)
}

func (e *Executor) newModelBufferCapture(st *engineState, engine string, prepared *llamaServingLaunch, inputs map[string]modelEstimateInput, command launchCommand, contexts ...context.Context) *modelBufferCapture {
	if st.manifest.FamilyOrDefault() != "llamacpp" {
		return nil
	}
	if prepared == nil {
		slog.Warn("model buffer capture unavailable", "stage", "capture-create", "reason", "unprepared")
		return nil
	}
	loads := modelBufferPreparedLoads(prepared)
	if len(loads) == 0 {
		slog.Warn("model buffer capture unavailable", "stage", "capture-create", "reason", "no-loads")
		return nil
	}
	c := &modelBufferCapture{owner: &e.modelBuffers, token: modelBufferHash(newOpID()), engineID: modelBufferHash(engine), managerID: modelBufferHash(e.switchIncarnation), prepared: prepared, children: map[int]*modelBufferChild{}, frozen: map[string]*modelBufferFrozen{}, charge: 16 << 10, wallHigh: time.Now().UnixNano()}
	c.owner.mu.Lock()
	if c.owner.closed || c.charge > modelBufferCaptureBytes-c.owner.bytes {
		c.owner.mu.Unlock()
		slog.Warn("model buffer capture unavailable", "stage", "capture-create", "reason", "budget")
		return nil
	}
	c.owner.bytes += c.charge
	c.owner.mu.Unlock()
	accepted := false
	defer func() {
		if !accepted {
			c.retire()
		}
	}()
	// File-only verification, never native enumeration or helper execution.
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	command, constrainErr := modelConstrainCommand(command, prepared.ModelLoads)
	var diagnostic modelBufferSourceDiagnostic
	source, err := modelReviewedNativeSource(ctx, st, command, &diagnostic)
	cancel()
	if constrainErr != nil {
		(modelBufferSourceDiagnostic{stage: "constrain-command", err: constrainErr}).warn()
	} else if err != nil {
		diagnostic.warn()
	}
	if err == nil && constrainErr == nil {
		c.sourceID = source
	}
	if len(prepared.ModelLoads) == 0 && c.sourceID == "" {
		return nil
	}
	var memory *modelEstimateMemory
	if len(contexts) > 0 {
		memory = modelBufferExistingMemory(contexts[0])
	}
	actualUUID, _ := modelReviewedUUID(prepared.ModelLoads)
	memoryCharge := 0
	if memory != nil {
		encoded, _ := json.Marshal(memory)
		memoryCharge = len(encoded) * 4
	}
	for _, load := range loads {
		if len(c.frozen) >= 64 {
			c.budgetOmitted = true
			break
		}
		in, ok := inputs[load.ModelID]
		if !ok || in.Allocation == nil {
			if load.Resolved == nil {
				continue
			}
			allocation := load.Resolved.Clone()
			in = modelEstimateInput{Allocation: &allocation, Bindings: modelEstimateBindings(allocation.Artifacts), BuildID: allocation.NativeBuildID, Support: modelEstimatorSupport}
		}
		in.Memory = memory // only the already available operation snapshot
		advisoryID := modelBufferHash(in)
		inputOmitted := false
		// The decoder exposes a detached bounded clone. No compiler-owned
		// mutable map/slice survives this handoff and no path enters history.
		charge := int(in.Facts.Usage.RetainedBytes)*4 + 8192 + len(in.Bindings)*512
		charge += memoryCharge
		c.owner.mu.Lock()
		available := modelBufferCaptureBytes - c.owner.bytes
		if charge < 0 || charge > available-modelBufferChildCharge {
			inputOmitted = true
			in.Facts.Shards = nil
			in.Facts.Usage.RetainedBytes = 0
			charge = 8192 + len(in.Bindings)*512 + memoryCharge
			if charge > available {
				c.budgetOmitted = true
				c.owner.mu.Unlock()
				continue
			}
		}
		c.charge += charge
		c.owner.bytes += charge
		c.owner.mu.Unlock()
		in.Facts = in.Facts.Clone()
		in.Bindings = slices.Clone(in.Bindings)
		allocation := in.Allocation.Clone()
		in.Allocation = &allocation
		if in.SWAFull != nil {
			value := *in.SWAFull
			in.SWAFull = &value
		}
		f := &modelBufferFrozen{input: in, options: allocation.Options.Clone(), context: allocation.PerSlotContext, slots: allocation.Slots}
		if inputOmitted {
			f.baselineDone = true
			for _, name := range modelconfig.MemoryComponentNames {
				f.baseline = append(f.baseline, modelconfig.BufferBaseline{Component: name, Status: "unknown"})
			}
		}
		f.requested = load.Requested.Clone()
		modelBufferFreezeTopology(f, memory, actualUUID)
		if inputOmitted {
			f.input = modelEstimateInput{}
		}
		f.options.Context = &modelconfig.Context{Mode: "fixed", Tokens: f.context}
		slots := f.slots
		f.options.Slots = &slots
		placement := allocation.Clone().GPU
		f.options.GPU = &placement
		f.optionUnknowns = []string{"mmap", "load-mode", "speculation"}
		if f.options.FlashAttention == nil || *f.options.FlashAttention == "inherit" {
			f.optionUnknowns = append(f.optionUnknowns, "flash-attention")
		}
		if f.options.BatchSize == nil {
			f.optionUnknowns = append(f.optionUnknowns, "batch")
		}
		if f.options.UBatchSize == nil {
			f.optionUnknowns = append(f.optionUnknowns, "ubatch")
		}
		if f.options.CacheTypeK == nil {
			f.optionUnknowns = append(f.optionUnknowns, "cache-k")
		}
		if f.options.CacheTypeV == nil {
			f.optionUnknowns = append(f.optionUnknowns, "cache-v")
		}
		if f.options.KVOffload == nil {
			f.optionUnknowns = append(f.optionUnknowns, "kv-offload")
		}
		if f.options.OpOffload == nil {
			f.optionUnknowns = append(f.optionUnknowns, "op-offload")
		}
		if f.options.GPU == nil || f.options.GPU.Mode == "native-default" {
			f.optionUnknowns = append(f.optionUnknowns, "placement")
		}
		for _, a := range in.Bindings {
			a.Assurance = "sampled-local-binding"
			f.artifacts = append(f.artifacts, a)
		}
		f.optionsID = modelBufferHash(struct {
			Options modelconfig.RequestedLoad
			Preset  string
		}{f.options, modelLoadDigest(prepared.Snapshot)})
		f.configurationKey = modelConfigurationKey(in, c.sourceID, f.optionsID)
		f.inputID = modelBufferHash(struct {
			Advisory, Source, Options, Revision, Topology string
		}{advisoryID, c.sourceID, f.optionsID, modelEstimatorRevision, f.topologyID})
		c.frozen[load.ModelID] = f
	}
	c.owner.mu.Lock()
	defer c.owner.mu.Unlock()
	if c.owner.closed {
		return nil
	}
	accepted = true
	return c
}

func (c *modelBufferCapture) bind(proc *managedProc) {
	if c == nil {
		return
	}
	c.mu.Lock()
	c.proc = proc
	c.mu.Unlock()
}

// Called with st.mu held at the existing readiness publication boundary.
func (c *modelBufferCapture) publishRun(run string) {
	if c == nil {
		return
	}
	c.mu.Lock()
	if !c.retired {
		c.run = run
	}
	c.mu.Unlock()
}
func (c *modelBufferCapture) retire() {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.retired {
		return
	}
	c.retired = true
	c.retiredFlag.Store(true)
	for _, child := range c.children {
		child.sample.Current = false
	}
	// Completed checkpoints are owned by their publication jobs/history.
	// Retiring a run must not retain another unbounded in-memory history.
	c.children = nil
	c.frozen = nil
	c.owner.mu.Lock()
	c.owner.releaseLocked(c)
	c.owner.mu.Unlock()
}

// The owner lock also accounts checkpoints held by the worker. A retired
// capture cannot free capacity still retained by a queued or publishing job.
func (o *modelBufferOwner) releaseLocked(c *modelBufferCapture) {
	if c.retiredFlag.Load() && c.jobs.Load() == 0 && c.released.CompareAndSwap(false, true) {
		o.bytes -= c.charge
		o.children -= c.childCharge
	}
}

func modelBufferStream(stream string) int {
	if stream == "stdout" {
		return 0
	}
	if stream == "stderr" {
		return 1
	}
	return -1
}

func (c *modelBufferCapture) observe(frame managedOutputFrame) {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.retired {
		return
	}
	i := modelBufferStream(frame.Stream)
	if i < 0 {
		c.poison()
		return
	}
	hash := modelBufferHash(struct{ Line, Failure string }{frame.Line, frame.Failure})
	if frame.Sequence == c.sequence[i] && hash == c.lastHash[i] {
		return
	}
	invalid := frame.Sequence == 0 || frame.Sequence != c.sequence[i]+1 || frame.Failure != ""
	if invalid {
		c.poison()
	}
	if frame.Sequence > c.sequence[i] {
		c.sequence[i], c.lastHash[i] = frame.Sequence, hash
	}
	if invalid {
		return
	}
	line, ok := modelBufferMessage(frame.Line)
	if !ok {
		c.poison()
		return
	}
	if match := modelBufferSpawn.FindStringSubmatch(line); match != nil {
		c.boundary++
		port, err := strconv.Atoi(match[2])
		if err != nil || port <= 0 || port > 65535 || match[2] != strconv.Itoa(port) {
			c.poison()
			return
		}
		if c.usedPorts[port/64]&(1<<uint(port%64)) != 0 {
			if old := c.children[port]; old != nil {
				old.poisoned = true
				old.sample.Current = false
			}
			return
		}
		if c.frozen[match[1]] == nil {
			return
		}
		for _, old := range c.children {
			if old.model == match[1] && !old.sleeping {
				old.poisoned = true
				old.sample.Current = false
				return
			}
		}
		c.owner.mu.Lock()
		if c.owner.children >= 64 || c.owner.bytes > modelBufferCaptureBytes-modelBufferChildCharge {
			c.owner.mu.Unlock()
			c.frozen[match[1]].captureBudget = true
			c.usedPorts[port/64] |= 1 << uint(port%64)
			return
		}
		c.owner.children++
		c.childCharge++
		c.owner.bytes += modelBufferChildCharge
		c.charge += modelBufferChildCharge
		c.owner.mu.Unlock()
		c.usedPorts[port/64] |= 1 << uint(port%64)
		c.children[port] = &modelBufferChild{model: strings.Clone(match[1]), port: port, generation: 1, phase: "initial"}
		c.frozen[match[1]].captureBudget = false
		return
	}
	if match := modelBufferExit.FindStringSubmatch(line); match != nil {
		c.boundary++
		for port, child := range c.children {
			if child.model == match[1] {
				child.sample.Current = false
				delete(c.children, port)
				c.owner.mu.Lock()
				c.owner.children--
				c.childCharge--
				c.owner.bytes -= modelBufferChildCharge
				c.charge -= modelBufferChildCharge
				c.owner.mu.Unlock()
			}
		}
		return
	}
	match := modelBufferPort.FindStringSubmatch(line)
	if match == nil {
		return
	}
	port, _ := strconv.Atoi(match[1])
	if port <= 0 || port > 65535 || !strings.HasPrefix(line, fmt.Sprintf("[%5d] ", port)) {
		c.poison()
		return
	}
	child := c.children[port]
	if child == nil {
		// Cross-stream output preceding its parent cannot establish ordering.
		// A later spawn on this port is deliberately unsupported for this run.
		if port > 0 && port < 65536 {
			c.usedPorts[port/64] |= 1 << uint(port%64)
		} else {
			c.poison()
		}
		return
	}
	// The pinned router combines each child's pipes, then forwards complete
	// frames on one parent stream. Cross-parent-stream movement is ambiguous.
	if child.stream == "" {
		child.stream = frame.Stream
	}
	if frame.Stream != child.stream {
		child.poisoned = true
		child.sample.Current = false
		return
	}
	message, ok := modelBufferMessage(match[2])
	if !ok {
		child.poisoned = true
		return
	}
	if sleep := modelBufferSleep.FindStringSubmatch(message); sleep != nil {
		c.boundary++
		child.sample.Current = false
		if sleep[1] == "entering" {
			child.sleeping = true
			return
		}
		if !child.sleeping {
			child.poisoned = true
			return
		}
		child.generation++
		child.phase = "wakeup"
		child.parser = modelBufferParser{}
		child.sequence = [2]uint64{}
		child.loading = false
		child.ready = false
		child.sleeping = false
		child.sample = modelconfig.BufferObservation{}
		child.checkpoint = nil
		return
	}
	if modelBufferLoading.MatchString(message) {
		c.boundary++
		if child.loading {
			child.poisoned = true
			child.sample.Current = false
			return
		}
		child.loading = true
		child.startedAt = time.Now()
		return
	}
	if modelBufferReady.MatchString(message) {
		c.boundary++
		if !child.loading || child.ready || child.phase != "initial" {
			child.poisoned = true
		} else {
			child.ready = true
		}
		return
	}
	if !child.loading || child.sleeping || child.ready {
		return
	}
	child.sequence[i]++
	beforeRows := len(child.parser.rows)
	beforeReserve := child.parser.inReserve
	child.parser.consume(managedOutputFrame{Stream: frame.Stream, Sequence: child.sequence[i], Line: message}, frame.Sequence)
	if beforeRows != len(child.parser.rows) || beforeReserve != child.parser.inReserve {
		c.boundary++
	}
}

func (c *modelBufferCapture) poison() {
	for _, child := range c.children {
		child.poisoned = true
		child.sample.Current = false
	}
}

// Called with st.mu held. No disk or native operation and no declaration path.
func (c *modelBufferCapture) currentLocked(st *engineState) bool {
	if c == nil || c.proc == nil || c.retiredFlag.Load() {
		return false
	}
	select {
	case <-c.proc.done:
		return false
	default:
	}
	return c != nil && st.bufferCapture == c && st.running && !st.stopping && !st.adopted && !st.reclaimFailed && st.reclaimPID == 0 && st.proc == c.proc && st.runID == c.run && c.run != "" && st.plat.Runtime.servingLaunch == c.prepared && modelconfig.EqualLoads(st.plat.Runtime.ModelLoads, c.prepared.ModelLoads) && (len(c.prepared.ModelLoads) > 0 || st.plat.Runtime.Serving.Equal(c.prepared.Config))
}

func (e *Executor) observeModelBuffers(ctx context.Context, st *engineState, c *modelBufferCapture, sweep uint64, loaded []string, known bool, raw json.RawMessage, props map[string]modelBufferProps, boundary ...uint64) {
	if c == nil {
		return
	}
	st.mu.Lock()
	current := c.currentLocked(st)
	command := st.plat.Runtime.servingCommand
	st.mu.Unlock()
	if !current {
		return
	}
	if !known || len(loaded) == 0 || !llamaLoadedSemantics(raw, loaded) || ctx.Err() != nil {
		c.mu.Lock()
		if sweep >= c.lastSweep {
			c.lastSweep = sweep
			for _, child := range c.children {
				child.sample.Current = false
			}
		}
		c.mu.Unlock()
		return
	}
	command, constrainErr := modelConstrainCommand(command, c.prepared.ModelLoads)
	sourceCtx, cancelSource := context.WithTimeout(ctx, 200*time.Millisecond)
	source, err := modelReviewedNativeSource(sourceCtx, st, command)
	cancelSource()
	bindingsCurrent := modelBufferBindingsCurrent(ctx, modelBufferPreparedLoads(c.prepared))
	st.mu.Lock()
	stillCurrent := c.currentLocked(st)
	st.mu.Unlock()
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.retired || sweep < c.lastSweep {
		return
	}
	c.lastSweep = sweep
	if wall := time.Now().UnixNano(); wall < c.wallHigh {
		c.clockUncertain = true
	} else {
		c.wallHigh = wall
	}
	for _, child := range c.children {
		child.sample.Current = false
	}
	if len(boundary) > 0 && boundary[0] != c.boundary {
		return
	}
	if c.clockUncertain || !stillCurrent || !bindingsCurrent || err != nil || constrainErr != nil || source == "" || source != c.sourceID || !known || !llamaLoadedSemantics(raw, loaded) || ctx.Err() != nil {
		if stillCurrent && !c.clockUncertain && ctx.Err() == nil {
			reason := "source-unavailable"
			if source != "" && source != c.sourceID {
				reason = "source-changed"
			}
			for _, child := range c.children {
				child.sample = modelconfig.BufferObservation{Status: "unknown", Reason: reason}
			}
		}
		return
	}
	c.confirmedAt = time.Now()
	for port, child := range c.children {
		f := c.frozen[child.model]
		p, ok := props[child.model]
		if !ok || !slices.Contains(loaded, child.model) || child.sleeping || child.poisoned || !child.loading || child.phase == "initial" && !child.ready {
			continue
		}
		if !modelBufferBuildSupported(p.Build) || p.Build != p.RouterBuild || p.Context != int64(f.context) || p.Slots != int64(f.slots) {
			if !c.unsupportedLogged {
				c.unsupportedLogged = true
				switch {
				case !modelBufferBuildSupported(p.Build):
					slog.Warn("model buffer capture unavailable", "stage", "build-support", "build", modelBufferDiagnosticBuild(p.Build))
				case p.Build != p.RouterBuild:
					slog.Warn("model buffer capture unavailable", "stage", "router-build", "build", modelBufferDiagnosticBuild(p.Build), "router_build", modelBufferDiagnosticBuild(p.RouterBuild))
				case p.Context != int64(f.context):
					slog.Warn("model buffer capture unavailable", "stage", "context", "context", p.Context, "frozen_context", f.context)
				case p.Slots != int64(f.slots):
					slog.Warn("model buffer capture unavailable", "stage", "slots", "slots", p.Slots, "frozen_slots", f.slots)
				}
			}
			child.sample = modelconfig.BufferObservation{Status: "unknown", Reason: "unsupported-build"}
			child.poisoned = true
			continue
		}
		if child.parser.inReserve || !child.parser.reserveComplete || len(child.parser.rows) == 0 {
			continue
		}
		if child.sample.SampleID != "" {
			child.sample.Current = child.sample.Status != "unknown"
			if child.sample.Persistence == "failed" || child.checkpoint != nil && child.checkpoint.omitted.Load() {
				child.sample.Persistence = "pending"
				job := &modelBufferCheckpoint{capture: c, port: port, model: child.model, frozen: f, sample: modelBufferStoredSample{ModelIdentity: modelBufferHash(struct{ Engine, Model string }{c.engineID, child.model}), Observation: child.sample.Clone()}}
				child.checkpoint = job
				e.modelBuffers.enqueue(e, job)
			}
			continue
		}
		o := modelconfig.BufferObservation{Status: "partial", Reason: "unknown-baseline", RecordedAt: child.startedAt.UTC().Format(time.RFC3339Nano), Current: true, Persistence: "pending", UpstreamPin: modelBufferPin, BuildAssurance: "recognized-self-reported-pin", EstimatorRevision: modelEstimatorRevision, InputIdentity: f.inputID, SourceIdentity: c.sourceID, OptionsIdentity: f.optionsID, LoadPhase: child.phase, PerSlotContext: f.context, Slots: f.slots, CapturedOptions: &f.options, Artifacts: slices.Clone(f.artifacts), Rows: slices.Clone(child.parser.rows)}
		o.SampleID = modelBufferHash(struct {
			Manager, Token, Engine, Run, Model, Input, Phase string
			Port                                             int
			Generation                                       uint64
		}{c.managerID, c.token, c.engineID, modelBufferHash(c.run), modelBufferHash(child.model), f.inputID, child.phase, port, child.generation})
		if len(c.prepared.ModelLoads) > 0 {
			o.RequestedOptions = &f.requested
		}
		o.OptionUnknowns = slices.Clone(f.optionUnknowns)
		o.TopologyIdentity = f.topologyID
		o.ConfigurationKey = f.configurationKey
		for i := range o.Rows {
			o.Rows[i].Stream = child.stream
			switch o.Rows[i].Label {
			case "CPU", "CPU_Mapped", "CUDA_Host":
				if !f.hostValidUntil.IsZero() && !child.startedAt.After(f.hostValidUntil) {
					o.Rows[i].PoolIdentity = f.hostPoolID
				}
			case "CUDA0":
				if !f.deviceValidUntil.IsZero() && !child.startedAt.After(f.deviceValidUntil) {
					o.Rows[i].PoolIdentity = f.devicePoolID
					o.Rows[i].DeviceBinding = f.deviceBindingID
				}
			}
			if !slices.ContainsFunc(f.artifacts, func(a modelconfig.ArtifactSummary) bool { return a.Role != "model" }) {
				o.Rows[i].Role = "model"
			} else {
				o.Rows[i].Role = "mixed"
			}
		}
		if !slices.ContainsFunc(o.Rows, func(row modelconfig.BufferReportRow) bool { return row.PoolIdentity != "" }) {
			o.TopologyIdentity = ""
		}
		if child.parser.partial {
			o.Reasons = slices.Clone(child.parser.reasons)
			if len(o.Reasons) == 0 {
				o.Reasons = append(o.Reasons, "missing-boundary")
			}
		}
		child.sample = o
		if child.phase == "wakeup" {
			child.ready = true
		}
		job := &modelBufferCheckpoint{capture: c, port: port, model: child.model, frozen: f, sample: modelBufferStoredSample{ModelIdentity: modelBufferHash(struct{ Engine, Model string }{c.engineID, child.model}), Observation: o.Clone()}}
		child.checkpoint = job
		e.modelBuffers.enqueue(e, job)
	}
}

func (o *modelBufferOwner) enqueue(e *Executor, job *modelBufferCheckpoint) {
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.closed {
		job.omitted.Store(true)
		return
	}
	if o.pending != nil && o.pending.sample.Observation.SampleID != job.sample.Observation.SampleID {
		pendingAt, _ := time.Parse(time.RFC3339Nano, o.pending.sample.Observation.RecordedAt)
		at, _ := time.Parse(time.RFC3339Nano, job.sample.Observation.RecordedAt)
		if at.Before(pendingAt) || at.Equal(pendingAt) && job.sample.Observation.SampleID < o.pending.sample.Observation.SampleID {
			job.omitted.Store(true)
			return
		}
	}
	job.capture.jobs.Add(1)
	if o.pending != nil && o.pending.sample.Observation.SampleID != job.sample.Observation.SampleID {
		o.pending.omitted.Store(true)
	}
	if o.pending != nil {
		old := o.pending.capture
		old.jobs.Add(-1)
		o.releaseLocked(old)
	}
	o.pending = job
	if !o.running {
		o.running = true
		o.done = make(chan struct{})
		go o.work(e, o.done)
	}
}

func (o *modelBufferOwner) work(e *Executor, done chan struct{}) {
	defer close(done)
	for {
		o.mu.Lock()
		job := o.pending
		o.pending = nil
		if job == nil {
			o.running = false
			o.mu.Unlock()
			return
		}
		o.mu.Unlock()
		c := job.capture
		c.mu.Lock()
		f := job.frozen
		input := f.input
		baselineDone := f.baselineDone
		baseline := slices.Clone(f.baseline)
		c.mu.Unlock()
		if !baselineDone {
			ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
			estimate := e.modelEstimates.estimate(ctx, input)
			cancel()
			if len(estimate.Components) <= 64 {
				baseline = modelBufferBaseline(estimate.Components)
			}
			if len(baseline) == 0 {
				for _, name := range modelconfig.MemoryComponentNames {
					baseline = append(baseline, modelconfig.BufferBaseline{Component: name, Status: "unknown"})
				}
			}
			c.mu.Lock()
			f.baseline = baseline
			f.baselineDone = true
			f.input = modelEstimateInput{}
			c.mu.Unlock()
		}
		job.sample.Observation.Baseline = baseline
		job.sample.Observation.Comparisons = modelBufferComparisons(job.sample.Observation)
		job.sample.Observation.Reason = "unmapped-device"
		modelBufferBoundSample(&job.sample)
		status := "history-unavailable"
		if o.store != nil {
			status = o.store.publish(job.sample)
		}
		c.mu.Lock()
		child := c.children[job.port]
		if child != nil && child.sample.SampleID == job.sample.Observation.SampleID {
			current := child.sample.Current && !c.retired && job.sample.Observation.Status != "unknown"
			child.sample = job.sample.Observation.Clone()
			child.sample.Current = current
			child.sample.Persistence = status
		}
		c.mu.Unlock()
		o.mu.Lock()
		c.jobs.Add(-1)
		o.releaseLocked(c)
		o.mu.Unlock()
	}
}

// A captured baseline is keyed by category, independent of pool placement.
func modelBufferBaseline(components []modelconfig.EstimateComponent) []modelconfig.BufferBaseline {
	var out []modelconfig.BufferBaseline
	rank := map[string]int{"excluded": 0, "estimated": 1, "partial": 2, "unknown": 3}
	for _, c := range components {
		i := slices.IndexFunc(out, func(b modelconfig.BufferBaseline) bool { return b.Component == c.Name })
		if i < 0 {
			out = append(out, modelconfig.BufferBaseline{Component: c.Name, Status: c.Status, Bytes: c.Bytes})
			continue
		}
		b := &out[i]
		if rank[c.Status] > rank[b.Status] {
			b.Status = c.Status
		}
		if b.Status == "unknown" {
			b.Bytes = ""
			continue
		}
		if c.Bytes == "" {
			continue
		}
		x, ex := strconv.ParseUint(b.Bytes, 10, 64)
		if b.Bytes == "" {
			x, ex = 0, nil
		}
		y, ey := strconv.ParseUint(c.Bytes, 10, 64)
		if ex != nil || ey != nil || y > ^uint64(0)-x {
			b.Status, b.Bytes = "unknown", ""
			continue
		}
		b.Bytes = strconv.FormatUint(x+y, 10)
	}
	return out
}

func modelBufferBoundSample(sample *modelBufferStoredSample) {
	o := &sample.Observation
	for {
		encoded, err := json.Marshal(sample)
		if err == nil && len(encoded) <= 64<<10 {
			return
		}
		if len(o.Rows) == 0 {
			*o = modelconfig.BufferObservation{Status: "unknown", Reason: "capture-budget", SampleID: o.SampleID, RecordedAt: o.RecordedAt, InputIdentity: o.InputIdentity, SourceIdentity: o.SourceIdentity}
			return
		}
		o.Rows = o.Rows[:len(o.Rows)-1]
		o.Status, o.Reason = "partial", "capture-budget"
		for i := range o.Comparisons {
			o.Comparisons[i] = modelconfig.BufferComparison{Component: o.Comparisons[i].Component, Status: "unknown"}
		}
	}
}

func modelBufferComparisons(o modelconfig.BufferObservation) []modelconfig.BufferComparison {
	var out []modelconfig.BufferComparison
	for _, name := range []string{"weights", "attention-cache", "compute-workspace"} {
		comparison := modelconfig.BufferComparison{Component: name, Status: "unknown"}
		var baseline *modelconfig.BufferBaseline
		for i := range o.Baseline {
			if o.Baseline[i].Component == name {
				if baseline != nil {
					baseline = nil
					break
				}
				baseline = &o.Baseline[i]
			}
		}
		if baseline != nil {
			comparison.EstimatedBytes = baseline.Bytes
		}
		// Only the CPU weights scope has a complete supported correspondence
		// here. Compute remains partial in the pinned estimator; inherited KV
		// options and unmapped devices cannot support component comparisons.
		compatible := name == "weights" && baseline != nil && baseline.Status == "estimated" && baseline.Bytes != "" && o.TopologyIdentity != "" && o.CapturedOptions != nil && o.CapturedOptions.GPU != nil && o.CapturedOptions.GPU.Mode == "cpu" && len(o.Artifacts) > 0 && !slices.ContainsFunc(o.Artifacts, func(a modelconfig.ArtifactSummary) bool { return a.Role != "model" }) && len(o.Reasons) == 0
		lo, hi := new(big.Int), new(big.Int)
		found := false
		for _, row := range o.Rows {
			if row.Component != "model" {
				continue
			}
			found = true
			if row.Label != "CPU" || row.Role != "model" || row.PoolIdentity == "" || row.LowerBytes == "" || row.UpperBytes == "" {
				compatible = false
				continue
			}
			l, ok := new(big.Int).SetString(row.LowerBytes, 10)
			if !ok {
				compatible = false
				continue
			}
			h, ok := new(big.Int).SetString(row.UpperBytes, 10)
			if !ok {
				compatible = false
				continue
			}
			lo.Add(lo, l)
			hi.Add(hi, h)
		}
		if compatible && found {
			b, ok := new(big.Int).SetString(baseline.Bytes, 10)
			if ok {
				comparison.Status = "compatible"
				comparison.LowerBytes = lo.String()
				comparison.UpperBytes = hi.String()
				comparison.DeltaLowerBytes = new(big.Int).Sub(lo, b).String()
				comparison.DeltaUpperBytes = new(big.Int).Sub(hi, b).String()
			}
		}
		out = append(out, comparison)
	}
	return out
}

// Read only an operation's already constructed private snapshot. This does
// not call the generic memory interface or a device runner, poll or helper.
func modelBufferExistingMemory(ctx context.Context) *modelEstimateMemory {
	op, _ := ctx.Value(modelAutoOperationKey{}).(modelAutoOperation)
	var memory *modelEstimateMemory
	switch owner := op.Memory.(type) {
	case modelAutoSnapshotMemory:
		memory = owner.memory
	case modelTargetMemoryOwner:
		memory, _ = owner.ModelAutoMemory(ctx)
	}
	if memory == nil || memory.Target == nil || memory.Target.Validate() != nil || memory.Target.At(noderec.MemoryBootClock) == nil {
		return nil
	}
	copy := *memory
	copy.Target = memory.Target.Clone()
	copy.Domains = slices.Clone(memory.Domains)
	for i := range copy.Domains {
		copy.Domains[i].Unknowns = slices.Clone(copy.Domains[i].Unknowns)
	}
	if memory.Map != nil {
		mapped := *memory.Map
		mapped.Devices = slices.Clone(memory.Map.Devices)
		if memory.Map.Binding != nil {
			binding := *memory.Map.Binding
			binding.Environment = nil
			mapped.Binding = &binding
		}
		copy.Map = &mapped
	}
	encoded, err := json.Marshal(copy)
	if err != nil || len(encoded) > 64<<10 {
		return nil
	}
	return &copy
}

func modelBufferFreezeTopology(f *modelBufferFrozen, memory *modelEstimateMemory, actualUUID string) {
	if memory == nil || memory.Target == nil {
		return
	}
	observed := memory.Target.At(noderec.MemoryBootClock)
	if observed == nil {
		return
	}
	host := ""
	hostRemaining := time.Duration(0)
	for _, pool := range observed.Domains {
		if pool.Kind == "system" || pool.Kind == "shared" {
			if host != "" {
				return
			}
			host = pool.ID
			hostRemaining = modelBufferPoolRemaining(pool)
		}
	}
	if host != "" && hostRemaining > 0 {
		f.hostPoolID = modelBufferHash(struct{ Host, Pool string }{memory.Target.HostID, host})
		f.hostValidUntil = time.Now().Add(hostRemaining)
	}
	mapID := ""
	if memory.Map != nil {
		mapID = memory.Map.ObservationID
	}
	f.topologyID = modelBufferHash(struct{ Host, Observation, Map, HostPool string }{memory.Target.HostID, memory.Target.ObservationID, mapID, host})
	if f.input.Allocation == nil || f.input.Allocation.DeviceBinding == nil || actualUUID == "" || memory.Map == nil || memory.Map.Unknown != "" || memory.Map.Binding == nil || memory.Target.MapID != mapID {
		return
	}
	a, b := f.input.Allocation.DeviceBinding, memory.Map.Binding
	if a.Validate() != nil || a.DeviceID != actualUUID || a.HostID != memory.Target.HostID || a.HostID != b.Host || a.DomainID != b.Pool || a.DeviceID != b.UUID || a.Selector != b.Selector || a.Digest != b.Digest {
		return
	}
	for _, pool := range observed.Domains {
		if pool.ID == b.Pool && pool.Kind == "dedicated" && slices.Contains(pool.DeviceIDs, b.UUID) {
			remaining := modelBufferPoolRemaining(pool)
			if remaining <= 0 {
				return
			}
			f.devicePoolID = modelBufferHash(struct{ Host, Pool string }{b.Host, b.Pool})
			f.deviceBindingID = modelBufferHash(struct{ Host, UUID, Selector, Digest string }{b.Host, b.UUID, b.Selector, b.Digest})
			f.deviceValidUntil = time.Now().Add(remaining)
			return
		}
	}
}

func modelBufferPoolRemaining(pool noderec.MemoryDomain) time.Duration {
	remaining := int64(0)
	for _, fact := range []noderec.MemoryFact{pool.Total, pool.Free} {
		if fact.Valid && fact.AgeMS >= 0 && fact.AgeMS <= noderec.MemoryFreshMS {
			remaining = max(remaining, noderec.MemoryFreshMS-fact.AgeMS)
		}
	}
	return time.Duration(remaining) * time.Millisecond
}

func (o *modelBufferOwner) close() {
	o.mu.Lock()
	o.closed = true
	done := o.done
	o.mu.Unlock()
	if done != nil {
		timer := time.NewTimer(time.Second)
		defer timer.Stop()
		select {
		case <-done:
		case <-timer.C:
		}
	}
}

func (e *Executor) observeModelBufferSweep(ctx context.Context, st *engineState, c *modelBufferCapture, sweep uint64, loaded []string, known bool, raw json.RawMessage, shared map[string]modelBufferProps, boundary ...uint64) {
	if c == nil {
		return
	}
	st.mu.Lock()
	current := c.currentLocked(st)
	port := st.port
	st.mu.Unlock()
	if !current {
		return
	}
	if c.sourceID == "" {
		return
	}
	if len(boundary) == 0 {
		boundary = []uint64{c.captureBoundary()}
	}
	c.mu.Lock()
	interested := false
	for _, child := range c.children {
		if slices.Contains(loaded, child.model) {
			interested = true
			break
		}
	}
	changed := c.boundary != boundary[0]
	c.mu.Unlock()
	if changed || !interested {
		e.clearModelBufferCurrent(c, sweep)
		return
	}
	if known && llamaLoadedSemantics(raw, loaded) && len(loaded) > 0 {
		_, router, routerErr := e.llamaServingPropsObserved(ctx, port, "")
		for _, model := range loaded {
			c.mu.Lock()
			frozen := c.frozen[model] != nil
			c.mu.Unlock()
			if !frozen {
				continue
			}
			p, ok := shared[model]
			if !ok {
				_, value, err := e.llamaServingPropsObserved(ctx, port, model)
				if err != nil {
					continue
				}
				p = value
			}
			if routerErr == nil {
				p.RouterBuild = router.Build
			}
			shared[model] = p
		}
	}
	e.observeModelBuffers(ctx, st, c, sweep, loaded, known, raw, shared, boundary...)
}

func (e *Executor) clearModelBufferCurrent(c *modelBufferCapture, sweep uint64) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if sweep < c.lastSweep {
		return
	}
	c.lastSweep = sweep
	for _, child := range c.children {
		child.sample.Current = false
	}
}

func (c *modelBufferCapture) captureBoundary() uint64 {
	if c == nil {
		return 0
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.boundary
}

func (e *Executor) projectModelBufferObservation(st *engineState, engine, model string, artifacts []modelconfig.ArtifactSummary) modelconfig.BufferObservation {
	identity := modelBufferHash(struct{ Engine, Model string }{modelBufferHash(engine), model})
	o := modelconfig.BufferObservation{Status: "unknown", Reason: "no-matching-sample"}
	if e.modelBuffers.store != nil {
		o = e.modelBuffers.store.newest(identity)
	}
	st.mu.Lock()
	c := st.bufferCapture
	current := c != nil && c.currentLocked(st)
	running := st.running
	st.mu.Unlock()
	if running && (c == nil || c.sourceID == "") {
		return modelconfig.BufferObservation{Status: "unknown", Reason: "unsupported-build"}
	}
	if current && o.SourceIdentity != "" && o.SourceIdentity != c.sourceID {
		o = modelconfig.BufferObservation{Status: "unknown", Reason: "source-changed"}
	}
	if c != nil {
		c.mu.Lock()
		if c.budgetOmitted && c.frozen[model] == nil {
			o = modelconfig.BufferObservation{Status: "unknown", Reason: "capture-budget"}
		}
		if f := c.frozen[model]; f != nil && f.captureBudget {
			o = modelconfig.BufferObservation{Status: "unknown", Reason: "capture-budget"}
		}
		if time.Now().UnixNano() < c.wallHigh {
			c.clockUncertain = true
		}
		for _, child := range c.children {
			if child.model == model && (child.sample.SampleID != "" || child.sample.Status == "unknown") {
				o = child.sample.Clone()
				if o.Persistence == "pending" {
					o = modelconfig.BufferObservation{Status: "unknown", Reason: "unknown-baseline", Persistence: "pending"}
				}
				if child.checkpoint != nil && child.checkpoint.omitted.Load() {
					o.Persistence = "pending"
					o.Reason = "capture-budget"
				}
				budget := e.loadedPollInterval + modelsTimeout
				if !current || c.retired || time.Now().Before(c.confirmedAt) || time.Since(c.confirmedAt) > budget {
					o.Current = false
				}
				if c.clockUncertain {
					o.Current = false
					o.Status = "partial"
					o.Reason = "clock-reversal"
				}
				break
			}
		}
		c.mu.Unlock()
	}
	if o.RecordedAt != "" {
		at, err := time.Parse(time.RFC3339Nano, o.RecordedAt)
		if err != nil || time.Since(at) > modelBufferHistoryAge {
			return modelconfig.BufferObservation{Status: "unknown", Reason: "no-matching-sample"}
		}
		if time.Now().Before(at) {
			o.Current = false
			o.Status = "partial"
			o.Reason = "clock-reversal"
		}
	}
	if len(o.Artifacts) > 0 && !reflect.DeepEqual(o.Artifacts, artifacts) {
		return modelconfig.BufferObservation{Status: "unknown", Reason: "source-changed"}
	}
	return o
}

func modelBufferBindingsCurrent(ctx context.Context, loads []modelconfig.ModelLoad) bool {
	count := 0
	for _, load := range loads {
		if load.Resolved == nil {
			return false
		}
		for _, a := range load.Resolved.Artifacts {
			count++
			if count > modelconfig.MaxArtifacts || ctx.Err() != nil {
				return false
			}
			b, err := (servingFileAccess{}).binding(ctx, a.Path)
			if err != nil || b.ResolvedTarget != a.ResolvedTarget || strconv.FormatInt(b.SizeBytes, 10) != a.Size || b.MtimeNS != a.MtimeNS || b.SampleSHA256 != a.SampleSHA256 {
				return false
			}
		}
	}
	return count > 0
}

// Adapt only an already compiled finite launch. No declaration becomes load
// authority here, and unavailable advisory facts remain an Unknown baseline.
// Native paths/roles come from the existing consumed-path owner, not a second
// flag table or allocation compiler.
func modelBufferPreparedLoads(prepared *llamaServingLaunch) []modelconfig.ModelLoad {
	if prepared == nil {
		return nil
	}
	if len(prepared.ModelLoads) > 0 {
		return prepared.ModelLoads
	}
	if prepared.Config == nil || len(prepared.Config.Models) > 64 {
		return nil
	}
	sections, err := parseLlamaServingPreset(prepared.Snapshot)
	if err != nil {
		return nil
	}
	var loads []modelconfig.ModelLoad
	count := 0
	for _, model := range prepared.Config.Models {
		var allocation *modelconfig.ResolvedLoad
		for _, envelope := range prepared.Envelopes {
			if envelope.ModelID != model.ModelID {
				continue
			}
			if envelope.PerSlotContextTokens <= 0 || envelope.PerSlotContextTokens > math.MaxInt32 || envelope.TotalContextTokens > math.MaxInt32 || envelope.Slots <= 0 || envelope.Slots > math.MaxInt32 {
				return nil
			}
			allocation = &modelconfig.ResolvedLoad{PerSlotContext: int32(envelope.PerSlotContextTokens), TotalContext: int32(envelope.TotalContextTokens), Slots: int32(envelope.Slots), GPU: modelconfig.GPU{Mode: "native-default"}}
		}
		if allocation == nil {
			return nil
		}
		options := maps.Clone(sections["*"])
		if options == nil {
			options = map[string]string{}
		}
		maps.Copy(options, sections[model.ModelID])
		paths, roles, err := llamaConsumedPaths(prepared.Source, options, model.ModelID, modelconfig.MaxArtifacts)
		if err != nil || len(paths) != len(model.Artifacts) {
			return nil
		}
		for i, artifact := range model.Artifacts {
			count++
			if count > modelconfig.MaxArtifacts || artifact.Binding == nil || artifact.Binding.SourcePath != paths[i] {
				return nil
			}
			b := artifact.Binding
			allocation.Artifacts = append(allocation.Artifacts, modelconfig.ArtifactBinding{Role: roles[i], Path: b.SourcePath, ResolvedTarget: b.ResolvedTarget, Size: strconv.FormatInt(b.SizeBytes, 10), MtimeNS: b.MtimeNS, SampleSHA256: b.SampleSHA256})
		}
		loads = append(loads, modelconfig.ModelLoad{ModelID: model.ModelID, Resolved: allocation})
	}
	return loads
}

// Measure the same standard descriptors and fallback estimate used by Inspect.
// Observation pressure can remove only this optional field, never original facts.
func attachModelBufferObservation(result modelconfig.InspectResult, o modelconfig.BufferObservation) modelconfig.InspectResult {
	original := result
	options, err := modelInspectionOptions(result.Options)
	if err != nil {
		return original
	}
	fit := func(value modelconfig.BufferObservation) bool {
		encoded, err := json.Marshal(value)
		if err != nil || len(encoded) > 4096 || value.Validate() != nil {
			return false
		}
		candidate := result.Clone()
		candidate.Options = options
		candidate.Estimate = modelEstimateUnknown("estimate omitted: complete inspection reply size limit")
		candidate.LoadObservation = &value
		encoded, err = json.Marshal(candidate)
		return err == nil && len(encoded) <= modelconfig.MaxPayloadBytes-1024
	}
	for {
		if fit(o) {
			result.LoadObservation = &o
			return result
		}
		if len(o.Rows) == 0 {
			break
		}
		o.Rows = o.Rows[:len(o.Rows)-1]
		o.Status = "partial"
		o.Reason = "response-budget"
		// A reduced allocation list has no component-level comparison coverage.
		for i := range o.Comparisons {
			o.Comparisons[i] = modelconfig.BufferComparison{Component: o.Comparisons[i].Component, Status: "unknown"}
		}
	}
	omitted := modelconfig.BufferObservation{Status: "unknown", Reason: "response-budget"}
	if fit(omitted) {
		result.LoadObservation = &omitted
		return result
	}
	return original
}
