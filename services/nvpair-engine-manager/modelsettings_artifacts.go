// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"nvpair-shared/gguffacts"
	"nvpair-shared/modelconfig"
)

// inspectionNumericJSON preserves scalar integer precision and rejects arrays,
// booleans, and non-finite floats rather than inventing a recommendation.
func inspectionNumericJSON(n gguffacts.Numeric) (json.RawMessage, bool) {
	if n.Array != nil {
		return nil, false
	}
	var value string
	switch n.Type {
	case 0:
		value = strconv.FormatUint(uint64(uint8(n.Bits)), 10)
	case 1:
		value = strconv.FormatInt(int64(int8(n.Bits)), 10)
	case 2:
		value = strconv.FormatUint(uint64(uint16(n.Bits)), 10)
	case 3:
		value = strconv.FormatInt(int64(int16(n.Bits)), 10)
	case 4:
		value = strconv.FormatUint(uint64(uint32(n.Bits)), 10)
	case 5:
		value = strconv.FormatInt(int64(int32(n.Bits)), 10)
	case 6:
		f := float64(math.Float32frombits(uint32(n.Bits)))
		if math.IsNaN(f) || math.IsInf(f, 0) {
			return nil, false
		}
		value = strconv.FormatFloat(f, 'g', -1, 32)
	case 10:
		value = strconv.FormatUint(n.Bits, 10)
	case 11:
		value = strconv.FormatInt(int64(n.Bits), 10)
	case 12:
		f := math.Float64frombits(n.Bits)
		if math.IsNaN(f) || math.IsInf(f, 0) {
			return nil, false
		}
		value = strconv.FormatFloat(f, 'g', -1, 64)
	default:
		return nil, false
	}
	return json.RawMessage(value), true
}

// mergeModelInspectionOptions overlays only matching descriptors so the
// standard option list remains authoritative and no duplicate fields are sent.
func mergeModelInspectionOptions(recommendations, standard []modelconfig.OptionDescriptor) []modelconfig.OptionDescriptor {
	byField := make(map[string]modelconfig.OptionDescriptor, len(recommendations))
	for _, option := range recommendations {
		if option.Default != nil && option.DefaultSource != "" {
			byField[option.Field] = option
		}
	}
	out := make([]modelconfig.OptionDescriptor, 0, len(standard))
	for _, option := range standard {
		option = option.Clone()
		if recommendation, ok := byField[option.Field]; ok && option.Supported && option.Owner == "request-policy" {
			field := strings.TrimPrefix(option.Field, "generation.")
			number := json.Number(recommendation.Default)
			if err := (modelconfig.PolicyPatch{Generation: map[string]*json.Number{field: &number}}).Validate(); err == nil {
				option.Default = append(json.RawMessage(nil), recommendation.Default...)
				option.DefaultSource = recommendation.DefaultSource
			}
		}
		out = append(out, option)
	}
	return out
}

func modelInspectionOptions(recommendations []modelconfig.OptionDescriptor) ([]modelconfig.OptionDescriptor, error) {
	options, err := llamaModelSettingsOptions()
	if err != nil {
		return nil, err
	}
	standard := make([]modelconfig.OptionDescriptor, 0, len(options))
	for _, option := range options {
		standard = append(standard, option.Descriptor.Clone())
	}
	standard = append(standard, modelSettingsExtraOptions()...)
	return mergeModelInspectionOptions(recommendations, standard), nil
}

// The artifact ceiling is not permission to exceed the existing reply ceiling.
// Reserve room for the current small unavailable advisory estimate; Task 10's
// richer estimate must retain the final RPC owner's complete-payload check.
func inspectionReplyFits(result modelconfig.InspectResult) (bool, error) {
	options, err := modelInspectionOptions(result.Options)
	if err != nil {
		return false, err
	}
	result.Options = options
	data, err := json.Marshal(result)
	if err != nil {
		return false, err
	}
	return len(data) <= modelconfig.MaxPayloadBytes-1024, nil
}

const modelInspectionCacheEntries = 8
const modelInspectionCacheBytes = 64 << 20

// Reserve the complete entry backing array even when only one slot is live.
const modelInspectionCacheSlots = modelInspectionCacheEntries * 256
const modelInspectionCacheTTL = 5 * time.Minute
const modelInspectionBudget = 5 * time.Second

var errModelArtifactLimit = errors.New("model artifact discovery limit")
var errModelInspectionBusy = errors.New("model artifact inspection busy")
var errModelInspectionChanged = errors.New("model artifact source changed")

type modelInspectionEntry struct {
	key, source string
	facts       gguffacts.Facts
	created     time.Time
	bytes       uint64
}

// The active flag owns the entire cache and instance seams. Its zero value is
// ready, starts no goroutine, and admits no queue. Entries contain no handles.
type modelInspectionState struct {
	active    atomic.Bool
	entries   []modelInspectionEntry // least recently used first
	bytes     uint64
	open      func(string) (servingArtifactFile, error)
	now       func() time.Time
	lastUsage servingReadUsage
}

func (s *modelInspectionState) clock() time.Time {
	if s.now != nil {
		return s.now()
	}
	return time.Now()
}

func (s *modelInspectionState) remove(i int) {
	s.bytes -= s.entries[i].bytes
	copy(s.entries[i:], s.entries[i+1:])
	s.entries[len(s.entries)-1] = modelInspectionEntry{}
	s.entries = s.entries[:len(s.entries)-1]
}

func (s *modelInspectionState) cached(key, source string) (gguffacts.Facts, bool) {
	now := s.clock()
	for i := 0; i < len(s.entries); {
		entry := s.entries[i]
		if now.Before(entry.created) || now.Sub(entry.created) >= modelInspectionCacheTTL || entry.source == source && entry.key != key {
			s.remove(i)
			continue
		}
		i++
	}
	for i, entry := range s.entries {
		if entry.key != key {
			continue
		}
		copy(s.entries[i:], s.entries[i+1:])
		s.entries[len(s.entries)-1] = entry
		return entry.facts.Clone(), true
	}
	return gguffacts.Facts{}, false
}

func (s *modelInspectionState) remember(key, source string, facts gguffacts.Facts) {
	// Include entry/string/slice overhead in addition to the decoder's charge
	// for names, maps, arrays and tensor capacities. Never trust a wire size.
	if facts.RetainedSize() > modelInspectionCacheBytes {
		return
	}
	size := facts.RetainedSize() + uint64(len(key)+len(source)) + 256
	if size > modelInspectionCacheBytes-modelInspectionCacheSlots {
		return
	}
	for len(s.entries) >= modelInspectionCacheEntries || s.bytes > modelInspectionCacheBytes-modelInspectionCacheSlots-size {
		s.remove(0)
	}
	if s.entries == nil {
		s.entries = make([]modelInspectionEntry, 0, modelInspectionCacheEntries)
	}
	s.entries = append(s.entries, modelInspectionEntry{key, source, facts.Clone(), s.clock(), size})
	s.bytes += size
}

// snapshotLlamaState runs under opMu and mu, and detaches every launch input
// read after unlocking. Runtime process/observation pointers are never inspected.
func snapshotLlamaState(st *engineState) *engineState {
	st.mu.Lock()
	defer st.mu.Unlock()
	plat := *st.plat
	rt := &plat.Runtime
	rt.Serving = rt.Serving.Clone()
	rt.ModelLoads = modelconfig.CloneLoads(rt.ModelLoads)
	rt.Args = slices.Clone(rt.Args)
	rt.Env = maps.Clone(rt.Env)
	rt.Start = slices.Clone(rt.Start)
	for i := range rt.Start {
		rt.Start[i] = slices.Clone(rt.Start[i])
	}
	if rt.LaunchArgs != nil {
		value := slices.Clone(*rt.LaunchArgs)
		rt.LaunchArgs = &value
	}
	if rt.LaunchEnv != nil {
		value := slices.Clone(*rt.LaunchEnv)
		rt.LaunchEnv = &value
	}
	if rt.EditableLaunch != nil {
		value := *rt.EditableLaunch
		value.FixedArgs = slices.Clone(value.FixedArgs)
		value.Controls = slices.Clone(value.Controls)
		for i := range value.Controls {
			control := &value.Controls[i]
			control.Flags, control.Env = slices.Clone(control.Flags), slices.Clone(control.Env)
			if control.Implicit != nil {
				v := *control.Implicit
				control.Implicit = &v
			}
		}
		rt.EditableLaunch = &value
	}
	return &engineState{manifest: st.manifest, plat: &plat, binPath: st.binPath, installDir: st.installDir, modelsDir: st.modelsDir, gen: st.gen}
}

type modelInspectionSnapshot struct {
	state   *engineState
	command launchCommand
	store   engineModelStore
}

func inspectionSnapshot(st *engineState, engine string) (modelInspectionSnapshot, error) {
	if !st.opMu.TryLock() {
		return modelInspectionSnapshot{}, errModelInspectionBusy
	}
	defer st.opMu.Unlock()
	snapshot := snapshotLlamaState(st)
	command, err := launchForState(snapshot, snapshot.plat.Runtime.Port)
	if err != nil {
		return modelInspectionSnapshot{}, err
	}
	store := engineModelStore{engine: engine, adapter: modelStoreAdapter(snapshot.manifest.FamilyOrDefault()), root: expandPath(os.Getenv("LLAMA_CACHE"))}
	for key, value := range command.Env {
		if environmentKey(key) == "LLAMA_CACHE" && strings.TrimSpace(value) != "" {
			store.root = expandPath(value)
		}
	}
	if store.root == "" {
		store.root = snapshot.modelsDir
	}
	if store.adapter != "llamacpp" || !filepath.IsAbs(store.root) {
		return modelInspectionSnapshot{}, errNoArtifactBinding
	}
	store.root = filepath.Clean(store.root)
	return modelInspectionSnapshot{snapshot, command, store}, nil
}

func (s modelInspectionSnapshot) equal(other modelInspectionSnapshot) bool {
	return s.store == other.store && s.state.gen == other.state.gen && s.state.manifest == other.state.manifest &&
		reflect.DeepEqual(s.command, other.command) && reflect.DeepEqual(s.state.plat.Runtime.EditableLaunch, other.state.plat.Runtime.EditableLaunch) &&
		s.state.plat.Runtime.Serving.Equal(other.state.plat.Runtime.Serving) && modelconfig.EqualLoads(s.state.plat.Runtime.ModelLoads, other.state.plat.Runtime.ModelLoads)
}

func artifactPathSafe(path string) bool {
	if !filepath.IsAbs(path) || len(path) > 4096 || strings.ContainsAny(path, "\x00\r\n") {
		return false
	}
	for _, part := range strings.FieldsFunc(filepath.ToSlash(path), func(r rune) bool { return r == '/' }) {
		if strings.HasSuffix(strings.ToLower(part), ".partial") {
			return false
		}
	}
	return true
}

func validateResolvedArtifacts(list modelFileList) error {
	if len(list.Files) == 0 {
		return errModelNotFound
	}
	if len(list.Files) > maxCopyArtifacts {
		return errModelArtifactLimit
	}
	seen := make(map[string]bool, len(list.Files))
	seenNames := make(map[string]bool, len(list.Files))
	previous := -1
	for i, f := range list.Files {
		role := slices.Index([]string{"model", "draft", "companion"}, f.Role)
		if role < previous || role < 0 || i == 0 && role != 0 {
			return errNoArtifactBinding
		}
		previous = role
		if !validModelRelPath(f.Path) || !artifactPathSafe(f.configured) || !strings.HasSuffix(f.configured, ".gguf") || !artifactPathSafe(f.abs) || f.info == nil || !f.info.Mode().IsRegular() || f.Size <= 0 {
			return errNoArtifactBinding
		}
		key := copyPathKey(f.abs)
		if seen[key] {
			return errNoArtifactBinding
		}
		seen[key] = true
		name := copyPathKey(f.Path)
		if seenNames[name] {
			return errNoArtifactBinding
		}
		seenNames[name] = true
		for _, prior := range list.Files[:i] {
			if os.SameFile(prior.info, f.info) {
				return errNoArtifactBinding
			}
		}
		parts := llamaShardName.FindStringSubmatch(f.configured)
		if parts == nil {
			continue
		}
		count, err := strconv.Atoi(parts[3])
		index, indexErr := strconv.Atoi(parts[2])
		if err != nil || indexErr != nil || count < 1 || count > maxCopyArtifacts || index < 1 || index > count {
			return errNoArtifactBinding
		}
		start := i - index + 1
		if start < 0 || start+count > len(list.Files) {
			return errModelNotFound
		}
		for shard := 0; shard < count; shard++ {
			want := fmt.Sprintf("%s-%05d-of-%05d.gguf", parts[1], shard+1, count)
			if list.Files[start+shard].configured != want || list.Files[start+shard].Role != f.Role {
				return errModelNotFound
			}
		}
	}
	return nil
}

// Task 6 passes stored target-local bindings through the same resolver. These
// are read assertions, never serving approval or a portable content identity.
func resolvedLoadArtifacts(ctx context.Context, store engineModelStore, model string, load modelconfig.ModelLoad, access servingFileAccess) (modelFileList, error) {
	if err := load.Validate(); err != nil {
		return modelFileList{}, errNoArtifactBinding
	}
	if len(load.Resolved.Artifacts) > maxCopyArtifacts {
		return modelFileList{}, errModelArtifactLimit
	}
	list := modelFileList{Engine: store.engine, Model: model, Layout: llamaArtifactsLayout}
	for _, artifact := range load.Resolved.Artifacts {
		if !artifactPathSafe(artifact.Path) || !strings.HasSuffix(artifact.Path, ".gguf") || !artifactPathSafe(artifact.ResolvedTarget) {
			return modelFileList{}, errNoArtifactBinding
		}
		binding, err := access.binding(ctx, artifact.Path)
		if err != nil {
			return modelFileList{}, err
		}
		if binding.ResolvedTarget != artifact.ResolvedTarget || strconv.FormatInt(binding.SizeBytes, 10) != artifact.Size || binding.MtimeNS != artifact.MtimeNS || binding.SampleSHA256 != artifact.SampleSHA256 {
			return modelFileList{}, errModelInspectionChanged
		}
		info, err := os.Stat(binding.ResolvedTarget)
		if err != nil {
			return modelFileList{}, err
		}
		list.Files = append(list.Files, modelFile{Role: artifact.Role, Path: artifact.Role + "/" + filepath.Base(artifact.Path), Size: binding.SizeBytes, abs: binding.ResolvedTarget, configured: binding.SourcePath, sample: binding.SampleSHA256, info: info})
	}
	if err := validateResolvedArtifacts(list); err != nil {
		return modelFileList{}, err
	}
	list.binding = exportBinding(list, load)
	return list, nil
}

type inspectionHeaderReader struct {
	ctx   context.Context
	file  servingArtifactFile
	usage *servingReadUsage
}

func (r inspectionHeaderReader) ReadAt(p []byte, offset int64) (int, error) {
	if err := r.ctx.Err(); err != nil {
		return 0, err
	}
	n, err := r.file.ReadAt(p, offset)
	r.usage.HeaderBytes += uint64(n)
	return n, err
}

func decodeInspectionFacts(ctx context.Context, list modelFileList, access servingFileAccess) (gguffacts.Facts, error) {
	if len(list.Files) == 0 || len(list.Files) > maxCopyArtifacts {
		return gguffacts.Facts{}, errModelArtifactLimit
	}
	if access.usage == nil {
		access.usage = &servingReadUsage{}
	}
	files := make([]servingArtifactFile, 0, len(list.Files))
	defer func() {
		for _, file := range files {
			_ = file.Close()
		}
	}()
	artifacts := make([]gguffacts.Artifact, 0, len(list.Files))
	for _, f := range list.Files {
		if err := ctx.Err(); err != nil {
			return gguffacts.Facts{}, err
		}
		file, err := access.openFile(f.configured)
		if err != nil {
			return gguffacts.Facts{}, err
		}
		files = append(files, file)
		info, err := file.Stat()
		if err != nil || !sameCopyFile(f.info, info) {
			return gguffacts.Facts{}, errModelInspectionChanged
		}
		role := gguffacts.RoleMain
		if f.Role == "draft" {
			role = gguffacts.RoleDraft
		} else if f.Role == "companion" {
			role = gguffacts.RoleCompanion
		}
		artifacts = append(artifacts, gguffacts.Artifact{Role: role, Reader: inspectionHeaderReader{ctx, file, access.usage}, Size: f.Size})
	}
	facts, err := gguffacts.Decode(ctx, artifacts)
	if err != nil {
		return gguffacts.Facts{}, err
	}
	for i, file := range files {
		info, err := file.Stat()
		if err != nil || !sameCopyFile(list.Files[i].info, info) {
			return gguffacts.Facts{}, errModelInspectionChanged
		}
	}
	return facts, nil
}

func (e *Executor) inspectModelArtifacts(ctx context.Context, p modelconfig.InspectRequest, advisory ...*modelEstimateInput) (modelconfig.InspectResult, error) {
	// A separate read-only handoff; never stored in Task 5's facts cache or
	// used for admission, binding identity, discovery or source revalidation.
	for _, output := range advisory {
		if output != nil {
			*output = modelEstimateInput{}
		}
	}
	result := modelconfig.InspectResult{Engine: p.Engine, ModelID: p.ModelID}
	unavailable := func(status, reason string) (modelconfig.InspectResult, error) {
		if err := ctx.Err(); err != nil {
			return modelconfig.InspectResult{}, err
		}
		result.Status, result.Reason = status, reason
		result.Artifacts, result.Options = nil, nil
		result.Facts = &modelconfig.Facts{Status: status, Unknowns: []string{reason}}
		return result, nil
	}
	if p.Engine != "llamacpp" {
		return unavailable("unavailable", "Local GGUF inspection is supported for llama.cpp only.")
	}
	if !e.modelInspection.active.CompareAndSwap(false, true) {
		return unavailable("busy", "Another model facts inspection is active.")
	}
	defer e.modelInspection.active.Store(false)
	ctx, cancel := context.WithTimeout(ctx, modelInspectionBudget)
	defer cancel()
	usage := servingReadUsage{}
	defer func() { e.modelInspection.lastUsage = usage }()
	access := servingFileAccess{open: e.modelInspection.open, usage: &usage}
	if e.reg == nil || e.downloads == nil {
		return unavailable("unavailable", "Model content admission is unavailable.")
	}
	downloads, registry := e.downloads, e.reg
	// Aliases, drafts and companions can share files with any catalog key.
	// Admit the existing engine-wide read so an already unsafe alternate name
	// cannot be bypassed by choosing a different inspection alias.
	key := downloadContentKey{engine: p.Engine}
	release, err := downloads.acquireRead(key, false)
	if err != nil {
		return unavailable("busy", "Model content is active, unsafe or unavailable for reading.")
	}
	defer release()
	_, ok := registry.Get(p.Engine)
	if !ok {
		return unavailable("unavailable", "Engine configuration is unavailable.")
	}
	downloads.mu.Lock()
	source, revision := downloads.source, downloads.revision
	downloads.mu.Unlock()
	st, err := e.state(p.Engine)
	if err != nil {
		return unavailable("unavailable", "Engine configuration is unavailable.")
	}
	snapshot, err := inspectionSnapshot(st, p.Engine)
	if errors.Is(err, errModelInspectionBusy) {
		return unavailable("busy", "Engine configuration is being changed.")
	}
	if err != nil {
		return unavailable("unavailable", "Exact local model source is unavailable.")
	}
	list, _, err := resolveLlamaArtifacts(ctx, snapshot.state, snapshot.command, snapshot.store, p.ModelID, access)
	if err != nil {
		switch {
		case errors.Is(err, errModelArtifactLimit):
			return unavailable("unknown", "Artifact discovery exceeded its bounded scan or artifact limit.")
		case errors.Is(err, errNoArtifactBinding):
			return unavailable("ambiguous", "No unique safe exact artifact binding is available.")
		default:
			return unavailable("missing", "Configured artifacts are missing, unsupported, partial or changed.")
		}
	}
	cacheSource := p.Engine + "\x00" + p.ModelID
	cacheKey := exportBinding(list, struct {
		Revision   int
		Source     string
		Binding    string
		Generation int64
		Store      engineModelStore
		Command    launchCommand
		Loads      []modelconfig.ModelLoad
		Serving    any
	}{gguffacts.ReaderRevision, source, list.binding, snapshot.state.gen, snapshot.store, snapshot.command, snapshot.state.plat.Runtime.ModelLoads, snapshot.state.plat.Runtime.Serving})
	facts, hit := e.modelInspection.cached(cacheKey, cacheSource)
	if !hit {
		facts, err = decodeInspectionFacts(ctx, list, access)
		if err != nil {
			if errors.Is(err, gguffacts.ErrLimit) {
				return unavailable("unknown", "GGUF metadata exceeded its bounded parsing limit.")
			}
			return unavailable("unknown", "GGUF facts are unsupported, incomplete or changed.")
		}
	}
	// Cache hits repeat resolution, binding samples and all owner checks too.
	current, err := inspectionSnapshot(st, p.Engine)
	if err != nil || !snapshot.equal(current) {
		return unavailable("busy", "Engine configuration changed during inspection.")
	}
	after, _, err := resolveLlamaArtifacts(ctx, current.state, current.command, current.store, p.ModelID, access)
	if err != nil || list.binding != after.binding || len(list.Files) != len(after.Files) {
		return unavailable("unknown", "Artifact sources changed during inspection.")
	}
	for i, f := range list.Files {
		if !sameCopyFile(f.info, after.Files[i].info) {
			return unavailable("unknown", "Artifact files changed during inspection.")
		}
	}
	// Finish with a nonblocking configuration check after the last file I/O.
	current, err = inspectionSnapshot(st, p.Engine)
	if err != nil || !snapshot.equal(current) {
		return unavailable("busy", "Engine configuration changed during inspection.")
	}
	downloads.mu.Lock()
	changed := e.downloads != downloads || e.reg != registry || downloads.source != source || downloads.revision != revision || downloads.blockedLocked(key) || downloads.writers[p.Engine]
	downloads.mu.Unlock()
	if changed {
		return unavailable("busy", "Model content source changed during inspection.")
	}
	if err := ctx.Err(); err != nil {
		return modelconfig.InspectResult{}, err
	}
	result.Status = "resolved"
	result.Reason = "Target-local sampled bindings and bounded GGUF metadata; content and runtime qualification remain unverified."
	for _, f := range list.Files {
		identity := exportBinding(modelFileList{Files: []modelFile{f}}, gguffacts.ReaderRevision)
		result.Artifacts = append(result.Artifacts, modelconfig.ArtifactSummary{Role: f.Role, Identity: identity, Size: strconv.FormatInt(f.Size, 10), Assurance: "local head/tail samples and file identity; not full content verification"})
	}
	result.Facts, result.Options = projectInspectionFacts(facts)
	if list.Layout == llamaArtifactsLayout {
		result.Facts.Source = "target-local configured preset/receipt/typed binding"
	} else {
		result.Facts.Source = "target-local canonical llama.cpp cache"
	}
	fits, err := inspectionReplyFits(result)
	if err != nil {
		return modelconfig.InspectResult{}, err
	}
	if !fits {
		return unavailable("unknown", "The complete artifact facts exceed the bounded public inspection reply; no artifact subset was selected.")
	}
	if err := ctx.Err(); err != nil {
		return modelconfig.InspectResult{}, err
	}
	if !hit {
		e.modelInspection.remember(cacheKey, cacheSource, facts)
	}
	if err := ctx.Err(); err != nil {
		return modelconfig.InspectResult{}, err
	}
	for _, output := range advisory {
		if output != nil {
			// This is a launch snapshot, not a freshly sampled executable binding.
			// No effective allocation/build support is asserted by inspection.
			*output = modelEstimateInput{Facts: facts, Bindings: slices.Clone(result.Artifacts), BuildID: "unverified-launch-snapshot:" + exportBinding(list, snapshot.command)}
		}
	}
	var observedArtifacts []modelconfig.ArtifactSummary
	for _, load := range snapshot.state.plat.Runtime.ModelLoads {
		if load.ModelID == p.ModelID && load.Resolved != nil {
			observedArtifacts = modelEstimateBindings(load.Resolved.Artifacts)
			for i := range observedArtifacts {
				observedArtifacts[i].Assurance = "sampled-local-binding"
			}
			break
		}
	}
	if len(observedArtifacts) == 0 {
		var bindings []modelconfig.ArtifactBinding
		for _, file := range list.Files {
			bindings = append(bindings, modelconfig.ArtifactBinding{Role: file.Role, Path: file.configured, ResolvedTarget: file.abs, Size: strconv.FormatInt(file.Size, 10), MtimeNS: strconv.FormatInt(file.info.ModTime().UnixNano(), 10), SampleSHA256: file.sample})
		}
		observedArtifacts = modelEstimateBindings(bindings)
		for i := range observedArtifacts {
			observedArtifacts[i].Assurance = "sampled-local-binding"
		}
	}
	result = attachModelBufferObservation(result, e.projectModelBufferObservation(st, p.Engine, p.ModelID, observedArtifacts))
	return result, nil
}

func projectInspectionFacts(facts gguffacts.Facts) (*modelconfig.Facts, []modelconfig.OptionDescriptor) {
	result := &modelconfig.Facts{Status: "known", Dimensions: map[string]string{}}
	values := map[string]gguffacts.Numeric{}
	conflicts := map[string]bool{}
	architecture, architectureConflict := "", false
	var tensorCount, tensorBytes uint64
	types := map[uint32]uint64{}
	templateDigests := map[string][32]byte{}
	templateConflict := false
	var vocabulary, merges *uint64
	count := func(dst **uint64, v gguffacts.Count, name string) {
		if !v.Present {
			return
		}
		if *dst == nil {
			value := v.Value
			*dst = &value
		} else if **dst != v.Value {
			conflicts[name] = true
		}
	}
	for _, shard := range facts.Shards {
		if shard.Role != gguffacts.RoleMain {
			continue
		}
		if shard.Architecture != "" {
			if !inspectionArchitectureName(shard.Architecture) {
				architectureConflict = true
			} else if architecture == "" {
				architecture = shard.Architecture
			} else if architecture != shard.Architecture {
				architectureConflict = true
			}
		}
		for key, value := range shard.Numeric {
			if previous, ok := values[key]; ok && !reflect.DeepEqual(previous, value) {
				conflicts[key] = true
			} else {
				values[key] = value
			}
		}
		count(&vocabulary, shard.Vocabulary, "tokenizer.vocabulary_count")
		count(&merges, shard.Merges, "tokenizer.merge_count")
		for _, template := range shard.Templates {
			if previous, ok := templateDigests[template.Key]; ok && previous != template.SHA256 {
				templateConflict = true
			}
			templateDigests[template.Key] = template.SHA256
		}
		for _, tensor := range shard.Tensors {
			tensorCount++
			if tensor.Bytes > math.MaxUint64-tensorBytes {
				conflicts["tensor.bytes"] = true
			} else {
				tensorBytes += tensor.Bytes
			}
			if tensor.Bytes > math.MaxUint64-types[tensor.Type] {
				conflicts["tensor.types"] = true
			} else {
				types[tensor.Type] += tensor.Bytes
			}
		}
	}
	if !architectureConflict {
		result.Architecture = architecture
	}
	put := func(key string, value uint64) {
		if len(result.Dimensions) < 256 && len(key) <= 256 {
			result.Dimensions[key] = strconv.FormatUint(value, 10)
		}
	}
	put("tensor.count", tensorCount)
	if !conflicts["tensor.bytes"] {
		put("tensor.bytes", tensorBytes)
	}
	if !conflicts["tensor.types"] {
		for typ, bytes := range types {
			put("tensor.type."+strconv.FormatUint(uint64(typ), 10)+".bytes", bytes)
		}
	}
	if vocabulary != nil && !conflicts["tokenizer.vocabulary_count"] {
		put("tokenizer.vocabulary_count", *vocabulary)
	}
	if merges != nil && !conflicts["tokenizer.merge_count"] {
		put("tokenizer.merge_count", *merges)
	}
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		if conflicts[key] || architectureConflict || strings.HasPrefix(key, "general.sampling.") {
			continue
		}
		value := values[key]
		if unsigned, ok := value.Uint64(); ok {
			put(key, unsigned)
			if key == architecture+".context_length" && unsigned > 0 && unsigned <= math.MaxInt32 {
				v := int32(unsigned)
				result.ContextTokens = &v
			}
			if key == architecture+".block_count" && unsigned <= math.MaxInt32 {
				v := int32(unsigned)
				result.LayerCount = &v
			}
		}
		if value.Array != nil {
			for i, bits := range value.Array {
				if len(result.Dimensions) >= 256 {
					break
				}
				scalar := gguffacts.Numeric{Type: value.Type, Bits: bits}
				if unsigned, ok := scalar.Uint64(); ok {
					put(key+"."+strconv.Itoa(i), unsigned)
				}
			}
		}
	}
	if !templateConflict {
		present := len(templateDigests) != 0
		result.TemplatePresent = &present
		if digest, ok := templateDigests["tokenizer.chat_template"]; ok {
			result.TemplateDigest = hex.EncodeToString(digest[:])
		}
	}
	if result.Architecture == "" {
		result.Unknowns = append(result.Unknowns, "main architecture unavailable or conflicting")
	}
	if result.ContextTokens == nil {
		result.Unknowns = append(result.Unknowns, "main training context unavailable")
	}
	if result.LayerCount == nil {
		result.Unknowns = append(result.Unknowns, "main layer count unavailable")
	}
	if len(conflicts) > 0 || templateConflict {
		result.Unknowns = append(result.Unknowns, "conflicting main-shard facts omitted")
	}
	if len(result.Dimensions) == 256 {
		result.Unknowns = append(result.Unknowns, "public dimension projection limit reached")
	}
	if len(result.Unknowns) != 0 {
		result.Status = "partial"
	}
	var recommendations []modelconfig.OptionDescriptor
	for key, field := range map[string]string{"general.sampling.temp": "temperature", "general.sampling.top_p": "top_p", "general.sampling.top_k": "top_k", "general.sampling.min_p": "min_p", "general.sampling.penalty_repeat": "repeat_penalty"} {
		value, exists := values[key]
		if !exists || conflicts[key] || architectureConflict {
			continue
		}
		raw, ok := inspectionNumericJSON(value)
		if !ok {
			continue
		}
		number := json.Number(raw)
		if err := (modelconfig.PolicyPatch{Generation: map[string]*json.Number{field: &number}}).Validate(); err != nil {
			continue
		}
		recommendations = append(recommendations, modelconfig.OptionDescriptor{Field: "generation." + field, Default: raw, DefaultSource: "embedded GGUF " + key})
	}
	return result, recommendations
}

// Architecture metadata is an identifier, not free-form model text. Keep the
// public projection free of path-shaped or arbitrary metadata strings even
// when an otherwise valid GGUF supplies one as its architecture prefix.
func inspectionArchitectureName(name string) bool {
	if name == "" || len(name) > gguffacts.MaxNameBytes {
		return false
	}
	for _, c := range name {
		if c < 'a' || c > 'z' {
			if c < '0' || c > '9' {
				if c != '_' && c != '-' {
					return false
				}
			}
		}
	}
	return true
}
