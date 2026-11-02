// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"math/big"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"
	"strings"

	"nvpair-shared/gguffacts"
	"nvpair-shared/modelconfig"
)

const strataPlanMetadataLimit = 64 << 20
const strataPlanExecutableLimit = 512 << 20

var errStrataMemoryPlanDrift = errors.New("Strata memory source changed during inspection")
var errStrataMemoryPlanResponse = errors.New("Strata memory report unavailable or invalid")

type strataMemoryPlanSource struct {
	state      *engineState
	selection  []byte
	command    launchCommand
	invocation memoryPlanCommand
	input      *strataServingInput
	binding    strataMemoryPlanBinding
}

// Bind the selected in-memory manifest and effective platform, including the
// resolved installation paths. Detach them while holding only the state lock.
func strataMemoryPlanSelection(st *engineState) (*engineState, []byte, error) {
	st.mu.Lock()
	defer st.mu.Unlock()
	if st.manifest == nil || st.plat == nil {
		return nil, nil, errStrataMemoryPlan
	}
	var selected struct {
		Manifest             *Manifest
		Platform             *Platform
		Bin, Install, Models string
	}
	selected.Manifest, selected.Platform = st.manifest, st.plat
	selected.Bin, selected.Install, selected.Models = st.binPath, st.installDir, st.modelsDir
	data, err := json.Marshal(selected)
	if err != nil || json.Unmarshal(data, &selected) != nil {
		return nil, nil, errStrataMemoryPlan
	}
	return &engineState{manifest: selected.Manifest, plat: selected.Platform, binPath: selected.Bin, installDir: selected.Install, modelsDir: selected.Models}, data, nil
}

func prepareStrataMemoryPlanSource(ctx context.Context, st *engineState, access strataServingAccess) (*strataMemoryPlanSource, error) {
	snapshot, selected, err := strataMemoryPlanSelection(st)
	if err != nil || snapshot.manifest.MemoryPlan == nil || snapshot.manifest.MemoryPlan.Adapter != "strata" || snapshot.plat.MemoryPlan == nil {
		return nil, errStrataMemoryPlan
	}
	invocation := *snapshot.plat.MemoryPlan
	invocation.Command = append([]string(nil), invocation.Command...)
	if invocation.validate(runtime.GOOS+"/"+runtime.GOARCH) != nil {
		return nil, errStrataMemoryPlan
	}
	command, err := launchForState(snapshot, snapshot.plat.Runtime.Port)
	if err != nil || command.Bin != invocation.Command[0] || len(command.Args) == 0 || command.Args[0] != filepath.Join(invocation.WorkingDirectory, "serve", "server.py") {
		return nil, errStrataMemoryPlan
	}
	input, err := deriveStrataServingInputExpected(ctx, command, snapshot.plat.Runtime, access, nil)
	if err != nil {
		return nil, errStrataMemoryPlan
	}
	if _, err := strataMemoryPlanFiniteRequest(input); err != nil {
		return nil, err
	}
	if err := prepareStrataInputRuntimeHashes(ctx, input); err != nil {
		return nil, errStrataMemoryPlan
	}
	source := &strataMemoryPlanSource{state: snapshot, selection: selected, command: command, invocation: invocation, input: input}
	digest, err := source.runtimeDigest(ctx, access.files)
	if err != nil {
		return nil, err
	}
	source.binding, err = strataMemoryPlanIdentityBinding(ctx, input, digest, access.files)
	if err != nil {
		return nil, err
	}
	return source, nil
}

func strataMemoryPlanCheckedHash(ctx context.Context, file sourceEvidence, access servingFileAccess) (string, error) {
	if file.size < 0 || file.size > strataPlanExecutableLimit {
		return "", errStrataMemoryPlan
	}
	f, err := access.checkedOpen(ctx, file.configured, file.target, file.info)
	if err != nil {
		return "", errStrataMemoryPlan
	}
	defer f.Close()
	h := sha256.New()
	n, err := io.Copy(h, io.NewSectionReader(servingContextReaderAt{ctx: ctx, reader: f, expected: file}, 0, file.size))
	opened, statErr := f.Stat()
	if err != nil || n != file.size || statErr != nil || !sameSource(file.info, opened, false) || validateSource(file) != nil || ctx.Err() != nil {
		return "", errStrataMemoryPlan
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func (source *strataMemoryPlanSource) runtimeDigest(ctx context.Context, access servingFileAccess) (string, error) {
	evidence := source.input.launch.evidence.artifacts
	moduleDigest, err := strataMemoryPlanModuleDigestAccess(ctx, source.invocation.WorkingDirectory, evidence, access)
	if err != nil {
		return "", err
	}
	type runtimeFile struct{ Path, SHA256 string }
	files := []runtimeFile{}
	declared := map[string]bool{}
	for _, artifact := range source.input.launch.Config.Models[0].Artifacts {
		if !strings.HasPrefix(artifact.Role, "runtime:") || artifact.Binding == nil {
			continue
		}
		path := artifact.Binding.SourcePath
		file, ok := evidence[path]
		if !ok {
			return "", errStrataMemoryPlan
		}
		sum, err := strataMemoryPlanCheckedHash(ctx, file, access)
		if err != nil || sum != artifact.SHA256 {
			return "", errStrataMemoryPlan
		}
		files = append(files, runtimeFile{path, sum})
		declared[path] = true
	}
	if !declared[source.command.Bin] {
		return "", errStrataMemoryPlan
	}
	for _, name := range strataMemoryPlanModules {
		if !declared[filepath.Join(source.invocation.WorkingDirectory, filepath.FromSlash(name))] {
			return "", errStrataMemoryPlan
		}
	}
	data, err := json.Marshal(struct {
		Selection        []byte
		Modules, Serving string
		Files            []runtimeFile
	}{source.selection, moduleDigest, source.input.launch.Digest, files})
	if err != nil {
		return "", errStrataMemoryPlan
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]), nil
}

func (source *strataMemoryPlanSource) recheck(ctx context.Context, st *engineState, access strataServingAccess) error {
	_, current, err := strataMemoryPlanSelection(st)
	if err != nil || !bytes.Equal(current, source.selection) || recheckStrataServingInput(ctx, source.command, source.state.plat.Runtime, source.input, access) != nil {
		return errStrataMemoryPlanDrift
	}
	digest, err := source.runtimeDigest(ctx, access.files)
	if err != nil || digest != source.binding.RuntimeSourceSHA256 {
		return errStrataMemoryPlanDrift
	}
	binding, err := strataMemoryPlanIdentityBinding(ctx, source.input, digest, access.files)
	if err != nil || !reflect.DeepEqual(binding, source.binding) {
		return errStrataMemoryPlanDrift
	}
	return nil
}

func (source *strataMemoryPlanSource) query(ctx context.Context, e *Executor, st *engineState, access strataServingAccess) (modelconfig.MemoryPlan, error) {
	ctx, cancel := context.WithTimeout(ctx, memoryPlanBudget)
	defer cancel()
	if err := source.recheck(ctx, st, access); err != nil {
		return modelconfig.MemoryPlan{}, err
	}
	request, err := json.Marshal(source.binding.Request)
	if err != nil {
		return modelconfig.MemoryPlan{}, errStrataMemoryPlan
	}
	response, err := e.runMemoryPlanCommand(ctx, source.invocation, request)
	if err != nil {
		return modelconfig.MemoryPlan{}, err
	}
	return source.complete(ctx, st, response, access)
}

func (source *strataMemoryPlanSource) complete(ctx context.Context, st *engineState, response []byte, access strataServingAccess) (modelconfig.MemoryPlan, error) {
	if err := source.recheck(ctx, st, access); err != nil {
		return modelconfig.MemoryPlan{}, err
	}
	plan, err := decodeStrataMemoryPlan(response, 0, source.binding)
	if err != nil {
		return modelconfig.MemoryPlan{}, errStrataMemoryPlanResponse
	}
	return plan, nil
}

// The inspection owner supplies the selected target state. Admission and the
// shared deadline cover preparation, invocation, cleanup and final rechecks.
func (e *Executor) queryStrataMemoryPlan(ctx context.Context, st *engineState) (modelconfig.MemoryPlan, error) {
	ctx, cancel := context.WithTimeout(ctx, memoryPlanBudget)
	defer cancel()
	snapshot, _, err := strataMemoryPlanSelection(st)
	if err != nil || snapshot.plat.Runtime.Serving == nil || len(snapshot.plat.Runtime.Serving.Models) != 1 {
		return modelconfig.MemoryPlan{}, errStrataMemoryPlan
	}
	engine, model := snapshot.manifest.Engine, snapshot.plat.Runtime.Serving.Models[0].ModelID
	release, err := e.acquireModelRead(engine, model, false)
	if err != nil {
		return modelconfig.MemoryPlan{}, errStrataMemoryPlan
	}
	defer release()
	source, err := prepareStrataMemoryPlanSource(ctx, st, strataServingAccess{})
	if err != nil || source.state.manifest.Engine != engine || source.input.launch.Config.Models[0].ModelID != model {
		return modelconfig.MemoryPlan{}, errStrataMemoryPlan
	}
	return source.query(ctx, e, st, strataServingAccess{})
}

// Construct expectations from the admitted selection, never from a provider's
// response. The caller separately binds its selected manifest and module closure.
func strataMemoryPlanIdentityBinding(ctx context.Context, input *strataServingInput, runtimeDigest string, access servingFileAccess) (strataMemoryPlanBinding, error) {
	var binding strataMemoryPlanBinding
	if input == nil || input.launch == nil || input.launch.evidence == nil || !strataPlanDigest(runtimeDigest) {
		return binding, errStrataMemoryPlan
	}
	request, err := strataMemoryPlanFiniteRequest(input)
	if err != nil {
		return binding, err
	}
	build, executableDigest, err := strataMemoryPlanNativeIdentity(ctx, input, access)
	if err != nil {
		return binding, err
	}
	inputs, err := strataMemoryPlanInputIdentities(ctx, input, access)
	if err != nil {
		return binding, err
	}
	for _, file := range input.launch.evidence.artifacts {
		if validateSource(file) != nil {
			return binding, errStrataMemoryPlan
		}
	}
	return strataMemoryPlanBinding{Request: request, Build: build, ExecutableSHA256: executableDigest, RuntimeSourceSHA256: runtimeDigest, Inputs: inputs}, nil
}

func strataMemoryPlanNativeIdentity(ctx context.Context, input *strataServingInput, access servingFileAccess) (strataMemoryPlanBuild, string, error) {
	var build strataMemoryPlanBuild
	path := input.config.Exe
	file, ok := input.launch.evidence.artifacts[path]
	if !ok || file.configured != path || file.size <= 0 || file.size > strataPlanExecutableLimit {
		return build, "", errStrataMemoryPlan
	}
	// Use the existing checked runtime hash cache, with the original evidence.
	sum, err := fileHashes.sumBoundKey(ctx, file, path, nil)
	if err != nil {
		return build, "", errStrataMemoryPlan
	}
	declared := false
	for _, artifact := range input.launch.Config.Models[0].Artifacts {
		if artifact.Binding != nil && artifact.Binding.SourcePath == path && artifact.Role == "runtime:native-exe" && artifact.SHA256 == sum {
			declared = true
		}
	}
	if !declared {
		return build, "", errStrataMemoryPlan
	}
	f, err := access.checkedOpen(ctx, path, file.target, file.info)
	if err != nil {
		return build, "", errStrataMemoryPlan
	}
	defer f.Close()
	r := servingContextReaderAt{ctx: ctx, reader: f, expected: file}
	marker := []byte("STRATA_BUILD/1:")
	// A bounded overlapping scan finds records even across chunk boundaries,
	// without retaining an executable-sized buffer or executing identity commands.
	buf := make([]byte, 64<<10)
	count := 0
	content := sha256.New()
	var hashedThrough int64
	for offset := int64(0); offset < file.size; {
		n := min(int64(len(buf)), file.size-offset)
		if _, err := r.ReadAt(buf[:n], offset); err != nil {
			return build, "", errStrataMemoryPlan
		}
		chunk := buf[:n]
		// The scan already reads every byte. Bind those bytes too, so a warm
		// runtime hash cannot hide an in-place rewrite preserving file stats.
		_, _ = content.Write(chunk[hashedThrough-offset:])
		hashedThrough = offset + n
		for start := 0; start < len(chunk); {
			i := bytes.Index(chunk[start:], marker)
			if i < 0 {
				break
			}
			i += start
			position := offset + int64(i+len(marker))
			raw, err := io.ReadAll(io.NewSectionReader(r, position, min(int64(16385), file.size-position)))
			if err != nil {
				return build, "", errStrataMemoryPlan
			}
			end := bytes.IndexByte(raw, '\n')
			if end > 0 && end <= 16384 && !bytes.ContainsRune(raw[:end], 0) {
				var schema struct {
					SchemaVersion int `json:"schema_version"`
				}
				if json.Unmarshal(raw[:end], &schema) == nil && schema.SchemaVersion == 1 {
					count++
					if count != 1 || strataPlanDecode(raw[:end], &build) != nil || build.Backend != "cuda" {
						return build, "", errStrataMemoryPlan
					}
				}
			}
			start = i + len(marker)
		}
		if offset+n == file.size {
			break
		}
		offset += n - int64(len(marker)-1)
	}
	opened, statErr := f.Stat()
	if count != 1 || hex.EncodeToString(content.Sum(nil)) != sum || statErr != nil || !sameSource(file.info, opened, false) || validateSource(file) != nil || ctx.Err() != nil {
		return build, "", errStrataMemoryPlan
	}
	return build, sum, nil
}

func strataMemoryPlanInputIdentities(ctx context.Context, input *strataServingInput, access servingFileAccess) ([]strataMemoryPlanInputIdentity, error) {
	evidence := input.launch.evidence.artifacts
	pack, native := input.options["--pack"], input.options["--native"]
	type selected struct {
		path           string
		metadata, gguf bool
	}
	paths := []selected{{input.options["--expert-profile"], true, false}, {filepath.Join(pack, "index.txt"), true, false}, {filepath.Join(pack, "native_experts.txt"), true, false}}
	for _, name := range []string{"dense.bin", "embd.bin", "experts.bin", "extra.bin"} {
		path := filepath.Join(pack, name)
		if _, err := os.Lstat(path); err == nil {
			paths = append(paths, selected{path: path})
		} else if !os.IsNotExist(err) {
			return nil, errStrataMemoryPlan
		}
	}
	shards, err := llamaArtifactPaths(native)
	if err != nil || len(shards) > 256 {
		return nil, errStrataMemoryPlan
	}
	// The shared owner validates the table syntax and safe local shard names.
	// Preserve the engine's first-encounter order when extending the main family.
	tableFile, ok := evidence[filepath.Join(pack, "native_experts.txt")]
	if !ok {
		return nil, errStrataMemoryPlan
	}
	table, _, err := access.sourceExpected(ctx, tableFile, strataPlanMetadataLimit, false)
	if err != nil || strataExpertTable(table, native, func(string, string) error { return nil }) != nil {
		return nil, errStrataMemoryPlan
	}
	if _, err := os.Lstat(filepath.Join(pack, "experts.bin")); os.IsNotExist(err) {
		seen := map[string]bool{}
		for _, path := range shards {
			seen[path] = true
		}
		for _, line := range strings.Split(string(table), "\n") {
			fields := strings.Fields(line)
			if len(fields) != 9 || strings.HasPrefix(line, "#") {
				continue
			}
			for _, name := range strings.Split(fields[8], ",") {
				path := native
				if name != "" {
					path = filepath.Join(filepath.Dir(native), name)
				}
				if !seen[path] {
					shards = append(shards, path)
					seen[path] = true
				}
			}
		}
	}
	for _, path := range shards {
		paths = append(paths, selected{path: path, gguf: true})
	}
	if len(paths) > 256 {
		return nil, errStrataMemoryPlan
	}
	identities := make([]strataMemoryPlanInputIdentity, 0, len(paths))
	selectedPaths := map[string]bool{}
	for _, selection := range paths {
		file, ok := evidence[selection.path]
		if !ok || file.configured != selection.path || selectedPaths[selection.path] {
			return nil, errStrataMemoryPlan
		}
		identity, err := strataMemoryPlanFileIdentity(ctx, file, selection.metadata, selection.gguf, access)
		if err != nil {
			return nil, err
		}
		identities = append(identities, identity)
		selectedPaths[selection.path] = true
	}
	for _, artifact := range input.launch.Config.Models[0].Artifacts {
		if !strings.HasPrefix(artifact.Role, "runtime:") && (artifact.Binding == nil || !selectedPaths[artifact.Binding.SourcePath]) {
			return nil, errStrataMemoryPlan
		}
	}
	return identities, nil
}

func strataMemoryPlanFileIdentity(ctx context.Context, file sourceEvidence, metadata, gguf bool, access servingFileAccess) (strataMemoryPlanInputIdentity, error) {
	identity := strataMemoryPlanInputIdentity{Path: file.configured, SizeBytes: uint64(file.size)}
	f, err := access.checkedOpen(ctx, file.configured, file.target, file.info)
	if err != nil {
		return identity, errStrataMemoryPlan
	}
	defer f.Close()
	// Reuse the existing platform handle identity owner. A non-OS test reader
	// without an independently inspectable handle cannot establish a file ID.
	osFile, ok := f.(*os.File)
	if !ok {
		return identity, errStrataMemoryPlan
	}
	id, err := strataFileID(osFile)
	if err != nil {
		return identity, errStrataMemoryPlan
	}
	parts := strings.Split(id, ":")
	if len(parts) != 2 && len(parts) != 3 {
		return identity, errStrataMemoryPlan
	}
	identity.FileID = make([]uint64, 2)
	for i, part := range parts {
		value, err := strconv.ParseUint(part, 16, 64)
		if err != nil {
			return identity, errStrataMemoryPlan
		}
		if i == 0 {
			identity.FileID[0] = value
		} else if len(parts) == 2 {
			identity.FileID[1] = value
		} else if i == 1 {
			identity.FileID[1] = value << 32
		} else {
			identity.FileID[1] |= value
		}
	}
	r := servingContextReaderAt{ctx: ctx, reader: f, expected: file}
	if gguf {
		var head [8]byte
		if _, err := r.ReadAt(head[:], 0); err != nil || !bytes.Equal(head[:], []byte{'G', 'G', 'U', 'F', 3, 0, 0, 0}) {
			return identity, errStrataMemoryPlan
		}
		facts, err := gguffacts.Decode(ctx, []gguffacts.Artifact{{Role: gguffacts.RoleMain, Reader: io.NewSectionReader(r, 0, strataPlanMetadataLimit), Size: file.size}})
		if err != nil || facts.Usage.ScannedHeaderBytes > strataPlanMetadataLimit {
			return identity, errStrataMemoryPlan
		}
		n := facts.Usage.ScannedHeaderBytes
		h := sha256.New()
		if _, err := io.Copy(h, io.NewSectionReader(r, 0, int64(n))); err != nil {
			return identity, errStrataMemoryPlan
		}
		sum := hex.EncodeToString(h.Sum(nil))
		identity.MetadataBytes, identity.MetadataSHA256 = &n, &sum
	} else if metadata {
		data, _, err := access.sourceExpected(ctx, file, strataPlanMetadataLimit, true)
		if err != nil {
			return identity, errStrataMemoryPlan
		}
		sum := sha256.Sum256(data)
		value := hex.EncodeToString(sum[:])
		identity.ContentSHA256 = &value
	}
	opened, statErr := f.Stat()
	if statErr != nil || !sameSource(file.info, opened, false) || validateSource(file) != nil || ctx.Err() != nil {
		return identity, errStrataMemoryPlan
	}
	return identity, nil
}

// This is the reviewed pure-mode import closure, not an attestation of the
// interpreter, system hooks or third-party imports.
var strataMemoryPlanModules = []string{
	"serve/__init__.py", "serve/server.py", "serve/memory_plan.py", "serve/launch_plan.py",
	"serve/mcp.py", "serve/runconfig.py", "serve/credentials.py", "serve/winjob.py",
	"serve/structured.py", "tools/build_provenance.py", "tools/gguf_reader.py",
	"tools/prepare_unsloth_q2.py", "setup.py",
}

func strataMemoryPlanFiniteRequest(input *strataServingInput) (strataMemoryPlanRequest, error) {
	var request strataMemoryPlanRequest
	if input == nil || input.launch == nil || input.launch.Config == nil || len(input.launch.Config.Models) != 1 || input.config.GPU == nil || input.config.Parallel != 1 || len(input.config.LibDirs) != 0 || input.config.FitMaxTokens != nil && *input.config.FitMaxTokens {
		return request, errStrataMemoryPlan
	}
	o := input.options
	allowed := map[string]bool{"--serve": true, "--max-context": true, "--kv": true, "--prefill": true, "--native": true, "--pack": true, "--expert-profile": true, "--resident-budget-gib": true, "--vram-reserve-mib": true, "--expert-cache": true, "--kv-resident": true, "--spec": true, "--no-prefill-borrow": true}
	for key := range o {
		if !allowed[key] {
			return request, errStrataMemoryPlan
		}
	}
	uintOption := func(value string, scale uint64) (uint64, error) {
		if value == "" || strings.HasPrefix(value, "+") {
			return 0, errStrataMemoryPlan
		}
		n, ok := new(big.Rat).SetString(value)
		if !ok || n.Sign() < 0 {
			return 0, errStrataMemoryPlan
		}
		n.Mul(n, new(big.Rat).SetInt(new(big.Int).SetUint64(scale)))
		if !n.IsInt() || !n.Num().IsUint64() {
			return 0, errStrataMemoryPlan
		}
		return n.Num().Uint64(), nil
	}
	values := make([]uint64, 8)
	for i, entry := range []struct {
		value string
		scale uint64
	}{
		{o["--max-context"], 1}, {o["--prefill"], 1}, {o["--resident-budget-gib"], 1 << 30},
		{input.config.Env["STRATA_FILE_CACHE_MIB"], 1 << 20}, {input.config.Env["STRATA_RESIDENT_HEADROOM_GIB"], 1 << 30},
		{o["--vram-reserve-mib"], 1 << 20}, {o["--expert-cache"], 1}, {o["--kv-resident"], 1},
	} {
		var err error
		values[i], err = uintOption(entry.value, entry.scale)
		if err != nil {
			return request, err
		}
	}
	if values[0] > 1048576 || values[1] > 32768 || o["--spec"] != "2" || o["--no-prefill-borrow"] != "true" || *input.config.GPU < 0 {
		return request, errStrataMemoryPlan
	}
	one, micro := int32(1), int32(values[1])
	c := modelconfig.MemoryPlanConfig{ContextTokens: int32(values[0]), KVFormat: o["--kv"], BatchSize: &one, ParallelSlots: &one, Microbatch: &modelconfig.MemoryPlanMicrobatch{Mode: "fixed", Tokens: &micro}, Devices: []modelconfig.MemoryPlanSelector{{Backend: "cuda", Selector: strconv.FormatInt(*input.config.GPU, 10)}}, Placement: "hybrid"}
	s := strataMemoryPlanOptions{Executable: input.config.Exe, WorkingDirectory: input.config.CWD, Pack: o["--pack"], NativeModel: o["--native"], ExpertProfile: o["--expert-profile"], ResidentBudgetBytes: values[2], FileCacheBytes: values[3], HostHeadroomBytes: values[4], VRAMReserveBytes: values[5], CacheSlots: strataMemoryPlanChoice{Value: values[6]}, KVResidentCells: values[7], Spec: 2, NoPrefillBorrow: true, Environment: map[string]string{}}
	for key, value := range input.config.Env {
		if key == "STRATA_FILE_CACHE_MIB" || key == "STRATA_RESIDENT_HEADROOM_GIB" {
			continue
		}
		if !strataPlanEnvironment(key, value) {
			return request, errStrataMemoryPlan
		}
		s.Environment[key] = value
	}
	data, err := encodeStrataMemoryPlanRequest(c, strataMemoryPlanModelRequest{ID: input.launch.Config.Models[0].ModelID}, s)
	if err != nil {
		return request, err
	}
	if err := json.Unmarshal(data, &request); err != nil {
		return request, errStrataMemoryPlan
	}
	return request, nil
}

func strataMemoryPlanModuleDigest(ctx context.Context, root string, evidence map[string]sourceEvidence) (string, error) {
	return strataMemoryPlanModuleDigestAccess(ctx, root, evidence, servingFileAccess{})
}

func strataMemoryPlanModuleDigestAccess(ctx context.Context, root string, evidence map[string]sourceEvidence, access servingFileAccess) (string, error) {
	if !filepath.IsAbs(root) || filepath.Clean(root) != root {
		return "", errStrataMemoryPlan
	}
	if _, err := os.Lstat(filepath.Join(root, "tools", "__init__.py")); !os.IsNotExist(err) {
		return "", errStrataMemoryPlan
	}
	type item struct{ Name, SHA256 string }
	items := make([]item, 0, len(strataMemoryPlanModules))
	for _, name := range strataMemoryPlanModules {
		path := filepath.Join(root, filepath.FromSlash(name))
		file, ok := evidence[path]
		if !ok || file.configured != path || validateSource(file) != nil {
			return "", errStrataMemoryPlan
		}
		sum, err := strataMemoryPlanCheckedHash(ctx, file, access)
		if err != nil || validateSource(file) != nil {
			return "", errStrataMemoryPlan
		}
		items = append(items, item{name, sum})
	}
	data, err := json.Marshal(items)
	if err != nil {
		return "", errStrataMemoryPlan
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]), nil
}
