// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode/utf8"
	"unsafe"

	"nvpair-shared/secretenv"
)

const modelDevicesHelperArg = "--private-model-devices"
const modelDevicesLimit = 64
const modelDevicesBytes = 64 << 10
const modelDevicesBudget = 3 * time.Second
const modelDeviceReportUnknown = "device report not recognized for this build"
const modelDeviceSeveralGPUs = "several GPUs: reviewed Auto supports one GPU; choose GPUs with extra options"

// This is an internal supplied-operation snapshot, not a public request or a
// second telemetry owner. Task 5 supplies topology from the target broker and
// the already compiled selection/environment, under its existing fences.
type modelDevicePool struct {
	ID, Kind, UUID, PCI, LUID string
	Unsupported               string // MIG, virtual aliases and mixed layouts remain Unknown.
}
type modelDeviceTopology struct {
	Host, Run, Observation string
	Pools                  []modelDevicePool
}
type modelDeviceLibrary struct{ Name, SHA256 string }
type modelDeviceRuntime struct {
	Build, Root string
	Libraries   []modelDeviceLibrary
}
type modelDeviceOperation struct {
	Runtime           modelDeviceRuntime
	Topology          modelDeviceTopology
	Selector          string
	Environment       []string // complete effective child environment, not overrides
	NativeInitAllowed bool     // supplied by the operation's Pause/activation owner
}
type modelNativeDevice struct {
	Selector, PCI, UUID, LUID, Pool, Unknown string
	Name                                     string
	Total                                    uint64
	Kind                                     int32
}
type modelDeviceReply struct {
	Devices []modelNativeDevice
	Commit  string
	Unknown string
}
type modelDeviceBinding struct {
	Host, Pool, UUID, PCI, LUID, Selector, Digest string
	Environment                                   []string
}
type modelDeviceMap struct {
	ObservationID, HostPool, Commit, Unknown string
	Devices                                  []modelNativeDevice
	Binding                                  *modelDeviceBinding
}
type modelDeviceRunner func(context.Context, modelDeviceRuntime, []string) (modelDeviceReply, error)

type modelDeviceCheck struct{ Commit, Unknown string }

// Cache build recognition only. Physical inventories, visibility verification,
// topology joins and bindings belong to each operation and are never cached.
var modelDeviceChecks = struct {
	sync.Mutex
	builds map[string]modelDeviceCheck
}{builds: make(map[string]modelDeviceCheck)}

func modelDeviceCheckedBuild(build string) (modelDeviceCheck, bool) {
	modelDeviceChecks.Lock()
	defer modelDeviceChecks.Unlock()
	checked, ok := modelDeviceChecks.builds[build]
	return checked, ok
}

func modelDeviceRememberCheck(build string, reply modelDeviceReply) {
	modelDeviceChecks.Lock()
	defer modelDeviceChecks.Unlock()
	modelDeviceChecks.builds[build] = modelDeviceCheck{Commit: reply.Commit, Unknown: reply.Unknown}
}

var modelDevicePCI = regexp.MustCompile(`(?i)^([0-9a-f]{4}|[0-9a-f]{8}):([0-9a-f]{2}):([0-9a-f]{2})\.([0-7])$`)
var modelDeviceUUID = regexp.MustCompile(`(?i)^GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$`)

func canonicalModelDeviceUUID(value string) string {
	if !modelDeviceUUID.MatchString(value) || strings.EqualFold(value, "GPU-00000000-0000-0000-0000-000000000000") {
		return ""
	}
	return "GPU-" + strings.ToLower(value[4:])
}

func canonicalModelDevicePCI(value string) string {
	m := modelDevicePCI.FindStringSubmatch(value)
	if m == nil || len(m[1]) == 8 && m[1][:4] != "0000" {
		return ""
	}
	device, err := strconv.ParseUint(m[3], 16, 8)
	if err != nil || device > 31 {
		return ""
	}
	return strings.ToLower(m[1][len(m[1])-4:] + ":" + m[2] + ":" + m[3] + "." + m[4])
}

func modelDeviceDigest(value any) (string, error) {
	b, err := json.Marshal(value)
	if err != nil || len(b) > modelDevicesBytes {
		return "", errors.New("device identity exceeds bound")
	}
	d := sha256.Sum256(b)
	return hex.EncodeToString(d[:]), nil
}

// mapModelSettingsDevices is callable without inspection/settings hooks. It is
// review-only. Apply, recovery and start must replay Binding and never call it.
func mapModelSettingsDevices(ctx context.Context, op modelDeviceOperation, run modelDeviceRunner) modelDeviceMap {
	out := modelDeviceMap{}
	fail := func(reason string) modelDeviceMap {
		out.Unknown, out.Devices, out.Binding = reason, nil, nil
		return out
	}
	if ctx.Err() != nil {
		return fail("device operation canceled")
	}
	t := op.Topology
	if t.Host == "" || t.Run == "" || t.Observation == "" || len(t.Pools) == 0 || len(t.Pools) > modelDevicesLimit {
		return fail("missing or oversized topology identity")
	}
	if len(t.Host) > 256 || len(t.Run) > 256 || len(t.Observation) > 256 {
		return fail("topology identity bound")
	}
	for _, p := range t.Pools {
		for _, v := range []string{p.ID, p.Kind, p.UUID, p.PCI, p.LUID, p.Unsupported} {
			if len(v) > 256 {
				return fail("physical identity bound")
			}
		}
	}
	// Sort a copy: ordering is neither identity nor a physical matching proof.
	t.Pools = append([]modelDevicePool(nil), t.Pools...)
	sort.Slice(t.Pools, func(i, j int) bool { return t.Pools[i].ID < t.Pools[j].ID })
	var err error
	out.ObservationID, err = modelDeviceDigest(t)
	if err != nil {
		return fail("invalid topology identity")
	}
	ids, uuids, pcis, luids := map[string]bool{}, map[string]bool{}, map[string]bool{}, map[string]bool{}
	hasShared, hasDedicated := false, false
	dedicated := 0
	for _, p := range t.Pools {
		hasShared = hasShared || p.Kind == "shared"
		hasDedicated = hasDedicated || p.Kind == "dedicated"
		if p.Kind == "dedicated" {
			dedicated++
		}
		if p.Kind == "host" || p.Kind == "shared" {
			if out.HostPool != "" {
				return fail("ambiguous host/shared pool")
			}
			out.HostPool = p.ID
		}
	}
	if op.Selector == "CPU" {
		if out.HostPool == "" {
			return fail("missing host/shared pool")
		}
		return out
	}
	if dedicated > 1 {
		return fail(modelDeviceSeveralGPUs)
	}
	if hasShared && hasDedicated {
		return fail("unsupported mixed UMA/discrete topology")
	}
	for i := range t.Pools {
		p := &t.Pools[i]
		if p.ID == "" || ids[p.ID] {
			return fail("duplicate physical pool")
		}
		ids[p.ID] = true
		if p.Kind == "host" || p.Kind == "shared" {
			continue
		}
		p.PCI = canonicalModelDevicePCI(p.PCI)
		p.UUID = canonicalModelDeviceUUID(p.UUID)
		p.LUID = strings.ToLower(p.LUID)
		for _, pair := range []struct {
			key  string
			seen map[string]bool
		}{{p.UUID, uuids}, {p.PCI, pcis}, {p.LUID, luids}} {
			if pair.key != "" && pair.seen[pair.key] {
				return fail("duplicate physical alias")
			}
			pair.seen[pair.key] = true
		}
	}
	if !op.NativeInitAllowed {
		return fail("native initialization prohibited by target activation/Pause")
	}
	if op.Runtime.Build == "" || len(op.Runtime.Libraries) == 0 {
		return fail("unsupported native build/library identity")
	}
	if checked, ok := modelDeviceCheckedBuild(op.Runtime.Build); ok && checked.Unknown != "" {
		out.Commit = checked.Commit
		return fail(checked.Unknown)
	}
	op.Runtime.Libraries = append([]modelDeviceLibrary(nil), op.Runtime.Libraries...)
	sort.Slice(op.Runtime.Libraries, func(i, j int) bool { return op.Runtime.Libraries[i].Name < op.Runtime.Libraries[j].Name })
	if _, err := modelDeviceDigest(op.Runtime); err != nil {
		return fail("native library identity exceeds bound")
	}
	env, err := modelDeviceEnvironment(op.Environment, "")
	if err != nil {
		return fail(err.Error())
	}
	ctx, cancel := context.WithTimeout(ctx, modelDevicesBudget)
	defer cancel()
	if run == nil {
		run = runModelDeviceProcess
	}
	reply, err := run(ctx, op.Runtime, env)
	if err != nil || ctx.Err() != nil {
		return fail("native metadata helper unavailable or canceled")
	}
	out.Commit = reply.Commit
	if reply.Unknown != "" || len(reply.Devices) > modelDevicesLimit {
		reply.Unknown = modelDeviceReportUnknown
		modelDeviceRememberCheck(op.Runtime.Build, reply)
		return fail(modelDeviceReportUnknown)
	}
	modelDeviceRememberCheck(op.Runtime.Build, reply)
	var selected *modelNativeDevice
	seen := map[string]bool{}
	nativePCI, nativeUUID := map[string]bool{}, map[string]bool{}
	for i := range reply.Devices {
		d := &reply.Devices[i]
		d.Pool = "" // supplied native output never supplies physical authority
		if seen[d.Selector] {
			return fail("duplicate native selector")
		}
		seen[d.Selector] = true
		if d.Kind == 0 {
			d.Pool = out.HostPool
		} else if d.Kind == 1 && strings.HasPrefix(d.Selector, "CUDA") {
			pci, uuid := canonicalModelDevicePCI(d.PCI), canonicalModelDeviceUUID(d.UUID)
			if pci != "" && nativePCI[pci] || uuid != "" && nativeUUID[uuid] {
				return fail("duplicate native physical alias")
			}
			nativePCI[pci] = true
			nativeUUID[uuid] = true
			if d.Unknown == "" {
				for _, p := range t.Pools {
					if pci != "" && uuid != "" && p.PCI == pci && p.UUID == uuid && p.Kind == "dedicated" && p.Unsupported == "" && (runtime.GOOS != "windows" || p.LUID != "" && strings.EqualFold(p.LUID, d.LUID)) {
						d.Pool = p.ID
					}
				}
				if d.Pool == "" {
					d.Unknown = "unmatched physical PCI/UUID/LUID"
				}
			}
		} else if d.Unknown == "" {
			d.Unknown = "unsupported native identity"
		}
		if d.Selector == op.Selector {
			selected = d
		}
	}
	if selected == nil || selected.Kind != 1 || selected.Unknown != "" || !strings.HasPrefix(selected.Selector, "CUDA") {
		return fail("unsupported selected native device")
	}
	pci := canonicalModelDevicePCI(selected.PCI)
	if pci == "" || canonicalModelDeviceUUID(selected.UUID) == "" {
		return fail("missing exact CUDA PCI/full UUID identity")
	}
	var pool *modelDevicePool
	for i := range t.Pools {
		p := &t.Pools[i]
		if p.Kind == "dedicated" && p.PCI == pci && strings.EqualFold(p.UUID, selected.UUID) {
			if pool != nil {
				return fail("ambiguous physical match")
			}
			pool = p
		}
	}
	if pool == nil || pool.Unsupported != "" {
		return fail("unmatched or unsupported physical device")
	}
	if runtime.GOOS == "windows" && (pool.LUID == "" || !strings.EqualFold(pool.LUID, selected.LUID)) {
		return fail("CUDA/DXGI LUID mismatch")
	}
	env, err = modelDeviceEnvironment(op.Environment, pool.UUID)
	if err != nil {
		return fail(err.Error())
	}
	verified, err := run(ctx, op.Runtime, env)
	if err != nil || ctx.Err() != nil {
		return fail("UUID visibility verification unavailable")
	}
	if verified.Unknown != "" || len(verified.Devices) > modelDevicesLimit {
		verified.Unknown = modelDeviceReportUnknown
		modelDeviceRememberCheck(op.Runtime.Build, verified)
		return fail(modelDeviceReportUnknown)
	}
	var gpu *modelNativeDevice
	for i := range verified.Devices {
		d := &verified.Devices[i]
		if d.Kind == 0 {
			continue
		}
		if gpu != nil || d.Kind != 1 {
			return fail("UUID visibility is not one CUDA device")
		}
		gpu = d
	}
	if gpu == nil || gpu.Selector != "CUDA0" || gpu.Unknown != "" || canonicalModelDevicePCI(gpu.PCI) != pool.PCI || !strings.EqualFold(gpu.UUID, pool.UUID) || runtime.GOOS == "windows" && !strings.EqualFold(gpu.LUID, pool.LUID) {
		return fail("UUID-constrained CUDA0 identity mismatch")
	}
	b := &modelDeviceBinding{Host: t.Host, Pool: pool.ID, UUID: pool.UUID, PCI: pool.PCI, LUID: pool.LUID, Selector: "CUDA0", Environment: env}
	// Deliberately excludes sample time, memory values, producer run and the
	// unrestricted inventory's selector. Replay uses only constrained CUDA0.
	b.Digest, err = modelDeviceDigest(struct {
		Host, Pool, UUID, PCI, LUID, Selector, Build string
		Libraries                                    []modelDeviceLibrary
		Environment                                  []string
	}{b.Host, b.Pool, b.UUID, b.PCI, b.LUID, b.Selector, op.Runtime.Build, op.Runtime.Libraries, env})
	if err != nil {
		return fail("binding identity exceeds bound")
	}
	out.Binding = b
	out.Devices = reply.Devices
	return out
}

// Restrict inherited loader injection and MPS rather than allowing a helper to
// observe a different program/library graph. The rest of the effective env is
// preserved and bound (including CUDA_DEVICE_ORDER and visibility).
func modelDeviceEnvironment(input []string, uuid string) ([]string, error) {
	if len(input) > 512 {
		return nil, errors.New("device environment bound")
	}
	values := map[string]string{}
	bytes := 0
	for _, kv := range secretenv.Remove(input) {
		bytes += len(kv)
		if bytes > modelDevicesBytes {
			return nil, errors.New("device environment bound")
		}
		k, v, ok := strings.Cut(kv, "=")
		if runtime.GOOS == "windows" {
			k = strings.ToUpper(k)
		}
		if !ok || k == "" || strings.ContainsAny(k+v, "\x00\r\n") {
			return nil, errors.New("invalid device environment")
		}
		if _, exists := values[k]; exists {
			return nil, errors.New("duplicate effective environment key")
		}
		mpsExclusion := runtime.GOOS == "linux" && k == "CUDA_MPS_PIPE_DIRECTORY" && v == "/dev/null"
		if k == "LD_PRELOAD" || k == "LD_AUDIT" || strings.HasPrefix(k, "DYLD_") || strings.HasPrefix(k, "CUDA_MPS_") && !mpsExclusion || strings.Contains(strings.ToUpper(v), "MIG-") {
			return nil, errors.New("unsupported native injection/MIG/MPS environment")
		}
		values[k] = v
	}
	if uuid != "" {
		values["CUDA_VISIBLE_DEVICES"] = uuid
	}
	if runtime.GOOS == "linux" {
		// MPS otherwise connects implicitly through /tmp/nvidia-mps. A
		// character device cannot contain its control pipe. Keep this
		// exclusion in the replay environment as well as both review runs.
		info, err := os.Lstat("/dev/null")
		if err != nil || info.Mode()&os.ModeCharDevice == 0 {
			return nil, errors.New("cannot exclude implicit CUDA MPS")
		}
		values["CUDA_MPS_PIPE_DIRECTORY"] = "/dev/null"
	}
	out := make([]string, 0, len(values))
	for k, v := range values {
		out = append(out, k+"="+v)
	}
	sort.Strings(out)
	if _, err := modelDeviceDigest(out); err != nil {
		return nil, err
	}
	return out, nil
}

// The existing selected runtime/install owner supplies st, never public paths.
// No engine is launched and no library is loaded in the manager process.
func modelDeviceRuntimeSnapshot(ctx context.Context, st *engineState) (modelDeviceRuntime, error) {
	runtime, _, err := modelDeviceRuntimeSnapshotBound(ctx, st, servingFileAccess{}, nil)
	return runtime, err
}

// Library identities are request-local and retained for Auto replay.
func modelDeviceRuntimeSnapshotBound(ctx context.Context, st *engineState, access servingFileAccess, expected map[string]sourceEvidence) (modelDeviceRuntime, map[string]sourceEvidence, error) {
	bin, err := nativeBuildExecutable(st)
	if err != nil {
		return modelDeviceRuntime{}, nil, err
	}
	executable, err := captureSource(bin, false)
	if err != nil {
		return modelDeviceRuntime{}, nil, err
	}
	r, files, err := nativeBuildSnapshotExpected(ctx, executable, access, expected)
	if err == nil && len(r.Libraries) == 0 {
		return r, nil, errors.New("selected runtime has no metadata libraries")
	}
	return r, files, err
}

// File-only callers and device review share the selected directory's library set.
func modelDeviceLibrarySnapshotBound(ctx context.Context, r modelDeviceRuntime, access servingFileAccess, expected map[string]sourceEvidence) (modelDeviceRuntime, map[string]sourceEvidence, error) {
	root := r.Root
	if err := ctx.Err(); err != nil {
		return r, nil, err
	}
	directory, err := captureSource(root, true)
	if err != nil {
		return r, nil, err
	}
	entries, err := os.ReadDir(root)
	if err != nil {
		return r, nil, err
	}
	files := make(map[string]sourceEvidence)
	for _, entry := range entries {
		name := entry.Name()
		if !modelDeviceLibraryName(name) {
			continue
		}
		if len(r.Libraries) >= modelDevicesLimit || ctx.Err() != nil {
			return r, nil, errors.New("native library snapshot bound/cancellation")
		}
		var file sourceEvidence
		if expected == nil {
			file, err = captureSource(filepath.Join(root, name), false)
		} else {
			var ok bool
			file, ok = expected[name]
			if !ok {
				return r, nil, errors.New("native library set changed")
			}
		}
		if err != nil {
			return r, nil, err
		}
		digest, err := modelDeviceHashFileExpected(ctx, root, name, access, file)
		if err != nil {
			return r, nil, err
		}
		files[name] = file
		r.Libraries = append(r.Libraries, modelDeviceLibrary{Name: name, SHA256: digest})
	}
	if expected != nil && len(files) != len(expected) {
		return r, nil, errors.New("native library set changed")
	}
	for _, file := range files {
		if err := validateSource(file); err != nil {
			return r, nil, err
		}
	}
	if err := validateSource(directory); err != nil {
		return r, nil, err
	}
	if err := ctx.Err(); err != nil {
		return r, nil, err
	}
	return r, files, nil
}
func modelDeviceLibraryName(name string) bool {
	n := strings.ToLower(name)
	return (strings.HasSuffix(n, ".dll") || strings.HasSuffix(n, ".dylib") || strings.Contains(n, ".so")) && (strings.HasPrefix(n, "ggml") || strings.HasPrefix(n, "libggml") || strings.HasPrefix(n, "libllama") || strings.HasPrefix(n, "llama") || strings.HasPrefix(n, "cudart") || strings.HasPrefix(n, "cublas") || strings.HasPrefix(n, "libcudart") || strings.HasPrefix(n, "libcublas"))
}
func modelDeviceHashFile(ctx context.Context, root, name string) (string, error) {
	if _, err := modelDeviceLibraryPath(root, name); err != nil {
		return "", err
	}
	file, err := captureSource(filepath.Join(root, name), false)
	if err != nil {
		return "", err
	}
	return modelDeviceHashFileExpected(ctx, root, name, servingFileAccess{}, file)
}

func modelDeviceHashFileExpected(ctx context.Context, root, name string, access servingFileAccess, expected sourceEvidence) (string, error) {
	p, err := modelDeviceLibraryPath(root, name)
	if err != nil {
		return "", err
	}
	if expected.configured != filepath.Join(root, name) || expected.target != p {
		return "", errors.New("library identity changed")
	}
	return nativeBuildHashFileExpected(ctx, access, expected)
}

func modelDeviceLibraryPath(root, name string) (string, error) {
	if filepath.Base(name) != name || !modelDeviceLibraryName(name) {
		return "", errors.New("invalid selected library name")
	}
	p, err := resolveExistingPath(filepath.Join(root, name))
	if err != nil {
		return "", err
	}
	rel, err := filepath.Rel(root, p)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", errors.New("library escapes selected install")
	}
	return p, nil
}

type modelDeviceOutputBudget struct {
	mu       sync.Mutex
	used     int
	cancel   context.CancelFunc
	exceeded bool
	limit    int
}
type modelDeviceBoundedWriter struct {
	data   bytes.Buffer
	budget *modelDeviceOutputBudget
}

func (w *modelDeviceBoundedWriter) Write(p []byte) (int, error) {
	w.budget.mu.Lock()
	defer w.budget.mu.Unlock()
	limit := w.budget.limit
	if limit == 0 {
		limit = modelDevicesBytes
	}
	if len(p) > limit-w.budget.used {
		w.budget.exceeded = true
		w.budget.cancel()
		return 0, errors.New("device helper output bound")
	}
	w.budget.used += len(p)
	return w.data.Write(p)
}
func runModelDeviceProcess(ctx context.Context, selected modelDeviceRuntime, env []string) (modelDeviceReply, error) {
	if runtime.GOOS == "linux" {
		found := false
		for _, kv := range env {
			if strings.HasPrefix(kv, "LD_LIBRARY_PATH=") {
				found = kv == "LD_LIBRARY_PATH="+selected.Root
			}
		}
		if !found {
			return modelDeviceReply{}, errors.New("selected native library search path is not isolated")
		}
	}
	exe, err := os.Executable()
	if err != nil {
		return modelDeviceReply{}, err
	}
	return runModelDeviceCommand(ctx, exe, []string{modelDevicesHelperArg}, selected, env)
}
func runModelDeviceCommand(ctx context.Context, exe string, args []string, selected modelDeviceRuntime, env []string) (modelDeviceReply, error) {
	// Reserve pipe-drain time inside the hard parent wall-clock ceiling.
	deadline := time.Now().Add(modelDevicesBudget)
	if parent, ok := ctx.Deadline(); ok && parent.Before(deadline) {
		deadline = parent
	}
	ctx, cancel := context.WithDeadline(ctx, deadline.Add(-100*time.Millisecond))
	defer cancel()
	input, err := json.Marshal(selected)
	if err != nil || len(input) > modelDevicesBytes {
		return modelDeviceReply{}, errors.New("helper input bound")
	}
	cmd := exec.CommandContext(ctx, exe, args...)
	if selected.Root != "" {
		cmd.Dir = selected.Root
	}
	cmd.Env = secretenv.Remove(env)
	cmd.Stdin = bytes.NewReader(input)
	cmd.WaitDelay = 100 * time.Millisecond
	configureSysProcAttr(cmd)
	budget := &modelDeviceOutputBudget{cancel: cancel}
	w := &modelDeviceBoundedWriter{budget: budget}
	cmd.Stderr = w
	// stderr shares the output ceiling, but is never parsed or exposed. Native
	// libraries may log to stderr; keep the structured stdout separate.
	stdout := &modelDeviceBoundedWriter{budget: budget}
	cmd.Stdout = stdout
	err = cmd.Run() // Run always Waits/reaps, including cancellation/output overflow.
	if err != nil || ctx.Err() != nil || budget.exceeded {
		return modelDeviceReply{}, errors.New("native metadata helper failed/bounded")
	}
	var reply modelDeviceReply
	d := json.NewDecoder(bytes.NewReader(stdout.data.Bytes()))
	d.DisallowUnknownFields()
	if err := d.Decode(&reply); err != nil {
		return reply, errors.New("invalid device metadata")
	}
	if d.Decode(new(json.RawMessage)) != io.EOF || len(reply.Devices) > modelDevicesLimit {
		return modelDeviceReply{}, errors.New("device metadata bound")
	}
	return reply, nil
}

func runModelDevicesHelper(input io.Reader, output io.Writer) int {
	if runtime.GOOS == "linux" {
		info, err := os.Lstat("/dev/null")
		if os.Getenv("CUDA_MPS_PIPE_DIRECTORY") != "/dev/null" || err != nil || info.Mode()&os.ModeCharDevice == 0 {
			return 2
		}
	}
	var selected modelDeviceRuntime
	raw, err := io.ReadAll(io.LimitReader(input, modelDevicesBytes+1))
	if err != nil || len(raw) > modelDevicesBytes {
		return 2
	}
	d := json.NewDecoder(bytes.NewReader(raw))
	d.DisallowUnknownFields()
	if d.Decode(&selected) != nil || d.Decode(new(json.RawMessage)) != io.EOF {
		return 2
	}
	ctx, cancel := context.WithTimeout(context.Background(), modelDevicesBudget)
	defer cancel()
	if selected.Build == "" || !filepath.IsAbs(selected.Root) || len(selected.Libraries) == 0 || len(selected.Libraries) > modelDevicesLimit {
		return 2
	}
	seen := map[string]bool{}
	for _, lib := range selected.Libraries {
		if seen[lib.Name] {
			return 2
		}
		seen[lib.Name] = true
		h, err := modelDeviceHashFile(ctx, selected.Root, lib.Name)
		if err != nil || h != lib.SHA256 {
			return 2
		}
	}
	entries, err := os.ReadDir(selected.Root)
	if err != nil {
		return 2
	}
	for _, entry := range entries {
		if strings.HasPrefix(strings.ToLower(entry.Name()), "libcuda.so") || strings.EqualFold(entry.Name(), "nvcuda.dll") {
			return 2
		}
		if modelDeviceLibraryName(entry.Name()) && !seen[entry.Name()] {
			return 2
		}
	}
	reply := enumerateModelNativeDevices(selected)
	b, err := json.Marshal(reply)
	if err != nil || len(b) > modelDevicesBytes {
		return 2
	}
	if _, err = output.Write(b); err != nil {
		return 2
	}
	return 0
}

// This process-local hard exit remains armed when the parent dies and cannot
// cancel or reap us. Unlike context cancellation it also bounds a stuck native
// call. Only private helper/sentinel entry points may arm it.
func armModelDeviceDeadline() {
	time.AfterFunc(modelDevicesBudget, func() { os.Exit(124) })
}

// ggml_backend_dev_props on the supported 64-bit ABI. Name and total memory
// are checked against the driver, never used as telemetry or placement authority.
// https://github.com/ggml-org/llama.cpp/blob/0ec4027a2a0183de87c007069adddc06e665f7de/ggml/include/ggml-backend.h
type modelGGMLProps struct {
	Name, Description unsafe.Pointer
	Free, Total       uintptr
	Kind              int32
	_                 uint32
	ID                unsafe.Pointer
	Caps              [5]byte
	_                 [3]byte
}
type modelDeviceSymbols func(string) (uintptr, error)
type modelDeviceRegister func(any, uintptr)
type modelDeviceLibraryOpener func(modelDeviceRuntime) (modelDeviceSymbols, modelDeviceRegister, error)
type modelDeviceDriverOpener func() (modelDeviceSymbols, modelDeviceRegister, error)

func enumerateModelNativeDevices(selected modelDeviceRuntime) modelDeviceReply {
	return enumerateModelNativeDevicesWith(selected, openModelDeviceLibraries, openModelCUDADriver)
}

func enumerateModelNativeDevicesWith(selected modelDeviceRuntime, openLibraries modelDeviceLibraryOpener, openDriver modelDeviceDriverOpener) (reply modelDeviceReply) {
	defer func() {
		if recover() != nil {
			reply.Unknown = modelDeviceReportUnknown
		}
		if reply.Unknown != "" {
			reply.Devices = nil
			reply.Unknown = modelDeviceReportUnknown
		}
	}()
	if unsafe.Sizeof(uintptr(0)) != 8 || unsafe.Sizeof(modelGGMLProps{}) != 56 || unsafe.Offsetof(modelGGMLProps{}.ID) != 40 {
		return modelDeviceReply{Unknown: "unsupported native ABI"}
	}
	symbol, register, err := openLibraries(selected)
	if err != nil {
		return modelDeviceReply{Unknown: "selected native libraries unavailable"}
	}
	reply.Commit, err = readModelDeviceCommit(symbol, register)
	if err != nil {
		reply.Unknown = modelDeviceReportUnknown
		return reply
	}
	if err := loadModelDeviceCUDA(selected, symbol, register); err != nil {
		reply.Unknown = modelDeviceReportUnknown
		return reply
	}
	var count func() uintptr
	var get func(uintptr) uintptr
	var props func(uintptr, *modelGGMLProps)
	for _, f := range []struct {
		name string
		fn   any
	}{{"ggml_backend_dev_count", &count}, {"ggml_backend_dev_get", &get}, {"ggml_backend_dev_get_props", &props}} {
		p, e := symbol(f.name)
		if e != nil || p == 0 {
			reply.Unknown = modelDeviceReportUnknown
			return reply
		}
		register(f.fn, p)
	}
	n := count()
	if n > modelDevicesLimit {
		return modelDeviceReply{Unknown: "native device count bound"}
	}
	for i := uintptr(0); i < n; i++ {
		var p modelGGMLProps
		dev := get(i)
		if dev == 0 {
			return modelDeviceReply{Unknown: "null native device"}
		}
		props(dev, &p)
		d := modelNativeDevice{Selector: modelDeviceCString(p.Name), PCI: modelDeviceCString(p.ID), Name: modelDeviceCString(p.Description), Total: uint64(p.Total), Kind: p.Kind}
		if d.Selector == "" {
			reply.Unknown = modelDeviceReportUnknown
			return reply
		}
		if p.Kind != 0 {
			d.Unknown = "unsupported native identity"
		}
		reply.Devices = append(reply.Devices, d)
	}
	needsCUDA := false
	for _, d := range reply.Devices {
		needsCUDA = needsCUDA || strings.HasPrefix(d.Selector, "CUDA")
	}
	if !needsCUDA {
		return reply
	}
	driver, registerDriver, err := openDriver()
	if err != nil {
		reply.Unknown = modelDeviceReportUnknown
		return reply
	}
	return joinModelCUDADevices(reply, driver, registerDriver)
}
func modelDeviceCString(p unsafe.Pointer) string {
	if p == nil {
		return ""
	}
	b := make([]byte, 0, 256)
	for i := 0; i < 256; i++ {
		c := *(*byte)(unsafe.Add(p, i))
		if c == 0 {
			return modelDeviceString(b)
		}
		b = append(b, c)
	}
	return ""
}

func modelDeviceString(b []byte) string {
	if !utf8.Valid(b) {
		return ""
	}
	for _, c := range b {
		if c < 32 || c == 127 {
			return ""
		}
	}
	return string(b)
}

func modelDeviceTerminatedString(b []byte) string {
	end := bytes.IndexByte(b, 0)
	if end < 0 {
		return ""
	}
	return modelDeviceString(b[:end])
}

// ggml_backend_load checks GGML_BACKEND_API_VERSION before registering the
// selected dynamic engine. It is given one exact path, never a search path.
// https://github.com/ggml-org/llama.cpp/blob/0ec4027a2a0183de87c007069adddc06e665f7de/ggml/src/ggml-backend-reg.cpp
func loadModelDeviceCUDA(selected modelDeviceRuntime, symbol modelDeviceSymbols, register modelDeviceRegister) error {
	var cuda string
	for _, lib := range selected.Libraries {
		n := strings.ToLower(lib.Name)
		if n == "ggml-cuda.dll" || n == "libggml-cuda.so" || n == "libggml-cuda.dylib" {
			if cuda != "" {
				return errors.New("ambiguous CUDA library")
			}
			cuda = filepath.Join(selected.Root, lib.Name)
		}
	}
	if cuda == "" {
		return nil
	}
	p, err := symbol("ggml_backend_load")
	if err != nil || p == 0 {
		return errors.New("unsupported ggml loader export")
	}
	var load func(string) uintptr
	register(&load, p)
	if load(cuda) == 0 {
		return errors.New("unsupported CUDA engine ABI")
	}
	return nil
}

func readModelDeviceCommit(symbol modelDeviceSymbols, register modelDeviceRegister) (string, error) {
	p, err := symbol("ggml_commit")
	if err != nil || p == 0 {
		return "", errors.New("unsupported ggml revision export")
	}
	var commit func() unsafe.Pointer
	register(&commit, p)
	got := modelDeviceCString(commit())
	if got == "" {
		return "", errors.New("undecodable ggml revision")
	}
	return got, nil
}

// Driver ordinals only enumerate handles. PCI joins ggml to the driver; the
// parent then checks the driver's physical UUID against supplied topology.
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html
func joinModelCUDADevices(reply modelDeviceReply, symbol modelDeviceSymbols, register modelDeviceRegister) modelDeviceReply {
	fail := func() modelDeviceReply {
		return modelDeviceReply{Commit: reply.Commit, Unknown: modelDeviceReportUnknown}
	}
	var init func(uint32) int32
	var count func(*int32) int32
	var get func(*int32, int32) int32
	var pci func(*byte, int32, int32) int32
	var uuid func(*[16]byte, int32) int32
	var physicalUUID func(*[16]byte, int32) int32
	var luid func(*[8]byte, *uint32, int32) int32
	var name func(*byte, int32, int32) int32
	var total func(*uintptr, int32) int32
	functions := []struct {
		name string
		fn   any
	}{{"cuInit", &init}, {"cuDeviceGetCount", &count}, {"cuDeviceGet", &get}, {"cuDeviceGetPCIBusId", &pci}, {"cuDeviceGetUuid_v2", &uuid}, {"cuDeviceGetUuid", &physicalUUID}, {"cuDeviceGetName", &name}, {"cuDeviceTotalMem_v2", &total}}
	if runtime.GOOS == "windows" {
		functions = append(functions, struct {
			name string
			fn   any
		}{"cuDeviceGetLuid", &luid})
	}
	for _, f := range functions {
		p, e := symbol(f.name)
		if e != nil || p == 0 {
			return fail()
		}
		register(f.fn, p)
	}
	var n int32
	if init(0) != 0 || count(&n) != 0 || n < 0 || n > modelDevicesLimit {
		return fail()
	}
	identities := map[string]modelNativeDevice{}
	for i := int32(0); i < n; i++ {
		var dev int32
		var rawPCI [32]byte
		var rawUUID [16]byte
		var parentUUID [16]byte
		var rawName [256]byte
		var memory uintptr
		if get(&dev, i) != 0 || pci(&rawPCI[0], 32, dev) != 0 || uuid(&rawUUID, dev) != 0 || physicalUUID(&parentUUID, dev) != 0 || name(&rawName[0], int32(len(rawName)), dev) != 0 || total(&memory, dev) != 0 {
			return fail()
		}
		if rawUUID != parentUUID || rawUUID == [16]byte{} {
			return fail()
		}
		key := canonicalModelDevicePCI(modelDeviceTerminatedString(rawPCI[:]))
		deviceName := modelDeviceTerminatedString(rawName[:])
		if key == "" || deviceName == "" || memory == 0 {
			return fail()
		}
		if _, ok := identities[key]; ok {
			return fail()
		}
		u := hex.EncodeToString(rawUUID[:])
		identity := modelNativeDevice{PCI: key, UUID: "GPU-" + u[:8] + "-" + u[8:12] + "-" + u[12:16] + "-" + u[16:20] + "-" + u[20:], Name: deviceName, Total: uint64(memory)}
		if luid != nil {
			var raw [8]byte
			var mask uint32
			if luid(&raw, &mask, dev) != 0 || mask == 0 || mask&(mask-1) != 0 {
				return fail()
			}
			identity.LUID = fmt.Sprintf("luid_0x%08x_0x%08x_phys_0", binary.LittleEndian.Uint32(raw[4:]), binary.LittleEndian.Uint32(raw[:4]))
		}
		identities[key] = identity
	}
	seen := map[string]bool{}
	for i := range reply.Devices {
		d := &reply.Devices[i]
		if !strings.HasPrefix(d.Selector, "CUDA") {
			continue
		}
		if d.Kind != 1 {
			return fail()
		}
		key := canonicalModelDevicePCI(d.PCI)
		identity, ok := identities[key]
		if !ok || seen[key] || d.Name == "" || d.Name != identity.Name || d.Total == 0 || max(d.Total, identity.Total)-min(d.Total, identity.Total) > 1<<20 {
			return fail()
		}
		seen[key] = true
		d.PCI = identity.PCI
		d.UUID = identity.UUID
		d.LUID = identity.LUID
		d.Unknown = ""
	}
	return reply
}
