// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"
	"unsafe"
)

const modelDevicesFixtureUUID = "GPU-12345678-1234-1234-1234-123456789abc"

func TestModelSettingsDevicesBoundLibraryRead(t *testing.T) {
	root, err := resolveExistingPath(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(root, "libggml.dylib")
	writeTestFile(t, path, []byte("original library"))
	evidence, err := captureSource(path, false)
	if err != nil {
		t.Fatal(err)
	}
	reads, closes := 0, 0
	access := servingFileAccess{open: func(name string) (servingArtifactFile, error) {
		file, err := os.Open(name)
		if err != nil {
			return nil, err
		}
		return &countedLinkedFile{File: file, reads: &reads, closes: &closes}, nil
	}}
	if _, err := modelDeviceHashFileExpected(context.Background(), root, "libggml.dylib", access, evidence); err != nil || reads == 0 {
		t.Fatalf("positive hash reads=%d error=%v", reads, err)
	}
	if err := os.Rename(path, path+".old"); err != nil {
		t.Fatal(err)
	}
	writeTestFile(t, path, []byte("replacement library"))
	reads = 0
	if _, err := modelDeviceHashFileExpected(context.Background(), root, "libggml.dylib", access, evidence); err == nil {
		t.Fatal("replacement library accepted")
	}
	if reads != 0 {
		t.Fatalf("replacement library read %d times", reads)
	}
}

const modelDevicesFixtureLUID = "luid_0x00000000_0x00001234_phys_0"

func modelDevicesFixtureOperation(t *testing.T) modelDeviceOperation {
	t.Helper()
	build := "fixture:" + t.Name()
	t.Cleanup(func() {
		modelDeviceChecks.Lock()
		delete(modelDeviceChecks.builds, build)
		modelDeviceChecks.Unlock()
	})
	return modelDeviceOperation{
		Runtime: modelDeviceRuntime{Build: build, Libraries: []modelDeviceLibrary{{Name: "ggml.dll", SHA256: strings.Repeat("a", 64)}}},
		Topology: modelDeviceTopology{Host: "host-a", Run: "run-a", Observation: "observation-a", Pools: []modelDevicePool{
			{ID: "ram", Kind: "host"},
			{ID: "physical-a", Kind: "dedicated", UUID: modelDevicesFixtureUUID, PCI: "00000000:AB:00.0", LUID: modelDevicesFixtureLUID},
		}},
		Selector: "CUDA7", Environment: []string{"CUDA_DEVICE_ORDER=PCI_BUS_ID"}, NativeInitAllowed: true,
	}
}
func modelDevicesFixtureRunner(ctx context.Context, _ modelDeviceRuntime, env []string) (modelDeviceReply, error) {
	if err := ctx.Err(); err != nil {
		return modelDeviceReply{}, err
	}
	selector := "CUDA7"
	for _, kv := range env {
		if kv == "CUDA_VISIBLE_DEVICES="+modelDevicesFixtureUUID {
			selector = "CUDA0"
		}
	}
	return modelDeviceReply{Commit: "abcdef0-custom", Devices: []modelNativeDevice{
		{Selector: "CPU", Kind: 0},
		{Selector: selector, Kind: 1, PCI: "0000:ab:00.0", UUID: modelDevicesFixtureUUID, LUID: modelDevicesFixtureLUID},
	}}, nil
}

func TestModelSettingsDevicesPhysicalSelection(t *testing.T) {
	op := modelDevicesFixtureOperation(t)
	first := mapModelSettingsDevices(context.Background(), op, modelDevicesFixtureRunner)
	if first.Unknown != "" || first.Binding == nil {
		t.Fatalf("mapping = %+v", first)
	}
	if first.Binding.UUID != modelDevicesFixtureUUID || first.Binding.Selector != "CUDA0" || first.Binding.Pool != "physical-a" {
		t.Fatalf("binding = %+v", first.Binding)
	}
	wantEnv := []string{"CUDA_DEVICE_ORDER=PCI_BUS_ID", "CUDA_VISIBLE_DEVICES=" + modelDevicesFixtureUUID}
	if runtime.GOOS == "linux" {
		wantEnv = []string{"CUDA_DEVICE_ORDER=PCI_BUS_ID", "CUDA_MPS_PIPE_DIRECTORY=/dev/null", "CUDA_VISIBLE_DEVICES=" + modelDevicesFixtureUUID}
	}
	if !reflect.DeepEqual(first.Binding.Environment, wantEnv) {
		t.Fatalf("environment = %v", first.Binding.Environment)
	}
	// The source topology has no marketing names or capacity matching fields.
	// Reordering identical-name hardware can only affect native selectors.
	op.Topology.Pools[0], op.Topology.Pools[1] = op.Topology.Pools[1], op.Topology.Pools[0]
	op.Selector = "CUDA2"
	run := func(ctx context.Context, r modelDeviceRuntime, env []string) (modelDeviceReply, error) {
		reply, err := modelDevicesFixtureRunner(ctx, r, env)
		for i := range reply.Devices {
			if reply.Devices[i].Selector == "CUDA7" {
				reply.Devices[i].Selector = "CUDA2"
			}
		}
		return reply, err
	}
	second := mapModelSettingsDevices(context.Background(), op, run)
	if second.Unknown != "" || second.Binding == nil || second.Binding.Digest != first.Binding.Digest || second.ObservationID != first.ObservationID {
		t.Fatalf("reordered mapping = %+v", second)
	}
}

func TestModelSettingsDevicesObservationAndBindingIdentity(t *testing.T) {
	baseOp := modelDevicesFixtureOperation(t)
	base := mapModelSettingsDevices(context.Background(), baseOp, modelDevicesFixtureRunner)
	if base.Binding == nil {
		t.Fatalf("base = %+v", base)
	}
	test := func(name string, change func(*modelDeviceOperation), sameBinding bool) {
		t.Run(name, func(t *testing.T) {
			op := modelDevicesFixtureOperation(t)
			op.Runtime.Build = baseOp.Runtime.Build
			change(&op)
			got := mapModelSettingsDevices(context.Background(), op, modelDevicesFixtureRunner)
			if got.Binding == nil {
				t.Fatalf("mapping = %+v", got)
			}
			if (got.Binding.Digest == base.Binding.Digest) != sameBinding {
				t.Fatalf("binding digest = %s, base %s", got.Binding.Digest, base.Binding.Digest)
			}
			if sameBinding && got.ObservationID == base.ObservationID {
				t.Fatal("observation identity did not change")
			}
		})
	}
	test("producer restart", func(op *modelDeviceOperation) { op.Topology.Run = "run-b" }, true)
	test("new topology observation", func(op *modelDeviceOperation) { op.Topology.Observation = "observation-b" }, true)
	test("changed library content", func(op *modelDeviceOperation) { op.Runtime.Libraries[0].SHA256 = strings.Repeat("b", 64) }, false)
	test("changed effective visibility environment", func(op *modelDeviceOperation) { op.Environment = []string{"CUDA_DEVICE_ORDER=FASTEST_FIRST"} }, false)
	test("changed host", func(op *modelDeviceOperation) { op.Topology.Host = "host-b" }, false)
}

func TestModelSettingsDevicesSeveralGPUsDoNotInitialize(t *testing.T) {
	op := modelDevicesFixtureOperation(t)
	other := op.Topology.Pools[1]
	other.ID = "physical-b"
	other.PCI = "0000:ac:00.0"
	other.UUID = "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
	other.LUID = "luid_0x00000000_0x00005678_phys_0"
	op.Topology.Pools = append(op.Topology.Pools, other)
	runner := func(context.Context, modelDeviceRuntime, []string) (modelDeviceReply, error) {
		t.Fatal("several-GPU topology initialized native metadata")
		return modelDeviceReply{}, nil
	}
	first := mapModelSettingsDevices(context.Background(), op, runner)
	op.Topology.Pools[1], op.Topology.Pools[2] = op.Topology.Pools[2], op.Topology.Pools[1]
	op.Selector = "CUDA8"
	second := mapModelSettingsDevices(context.Background(), op, runner)
	for _, got := range []modelDeviceMap{first, second} {
		if got.Unknown != "several GPUs: reviewed Auto supports one GPU; choose GPUs with extra options" || got.Binding != nil || len(got.Devices) != 0 {
			t.Fatalf("several-GPU mapping = %+v", got)
		}
	}
}

func TestModelSettingsDevicesUnknownInputs(t *testing.T) {
	test := func(name string, change func(*modelDeviceOperation)) {
		t.Run(name, func(t *testing.T) {
			op := modelDevicesFixtureOperation(t)
			change(&op)
			got := mapModelSettingsDevices(context.Background(), op, modelDevicesFixtureRunner)
			if got.Unknown == "" || got.Binding != nil {
				t.Fatalf("expected Unknown, got %+v", got)
			}
		})
	}
	test("missing build identity", func(op *modelDeviceOperation) { op.Runtime.Build = "" })
	test("Pause prohibits native initialization", func(op *modelDeviceOperation) { op.NativeInitAllowed = false })
	test("split selector", func(op *modelDeviceOperation) { op.Selector = "CUDA7,CUDA8" })
	test("missing UUID", func(op *modelDeviceOperation) { op.Topology.Pools[1].UUID = "" })
	test("abbreviated UUID", func(op *modelDeviceOperation) { op.Topology.Pools[1].UUID = "gpu-1234" })
	test("duplicate physical alias", func(op *modelDeviceOperation) {
		p := op.Topology.Pools[1]
		p.ID = "alias"
		op.Topology.Pools = append(op.Topology.Pools, p)
	})
	test("ambiguous PCI", func(op *modelDeviceOperation) {
		p := op.Topology.Pools[1]
		p.ID = "other"
		p.UUID = "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
		p.LUID = "other"
		op.Topology.Pools = append(op.Topology.Pools, p)
	})
	test("mixed shared/discrete", func(op *modelDeviceOperation) { op.Topology.Pools[0].Kind = "shared" })
	test("MIG topology", func(op *modelDeviceOperation) { op.Topology.Pools[1].Unsupported = "MIG" })
	test("MPS environment", func(op *modelDeviceOperation) {
		op.Environment = append(op.Environment, "CUDA_MPS_PIPE_DIRECTORY=/tmp/mps")
	})
	test("MIG visibility", func(op *modelDeviceOperation) {
		op.Environment = append(op.Environment, "CUDA_VISIBLE_DEVICES=MIG-GPU-1234/1/2")
	})
	test("loader injection", func(op *modelDeviceOperation) { op.Environment = append(op.Environment, "LD_PRELOAD=/tmp/foreign.so") })
	if runtime.GOOS == "windows" {
		test("LUID mismatch", func(op *modelDeviceOperation) { op.Topology.Pools[1].LUID = "different" })
	}
}

func TestModelSettingsDevicesUnknownNativeResults(t *testing.T) {
	test := func(name string, change func(*modelDeviceReply), onlyVerification bool) {
		t.Run(name, func(t *testing.T) {
			calls := 0
			run := func(ctx context.Context, r modelDeviceRuntime, env []string) (modelDeviceReply, error) {
				calls++
				reply, err := modelDevicesFixtureRunner(ctx, r, env)
				if !onlyVerification || calls == 2 {
					change(&reply)
				}
				return reply, err
			}
			got := mapModelSettingsDevices(context.Background(), modelDevicesFixtureOperation(t), run)
			if got.Unknown == "" || got.Binding != nil || len(got.Devices) != 0 {
				t.Fatalf("expected Unknown, got %+v", got)
			}
		})
	}
	test("unsupported exports", func(r *modelDeviceReply) { r.Unknown = "unsupported CUDA exports" }, false)
	test("duplicate selector", func(r *modelDeviceReply) { r.Devices = append(r.Devices, r.Devices[1]) }, false)
	test("duplicate native physical alias", func(r *modelDeviceReply) { d := r.Devices[1]; d.Selector = "CUDA8"; r.Devices = append(r.Devices, d) }, false)
	test("device bound", func(r *modelDeviceReply) { r.Devices = make([]modelNativeDevice, modelDevicesLimit+1) }, false)
	test("CUDA0 changed physical UUID", func(r *modelDeviceReply) { r.Devices[1].UUID = "gpu-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa" }, true)
	test("verification retained native ordinal", func(r *modelDeviceReply) { r.Devices[1].Selector = "CUDA7" }, true)
	test("verification multiple devices", func(r *modelDeviceReply) { r.Devices = append(r.Devices, r.Devices[1]) }, true)
}

func TestModelSettingsDevicesCPUSharedPool(t *testing.T) {
	op := modelDevicesFixtureOperation(t)
	op.Selector = "CPU"
	op.NativeInitAllowed = false
	op.Topology.Pools = []modelDevicePool{{ID: "shared-uma", Kind: "shared"}}
	got := mapModelSettingsDevices(context.Background(), op, func(context.Context, modelDeviceRuntime, []string) (modelDeviceReply, error) {
		t.Fatal("CPU enumerated native devices")
		return modelDeviceReply{}, errors.New("unexpected enumeration")
	})
	if got.Unknown != "" || got.HostPool != "shared-uma" || got.Binding != nil || len(got.Devices) != 0 {
		t.Fatalf("CPU UMA = %+v", got)
	}
}

func TestModelSettingsDevicesCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	got := mapModelSettingsDevices(ctx, modelDevicesFixtureOperation(t), func(context.Context, modelDeviceRuntime, []string) (modelDeviceReply, error) {
		t.Fatal("canceled operation enumerated")
		return modelDeviceReply{}, nil
	})
	if got.Unknown == "" || got.Binding != nil {
		t.Fatalf("canceled mapping = %+v", got)
	}
}

func TestModelSettingsDevicesPCINormalization(t *testing.T) {
	for _, tc := range []struct{ input, want string }{{"00000000:AB:00.0", "0000:ab:00.0"}, {"0000:ab:00.0", "0000:ab:00.0"}, {"AB:00.0", ""}, {"00010000:ab:00.0", ""}, {"0000:ab:ff.0", ""}} {
		t.Run(tc.input, func(t *testing.T) {
			if got := canonicalModelDevicePCI(tc.input); got != tc.want {
				t.Fatalf("PCI = %q, want %q", got, tc.want)
			}
		})
	}
}

// TestMain dispatches this before fixture building or registry/service startup.
// It never calls the production native helper, a driver or a real library.
func runModelDevicesTestChild(mode string) int {
	if _, err := io.Copy(io.Discard, os.Stdin); err != nil {
		return 2
	}
	switch mode {
	case "self timeout":
		armModelDeviceDeadline()
		time.Sleep(time.Minute)
		return 2
	case "timeout":
		time.Sleep(time.Minute)
		return 2
	case "overflow":
		if _, err := os.Stdout.Write(bytes.Repeat([]byte("x"), modelDevicesBytes+1)); err != nil {
			return 2
		}
		return 0
	case "stderr overflow":
		if _, err := os.Stderr.Write(bytes.Repeat([]byte("x"), modelDevicesBytes+1)); err != nil {
			return 2
		}
		return 0
	case "valid":
		for _, v := range os.Environ() {
			if strings.HasPrefix(strings.ToUpper(v), "NVPAIR_PROVIDER_SECRET_") {
				return 2
			}
		}
		if err := json.NewEncoder(os.Stdout).Encode(modelDeviceReply{Devices: []modelNativeDevice{{Selector: "CPU", Kind: 0}}}); err != nil {
			return 2
		}
		return 0
	default:
		return 2
	}
}

func TestModelSettingsDevicesLoadedRevisionIsMetadata(t *testing.T) {
	for _, revision := range []string{"0ec4027a2a0183de87c007069adddc06e665f7de", "0ec4027", "unknown", "7fe450e-dirty", "0123456", "7fe450"} {
		t.Run(revision, func(t *testing.T) {
			lookup := func(name string) (uintptr, error) {
				if name != "ggml_commit" {
					t.Fatalf("unexpected property/initialization export %s", name)
				}
				return 1, nil
			}
			raw := append([]byte(revision), 0)
			register := func(fn any, _ uintptr) {
				p, ok := fn.(*func() unsafe.Pointer)
				if !ok {
					t.Fatalf("unexpected revision signature %T", fn)
				}
				*p = func() unsafe.Pointer { return unsafe.Pointer(&raw[0]) }
			}
			got, err := readModelDeviceCommit(lookup, register)
			if err != nil || got != revision {
				t.Fatalf("revision metadata = %q, %v; want %q", got, err, revision)
			}
		})
	}
	if _, err := readModelDeviceCommit(func(string) (uintptr, error) { return 0, errors.New("missing") }, func(any, uintptr) { t.Fatal("missing export registered") }); err == nil {
		t.Fatal("missing revision export accepted")
	}
}

func TestModelSettingsDevicesEnvironmentSemantics(t *testing.T) {
	env, err := modelDeviceEnvironment([]string{"cuda_visible_devices=ignored", "CUDA_DEVICE_ORDER=PCI_BUS_ID"}, "")
	if err != nil {
		t.Fatal(err)
	}
	joined := strings.Join(env, "\n")
	if runtime.GOOS == "windows" {
		if !strings.Contains(joined, "CUDA_VISIBLE_DEVICES=ignored") {
			t.Fatalf("Windows key normalization missing: %v", env)
		}
	} else if !strings.Contains(joined, "cuda_visible_devices=ignored") || strings.Contains(joined, "CUDA_VISIBLE_DEVICES=ignored") {
		t.Fatalf("Unix key spelling changed: %v", env)
	}
	if runtime.GOOS == "linux" && !strings.Contains(joined, "CUDA_MPS_PIPE_DIRECTORY=/dev/null") {
		t.Fatalf("implicit MPS was not excluded: %v", env)
	}
	if replay, err := modelDeviceEnvironment(env, ""); err != nil || !reflect.DeepEqual(replay, env) {
		t.Fatalf("safe effective environment cannot be reviewed again: %v, %v", replay, err)
	}
	if _, err := modelDeviceEnvironment([]string{"CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps"}, ""); err == nil {
		t.Fatal("explicit MPS accepted")
	}
}

func TestModelSettingsDevicesIndependentHelperDeadline(t *testing.T) {
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	// No runModelDeviceCommand parent timer: exercise the helper's own hard
	// exit, which also survives loss of the manager's cancellation goroutine.
	ctx, cancel := context.WithTimeout(context.Background(), 6*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, exe, "model-devices-test-child", "self timeout")
	cmd.Env = os.Environ()
	err = cmd.Run()
	var exit *exec.ExitError
	if ctx.Err() != nil || !errors.As(err, &exit) || exit.ExitCode() != 124 {
		t.Fatalf("independent helper deadline: err=%v context=%v", err, ctx.Err())
	}
}

func TestModelSettingsDevicesHelperBounds(t *testing.T) {
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	for _, mode := range []string{"valid", "timeout", "overflow", "stderr overflow"} {
		t.Run(mode, func(t *testing.T) {
			start := time.Now()
			got, err := runModelDeviceCommand(context.Background(), exe, []string{"model-devices-test-child", mode}, modelDeviceRuntime{}, append(os.Environ(), "NVPAIR_PROVIDER_SECRET_FIXTURE=private"))
			if mode == "valid" {
				if err != nil {
					t.Fatal(err)
				}
				if len(got.Devices) != 1 || got.Devices[0].Selector != "CPU" {
					t.Fatalf("reply = %+v", got)
				}
			} else if err == nil {
				t.Fatal("unbounded helper succeeded")
			}
			if time.Since(start) > 4*time.Second {
				t.Fatal("helper exceeded hard ceiling plus test scheduling margin")
			}
		})
	}
}

func TestModelSettingsDevicesPrivateModeRejectsInputBeforeNative(t *testing.T) {
	for _, input := range []string{"{}", `{"Build":"other"}`, strings.Repeat("x", modelDevicesBytes+1)} {
		var output bytes.Buffer
		if code := runModelDevicesHelper(strings.NewReader(input), &output); code == 0 || output.Len() != 0 {
			t.Fatalf("invalid private input: code %d, bytes %d", code, output.Len())
		}
	}
}

func TestModelSettingsDevicesHelperContextCancellation(t *testing.T) {
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
	defer cancel()
	start := time.Now()
	if _, err := runModelDeviceCommand(ctx, exe, []string{"model-devices-test-child", "timeout"}, modelDeviceRuntime{}, os.Environ()); err == nil {
		t.Fatal("canceled helper succeeded")
	}
	if time.Since(start) > 2*time.Second {
		t.Fatal("canceled helper was not reaped within bound")
	}
}

// Symbols are Go functions only. No native library, driver or helper process is
// opened by these fixtures, including the ABI and loader rejection cases.
func modelDeviceTestSymbols(t *testing.T, functions map[string]any) (modelDeviceSymbols, modelDeviceRegister) {
	t.Helper()
	var exports []any
	lookup := func(name string) (uintptr, error) {
		fn, ok := functions[name]
		if !ok {
			return 0, errors.New("missing fake symbol")
		}
		exports = append(exports, fn)
		return uintptr(len(exports)), nil
	}
	register := func(pointer any, address uintptr) {
		if address == 0 || address > uintptr(len(exports)) {
			t.Fatalf("invalid fake symbol address %d", address)
		}
		out, fn := reflect.ValueOf(pointer).Elem(), reflect.ValueOf(exports[address-1])
		if out.Type() != fn.Type() {
			t.Fatalf("fake symbol signature %v, want %v", fn.Type(), out.Type())
		}
		out.Set(fn)
	}
	return lookup, register
}

func modelDeviceTestABI(t *testing.T, selector string) (map[string]any, map[string]any) {
	t.Helper()
	selectorBytes := append([]byte(selector), 0)
	name := []byte("Synthetic CUDA GPU\x00")
	pci := []byte("0000:ab:00.0\x00")
	commit := []byte("abcdef0-custom\x00")
	uuid := [16]byte{0x12, 0x34, 0x56, 0x78, 0x12, 0x34, 0x12, 0x34, 0x12, 0x34, 0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc}
	ggml := map[string]any{
		"ggml_commit":            func() unsafe.Pointer { return unsafe.Pointer(&commit[0]) },
		"ggml_backend_load":      func(string) uintptr { return 1 },
		"ggml_backend_dev_count": func() uintptr { return 1 },
		"ggml_backend_dev_get":   func(uintptr) uintptr { return 1 },
		"ggml_backend_dev_get_props": func(_ uintptr, p *modelGGMLProps) {
			*p = modelGGMLProps{Name: unsafe.Pointer(&selectorBytes[0]), Description: unsafe.Pointer(&name[0]), ID: unsafe.Pointer(&pci[0]), Total: 8 << 30, Kind: 1}
		},
	}
	driver := map[string]any{
		"cuInit":           func(uint32) int32 { return 0 },
		"cuDeviceGetCount": func(n *int32) int32 { *n = 1; return 0 },
		"cuDeviceGet":      func(device *int32, ordinal int32) int32 { *device = ordinal; return 0 },
		"cuDeviceGetPCIBusId": func(out *byte, size int32, _ int32) int32 {
			copy(unsafe.Slice(out, int(size)), pci)
			return 0
		},
		"cuDeviceGetUuid_v2": func(out *[16]byte, _ int32) int32 { *out = uuid; return 0 },
		"cuDeviceGetUuid":    func(out *[16]byte, _ int32) int32 { *out = uuid; return 0 },
		"cuDeviceGetName": func(out *byte, size int32, _ int32) int32 {
			copy(unsafe.Slice(out, int(size)), name)
			return 0
		},
		"cuDeviceTotalMem_v2": func(out *uintptr, _ int32) int32 { *out = 8 << 30; return 0 },
		"cuDeviceGetLuid": func(out *[8]byte, mask *uint32, _ int32) int32 {
			*out, *mask = [8]byte{0x34, 0x12}, 1
			return 0
		},
	}
	return ggml, driver
}

func modelDeviceTestEnumerate(t *testing.T, selected modelDeviceRuntime, ggml, driver map[string]any) modelDeviceReply {
	t.Helper()
	return enumerateModelNativeDevicesWith(selected, func(modelDeviceRuntime) (modelDeviceSymbols, modelDeviceRegister, error) {
		symbol, register := modelDeviceTestSymbols(t, ggml)
		return symbol, register, nil
	}, func() (modelDeviceSymbols, modelDeviceRegister, error) {
		symbol, register := modelDeviceTestSymbols(t, driver)
		return symbol, register, nil
	})
}

func TestModelSettingsDevicesDriverCrossCheck(t *testing.T) {
	test := func(name string, mutate func(map[string]any, map[string]any), recognized bool) {
		t.Run(name, func(t *testing.T) {
			op := modelDevicesFixtureOperation(t)
			op.Runtime.Libraries = append(op.Runtime.Libraries, modelDeviceLibrary{Name: "ggml-cuda.dll"})
			ggml, driver := modelDeviceTestABI(t, "CUDA7")
			mutate(ggml, driver)
			got := modelDeviceTestEnumerate(t, op.Runtime, ggml, driver)
			if !recognized {
				if got.Unknown != "device report not recognized for this build" || len(got.Devices) != 0 {
					t.Fatalf("unrecognized report = %+v", got)
				}
				return
			}
			if got.Unknown != "" || got.Commit != "abcdef0-custom" || len(got.Devices) != 1 {
				t.Fatalf("recognized report = %+v", got)
			}
			d := got.Devices[0]
			if d.Selector != "CUDA7" || d.Name != "Synthetic CUDA GPU" || d.Total != 8589934592 || d.PCI != "0000:ab:00.0" || d.UUID != modelDevicesFixtureUUID || d.Unknown != "" {
				t.Fatalf("recognized device = %+v", d)
			}
			if runtime.GOOS == "windows" && d.LUID != modelDevicesFixtureLUID {
				t.Fatalf("driver LUID = %q", d.LUID)
			}
		})
	}
	test("agreement on another revision", func(map[string]any, map[string]any) {}, true)
	test("memory tolerance includes one MiB", func(_ map[string]any, driver map[string]any) {
		driver["cuDeviceTotalMem_v2"] = func(out *uintptr, _ int32) int32 { *out = (8 << 30) + (1 << 20); return 0 }
	}, true)
	test("memory lower tolerance includes one MiB", func(_ map[string]any, driver map[string]any) {
		driver["cuDeviceTotalMem_v2"] = func(out *uintptr, _ int32) int32 { *out = (8 << 30) - (1 << 20); return 0 }
	}, true)
	test("undecodable ggml name", func(ggml map[string]any, _ map[string]any) {
		original := ggml["ggml_backend_dev_get_props"].(func(uintptr, *modelGGMLProps))
		invalid := []byte{0xff, 0}
		ggml["ggml_backend_dev_get_props"] = func(device uintptr, p *modelGGMLProps) {
			original(device, p)
			p.Description = unsafe.Pointer(&invalid[0])
		}
	}, false)
	test("zero ggml total", func(ggml map[string]any, _ map[string]any) {
		original := ggml["ggml_backend_dev_get_props"].(func(uintptr, *modelGGMLProps))
		ggml["ggml_backend_dev_get_props"] = func(device uintptr, p *modelGGMLProps) {
			original(device, p)
			p.Total = 0
		}
	}, false)
	test("name mismatch", func(_ map[string]any, driver map[string]any) {
		driver["cuDeviceGetName"] = func(out *byte, size int32, _ int32) int32 {
			copy(unsafe.Slice(out, int(size)), "Another GPU\x00")
			return 0
		}
	}, false)
	test("memory exceeds upper tolerance", func(_ map[string]any, driver map[string]any) {
		driver["cuDeviceTotalMem_v2"] = func(out *uintptr, _ int32) int32 { *out = (8 << 30) + (1 << 20) + 1; return 0 }
	}, false)
	test("memory exceeds lower tolerance", func(_ map[string]any, driver map[string]any) {
		driver["cuDeviceTotalMem_v2"] = func(out *uintptr, _ int32) int32 { *out = (8 << 30) - (1 << 20) - 1; return 0 }
	}, false)
	test("undecodable driver name", func(_ map[string]any, driver map[string]any) {
		driver["cuDeviceGetName"] = func(out *byte, _ int32, _ int32) int32 { *out = 0xff; return 0 }
	}, false)
	test("unterminated driver name", func(_ map[string]any, driver map[string]any) {
		driver["cuDeviceGetName"] = func(out *byte, size int32, _ int32) int32 {
			copy(unsafe.Slice(out, int(size)), strings.Repeat("x", int(size)))
			return 0
		}
	}, false)
	test("zero driver memory", func(_ map[string]any, driver map[string]any) {
		driver["cuDeviceTotalMem_v2"] = func(out *uintptr, _ int32) int32 { *out = 0; return 0 }
	}, false)
	test("driver name call fails", func(_ map[string]any, driver map[string]any) {
		driver["cuDeviceGetName"] = func(*byte, int32, int32) int32 { return 1 }
	}, false)
	test("driver memory call fails", func(_ map[string]any, driver map[string]any) {
		driver["cuDeviceTotalMem_v2"] = func(*uintptr, int32) int32 { return 1 }
	}, false)
	test("missing driver name export", func(_ map[string]any, driver map[string]any) { delete(driver, "cuDeviceGetName") }, false)
	test("missing driver memory export", func(_ map[string]any, driver map[string]any) { delete(driver, "cuDeviceTotalMem_v2") }, false)
	test("missing commit export", func(ggml map[string]any, _ map[string]any) { delete(ggml, "ggml_commit") }, false)
	test("missing property export", func(ggml map[string]any, _ map[string]any) { delete(ggml, "ggml_backend_dev_get_props") }, false)
	test("missing loader export", func(ggml map[string]any, _ map[string]any) { delete(ggml, "ggml_backend_load") }, false)
	test("loader refuses selected module", func(ggml map[string]any, _ map[string]any) {
		ggml["ggml_backend_load"] = func(string) uintptr { return 0 }
	}, false)
}

func TestModelSettingsDevicesLibraryOpenFailure(t *testing.T) {
	got := enumerateModelNativeDevicesWith(modelDeviceRuntime{}, func(modelDeviceRuntime) (modelDeviceSymbols, modelDeviceRegister, error) {
		return nil, nil, errors.New("fake loader failure")
	}, func() (modelDeviceSymbols, modelDeviceRegister, error) {
		t.Fatal("driver opened after library failure")
		return nil, nil, errors.New("unexpected driver open")
	})
	if got.Unknown != "device report not recognized for this build" || len(got.Devices) != 0 {
		t.Fatalf("loader failure = %+v", got)
	}
}

func TestModelSettingsDevicesDriverOpenFailure(t *testing.T) {
	ggml, _ := modelDeviceTestABI(t, "CUDA0")
	got := enumerateModelNativeDevicesWith(modelDeviceRuntime{}, func(modelDeviceRuntime) (modelDeviceSymbols, modelDeviceRegister, error) {
		symbol, register := modelDeviceTestSymbols(t, ggml)
		return symbol, register, nil
	}, func() (modelDeviceSymbols, modelDeviceRegister, error) {
		return nil, nil, errors.New("fake driver open failure")
	})
	if got.Unknown != "device report not recognized for this build" || len(got.Devices) != 0 || got.Commit != "abcdef0-custom" {
		t.Fatalf("driver open failure = %+v", got)
	}
}

func TestModelSettingsDevicesSnapshotAcceptsFetchAndOverride(t *testing.T) {
	test := func(name string, override bool) {
		t.Run(name, func(t *testing.T) {
			f := newInspectionFixture(t, 4096)
			root := t.TempDir()
			f.st.installDir = root
			if override {
				f.st.installDir = t.TempDir()
				f.st.plat.Install = nil
			} else {
				f.st.plat.Install = &Install{Fetch: &Fetch{URL: "https://fixture.invalid/new-build.tgz"}, Run: []string{"never-run"}}
			}
			f.st.binPath = filepath.Join(root, "llama-server")
			writeTestFile(t, f.st.binPath, []byte("synthetic server "+t.Name()))
			writeTestFile(t, filepath.Join(root, "libggml.so"), []byte("synthetic metadata library"))
			got, err := modelDeviceRuntimeSnapshot(context.Background(), f.st)
			if err != nil {
				t.Fatal(err)
			}
			wantID, err := nativeBuildID(context.Background(), f.st)
			if err != nil {
				t.Fatal(err)
			}
			wantRoot, err := resolveExistingPath(root)
			if err != nil {
				t.Fatal(err)
			}
			if got.Build == "" || got.Build != wantID || got.Root != wantRoot || len(got.Libraries) != 1 || got.Libraries[0].Name != "libggml.so" {
				t.Fatalf("snapshot = %+v, want build %s in %s", got, wantID, wantRoot)
			}
		})
	}
	test("install fetch", false)
	test("override outside managed directory", true)
}

func TestModelSettingsDevicesRecognitionCacheKeepsMapsOperationScoped(t *testing.T) {
	op := modelDevicesFixtureOperation(t)
	first := mapModelSettingsDevices(context.Background(), op, modelDevicesFixtureRunner)
	checked, ok := modelDeviceCheckedBuild(op.Runtime.Build)
	if !ok || checked.Unknown != "" || checked.Commit != "abcdef0-custom" || first.Binding == nil {
		t.Fatalf("recognition = %+v, %v; map = %+v", checked, ok, first)
	}
	op.Topology.Host, op.Topology.Run, op.Topology.Observation = "other-host", "other-run", "other-observation"
	calls := 0
	second := mapModelSettingsDevices(context.Background(), op, func(ctx context.Context, r modelDeviceRuntime, env []string) (modelDeviceReply, error) {
		calls++
		return modelDevicesFixtureRunner(ctx, r, env)
	})
	if calls != 2 || second.Binding == nil || second.Binding.Host != "other-host" || second.Binding.Digest == first.Binding.Digest || second.ObservationID == first.ObservationID {
		t.Fatalf("cached recognition reused a map: calls=%d, second=%+v", calls, second)
	}
	op.NativeInitAllowed = false
	paused := mapModelSettingsDevices(context.Background(), op, func(context.Context, modelDeviceRuntime, []string) (modelDeviceReply, error) {
		t.Fatal("cached recognition bypassed Pause")
		return modelDeviceReply{}, nil
	})
	if paused.Unknown != "native initialization prohibited by target activation/Pause" || paused.Binding != nil || len(paused.Devices) != 0 {
		t.Fatalf("paused cached build = %+v", paused)
	}
}

func TestModelSettingsDevicesCanceledRecognitionIsNotCached(t *testing.T) {
	op := modelDevicesFixtureOperation(t)
	ctx, cancel := context.WithCancel(context.Background())
	got := mapModelSettingsDevices(ctx, op, func(context.Context, modelDeviceRuntime, []string) (modelDeviceReply, error) {
		cancel()
		return modelDeviceReply{Unknown: "fake incomplete check"}, nil
	})
	if got.Unknown == "" || got.Binding != nil || len(got.Devices) != 0 {
		t.Fatalf("canceled check = %+v", got)
	}
	if checked, ok := modelDeviceCheckedBuild(op.Runtime.Build); ok {
		t.Fatalf("canceled check cached: %+v", checked)
	}
}
