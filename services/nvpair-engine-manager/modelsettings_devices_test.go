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
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"
)

const modelDevicesFixtureUUID = "GPU-12345678-1234-1234-1234-123456789abc"
const modelDevicesFixtureLUID = "luid_0x00000000_0x00001234_phys_0"

func modelDevicesFixtureOperation() modelDeviceOperation {
	return modelDeviceOperation{
		Runtime: modelDeviceRuntime{Build: modelDevicesBuild, Libraries: []modelDeviceLibrary{{Name: "ggml.dll", SHA256: strings.Repeat("a", 64)}}},
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
	return modelDeviceReply{Devices: []modelNativeDevice{
		{Selector: "CPU", Kind: 0},
		{Selector: selector, Kind: 1, PCI: "0000:ab:00.0", UUID: modelDevicesFixtureUUID, LUID: modelDevicesFixtureLUID},
	}}, nil
}

func TestModelSettingsDevicesPhysicalSelection(t *testing.T) {
	op := modelDevicesFixtureOperation()
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
	base := mapModelSettingsDevices(context.Background(), modelDevicesFixtureOperation(), modelDevicesFixtureRunner)
	if base.Binding == nil {
		t.Fatalf("base = %+v", base)
	}
	test := func(name string, change func(*modelDeviceOperation), sameBinding bool) {
		t.Run(name, func(t *testing.T) {
			op := modelDevicesFixtureOperation()
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

func TestModelSettingsDevicesReorderedIdenticalAdapters(t *testing.T) {
	op := modelDevicesFixtureOperation()
	other := op.Topology.Pools[1]
	other.ID = "physical-b"
	other.PCI = "0000:ac:00.0"
	other.UUID = "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
	other.LUID = "luid_0x00000000_0x00005678_phys_0"
	op.Topology.Pools = append(op.Topology.Pools, other)
	runner := func(swapped bool) modelDeviceRunner {
		return func(ctx context.Context, r modelDeviceRuntime, env []string) (modelDeviceReply, error) {
			reply, err := modelDevicesFixtureRunner(ctx, r, env)
			if reply.Devices[1].Selector == "CUDA0" {
				return reply, err
			}
			second := modelNativeDevice{Selector: "CUDA8", Kind: 1, PCI: other.PCI, UUID: other.UUID, LUID: other.LUID}
			if swapped {
				reply.Devices[1].Selector = "CUDA8"
				second.Selector = "CUDA7"
				reply.Devices = []modelNativeDevice{second, reply.Devices[1], reply.Devices[0]}
			} else {
				reply.Devices = append(reply.Devices, second)
			}
			return reply, err
		}
	}
	first := mapModelSettingsDevices(context.Background(), op, runner(false))
	op.Topology.Pools[1], op.Topology.Pools[2] = op.Topology.Pools[2], op.Topology.Pools[1]
	op.Selector = "CUDA8"
	second := mapModelSettingsDevices(context.Background(), op, runner(true))
	if first.Binding == nil || second.Binding == nil {
		t.Fatalf("maps = %+v / %+v", first, second)
	}
	if first.Binding.Digest != second.Binding.Digest || second.Binding.UUID != modelDevicesFixtureUUID {
		t.Fatalf("physical binding changed: %+v / %+v", first.Binding, second.Binding)
	}
}

func TestModelSettingsDevicesUnknownInputs(t *testing.T) {
	test := func(name string, change func(*modelDeviceOperation)) {
		t.Run(name, func(t *testing.T) {
			op := modelDevicesFixtureOperation()
			change(&op)
			got := mapModelSettingsDevices(context.Background(), op, modelDevicesFixtureRunner)
			if got.Unknown == "" || got.Binding != nil {
				t.Fatalf("expected Unknown, got %+v", got)
			}
		})
	}
	test("unsupported build", func(op *modelDeviceOperation) { op.Runtime.Build = "other" })
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
			got := mapModelSettingsDevices(context.Background(), modelDevicesFixtureOperation(), run)
			if got.Unknown == "" || got.Binding != nil {
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
	op := modelDevicesFixtureOperation()
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
	got := mapModelSettingsDevices(ctx, modelDevicesFixtureOperation(), func(context.Context, modelDeviceRuntime, []string) (modelDeviceReply, error) {
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

func TestModelSettingsDevicesLoadedRevision(t *testing.T) {
	for _, revision := range []string{modelDevicesCommit, modelDevicesCommit[:7], "unknown", "7fe450e-dirty", "0123456", "7fe450"} {
		t.Run(revision, func(t *testing.T) {
			lookup := func(name string) (uintptr, error) {
				if name != "ggml_commit" {
					t.Fatalf("unexpected property/initialization export %s", name)
				}
				return 1, nil
			}
			register := func(fn any, _ uintptr) {
				p, ok := fn.(*func() string)
				if !ok {
					t.Fatalf("unexpected revision signature %T", fn)
				}
				*p = func() string { return revision }
			}
			err := checkModelDeviceCommit(lookup, register)
			want := revision == modelDevicesCommit || revision == modelDevicesCommit[:7]
			if (err == nil) != want {
				t.Fatalf("revision %q accepted=%v, want %v", revision, err == nil, want)
			}
		})
	}
	if err := checkModelDeviceCommit(func(string) (uintptr, error) { return 0, errors.New("missing") }, func(any, uintptr) { t.Fatal("missing export registered") }); err == nil {
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
