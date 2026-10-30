// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build windows

package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"
	"unsafe"

	"golang.org/x/sys/windows"
)

func init() { workingSetTestChild = runWorkingSetTestChild }

type workingSetTestObservation struct {
	Role         string `json:"role"`
	PID          int    `json:"pid"`
	Maximum      uint64 `json:"maximum"`
	Flags        uint32 `json:"flags"`
	Peak         uint64 `json:"peak"`
	TouchedBytes int    `json:"touchedBytes"`
}

// PROCESS_MEMORY_COUNTERS: DWORD header, then pointer-sized counters.
type workingSetTestMemoryCounters struct {
	Size                  uint32
	PageFaultCount        uint32
	PeakWorkingSetSize    uintptr
	WorkingSetSize        uintptr
	QuotaPeakPagedPool    uintptr
	QuotaPagedPool        uintptr
	QuotaPeakNonPagedPool uintptr
	QuotaNonPagedPool     uintptr
	PagefileUsage         uintptr
	PeakPagefileUsage     uintptr
}

func runWorkingSetTestChild(mode string) error {
	capped := strings.HasSuffix(mode, "-capped")
	role := "child"
	if strings.HasPrefix(mode, "grandchild-") {
		role = "grandchild"
	}
	// Check the root immediately to prove enforcement before resume. The
	// grandchild allocates immediately, without waiting for its notification,
	// so a delayed cap cannot hide an excessive peak from the assertion.
	_, maximum, flags, err := processWorkingSetLimits(windows.CurrentProcess())
	if err != nil {
		return err
	}
	if capped && role == "child" && flags&0x6 != 0x6 {
		return fmt.Errorf("root resumed without working-set cap: max=%d flags=%#x", maximum, flags)
	}
	const allocationBytes = 256 << 20
	data := make([]byte, allocationBytes)
	defer runtime.KeepAlive(data)
	for i := 0; i < len(data); i += 4096 {
		data[i] = 1
	}
	var pagesTouched int
	for i := 0; i < len(data); i += 4096 {
		pagesTouched += int(data[i])
	}
	if pagesTouched != allocationBytes/4096 {
		return fmt.Errorf("private allocation touch count=%d", pagesTouched)
	}
	_, maximum, flags, err = processWorkingSetLimits(windows.CurrentProcess())
	if err != nil {
		return err
	}
	counters := workingSetTestMemoryCounters{Size: uint32(unsafe.Sizeof(workingSetTestMemoryCounters{}))}
	getMemoryInfo := windows.NewLazySystemDLL("psapi.dll").NewProc("GetProcessMemoryInfo")
	result, _, callErr := getMemoryInfo.Call(uintptr(windows.CurrentProcess()), uintptr(unsafe.Pointer(&counters)), uintptr(counters.Size))
	if result == 0 {
		return callErr
	}
	observation := workingSetTestObservation{Role: role, PID: os.Getpid(), Maximum: uint64(maximum), Flags: flags, Peak: uint64(counters.PeakWorkingSetSize), TouchedBytes: pagesTouched * 4096}
	if err := json.NewEncoder(os.Stdout).Encode(observation); err != nil {
		return err
	}
	if role == "grandchild" {
		return nil
	}
	grandchildMode := "grandchild-default"
	if capped {
		grandchildMode = "grandchild-capped"
	}
	cmd := exec.Command(os.Args[0], "working-set-test-child", grandchildMode)
	cmd.Env = childEnv(nil)
	configureSysProcAttr(cmd)
	cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
	return cmd.Run()
}

func workingSetTestTree(t *testing.T, limit uint64, mode string) []workingSetTestObservation {
	t.Helper()
	var observations []workingSetTestObservation
	var outputErrors []string
	var outputMu sync.Mutex
	mp, err := startManagedProcObserved(os.Args[0], []string{"working-set-test-child", mode}, nil, func(stream, line string) {
		outputMu.Lock()
		defer outputMu.Unlock()
		if stream != "stdout" {
			outputErrors = append(outputErrors, line)
			return
		}
		var observation workingSetTestObservation
		if err := json.Unmarshal([]byte(line), &observation); err != nil {
			outputErrors = append(outputErrors, err.Error())
			return
		}
		observations = append(observations, observation)
	}, nil, limit)
	if err != nil {
		t.Fatalf("start working-set helper: %v", err)
	}
	t.Cleanup(func() { mp.stop(0) })
	select {
	case <-mp.done:
	case <-time.After(45 * time.Second):
		t.Fatal("working-set helper did not finish")
	}
	if !mp.cmd.ProcessState.Success() || len(outputErrors) != 0 {
		t.Fatalf("working-set helper failed: state=%v errors=%v", mp.cmd.ProcessState, outputErrors)
	}
	if len(observations) != 2 || observations[0].Role != "child" || observations[1].Role != "grandchild" {
		t.Fatalf("want child and grandchild observations, got %+v", observations)
	}
	if observations[0].PID == observations[1].PID || observations[0].PID != mp.cmd.Process.Pid {
		t.Fatalf("observations must describe distinct root and descendant PIDs: %+v", observations)
	}
	for _, observation := range observations {
		if observation.TouchedBytes != 268435456 || observation.Peak == 0 {
			t.Fatalf("helper did not measure a touched 256 MiB allocation: %+v", observation)
		}
		t.Logf("%s pid=%d max=%d flags=%#x peak=%d touched=%d", observation.Role, observation.PID, observation.Maximum, observation.Flags, observation.Peak, observation.TouchedBytes)
		assertMemoryPlanProcessStopped(t, observation.PID)
	}
	return observations
}

func TestWorkingSetCapsRootAndGrandchild(t *testing.T) {
	for _, observation := range workingSetTestTree(t, 64<<20, "child-capped") {
		if observation.Maximum != 67108864 || observation.Flags&0x6 != 0x6 {
			t.Errorf("%s quota max=%d flags=%#x, want max=67108864 flags containing 0x6", observation.Role, observation.Maximum, observation.Flags)
		}
		if observation.Peak > 68157440 {
			t.Errorf("%s peak=%d, exceeds 65 MiB tolerance", observation.Role, observation.Peak)
		}
	}
}

func TestWorkingSetZeroLeavesDefaultQuotas(t *testing.T) {
	for _, observation := range workingSetTestTree(t, 0, "child-default") {
		if observation.Flags&0x4 != 0 {
			t.Errorf("%s hard working-set maximum enabled with no limit: flags=%#x", observation.Role, observation.Flags)
		}
	}
}

func TestWorkingSetRootCapFailureReapsSuspendedProcess(t *testing.T) {
	original := capRootWorkingSet
	t.Cleanup(func() { capRootWorkingSet = original })
	var pid uint32
	var handle windows.Handle
	capRootWorkingSet = func(root uint32, _ uint64) error {
		pid = root
		var err error
		handle, err = windows.OpenProcess(windows.SYNCHRONIZE|windows.PROCESS_QUERY_LIMITED_INFORMATION, false, root)
		if err != nil {
			return fmt.Errorf("capture failed root handle: %w", err)
		}
		return windows.ERROR_ACCESS_DENIED
	}
	mp, err := startManagedProcObserved(os.Args[0], []string{"working-set-test-child", "child-default"}, nil, nil, nil, 64<<20)
	if mp != nil || !errors.Is(err, windows.ERROR_ACCESS_DENIED) || pid == 0 || handle == 0 {
		t.Fatalf("want injected root failure after process creation: proc=%v err=%v pid=%d handle=%d", mp, err, pid, handle)
	}
	t.Cleanup(func() {
		if err := windows.CloseHandle(handle); err != nil {
			t.Errorf("close failed-root observation handle: %v", err)
		}
	})
	status, waitErr := windows.WaitForSingleObject(handle, 0)
	if waitErr != nil || status != windows.WAIT_OBJECT_0 {
		t.Fatalf("failed root survived return: pid=%d status=%d err=%v", pid, status, waitErr)
	}
	assertMemoryPlanProcessStopped(t, int(pid))
	t.Logf("forced root cap failure pid=%d: signaled and reaped before return", pid)
}

func TestWorkingSetNotificationIgnoresProcessOutsideJob(t *testing.T) {
	c, err := newWorkingSetContainment("fixture", 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = c.Close() })
	// A suspended process that never joined the job stands in for a PID that
	// was reused after the notified job member exited.
	cmd := exec.Command(os.Args[0], "working-set-test-child", "child-default")
	cmd.Env = childEnv(nil)
	configureSysProcAttr(cmd)
	cmd.SysProcAttr.CreationFlags |= windows.CREATE_SUSPENDED
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
	})
	c.capMember(uint32(cmd.Process.Pid))
	process, err := windows.OpenProcess(windows.SYNCHRONIZE|windows.PROCESS_QUERY_LIMITED_INFORMATION, false, uint32(cmd.Process.Pid))
	if err != nil {
		t.Fatal(err)
	}
	defer windows.CloseHandle(process)
	if status, err := windows.WaitForSingleObject(process, 0); err != nil || status != uint32(windows.WAIT_TIMEOUT) {
		t.Fatalf("process outside the job was terminated: status=%d err=%v", status, err)
	}
	_, maximum, flags, err := processWorkingSetLimits(process)
	if err != nil {
		t.Fatal(err)
	}
	if flags&0x4 != 0 || maximum == 67108864 {
		t.Fatalf("process outside the job was capped: max=%d flags=%#x", maximum, flags)
	}
}

func TestWorkingSetContainmentCloseIsIdempotent(t *testing.T) {
	c, err := newWorkingSetContainment("fixture", 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	if err := c.Close(); err != nil {
		t.Fatalf("close unused containment: %v", err)
	}
	if err := c.Close(); err != nil {
		t.Fatalf("repeat close: %v", err)
	}
	select {
	case <-c.done:
	default:
		t.Fatal("completion reader survived Close")
	}
}
