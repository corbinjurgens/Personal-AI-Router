// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build windows

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"
	"unsafe"

	"golang.org/x/sys/windows"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/modelconfig"
)

func init() { workingSetTestChild = runWorkingSetTestChild }

type workingSetTestObservation struct {
	Role         string        `json:"role"`
	PID          int           `json:"pid"`
	Maximum      uint64        `json:"maximum"`
	Flags        uint32        `json:"flags"`
	Peak         uint64        `json:"peak"`
	TouchedBytes int           `json:"touchedBytes"`
	CapDelay     time.Duration `json:"capDelay"`
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
	if mode == "exit" {
		return nil
	}
	if mode == "sentinel" {
		_, err := io.Copy(io.Discard, os.Stdin)
		return err
	}
	if mode == "child-sweep-capped" {
		// The parent releases this gate after the watcher's initial sweep,
		// making the later descendant depend on a periodic sweep alone.
		var gate [1]byte
		if _, err := io.ReadFull(os.Stdin, gate[:]); err != nil {
			return err
		}
	}
	capped := strings.HasSuffix(mode, "-capped")
	role := "child"
	if strings.HasPrefix(mode, "grandchild-") {
		role = "grandchild"
	}
	// Check the root immediately to prove enforcement before resume. The
	// notification-path grandchild allocates immediately so a delayed cap
	// cannot hide an excessive peak. The sweep fixture waits for its cap.
	_, maximum, flags, err := processWorkingSetLimits(windows.CurrentProcess())
	if err != nil {
		return err
	}
	var capDelay time.Duration
	if mode == "grandchild-sweep-capped" {
		started := time.Now()
		for uint64(maximum) != 67108864 || flags&0x6 != 0x6 {
			if time.Since(started) >= 5*time.Second {
				return fmt.Errorf("grandchild did not observe its cap within 5 s: max=%d flags=%#x", maximum, flags)
			}
			time.Sleep(10 * time.Millisecond)
			_, maximum, flags, err = processWorkingSetLimits(windows.CurrentProcess())
			if err != nil {
				return err
			}
		}
		capDelay = time.Since(started)
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
	observation := workingSetTestObservation{Role: role, PID: os.Getpid(), Maximum: uint64(maximum), Flags: flags, Peak: uint64(counters.PeakWorkingSetSize), TouchedBytes: pagesTouched * 4096, CapDelay: capDelay}
	if err := json.NewEncoder(os.Stdout).Encode(observation); err != nil {
		return err
	}
	if role == "grandchild" {
		if mode == "grandchild-sweep-capped" {
			_, err := io.Copy(io.Discard, os.Stdin)
			return err
		}
		return nil
	}
	grandchildMode := "grandchild-default"
	if capped {
		grandchildMode = "grandchild-capped"
	}
	if mode == "child-sweep-capped" {
		grandchildMode = "grandchild-sweep-capped"
	}
	cmd := exec.Command(os.Args[0], "working-set-test-child", grandchildMode)
	cmd.Env = childEnv(nil)
	configureSysProcAttr(cmd)
	cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
	if mode == "child-sweep-capped" {
		cmd.Stdin = os.Stdin
	}
	return cmd.Run()
}

func unnotifiedWorkingSetContainment(t *testing.T) *workingSetContainment {
	t.Helper()
	job, err := windows.CreateJobObject(nil, nil)
	if err != nil {
		t.Fatalf("create unnotified job: %v", err)
	}
	port, err := windows.CreateIoCompletionPort(windows.InvalidHandle, 0, 0, 1)
	if err != nil {
		if closeErr := windows.CloseHandle(job); closeErr != nil {
			t.Errorf("close unnotified job: %v", closeErr)
		}
		t.Fatalf("create unassociated completion port: %v", err)
	}
	c := &workingSetContainment{job: job, port: port, engine: "fixture", maxBytes: 64 << 20, stop: make(chan struct{}), done: make(chan struct{})}
	t.Cleanup(func() {
		if err := c.Close(); err != nil {
			t.Errorf("close unnotified containment: %v", err)
		}
	})
	return c
}

func startUnnotifiedWorkingSetTree(t *testing.T, c *workingSetContainment) io.Reader {
	t.Helper()
	cmd := exec.Command(os.Args[0], "working-set-test-child", "child-sweep-capped")
	cmd.Env = childEnv(nil)
	cmd.Stderr = os.Stderr
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	stdin, err := cmd.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := startSuspendedJobProcess(ctx, cmd, c.job, func(pid uint32) error {
		if err := capWorkingSet(pid, c.maxBytes); err != nil {
			return err
		}
		c.verifiedAt.Store(time.Now().UnixNano())
		return nil
	}, c.Close); err != nil {
		t.Fatalf("start unnotified working-set tree: %v", err)
	}
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := memoryPlanTerminateJob(ctx, c.job); err != nil {
			t.Errorf("terminate unnotified tree: %v", err)
		}
		if err := stdin.Close(); err != nil {
			t.Errorf("close unnotified tree input: %v", err)
		}
		if err := cmd.Wait(); err != nil {
			var exitErr *exec.ExitError
			if !errors.As(err, &exitErr) {
				t.Errorf("reap unnotified tree: %v", err)
			}
		}
	})
	rootVerifiedAt := c.verifiedAt.Load()
	c.start()
	deadline := time.Now().Add(2 * time.Second)
	for c.verifiedAt.Load() == rootVerifiedAt {
		if time.Now().After(deadline) {
			t.Fatal("initial working-set sweep did not finish")
		}
		time.Sleep(10 * time.Millisecond)
	}
	if _, err := stdin.Write([]byte{1}); err != nil {
		t.Fatalf("release working-set root after initial sweep: %v", err)
	}
	return stdout
}

func TestWorkingSetSweepCapsDescendantWithoutNotification(t *testing.T) {
	c := unnotifiedWorkingSetContainment(t)
	stdout := startUnnotifiedWorkingSetTree(t, c)
	type result struct {
		observations [2]workingSetTestObservation
		err          error
	}
	observed := make(chan result, 1)
	go func() {
		var got result
		decoder := json.NewDecoder(stdout)
		for i := range got.observations {
			if err := decoder.Decode(&got.observations[i]); err != nil {
				got.err = err
				break
			}
		}
		observed <- got
	}()
	select {
	case got := <-observed:
		if got.err != nil {
			t.Fatalf("read unnotified tree observations: %v", got.err)
		}
		if got.observations[0].Role != "child" || got.observations[1].Role != "grandchild" || got.observations[0].PID == got.observations[1].PID {
			t.Fatalf("want distinct root and grandchild observations: %+v", got.observations)
		}
		for _, observation := range got.observations {
			if observation.Maximum != 67108864 || observation.Flags&0x6 != 0x6 {
				t.Errorf("%s quota max=%d flags=%#x, want max=67108864 flags containing 0x6", observation.Role, observation.Maximum, observation.Flags)
			}
			if observation.TouchedBytes != 268435456 || observation.Peak == 0 || observation.Peak > 68157440 {
				t.Errorf("%s touched=%d peak=%d, want 256 MiB touched and peak at most 65 MiB", observation.Role, observation.TouchedBytes, observation.Peak)
			}
			if !pidAlive(observation.PID) {
				t.Errorf("%s exited before containment verification", observation.Role)
			}
			t.Logf("%s max=%d flags=%#x peak=%d capDelay=%s", observation.Role, observation.Maximum, observation.Flags, observation.Peak, observation.CapDelay)
		}
		if got.observations[1].CapDelay > 2*time.Second {
			t.Errorf("grandchild cap took %s, want at most 2 s", got.observations[1].CapDelay)
		}
		if !c.Verified() {
			t.Fatal("running tree containment is not verified")
		}
	case <-time.After(45 * time.Second):
		t.Fatal("unnotified tree did not report its cap")
	}
}

func TestWorkingSetExitedMemberIsVerifiedWithoutTermination(t *testing.T) {
	c := unnotifiedWorkingSetContainment(t)
	start := func(mode string) (*exec.Cmd, windows.Handle) {
		t.Helper()
		cmd := exec.Command(os.Args[0], "working-set-test-child", mode)
		cmd.Env = childEnv(nil)
		stdin, err := cmd.StdinPipe()
		if err != nil {
			t.Fatal(err)
		}
		var handle windows.Handle
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := startSuspendedJobProcess(ctx, cmd, c.job, func(pid uint32) error {
			var err error
			handle, err = openWorkingSetProcess(pid)
			return err
		}, c.Close); err != nil {
			t.Fatalf("start %s member: %v", mode, err)
		}
		t.Cleanup(func() {
			if err := stdin.Close(); err != nil {
				t.Errorf("close %s input: %v", mode, err)
			}
			if err := windows.TerminateProcess(handle, 1); err != nil {
				if status, waitErr := windows.WaitForSingleObject(handle, 0); waitErr != nil || status != windows.WAIT_OBJECT_0 {
					t.Errorf("terminate %s: %v", mode, err)
				}
			}
			if err := cmd.Wait(); err != nil {
				var exitErr *exec.ExitError
				if !errors.As(err, &exitErr) {
					t.Errorf("reap %s: %v", mode, err)
				}
			}
			if err := windows.CloseHandle(handle); err != nil {
				t.Errorf("close %s handle: %v", mode, err)
			}
		})
		return cmd, handle
	}
	short, shortHandle := start("exit")
	_, sentinelHandle := start("sentinel")
	if status, err := windows.WaitForSingleObject(shortHandle, 5000); err != nil || status != windows.WAIT_OBJECT_0 {
		t.Fatalf("short-lived child did not exit: status=%d err=%v", status, err)
	}
	verified, capped := c.capMember(uint32(short.Process.Pid))
	if !verified || capped {
		t.Errorf("exited member verified=%t capped=%t, want true, false", verified, capped)
	}
	if status, err := windows.WaitForSingleObject(sentinelHandle, 0); err != nil || status != uint32(windows.WAIT_TIMEOUT) {
		t.Fatalf("sentinel terminated after exited-member cap: status=%d err=%v", status, err)
	}
}

func TestWorkingSetVerifiedRequiresFreshEvidence(t *testing.T) {
	c := unnotifiedWorkingSetContainment(t)
	if c.Verified() {
		t.Error("zero timestamp must not be verified")
	}
	c.verifiedAt.Store(time.Now().UnixNano())
	if !c.Verified() {
		t.Error("fresh timestamp must be verified")
	}
	c.verifiedAt.Store(time.Now().Add(-2 * workingSetVerificationMaxAge).UnixNano())
	if c.Verified() {
		t.Error("stale timestamp must not be verified")
	}
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
	if verified, capped := c.capMember(uint32(cmd.Process.Pid)); !verified || capped {
		t.Errorf("nonmember verified=%t capped=%t, want true, false", verified, capped)
	}
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
	checkClose := func(t *testing.T, started bool) {
		t.Helper()
		c, err := newWorkingSetContainment("fixture", 64<<20)
		if err != nil {
			t.Fatal(err)
		}
		if started {
			c.start()
		}
		closed := make(chan error, 1)
		go func() { closed <- errors.Join(c.Close(), c.Close()) }()
		select {
		case err := <-closed:
			if err != nil {
				t.Fatalf("close containment twice: %v", err)
			}
		case <-time.After(2 * time.Second):
			t.Fatal("closing containment blocked")
		}
		select {
		case <-c.done:
		default:
			t.Fatal("completion reader survived Close")
		}
		if c.Verified() {
			t.Fatal("closed containment is still verified")
		}
	}
	t.Run("unstarted containment", func(t *testing.T) { checkClose(t, false) })
	t.Run("started containment", func(t *testing.T) { checkClose(t, true) })
}

func TestWorkingSetLimitSurvivesModelChangePreview(t *testing.T) {
	f := newModelLoadFixture(t)
	limit := 4096
	f.st.plat.Runtime.WorkingSetLimitMiB = &limit
	two := int32(2)
	preview := previewModelLoadFixture(t, f, "weak-model", modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{Slots: &two}})
	if preview.Settings.WorkingSetLimitMiB == nil || *preview.Settings.WorkingSetLimitMiB != 4096 {
		t.Fatalf("model change preview dropped the saved working-set limit: %v", preview.Settings.WorkingSetLimitMiB)
	}
}

func TestWorkingSetLimitEngineStartCapsProcess(t *testing.T) {
	e := workingSetSettingsExecutor(t)
	limit := 256
	request := settingsRequest(t, e)
	request.Settings.WorkingSetLimitMiB = &limit
	preview := previewSettings(t, e, request)
	if _, err := e.ConfigureLaunch(context.Background(), settings.Configure{Engine: "fake", Settings: preview.Settings}, func() error { return nil }); err != nil {
		t.Fatal(err)
	}
	if err := e.Start(context.Background(), "fake"); err != nil {
		t.Fatal(err)
	}
	st := settingsState(t, e)
	st.mu.Lock()
	pid := st.proc.cmd.Process.Pid
	st.mu.Unlock()
	process, err := windows.OpenProcess(windows.PROCESS_QUERY_LIMITED_INFORMATION, false, uint32(pid))
	if err != nil {
		t.Fatal(err)
	}
	defer windows.CloseHandle(process)
	_, maximum, flags, err := processWorkingSetLimits(process)
	if err != nil {
		t.Fatal(err)
	}
	if maximum != 268435456 || flags&0x6 != 0x6 {
		t.Fatalf("started engine quota max=%d flags=%#x, want max=268435456 flags containing 0x6", maximum, flags)
	}
}
