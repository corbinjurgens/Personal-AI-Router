// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build windows

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	"golang.org/x/sys/windows"
	"nvpair-shared/servicectl"
)

const (
	consoleTargetEnv = "NVPAIR_SERVICE_TEST_CONSOLE_PID"
	consoleReportEnv = "NVPAIR_SERVICE_TEST_CONSOLE_REPORT"
)

func TestWindowsAutostartRunRejectsMalformedServiceArguments(t *testing.T) {
	test := func(name string, args []string, diagnostic string) {
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			t.Setenv("LOCALAPPDATA", dir)
			t.Setenv(servicectl.EndpointEnv, testEndpoint(dir))
			args = append([]string{"autostart", "run", "--broker-path", filepath.Join(dir, "missing-broker.exe")}, args...)
			var stdout, stderr bytes.Buffer
			if code := run(args, &stdout, &stderr); code != 1 {
				t.Fatalf("exit = %d, want 1; stdout=%q stderr=%q", code, stdout.String(), stderr.String())
			}
			if stdout.Len() != 0 || !strings.Contains(stderr.String(), diagnostic) {
				t.Fatalf("stdout=%q stderr=%q, want validation diagnostic %q", stdout.String(), stderr.String(), diagnostic)
			}
			entries, err := os.ReadDir(dir)
			if err != nil {
				t.Fatal(err)
			}
			if len(entries) != 0 {
				t.Fatalf("invalid start created files: %v", entries)
			}
		})
	}
	test("unknown service flag", []string{"--no-such-flag"}, "flag provided but not defined")
	test("missing flag value", []string{"--log-dir"}, "flag needs an argument")
	test("unexpected positional argument", []string{"serve"}, `unexpected argument "serve"`)
}

func TestWindowsAutostartRunHasNoConsoleAfterStarterExits(t *testing.T) {
	f := newWindowsAutostartFixture(t)
	starter := f.cli(append([]string{"autostart", "run"}, f.args...)...)
	if starter.err != nil || starter.stdout != "" || starter.stderr != "" {
		t.Fatalf("starter: %+v", starter)
	}
	ready := f.ready()
	status := f.status()
	if status.PID == starter.pid || status.PID == ready.PID || status.BrokerPID != ready.PID {
		t.Fatalf("starter PID=%d, ready broker PID=%d, service status=%+v", starter.pid, ready.PID, status)
	}
	probe := f.inspectConsole(status.PID)
	// CREATE_NO_WINDOW can retain an attachable headless console on Windows.
	// A real console window, including a hidden one, must still be rejected.
	if probe.Window != 0 || probe.Visible || (!probe.Attached && probe.AttachError != uint32(windows.ERROR_INVALID_HANDLE)) {
		t.Fatalf("daemon has a console window or inspection failed: %+v", probe)
	}
	// Both status calls run after the starter was joined. The same daemon and
	// fake broker must still answer after the independent console inspection.
	after := f.status()
	if after.PID != status.PID || after.BrokerPID != status.BrokerPID || after.BrokerRestarts != 0 {
		t.Fatalf("daemon changed after starter exit: before=%+v after=%+v", status, after)
	}
}

func TestWindowsAutostartRunPreservesArgumentsAndWorkingDirectory(t *testing.T) {
	f := newWindowsAutostartFixture(t)
	f.startAutostart()
	ready := f.ready()
	status := f.status()
	var daemon serviceProcessRecord
	f.readJSON(filepath.Join(f.records, fmt.Sprintf("daemon-%d.json", status.PID)), &daemon)
	if daemon.PID != status.PID || !slices.Equal(daemon.Args, f.args) {
		t.Fatalf("daemon record=%+v, want PID=%d args=%q", daemon, status.PID, f.args)
	}
	wantBrokerArgs := append(slices.Clone(f.brokerArgs), "--log-level", "debug")
	if ready.Version != "proc" || ready.PID != status.BrokerPID || !slices.Equal(ready.Args, wantBrokerArgs) {
		t.Fatalf("broker app:ready=%+v, want args=%q and PID=%d", ready, wantBrokerArgs, status.BrokerPID)
	}
	f.assertDirectory(daemon.Cwd, filepath.Dir(f.exe))
	f.assertDirectory(ready.Cwd, filepath.Dir(f.exe))
	for _, name := range []string{"service.log", "broker.log"} {
		if _, err := os.Stat(filepath.Join(f.logDir, name)); err != nil {
			t.Fatalf("forwarded --log-dir did not receive %s: %v", name, err)
		}
	}
}

func TestWindowsAutostartRunKeepsConsoleCLIUsable(t *testing.T) {
	f := newWindowsAutostartFixture(t)
	f.startAutostart()
	f.ready()
	status := f.status()
	version := f.cli("--version")
	if version.err != nil || version.stdout != Version+"\n" || version.stderr != "" {
		t.Fatalf("version CLI: %+v", version)
	}
	call := f.cli("call", "echo", `{"text":"space and \"quotes\"","values":[1,2]}`)
	if call.err != nil || call.stderr != "" {
		t.Fatalf("call CLI: %+v", call)
	}
	var echoed struct {
		Text   string `json:"text"`
		Values []int  `json:"values"`
	}
	if err := json.Unmarshal([]byte(call.stdout), &echoed); err != nil {
		t.Fatalf("call output is not JSON: %v; %q", err, call.stdout)
	}
	if echoed.Text != `space and "quotes"` || !slices.Equal(echoed.Values, []int{1, 2}) {
		t.Fatalf("call output = %+v", echoed)
	}
	invalid := f.cli("call", "echo", "{")
	if invalid.code != 2 || invalid.stdout != "" || !strings.Contains(invalid.stderr, "params are not valid JSON") {
		t.Fatalf("invalid call CLI: %+v", invalid)
	}
	stop := f.cli("stop")
	if stop.err != nil || stop.stdout != "nvpair-service: stopped\n" || stop.stderr != "" {
		t.Fatalf("stop CLI: %+v", stop)
	}
	f.assertExited(status.BrokerPID)
	f.assertExited(status.PID)
	after := f.cli("status")
	if after.code != 3 || after.stdout != "nvpair-service: not running\n" || after.stderr != "" {
		t.Fatalf("status after stop: %+v", after)
	}
}

func TestWindowsConsoleInspectorDetectsHiddenForegroundConsole(t *testing.T) {
	f := newWindowsAutostartFixture(t)
	f.startForeground()
	f.ready()
	status := f.status()
	if status.PID != f.foreground.Process.Pid {
		t.Fatalf("foreground PID=%d, service PID=%d", f.foreground.Process.Pid, status.PID)
	}
	probe := f.inspectConsole(status.PID)
	if !probe.Attached || probe.AttachError != 0 || probe.Window == 0 || probe.Visible {
		t.Fatalf("hidden foreground console was not detected: %+v", probe)
	}
	f.assertAlive(status.PID)
	stop := f.cli("stop")
	if stop.err != nil || stop.stdout != "nvpair-service: stopped\n" || stop.stderr != "" {
		t.Fatalf("foreground stop CLI: %+v", stop)
	}
	f.assertExited(status.BrokerPID)
	f.assertExited(status.PID)
	if err := f.joinForeground(); err != nil {
		t.Fatalf("foreground command exit: %v", err)
	}
}

type consoleInspection struct {
	InspectorPID int     `json:"inspectorPid"`
	TargetPID    int     `json:"targetPid"`
	BeforeWindow uintptr `json:"beforeWindow"`
	Attached     bool    `json:"attached"`
	AttachError  uint32  `json:"attachError"`
	Window       uintptr `json:"window"`
	Visible      bool    `json:"visible"`
}

// This helper runs in the same fixed test image, in a separate no-console
// process. It only attaches itself to the exact target; it never changes the
// target's console. The ordinary test run has no target and does no work here.
func TestWindowsAutostartConsoleInspectorHelper(t *testing.T) {
	value := os.Getenv(consoleTargetEnv)
	if value == "" {
		return
	}
	pid, err := strconv.Atoi(value)
	if err != nil || pid <= 0 || pid == os.Getpid() {
		t.Fatalf("invalid console target %q: %v", value, err)
	}
	kernel := windows.NewLazySystemDLL("kernel32.dll")
	getWindow := kernel.NewProc("GetConsoleWindow")
	free := kernel.NewProc("FreeConsole")
	before, _, _ := getWindow.Call()
	if before != 0 {
		t.Fatalf("inspector unexpectedly started with console window %d", before)
	}
	// A zero HWND alone does not establish detachment. Explicitly detach the
	// disposable inspector before AttachConsole, as the baseline probe requires.
	if ok, _, callErr := free.Call(); ok == 0 {
		t.Fatalf("FreeConsole before inspection: %v", callErr)
	}
	attached, _, attachErr := kernel.NewProc("AttachConsole").Call(uintptr(pid))
	probe := consoleInspection{InspectorPID: os.Getpid(), TargetPID: pid, BeforeWindow: before, Attached: attached != 0}
	if probe.Attached {
		defer func() {
			if ok, _, callErr := free.Call(); ok == 0 {
				t.Errorf("FreeConsole after inspection: %v", callErr)
			}
		}()
	} else {
		var errno syscall.Errno
		if !errors.As(attachErr, &errno) {
			t.Fatalf("AttachConsole returned a non-Windows error: %v", attachErr)
		}
		probe.AttachError = uint32(errno)
	}
	probe.Window, _, _ = getWindow.Call()
	visible, _, _ := windows.NewLazySystemDLL("user32.dll").NewProc("IsWindowVisible").Call(probe.Window)
	probe.Visible = visible != 0
	data, err := json.Marshal(probe)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(os.Getenv(consoleReportEnv), data, 0o600); err != nil {
		t.Fatal(err)
	}
}

type windowsCLIResult struct {
	code           int
	pid            int
	stdout, stderr string
	err            error
}

type windowsBrokerReady struct {
	Version string   `json:"version"`
	PID     int      `json:"pid"`
	Cwd     string   `json:"cwd"`
	Args    []string `json:"args"`
}

type windowsAutostartFixture struct {
	t              *testing.T
	exe            string
	dir            string
	records        string
	endpoint       string
	logDir         string
	env            []string
	args           []string
	brokerArgs     []string
	handles        map[int]windows.Handle
	foreground     *exec.Cmd
	foregroundDone chan error
	foregroundOut  bytes.Buffer
	foregroundErr  bytes.Buffer
}

func newWindowsAutostartFixture(t *testing.T) *windowsAutostartFixture {
	t.Helper()
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	exe, err = filepath.EvalSymlinks(exe)
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	f := &windowsAutostartFixture{
		t: t, exe: exe, dir: dir, records: filepath.Join(dir, "process records"),
		endpoint: testEndpoint(dir), logDir: filepath.Join(dir, "forwarded logs"),
		handles: make(map[int]windows.Handle),
	}
	if err := os.Mkdir(f.records, 0o700); err != nil {
		t.Fatal(err)
	}
	f.env = windowsAutostartEnv(map[string]string{
		serviceCommandEnv: "1", fakeBrokerEnv: "1", serviceProcessDirEnv: f.records,
		servicectl.EndpointEnv: f.endpoint, "LOCALAPPDATA": filepath.Join(dir, "profile"),
		consoleTargetEnv: "", consoleReportEnv: "",
	})
	f.brokerArgs = []string{processBrokerArg, "", "space value", `say "hi"`, `dir\ x\`, "--systemd", "--endpoint", "broker-only"}
	f.args = append([]string{"--endpoint", f.endpoint, "--broker-path", exe, "--log-dir", f.logDir, "--log-level", "debug", "--"}, f.brokerArgs...)
	t.Cleanup(f.cleanup)
	return f
}

func windowsAutostartEnv(overrides map[string]string) []string {
	var env []string
	for _, value := range os.Environ() {
		key, _, _ := strings.Cut(value, "=")
		replaced := false
		for name := range overrides {
			if strings.EqualFold(name, key) {
				replaced = true
				break
			}
		}
		if !replaced {
			env = append(env, value)
		}
	}
	for key, value := range overrides {
		env = append(env, key+"="+value)
	}
	return env
}

func (f *windowsAutostartFixture) cli(args ...string) windowsCLIResult {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, f.exe, args...)
	cmd.Env, cmd.Dir, cmd.WaitDelay = f.env, f.dir, 2*time.Second
	if len(args) > 1 && args[0] == "autostart" && args[1] == "run" {
		// A hidden console is deliberate: removing CREATE_NO_WINDOW from the
		// daemon must fail even if the starter's console was never visible.
		cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true, CreationFlags: windows.CREATE_NEW_CONSOLE | windows.CREATE_NEW_PROCESS_GROUP}
	}
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	err := cmd.Run()
	result := windowsCLIResult{stdout: stdout.String(), stderr: stderr.String(), err: err}
	if cmd.Process != nil {
		result.pid = cmd.Process.Pid
	}
	if cmd.ProcessState != nil {
		result.code = cmd.ProcessState.ExitCode()
	} else {
		result.code = -1
	}
	return result
}

func (f *windowsAutostartFixture) startAutostart() {
	f.t.Helper()
	if result := f.cli(append([]string{"autostart", "run"}, f.args...)...); result.err != nil {
		f.t.Fatalf("starter: %+v", result)
	}
}

func (f *windowsAutostartFixture) startForeground() {
	f.t.Helper()
	cmd := exec.Command(f.exe, f.args...)
	cmd.Env, cmd.Dir, cmd.WaitDelay = f.env, f.dir, 2*time.Second
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true, CreationFlags: windows.CREATE_NEW_CONSOLE | windows.CREATE_NEW_PROCESS_GROUP}
	cmd.Stdout, cmd.Stderr = &f.foregroundOut, &f.foregroundErr
	if err := cmd.Start(); err != nil {
		f.t.Fatal(err)
	}
	f.foreground, f.foregroundDone = cmd, make(chan error, 1)
	go func() { f.foregroundDone <- cmd.Wait() }()
}

func (f *windowsAutostartFixture) ready() windowsBrokerReady {
	f.t.Helper()
	waitUntil(f.t, "isolated service listener", func() bool {
		ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
		defer cancel()
		return alreadyAnswering(ctx, f.endpoint)
	})
	c := (&harness{t: f.t, endpoint: f.endpoint}).dial()
	message := c.notification("app:ready")
	var ready windowsBrokerReady
	if err := json.Unmarshal(message.Params, &ready); err != nil {
		f.t.Fatalf("decode fake app:ready: %v", err)
	}
	if ready.Version != "proc" || ready.PID <= 0 {
		f.t.Fatalf("not the process fake broker's app:ready: %+v", ready)
	}
	f.track(ready.PID)
	return ready
}

func (f *windowsAutostartFixture) status() statusResult {
	f.t.Helper()
	result := f.cli("status")
	if result.err != nil || result.stderr != "" {
		f.t.Fatalf("status CLI: %+v", result)
	}
	var status statusResult
	if err := json.Unmarshal([]byte(result.stdout), &status); err != nil {
		f.t.Fatalf("status output is not JSON: %v; %q", err, result.stdout)
	}
	if status.PID <= 0 || status.BrokerPID <= 0 || status.BrokerRestarts != 0 {
		f.t.Fatalf("unexpected service status: %+v", status)
	}
	f.track(status.PID)
	f.track(status.BrokerPID)
	return status
}

func (f *windowsAutostartFixture) inspectConsole(pid int) consoleInspection {
	f.t.Helper()
	f.assertAlive(pid)
	report := filepath.Join(f.dir, fmt.Sprintf("console-%d.json", pid))
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, f.exe, "-test.run=^TestWindowsAutostartConsoleInspectorHelper$")
	cmd.Env = windowsAutostartEnv(map[string]string{
		serviceCommandEnv: "", fakeBrokerEnv: "", serviceProcessDirEnv: "",
		consoleTargetEnv: strconv.Itoa(pid), consoleReportEnv: report,
	})
	cmd.Dir, cmd.WaitDelay = f.dir, 2*time.Second
	cmd.SysProcAttr = &syscall.SysProcAttr{CreationFlags: windows.CREATE_NO_WINDOW}
	if output, err := cmd.CombinedOutput(); err != nil {
		f.t.Fatalf("console inspector: %v; %s", err, output)
	}
	var probe consoleInspection
	f.readJSON(report, &probe)
	if probe.TargetPID != pid || probe.InspectorPID != cmd.Process.Pid || probe.BeforeWindow != 0 {
		f.t.Fatalf("console inspector identity mismatch: %+v; target=%d inspector=%d", probe, pid, cmd.Process.Pid)
	}
	f.assertAlive(pid) // ERROR_INVALID_PARAMETER for a dead PID cannot pass.
	return probe
}

func (f *windowsAutostartFixture) readJSON(path string, value any) {
	f.t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		f.t.Fatal(err)
	}
	if err := json.Unmarshal(data, value); err != nil {
		f.t.Fatalf("decode %s: %v", path, err)
	}
}

func (f *windowsAutostartFixture) assertDirectory(got, want string) {
	f.t.Helper()
	resolved, err := filepath.EvalSymlinks(got)
	if err != nil {
		f.t.Fatal(err)
	}
	if !strings.EqualFold(filepath.Clean(resolved), filepath.Clean(want)) {
		f.t.Fatalf("working directory=%q, want %q", resolved, want)
	}
}

func (f *windowsAutostartFixture) track(pid int) windows.Handle {
	f.t.Helper()
	if handle, ok := f.handles[pid]; ok {
		return handle
	}
	paths, err := filepath.Glob(filepath.Join(f.records, fmt.Sprintf("*-%d.json", pid)))
	if err != nil || len(paths) != 1 {
		f.t.Fatalf("fixture PID %d process record: paths=%q err=%v", pid, paths, err)
	}
	var record serviceProcessRecord
	f.readJSON(paths[0], &record)
	if record.PID != pid {
		f.t.Fatalf("fixture PID %d record belongs to PID %d", pid, record.PID)
	}
	handle, err := openRecordedServiceProcess(record, f.exe)
	if err != nil {
		f.t.Fatalf("open fixture PID %d: %v", pid, err)
	}
	f.handles[pid] = handle
	return handle
}

func serviceTestProcessIdentity() (string, uint64, error) {
	return serviceTestHandleIdentity(windows.CurrentProcess())
}

func serviceTestHandleIdentity(handle windows.Handle) (string, uint64, error) {
	var created, exited, kernel, user windows.Filetime
	if err := windows.GetProcessTimes(handle, &created, &exited, &kernel, &user); err != nil {
		return "", 0, err
	}
	image := make([]uint16, 32768)
	size := uint32(len(image))
	if err := windows.QueryFullProcessImageName(handle, 0, &image[0], &size); err != nil {
		return "", 0, err
	}
	return windows.UTF16ToString(image[:size]), uint64(created.HighDateTime)<<32 | uint64(created.LowDateTime), nil
}

func openRecordedServiceProcess(record serviceProcessRecord, expectedImage string) (windows.Handle, error) {
	handle, err := windows.OpenProcess(windows.SYNCHRONIZE|windows.PROCESS_TERMINATE|windows.PROCESS_QUERY_LIMITED_INFORMATION, false, uint32(record.PID))
	if err != nil {
		return 0, err
	}
	image, created, err := serviceTestHandleIdentity(handle)
	if err == nil && (record.Created == 0 || created != record.Created || !strings.EqualFold(image, record.Image) || !strings.EqualFold(image, expectedImage)) {
		err = fmt.Errorf("fixture PID %d process identity changed", record.PID)
	}
	if err != nil {
		if closeErr := windows.CloseHandle(handle); closeErr != nil {
			return 0, errors.Join(err, closeErr)
		}
		return 0, err
	}
	return handle, nil
}

func TestWindowsAutostartCleanupValidatesRecordedIdentity(t *testing.T) {
	image, created, err := serviceTestProcessIdentity()
	if err != nil {
		t.Fatal(err)
	}
	record := serviceProcessRecord{PID: os.Getpid(), Image: image, Created: created}
	test := func(name string, candidate serviceProcessRecord, expectedImage string, accepted bool) {
		t.Run(name, func(t *testing.T) {
			handle, err := openRecordedServiceProcess(candidate, expectedImage)
			if handle != 0 {
				defer func() {
					if err := windows.CloseHandle(handle); err != nil {
						t.Error(err)
					}
				}()
			}
			if accepted {
				if err != nil || handle == 0 {
					t.Fatalf("matching identity: handle=%v err=%v", handle, err)
				}
			} else if err == nil || handle != 0 {
				t.Fatalf("mismatched identity admitted: handle=%v err=%v", handle, err)
			}
		})
	}
	test("matching executable and creation time", record, image, true)
	missingTime := record
	missingTime.Created = 0
	test("legacy record has no creation time", missingTime, image, false)
	reusedPID := record
	reusedPID.Created++
	test("same executable but reused PID", reusedPID, image, false)
	wrongImage := record
	wrongImage.Image = image + ".different"
	test("recorded executable differs", wrongImage, image, false)
	test("fixture executable differs", record, image+".different", false)
}

func (f *windowsAutostartFixture) assertAlive(pid int) {
	f.t.Helper()
	state, err := windows.WaitForSingleObject(f.track(pid), 0)
	if err != nil || state != uint32(windows.WAIT_TIMEOUT) {
		f.t.Fatalf("fixture PID %d is not alive: state=%d err=%v", pid, state, err)
	}
}

func (f *windowsAutostartFixture) assertExited(pid int) {
	f.t.Helper()
	state, err := windows.WaitForSingleObject(f.track(pid), 5000)
	if err != nil || state != windows.WAIT_OBJECT_0 {
		f.t.Fatalf("fixture PID %d did not exit: state=%d err=%v", pid, state, err)
	}
}

func (f *windowsAutostartFixture) joinForeground() error {
	if f.foregroundDone == nil {
		return nil
	}
	select {
	case err := <-f.foregroundDone:
		f.foregroundDone = nil
		return err
	case <-time.After(5 * time.Second):
		return errors.New("foreground Wait did not finish")
	}
}

func (f *windowsAutostartFixture) cleanup() {
	// Stop is bounded by cli's context. If startup or stop failed, records
	// identify every daemon/broker even before readiness. Newly opened handles
	// must match both the recorded creation time and the fixed executable path,
	// so PID reuse cannot target a replacement process. Kill daemons first,
	// then rescan for brokers so a restart during teardown cannot escape.
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	listening := alreadyAnswering(ctx, f.endpoint)
	cancel()
	if listening {
		if result := f.cli("stop"); result.err != nil {
			f.t.Errorf("cleanup stop: %+v", result)
		}
	}
	for _, role := range []string{"daemon", "broker"} {
		paths, err := filepath.Glob(filepath.Join(f.records, role+"-*.json"))
		if err != nil {
			f.t.Error(err)
			continue
		}
		for _, path := range paths {
			var record serviceProcessRecord
			data, err := os.ReadFile(path)
			if err != nil {
				f.t.Error(err)
				continue
			}
			if err := json.Unmarshal(data, &record); err != nil || record.PID <= 0 {
				f.t.Errorf("invalid process record %s: %v", path, err)
				continue
			}
			handle, ok := f.handles[record.PID]
			if !ok {
				handle, err = openRecordedServiceProcess(record, f.exe)
				if errors.Is(err, windows.ERROR_INVALID_PARAMETER) {
					continue // This recorded process already exited.
				}
				if err != nil {
					f.t.Errorf("open cleanup PID %d: %v", record.PID, err)
					continue
				}
				f.handles[record.PID] = handle
			}
			state, err := windows.WaitForSingleObject(handle, 1000)
			if err != nil {
				f.t.Errorf("wait cleanup PID %d: %v", record.PID, err)
				continue
			}
			if state != windows.WAIT_OBJECT_0 {
				f.t.Errorf("fixture PID %d survived bounded stop; using verified-handle kill fallback", record.PID)
				if err := windows.TerminateProcess(handle, 1); err != nil {
					f.t.Errorf("kill exact fixture PID %d: %v", record.PID, err)
					continue
				}
				if state, err := windows.WaitForSingleObject(handle, 5000); err != nil || state != windows.WAIT_OBJECT_0 {
					f.t.Errorf("join killed PID %d: state=%d err=%v", record.PID, state, err)
				}
			}
		}
	}
	if f.foregroundDone != nil {
		if err := f.foreground.Process.Kill(); err != nil && !errors.Is(err, os.ErrProcessDone) {
			f.t.Errorf("foreground cleanup: %v", err)
		}
		if err := f.joinForeground(); err != nil {
			f.t.Errorf("foreground cleanup Wait: %v", err)
		}
	}
	for pid, handle := range f.handles {
		if err := windows.CloseHandle(handle); err != nil {
			f.t.Errorf("close PID %d handle: %v", pid, err)
		}
	}
}
