// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build windows

package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os/exec"
	"path/filepath"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"golang.org/x/sys/windows"
)

const (
	quotaLimitsHardWSMaxEnable   = windows.QUOTA_LIMITS_HARDWS_MAX_ENABLE  // 0x4
	quotaLimitsHardWSMinDisable  = windows.QUOTA_LIMITS_HARDWS_MIN_DISABLE // 0x2
	workingSetLimitFlags         = quotaLimitsHardWSMaxEnable | quotaLimitsHardWSMinDisable
	jobObjectMsgNewProcess       = 6
	jobObjectMsgActiveZero       = 4
	workingSetSweepInterval      = 250 * time.Millisecond
	workingSetVerificationMaxAge = time.Second
)

var workingSetPrivilegeOnce sync.Once
var workingSetPrivilegeError error

// Only the synchronous root cap is replaceable by failure-injection tests.
var capRootWorkingSet = capWorkingSet

func enableWorkingSetPrivilege() error {
	var token windows.Token
	if err := windows.OpenProcessToken(windows.CurrentProcess(), windows.TOKEN_ADJUST_PRIVILEGES|windows.TOKEN_QUERY, &token); err != nil {
		return err
	}
	defer token.Close()
	name, err := windows.UTF16PtrFromString("SeIncreaseWorkingSetPrivilege")
	if err != nil {
		return err
	}
	var luid windows.LUID
	if err := windows.LookupPrivilegeValue(nil, name, &luid); err != nil {
		return err
	}
	privileges := windows.Tokenprivileges{PrivilegeCount: 1}
	privileges.Privileges[0] = windows.LUIDAndAttributes{Luid: luid, Attributes: windows.SE_PRIVILEGE_ENABLED}
	// A token without the privilege can report success but leave it disabled.
	// The retried set call remains the authority, and still fails closed.
	return windows.AdjustTokenPrivileges(token, false, &privileges, 0, nil, nil)
}

func processWorkingSetLimits(process windows.Handle) (min, max uintptr, flags uint32, err error) {
	// x/sys's getter discards BOOL/GetLastError. Initialize all outputs and
	// reject an absent result rather than using an unread minimum or readback.
	windows.GetProcessWorkingSetSizeEx(process, &min, &max, &flags)
	if min == 0 || max == 0 {
		err = errors.New("working-set quota query returned no limits")
	}
	return
}

// x/sys/windows has no IsProcessInJob wrapper.
var procIsProcessInJob = windows.NewLazySystemDLL("kernel32.dll").NewProc("IsProcessInJob")

func processInJob(process, job windows.Handle) (bool, error) {
	var result int32
	r, _, err := procIsProcessInJob.Call(uintptr(process), uintptr(job), uintptr(unsafe.Pointer(&result)))
	if r == 0 {
		return false, err
	}
	return result != 0, nil
}

func openWorkingSetProcess(pid uint32) (windows.Handle, error) {
	return windows.OpenProcess(windows.PROCESS_SET_QUOTA|windows.PROCESS_QUERY_LIMITED_INFORMATION|windows.PROCESS_TERMINATE|windows.SYNCHRONIZE, false, pid)
}

// JOBOBJECT_BASIC_PROCESS_ID_LIST has a DWORD header and a variable-length
// array of pointer-sized process IDs.
type workingSetJobProcessIDList struct {
	numberOfAssignedProcesses uint32
	numberOfProcessIDsInList  uint32
	processIDs                [1]uintptr
}

func jobMemberPIDs(job windows.Handle) ([]uint32, error) {
	for capacity := 16; ; {
		const headerSize = unsafe.Offsetof(workingSetJobProcessIDList{}.processIDs)
		const idSize = unsafe.Sizeof(uintptr(0))
		buffer := make([]uintptr, int(headerSize/idSize)+capacity)
		list := (*workingSetJobProcessIDList)(unsafe.Pointer(&buffer[0]))
		err := windows.QueryInformationJobObject(job, windows.JobObjectBasicProcessIdList, uintptr(unsafe.Pointer(list)), uint32(headerSize)+uint32(capacity)*uint32(idSize), nil)
		if errors.Is(err, windows.ERROR_MORE_DATA) {
			capacity = max(capacity*2, int(list.numberOfAssignedProcesses)+16)
			continue
		}
		if err != nil {
			return nil, err
		}
		if int(list.numberOfProcessIDsInList) > capacity {
			return nil, errors.New("job member query returned too many process IDs")
		}
		ids := unsafe.Slice(&list.processIDs[0], int(list.numberOfProcessIDsInList))
		pids := make([]uint32, len(ids))
		for i, pid := range ids {
			pids[i] = uint32(pid)
		}
		return pids, nil
	}
}

func capWorkingSet(pid uint32, maxBytes uint64) error {
	process, err := openWorkingSetProcess(pid)
	if errors.Is(err, windows.ERROR_INVALID_PARAMETER) {
		return nil // An already-exited job member needs no cap.
	}
	if err != nil {
		return err
	}
	defer windows.CloseHandle(process)
	return capProcessWorkingSet(process, maxBytes)
}

func capProcessWorkingSet(process windows.Handle, maxBytes uint64) error {
	if uint64(uintptr(maxBytes)) != maxBytes || maxBytes == 0 {
		return errors.New("working-set maximum is outside the process address range")
	}
	min, _, _, err := processWorkingSetLimits(process)
	if err != nil {
		return err
	}
	err = windows.SetProcessWorkingSetSizeEx(process, min, uintptr(maxBytes), workingSetLimitFlags)
	if errors.Is(err, windows.ERROR_PRIVILEGE_NOT_HELD) {
		workingSetPrivilegeOnce.Do(func() { workingSetPrivilegeError = enableWorkingSetPrivilege() })
		if workingSetPrivilegeError != nil {
			return errors.Join(err, workingSetPrivilegeError)
		}
		err = windows.SetProcessWorkingSetSizeEx(process, min, uintptr(maxBytes), workingSetLimitFlags)
	}
	if err != nil {
		return err
	}
	_, max, gotFlags, err := processWorkingSetLimits(process)
	if err != nil {
		return err
	}
	if uint64(max) != maxBytes || gotFlags&workingSetLimitFlags != workingSetLimitFlags {
		return fmt.Errorf("working-set cap readback mismatch: max=%d flags=%#x", max, gotFlags)
	}
	return nil
}

// Engine jobs deliberately have no limits, including kill-on-close. Closing
// the manager's handles must preserve the existing engine/orphan lifetime.
type workingSetContainment struct {
	job        windows.Handle
	port       windows.Handle
	engine     string
	maxBytes   uint64
	stop       chan struct{}
	done       chan struct{}
	once       sync.Once
	startOnce  sync.Once
	verifiedAt atomic.Int64
	closeErr   error
}

func newWorkingSetContainment(engine string, maxBytes uint64) (*workingSetContainment, error) {
	job, err := windows.CreateJobObject(nil, nil)
	if err != nil {
		return nil, err
	}
	port, err := windows.CreateIoCompletionPort(windows.InvalidHandle, 0, 0, 1)
	if err != nil {
		windows.CloseHandle(job)
		return nil, err
	}
	// JOBOBJECT_ASSOCIATE_COMPLETION_PORT, two pointer-sized fields.
	association := struct {
		key  uintptr
		port windows.Handle
	}{1, port}
	if _, err := windows.SetInformationJobObject(job, windows.JobObjectAssociateCompletionPortInformation, uintptr(unsafe.Pointer(&association)), uint32(unsafe.Sizeof(association))); err != nil {
		windows.CloseHandle(port)
		windows.CloseHandle(job)
		return nil, err
	}
	return &workingSetContainment{job: job, port: port, engine: engine, maxBytes: maxBytes, stop: make(chan struct{}), done: make(chan struct{})}, nil
}

func (c *workingSetContainment) start() {
	c.startOnce.Do(func() { go c.watch() })
}

func (c *workingSetContainment) Verified() bool {
	at := c.verifiedAt.Load()
	return at != 0 && time.Since(time.Unix(0, at)) <= workingSetVerificationMaxAge
}

func (c *workingSetContainment) sweep(onInterval bool) error {
	pids, err := jobMemberPIDs(c.job)
	if err != nil {
		return err
	}
	allVerified := true
	for _, pid := range pids {
		verified, capped := c.capMember(pid)
		allVerified = allVerified && verified
		if onInterval && capped {
			slog.Info("working-set interval sweep capped member", "engine", c.engine, "pid", pid)
		}
	}
	if allVerified {
		c.verifiedAt.Store(time.Now().UnixNano())
	} else {
		c.verifiedAt.Store(0)
	}
	return nil
}

func (c *workingSetContainment) watch() {
	// IOCP concurrency is tracked per native thread. Keep notifications and
	// their cap calls on one thread rather than inheriting scheduler work.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	defer close(c.done)
	defer c.verifiedAt.Store(0)
	sweep := func(onInterval bool) bool {
		if err := c.sweep(onInterval); err != nil {
			c.verifiedAt.Store(0)
			err = errors.Join(err, windows.TerminateJobObject(c.job, 1))
			slog.Warn("working-set sweep failed", "engine", c.engine, "error", err)
			return false
		}
		return true
	}
	lastSweep := time.Now()
	if !sweep(false) {
		return
	}
	for {
		select {
		case <-c.stop:
			return
		default:
		}
		var message uint32
		var key, pid uintptr
		// Job messages carry a PID value in the OVERLAPPED slot, not a Go
		// pointer. Keep it in integer storage, without dereferencing it.
		remaining := max(time.Duration(0), workingSetSweepInterval-time.Since(lastSweep))
		err := windows.GetQueuedCompletionStatus(c.port, &message, &key, (**windows.Overlapped)(unsafe.Pointer(&pid)), uint32(remaining/time.Millisecond))
		if err != nil && !errors.Is(err, windows.WAIT_TIMEOUT) {
			c.verifiedAt.Store(0)
			err = errors.Join(err, windows.TerminateJobObject(c.job, 1))
			slog.Warn("working-set notification failed", "engine", c.engine, "error", err)
			return
		}
		switch message {
		case jobObjectMsgActiveZero:
			return
		case jobObjectMsgNewProcess:
			if !sweep(false) {
				return
			}
		}
		if time.Since(lastSweep) >= workingSetSweepInterval {
			lastSweep = time.Now()
			if !sweep(true) {
				return
			}
		}
	}
}

// A notification can arrive after its process exited and the PID was reused,
// so the opened handle must still name a member of this job before it is
// capped or, on cap failure, terminated.
func (c *workingSetContainment) capMember(pid uint32) (verified, capped bool) {
	process, err := openWorkingSetProcess(pid)
	if errors.Is(err, windows.ERROR_INVALID_PARAMETER) {
		return true, false // Already exited.
	}
	if err != nil {
		// Without a handle the member can be neither capped nor terminated
		// alone, so fail closed on the engine tree, as a failed read does.
		err = errors.Join(err, windows.TerminateJobObject(c.job, 1))
	} else {
		defer windows.CloseHandle(process)
		member, memberErr := processInJob(process, c.job)
		if memberErr == nil && !member {
			return true, false
		}
		err = memberErr
		if err == nil {
			_, maximum, flags, queryErr := processWorkingSetLimits(process)
			if queryErr == nil && uint64(maximum) == c.maxBytes && flags&workingSetLimitFlags == workingSetLimitFlags {
				return true, false
			}
			err = capProcessWorkingSet(process, c.maxBytes)
			if err == nil {
				return true, true
			}
			if status, waitErr := windows.WaitForSingleObject(process, 0); waitErr == nil && status == windows.WAIT_OBJECT_0 {
				return true, false
			}
		}
		if err != nil {
			err = errors.Join(err, windows.TerminateProcess(process, 1))
		}
	}
	if err != nil {
		slog.Warn("working-set cap failed", "engine", c.engine, "pid", pid, "error", err)
	}
	return false, false
}

func (c *workingSetContainment) Close() error {
	c.once.Do(func() {
		close(c.stop)
		// Claim an unstarted watcher so Close never waits for work that was
		// not launched, and a later start cannot use closed handles.
		c.startOnce.Do(func() { close(c.done) })
		// Wake the reader before closing either handle. The timeout also bounds
		// shutdown if posting fails, avoiding a close/use race on reused handles.
		var wakeErr error
		select {
		case <-c.done:
		default:
			wakeErr = windows.PostQueuedCompletionStatus(c.port, 0, 0, nil)
		}
		<-c.done
		c.verifiedAt.Store(0)
		c.closeErr = errors.Join(wakeErr, windows.CloseHandle(c.port), windows.CloseHandle(c.job))
	})
	return c.closeErr
}

func startWorkingSetProcess(cmd *exec.Cmd, maxBytes uint64) (func() error, func() bool, error) {
	engine := filepath.Base(cmd.Path)
	c, err := newWorkingSetContainment(engine, maxBytes)
	if err != nil {
		return nil, nil, err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := startSuspendedJobProcess(ctx, cmd, c.job, func(pid uint32) error {
		err := capRootWorkingSet(pid, maxBytes)
		if err != nil {
			slog.Warn("working-set cap failed", "engine", engine, "pid", pid, "error", err)
		} else {
			c.verifiedAt.Store(time.Now().UnixNano())
		}
		return err
	}, c.Close); err != nil {
		return nil, nil, err
	}
	c.start()
	return c.Close, c.Verified, nil
}
