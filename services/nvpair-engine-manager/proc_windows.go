// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build windows

package main

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"
	"unsafe"

	"golang.org/x/sys/windows"
)

// taskkillTimeout bounds each taskkill invocation so a hung taskkill can't
// wedge StopAll (and the whole app shutdown) now that the broker no longer
// force-kills engine-manager on a timeout.
const taskkillTimeout = 5 * time.Second

// resolveNativeSourcePath follows junctions as well as symlinks, including
// reparse points in ancestor directories. Never retain an unresolved spelling.
func resolveNativeSourcePath(path string) (string, error) {
	resolved, _, err := windowsFinalPath(path)
	if err != nil {
		return "", err
	}
	// Keep the DOS/UNC spelling used by existing source bindings. The handle
	// resolver retains the extended prefix for its process-ownership callers.
	if strings.HasPrefix(resolved, `\\?\UNC\`) {
		return `\\` + strings.TrimPrefix(resolved, `\\?\UNC\`), nil
	}
	return strings.TrimPrefix(resolved, `\\?\`), nil
}

func managedInstallPath(binAbs, dirAbs string) bool {
	binPath, binInfo, err := windowsFinalPath(binAbs)
	if err != nil || !binInfo.Mode().IsRegular() {
		return false
	}
	_, dirInfo, err := windowsFinalPath(dirAbs)
	if err != nil || !dirInfo.IsDir() {
		return false
	}
	// EvalSymlinks only walks ModeSymlink; Windows junctions can instead be
	// ModeIrregular. Walk the kernel-resolved executable's actual ancestors,
	// matching directory file IDs rather than trusting a lexical path prefix.
	// This also rejects a junction inside the install that escapes outside it.
	for parent := filepath.Dir(binPath); ; {
		info, err := os.Stat(parent)
		if err != nil || !info.IsDir() {
			return false
		}
		if os.SameFile(info, dirInfo) {
			return true
		}
		next := filepath.Dir(parent)
		if next == parent {
			return false
		}
		parent = next
	}
}

// windowsFinalPath follows all name-surrogate reparse points using a metadata
// handle, returning both its normalized final path and the same handle's file
// identity. It requires no content access and never falls back to the spelling
// supplied by the caller when resolution is unavailable.
func windowsFinalPath(path string) (string, os.FileInfo, error) {
	name, err := windows.UTF16PtrFromString(path)
	if err != nil {
		return "", nil, err
	}
	h, err := windows.CreateFile(name, 0, windows.FILE_SHARE_READ|windows.FILE_SHARE_WRITE|windows.FILE_SHARE_DELETE, nil, windows.OPEN_EXISTING, windows.FILE_FLAG_BACKUP_SEMANTICS, 0)
	if err != nil {
		return "", nil, err
	}
	file := os.NewFile(uintptr(h), path)
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return "", nil, err
	}
	buf := make([]uint16, 32768)
	// Flags zero requests FILE_NAME_NORMALIZED and VOLUME_NAME_DOS, including
	// UNC paths. The extended-path prefix is kept for subsequent OS lookups.
	n, err := windows.GetFinalPathNameByHandle(h, &buf[0], uint32(len(buf)), 0)
	if err != nil {
		return "", nil, err
	}
	if n == 0 || n >= uint32(len(buf)) {
		return "", nil, windows.ERROR_INSUFFICIENT_BUFFER
	}
	return windows.UTF16ToString(buf[:n]), info, nil
}

// configureSysProcAttr hides the child's console window
// (HideWindow + CREATE_NO_WINDOW), matching every other NVPAIR subprocess.
func configureSysProcAttr(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{
		HideWindow:    true,
		CreationFlags: 0x08000000, // CREATE_NO_WINDOW
	}
}

// gracefulSignal stops the process tree. Windows has no SIGTERM, and the
// engines we spawn run windowless (CREATE_NO_WINDOW), so a non-/F taskkill only
// posts WM_CLOSE — which a windowless process can't receive ("can only be
// terminated forcefully"), i.e. it does nothing. The graceful and forced
// managed-process steps therefore both use taskkill /T /F on Windows.
func gracefulSignal(cmd *exec.Cmd) error {
	return taskkill(cmd, true)
}

func taskkill(cmd *exec.Cmd, force bool) error {
	if cmd == nil || cmd.Process == nil {
		return nil
	}
	args := []string{"/T", "/PID", strconv.Itoa(cmd.Process.Pid)}
	if force {
		args = append([]string{"/F"}, args...)
	}
	ctx, cancel := context.WithTimeout(context.Background(), taskkillTimeout)
	defer cancel()
	kc := exec.CommandContext(ctx, "taskkill", args...)
	kc.Env = childEnv(nil)
	configureSysProcAttr(kc) // hide taskkill's own console window
	return kc.Run()
}

var (
	iphlpapi                = windows.NewLazySystemDLL("iphlpapi.dll")
	procGetExtendedTcpTable = iphlpapi.NewProc("GetExtendedTcpTable")
)

const (
	// TCP_TABLE_OWNER_PID_LISTENER — listening sockets with their owning PID.
	tcpTableOwnerPIDListener = 3
	errInsufficientBuffer    = 122 // ERROR_INSUFFICIENT_BUFFER
)

// mibTCPRowOwnerPID mirrors MIB_TCPROW_OWNER_PID (IPv4). Every field is a
// DWORD, so the C layout is a run of uint32s with no padding.
type mibTCPRowOwnerPID struct {
	state      uint32
	localAddr  uint32
	localPort  uint32
	remoteAddr uint32
	remotePort uint32
	owningPID  uint32
}

// mibTCP6RowOwnerPID mirrors MIB_TCP6ROW_OWNER_PID (IPv6).
type mibTCP6RowOwnerPID struct {
	localAddr     [16]byte
	localScopeID  uint32
	localPort     uint32
	remoteAddr    [16]byte
	remoteScopeID uint32
	remotePort    uint32
	state         uint32
	owningPID     uint32
}

// pidOnPort returns the PID of the process listening on the given TCP port
// (IPv4 first, then IPv6) plus that process's full image path. ok is false
// when nothing is listening or the owner can't be resolved, so a caller can
// fail closed (decline the stop). It is precise to a single port: a genuine
// desktop app on a different port is never surfaced.
func pidOnPort(port int) (pid int, image string, ok bool) {
	if p, found := listenerPID(windows.AF_INET, port); found {
		return p, imagePathForPID(p), true
	}
	if p, found := listenerPID(windows.AF_INET6, port); found {
		return p, imagePathForPID(p), true
	}
	return 0, "", false
}

// listenerPID walks the owner-PID TCP table for one address family and
// returns the PID whose local port matches. The table is fetched with the
// standard two-call sizing dance (NULL buffer to learn the size, then a
// real buffer).
func listenerPID(family uint32, port int) (int, bool) {
	var size uint32
	r, _, _ := procGetExtendedTcpTable.Call(0, uintptr(unsafe.Pointer(&size)), 0, uintptr(family), tcpTableOwnerPIDListener, 0)
	if r != errInsufficientBuffer && r != 0 {
		return 0, false
	}
	if size == 0 {
		return 0, false
	}
	buf := make([]byte, size)
	r, _, _ = procGetExtendedTcpTable.Call(uintptr(unsafe.Pointer(&buf[0])), uintptr(unsafe.Pointer(&size)), 0, uintptr(family), tcpTableOwnerPIDListener, 0)
	if r != 0 {
		return 0, false
	}
	// Both MIB_*TABLE_OWNER_PID structs are { DWORD dwNumEntries; row[] }, so
	// the rows begin at offset 4 (the row structs are 4-byte aligned).
	n := *(*uint32)(unsafe.Pointer(&buf[0]))
	if n == 0 {
		return 0, false
	}
	if family == windows.AF_INET {
		rows := unsafe.Slice((*mibTCPRowOwnerPID)(unsafe.Pointer(&buf[4])), int(n))
		for i := range rows {
			if ntohsPort(rows[i].localPort) == port {
				return int(rows[i].owningPID), true
			}
		}
		return 0, false
	}
	rows := unsafe.Slice((*mibTCP6RowOwnerPID)(unsafe.Pointer(&buf[4])), int(n))
	for i := range rows {
		if ntohsPort(rows[i].localPort) == port {
			return int(rows[i].owningPID), true
		}
	}
	return 0, false
}

// ntohsPort converts the local-port DWORD (the port number in network byte
// order, held in the low 16 bits) to a host-order int.
func ntohsPort(dw uint32) int {
	return int(((dw & 0xFF) << 8) | ((dw >> 8) & 0xFF))
}

// imagePathForPID resolves a PID's full executable path. Empty on any error
// (access denied, exited) so the image check fails closed.
func imagePathForPID(pid int) string {
	if pid <= 0 {
		return ""
	}
	h, err := windows.OpenProcess(windows.PROCESS_QUERY_LIMITED_INFORMATION, false, uint32(pid))
	if err != nil {
		return ""
	}
	defer windows.CloseHandle(h)
	buf := make([]uint16, 1024)
	size := uint32(len(buf))
	if err := windows.QueryFullProcessImageName(h, 0, &buf[0], &size); err != nil {
		return ""
	}
	return windows.UTF16ToString(buf[:size])
}

// signalPID stops the PID's process tree with taskkill /T /F, matching
// gracefulSignal. Keep the force argument for the shared platform interface.
func signalPID(pid int, force bool) error {
	return signalPIDContext(context.Background(), pid, force)
}

// Orphans use the same forceful tree stop as gracefulSignal: a windowless
// console engine cannot receive the WM_CLOSE sent by a non-/F taskkill.
func signalPIDContext(ctx context.Context, pid int, force bool) error {
	if pid <= 0 {
		return nil
	}
	args := []string{"/F", "/T", "/PID", strconv.Itoa(pid)}
	ctx, cancel := context.WithTimeout(ctx, taskkillTimeout)
	defer cancel()
	kc := exec.CommandContext(ctx, "taskkill", args...)
	kc.Env = childEnv(nil)
	configureSysProcAttr(kc) // hide taskkill's own console window
	err := kc.Run()
	if ctx.Err() != nil {
		return ctx.Err()
	}
	return err
}

// pidAlive reports whether the PID is still running (WaitForSingleObject with
// a zero timeout: WAIT_TIMEOUT means it hasn't signaled exit). A handle we
// can't open is treated as gone.
func pidAlive(pid int) bool {
	if pid <= 0 {
		return false
	}
	h, err := windows.OpenProcess(windows.SYNCHRONIZE, false, uint32(pid))
	if err != nil {
		return false
	}
	defer windows.CloseHandle(h)
	s, err := windows.WaitForSingleObject(h, 0)
	if err != nil {
		return true
	}
	return s == uint32(windows.WAIT_TIMEOUT)
}
