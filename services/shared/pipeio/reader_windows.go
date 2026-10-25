// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package pipeio

import (
	"errors"
	"io"
	"os"
	"runtime"
	"sync"
	"syscall"

	"golang.org/x/sys/windows"
)

var cancelSynchronousIO = windows.NewLazySystemDLL("kernel32.dll").NewProc("CancelSynchronousIo")

// NewReader takes ownership of a file used as a synchronous byte-pipe reader,
// such as stdin or exec.Cmd.StdoutPipe. Close cancels and joins an active native
// read before closing the file. The caller must not otherwise read or close it.
// Regular files and consoles retain their ordinary os.File behavior.
func NewReader(file *os.File) (io.ReadCloser, error) {
	raw, err := file.SyscallConn()
	if err != nil {
		return nil, err
	}
	var kind uint32
	var typeErr error
	if err := raw.Control(func(handle uintptr) {
		kind, typeErr = windows.GetFileType(windows.Handle(handle))
	}); err != nil {
		return nil, err
	}
	if typeErr != nil {
		return nil, typeErr
	}
	if kind != windows.FILE_TYPE_PIPE {
		return file, nil
	}
	if err := cancelSynchronousIO.Find(); err != nil {
		return nil, err
	}
	r := &reader{file: file, raw: raw}
	r.idle.L = &r.mu
	return r, nil
}

type reader struct {
	file *os.File
	raw  syscall.RawConn

	readMu sync.Mutex
	mu     sync.Mutex
	idle   sync.Cond
	thread windows.Handle
	closed bool

	closeOnce sync.Once
	closeErr  error
}

func (r *reader) Read(b []byte) (int, error) {
	r.readMu.Lock()
	defer r.readMu.Unlock()
	r.mu.Lock()
	if r.closed {
		r.mu.Unlock()
		return 0, os.ErrClosed
	}
	if len(b) == 0 {
		r.mu.Unlock()
		return 0, nil
	}
	// The actual ReadFile must run on this thread. Calling os.File.Read here
	// would depend on Go's choice of thread for synchronous pipe operations.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	var thread windows.Handle
	if err := windows.DuplicateHandle(windows.CurrentProcess(), windows.CurrentThread(),
		windows.CurrentProcess(), &thread, windows.THREAD_TERMINATE, false, 0); err != nil {
		r.mu.Unlock()
		return 0, err
	}
	r.thread = thread
	r.mu.Unlock()
	var n uint32
	var readErr error
	err := r.raw.Control(func(handle uintptr) {
		readErr = windows.ReadFile(windows.Handle(handle), b, &n, nil)
	})
	r.mu.Lock()
	// Cancellation holds mu while using the handle. Keep this thread pinned
	// until it can no longer be targeted, so unrelated I/O cannot be cancelled.
	r.thread = 0
	closeErr := windows.CloseHandle(thread)
	r.idle.Broadcast()
	r.mu.Unlock()
	if err != nil {
		return int(n), err
	}
	if errors.Is(readErr, windows.ERROR_OPERATION_ABORTED) {
		// Preserve the native cause as well as the ordinary closed-read identity.
		return int(n), errors.Join(os.ErrClosed, readErr)
	}
	if errors.Is(readErr, windows.ERROR_BROKEN_PIPE) || errors.Is(readErr, windows.ERROR_HANDLE_EOF) || (readErr == nil && n == 0) {
		readErr = io.EOF
	}
	if readErr == nil {
		readErr = closeErr
	}
	return int(n), readErr
}

func (r *reader) Close() error {
	r.closeOnce.Do(func() {
		r.mu.Lock()
		r.closed = true
		for r.thread != 0 {
			ok, _, err := cancelSynchronousIO.Call(uintptr(r.thread))
			if ok != 0 {
				for r.thread != 0 {
					r.idle.Wait()
				}
				break
			}
			if !errors.Is(err, windows.ERROR_NOT_FOUND) {
				r.closeErr = os.NewSyscallError("CancelSynchronousIo", err)
				r.mu.Unlock()
				return
			}
			// The read was admitted but has not submitted ReadFile yet (or just
			// completed). Fence admission and yield until that same read can be
			// cancelled or has left; a single cancellation loses this window.
			r.mu.Unlock()
			runtime.Gosched()
			r.mu.Lock()
		}
		r.mu.Unlock()
		r.closeErr = r.file.Close()
	})
	return r.closeErr
}
