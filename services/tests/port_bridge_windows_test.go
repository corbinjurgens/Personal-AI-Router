// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"unsafe"

	"golang.org/x/sys/windows"
)

func runOwnedPortWorker(binary string, args []string) error {
	// Assign the adapter before spawning, so the worker inherits membership
	// without a start/assignment race. A non-inheritable job handle ensures
	// terminating this adapter closes the last handle and kills its worker.
	job, err := windows.CreateJobObject(nil, nil)
	if err != nil {
		return fmt.Errorf("create port fixture job: %w", err)
	}
	limits := windows.JOBOBJECT_EXTENDED_LIMIT_INFORMATION{}
	limits.BasicLimitInformation.LimitFlags = windows.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
	if _, err := windows.SetInformationJobObject(job, windows.JobObjectExtendedLimitInformation, uintptr(unsafe.Pointer(&limits)), uint32(unsafe.Sizeof(limits))); err != nil {
		return errors.Join(fmt.Errorf("configure port fixture job: %w", err), windows.CloseHandle(job))
	}
	if err := windows.AssignProcessToJobObject(job, windows.CurrentProcess()); err != nil {
		return errors.Join(fmt.Errorf("assign port fixture adapter to job: %w", err), windows.CloseHandle(job))
	}
	// Keep the sole handle open until the adapter's os.Exit, including error
	// paths. Closing it here would also terminate the adapter itself.
	cmd := exec.Command(binary, args...)
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
	return cmd.Run()
}
