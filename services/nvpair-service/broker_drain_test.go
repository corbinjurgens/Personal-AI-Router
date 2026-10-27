// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"io"
	"testing"
	"time"
)

// observedReader reports when each blocked read has been released.
type observedReader struct {
	reader io.Reader
	done   chan struct{}
}

func (r observedReader) Read(p []byte) (int, error) {
	n, err := r.reader.Read(p)
	if err != nil {
		close(r.done)
	}
	return n, err
}

func TestRunBrokerBoundsBothOutputDrainPhases(t *testing.T) {
	stdoutR, stdoutW := io.Pipe()
	stderrR, stderrW := io.Pipe()
	stdoutDone, stderrDone := make(chan struct{}), make(chan struct{})
	closedOutput := make(chan struct{})
	runDone := make(chan struct{})
	proc := &brokerProc{
		stdout:      observedReader{stdoutR, stdoutDone},
		stderr:      observedReader{stderrR, stderrDone},
		wait:        func() int { return 0 },
		closeOutput: func() { close(closedOutput) },
	}
	run := &brokerRun{proc: proc, exited: make(chan struct{})}
	s := newService(config{})
	go func() { s.runBroker(run); close(runDone) }()
	t.Cleanup(func() {
		_ = stdoutW.Close()
		_ = stderrW.Close()
		<-stdoutDone
		<-stderrDone
		<-runDone
	})

	select {
	case <-closedOutput:
	case <-time.After(outputDrainTimeout + 2*time.Second):
		t.Fatal("graceful drain did not reach closeOutput with both pumps blocked")
	}
	select {
	case <-runDone:
	case <-time.After(outputDrainTimeout + 2*time.Second):
		t.Fatal("drain after closeOutput did not finish with both pumps blocked")
	}
}
