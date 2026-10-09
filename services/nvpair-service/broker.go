// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sync"
	"time"

	"nvpair-shared/jsonrpc"
)

// brokerName is the binary the service supervises, resolved beside its own
// executable like every other sibling in the bundle.
const brokerName = "nvpair-ui-broker"

// Timings for the broker's lifecycle.
const (
	// defaultStopGrace mirrors nvpair-tui's shutdownGrace: the broker's own
	// teardown budget (teardownBudget plus workloadHistoryFlushJoinTimeout in
	// nvpair-ui-broker, 15 s together) plus headroom, counted from after the
	// shutdown request. Killing sooner orphans workers that are still inside
	// their budget and drops the final workload-history flush.
	defaultStopGrace = 18 * time.Second
	// defaultShutdownRequestTimeout bounds the wait for the broker's answer to
	// shutdown before its stdin is closed, as nvpair-tui does.
	defaultShutdownRequestTimeout = 2 * time.Second
	// outputDrainTimeout bounds how long the pumps may keep reading after the
	// broker has exited, in case a straggler inherited its stdout or stderr.
	outputDrainTimeout = 2 * time.Second
	// stableRun is how long a broker must stay up for the restart backoff to
	// start over from its first step.
	stableRun = time.Minute
	// maxBackoff caps the restart backoff.
	maxBackoff = 30 * time.Second
	// maxLogLine caps one broker stderr line; the rest of a longer line is
	// dropped rather than split into fragments that look like separate records.
	maxLogLine = 1 << 20
	// brokerLogSource names the broker in service/log notifications.
	brokerLogSource = "nvpair-ui-broker"
)

// brokerProc is one started broker process, abstracted so tests can stand an
// in-process fake in for it.
type brokerProc struct {
	stdin  io.WriteCloser
	stdout io.Reader
	stderr io.Reader
	pid    int
	// wait blocks until the process exits and returns its exit code (-1 when
	// unknown). It is called exactly once.
	wait func() int
	kill func() error
	// closeOutput force-closes stdout and stderr so the pumps return even if a
	// straggler holds the write ends open.
	closeOutput func()
}

// spawnFunc starts a broker.
type spawnFunc func() (*brokerProc, error)

// brokerRun is the supervision state of one broker process.
type brokerRun struct {
	proc   *brokerProc
	wmu    sync.Mutex
	exited chan struct{}
	code   int
}

func (r *brokerRun) writeFrame(frame []byte) error {
	r.wmu.Lock()
	defer r.wmu.Unlock()
	select {
	case <-r.exited:
		return errors.New("broker has exited")
	default:
	}
	for off := 0; off < len(frame); {
		n, err := r.proc.stdin.Write(frame[off:])
		if err != nil {
			return err
		}
		if n == 0 {
			return io.ErrShortWrite
		}
		off += n
	}
	return nil
}

func (r *brokerRun) writeMessage(msg *jsonrpc.Message) error {
	data, err := json.Marshal(msg)
	if err != nil {
		return err
	}
	return r.writeFrame(append(data, '\n'))
}

// closeStdin closes the broker's stdin without taking the write lock: a writer
// stuck behind a broker that stopped reading holds that lock, and closing the
// pipe is what unblocks it.
func (r *brokerRun) closeStdin() {
	_ = r.proc.stdin.Close()
}

// defaultBackoff doubles from one second up to maxBackoff.
func defaultBackoff(attempt int) time.Duration {
	d := time.Second
	for i := 1; i < attempt && d < maxBackoff; i++ {
		d *= 2
	}
	if d > maxBackoff {
		d = maxBackoff
	}
	return d
}

// superviseBroker runs the broker, restarting it with backoff whenever it exits
// without being asked to, until the service stops.
func (s *Service) superviseBroker() {
	defer close(s.brokerDone)
	attempt := 0
	for {
		if s.isStopping() {
			return
		}
		proc, err := s.cfg.spawn()
		if err != nil {
			attempt++
			slog.Error("could not start broker", "err", err, "attempt", attempt)
			if !s.sleep(s.cfg.backoff(attempt)) {
				return
			}
			continue
		}

		run := &brokerRun{proc: proc, exited: make(chan struct{})}
		started := time.Now()
		s.mu.Lock()
		if s.stopping {
			// stop() ran while this broker was starting and saw no broker to
			// shut down, so this one is ours to end.
			s.mu.Unlock()
			_ = proc.kill()
			proc.wait()
			proc.closeOutput()
			return
		}
		s.broker = run
		restarted := s.runs > 0
		s.runs++
		if restarted {
			s.restarts++
		}
		restarts := s.restarts
		s.mu.Unlock()
		slog.Info("broker started", "pid", proc.pid)
		if restarted {
			s.broadcast(notifyBrokerRestarted, map[string]int{"brokerPid": proc.pid, "restarts": restarts})
		}

		s.runBroker(run)

		if s.isStopping() {
			slog.Info("broker stopped", "code", run.code)
			return
		}
		if time.Since(started) >= stableRun {
			attempt = 0
		}
		attempt++
		delay := s.cfg.backoff(attempt)
		slog.Warn("broker exited unexpectedly; restarting", "code", run.code, "attempt", attempt, "delay", delay)
		if !s.sleep(delay) {
			return
		}
	}
}

// runBroker pumps one broker's output until it exits, then clears everything
// that belonged to it.
func (s *Service) runBroker(run *brokerRun) {
	stdoutDone := make(chan struct{})
	stderrDone := make(chan struct{})
	go func() {
		defer close(stdoutDone)
		s.pumpStdout(run)
	}()
	go func() {
		defer close(stderrDone)
		s.pumpStderr(run.proc.stderr)
	}()
	run.code = run.proc.wait()
	close(run.exited)

	drain := time.After(outputDrainTimeout)
	for _, done := range []chan struct{}{stdoutDone, stderrDone} {
		select {
		case <-done:
		case <-drain:
		}
	}
	run.proc.closeOutput()
	// Bounded again: on Windows closing a pipe does not interrupt a read that
	// is already blocked, so a straggler holding the write end could otherwise
	// stall supervision. A pump left behind exits when that straggler does.
	drain = time.After(outputDrainTimeout)
	for _, done := range []chan struct{}{stdoutDone, stderrDone} {
		select {
		case <-done:
		case <-drain:
		}
	}
	s.brokerExited(run)
}

// brokerExited forgets the state of a broker that has gone: its app:ready and
// baselines, its subscriptions (kept for the next broker to restore), and every
// request still waiting on it, which is answered with an error so no client
// waits forever.
func (s *Service) brokerExited(run *brokerRun) {
	s.mu.Lock()
	if s.broker == run {
		s.broker = nil
	}
	s.readyFrame = nil
	s.baselines = make(map[string][]byte)
	for m := range s.subscribed {
		s.resubscribe = appendUnique(s.resubscribe, m)
	}
	s.subscribed = make(map[string]bool)
	var orphaned []*pendingRequest
	for id, p := range s.pending {
		if p.run == run {
			orphaned = append(orphaned, p)
			delete(s.pending, id)
		}
	}
	s.mu.Unlock()

	for _, p := range orphaned {
		if p.internal != nil {
			p.internal(nil)
			continue
		}
		id := p.origID
		p.client.respondError(&id, codeUnavailable, fmt.Sprintf("broker exited before answering %s", p.method))
	}
}

func appendUnique(list []string, v string) []string {
	for _, x := range list {
		if x == v {
			return list
		}
	}
	return append(list, v)
}

// sleep waits d unless the service starts stopping first; it reports whether
// the full wait elapsed.
func (s *Service) sleep(d time.Duration) bool {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return true
	case <-s.stopCh:
		return false
	}
}

// pumpStdout reads the broker's JSON-RPC frames.
func (s *Service) pumpStdout(run *brokerRun) {
	sc := bufio.NewScanner(run.proc.stdout)
	sc.Buffer(make([]byte, 0, 64*1024), jsonrpc.WorkerFrameBytes)
	for sc.Scan() {
		raw := sc.Bytes()
		var msg jsonrpc.Message
		if err := json.Unmarshal(raw, &msg); err != nil || msg.JSONRPC != "2.0" {
			slog.Debug("ignoring malformed broker frame")
			continue
		}
		switch {
		case msg.IsResponse():
			s.routeResponse(&msg)
		case msg.IsNotification():
			frame := make([]byte, len(raw)+1)
			copy(frame, raw)
			frame[len(raw)] = '\n'
			s.handleBrokerNotification(run, msg.Method, frame)
		case msg.IsRequest():
			// The broker's parent protocol has no upward requests; answer so
			// a future one fails fast instead of hanging.
			_ = run.writeMessage(&jsonrpc.Message{JSONRPC: "2.0", ID: msg.ID,
				Error: &jsonrpc.RPCError{Code: codeMethodNotFound, Message: "nvpair-service handles no requests from the broker"}})
		}
	}
	if err := sc.Err(); err != nil {
		// The stream cannot be resynchronised; a broker that keeps running
		// behind it would be unreachable, so end it and let supervision
		// restart it.
		slog.Error("broker stdout stream broken; stopping the broker", "err", err)
		_ = run.proc.kill()
	}
}

// pumpStderr writes each broker stderr line to the log file and broadcasts it
// as service/log.
func (s *Service) pumpStderr(r io.Reader) {
	br := bufio.NewReaderSize(r, 64*1024)
	var line []byte
	for {
		chunk, isPrefix, err := br.ReadLine()
		if len(chunk) > 0 && len(line) < maxLogLine {
			room := maxLogLine - len(line)
			if len(chunk) > room {
				chunk = chunk[:room]
			}
			line = append(line, chunk...)
		}
		if err != nil {
			if len(line) > 0 {
				s.brokerLogLine(line)
			}
			return
		}
		if isPrefix {
			continue
		}
		s.brokerLogLine(line)
		line = line[:0]
	}
}

// logParams is the service/log payload.
type logParams struct {
	Source string `json:"source"`
	Stream string `json:"stream"`
	Text   string `json:"text"`
}

func (s *Service) brokerLogLine(line []byte) {
	if s.log != nil {
		_, _ = s.log.Write(append(append([]byte(nil), line...), '\n'))
	}
	s.broadcast(notifyLog, logParams{Source: brokerLogSource, Stream: "stderr", Text: string(line)})
}

// shutdownBroker stops a broker the way nvpair-tui did when it owned one: ask
// with shutdown, close stdin as a second signal, give it stopGrace to exit, and
// kill it after that. The broker tears its workers down in its own order —
// proxy first, then engines, then the rest — so nothing here gets ahead of it.
func (s *Service) shutdownBroker(run *brokerRun) {
	// The request is written from its own goroutine: a broker that has stopped
	// reading would otherwise block the write, and with it the whole stop.
	answered := make(chan struct{})
	go func() {
		<-s.callBroker(run, methodShutdown)
		close(answered)
	}()
	select {
	case <-answered:
	case <-time.After(s.cfg.shutdownRequestTimeout):
	case <-run.exited:
	}
	run.closeStdin()
	select {
	case <-run.exited:
	case <-time.After(s.cfg.stopGrace):
		slog.Warn("broker did not exit in time; killing it", "pid", run.proc.pid, "grace", s.cfg.stopGrace)
		_ = run.proc.kill()
		<-run.exited
	}
}

// resolveBrokerPath honours an explicit path, otherwise looks beside this
// executable. There is no PATH fallback: a stale broker found on PATH is worse
// than a clear error.
func resolveBrokerPath(override string) (string, error) {
	if override != "" {
		if _, err := os.Stat(override); err != nil {
			return "", fmt.Errorf("broker binary %q not accessible: %w", override, err)
		}
		return filepath.Abs(override)
	}
	bin := brokerName
	if runtime.GOOS == "windows" {
		bin += ".exe"
	}
	exe, err := os.Executable()
	if err != nil {
		return "", fmt.Errorf("locate own executable: %w", err)
	}
	candidate := filepath.Join(filepath.Dir(exe), bin)
	if _, err := os.Stat(candidate); err != nil {
		return "", fmt.Errorf("%s not found next to nvpair-service (use --broker-path to override): %w", bin, err)
	}
	return candidate, nil
}

// execSpawner starts the real broker binary with the working directory set to
// its own directory, which is where it resolves its sibling workers.
//
// The pipes are created here rather than with exec's StdoutPipe helpers so the
// read ends belong to the service: Wait can then run alongside the readers
// without closing a pipe that still holds the broker's last frames.
func execSpawner(brokerPath string, args []string) spawnFunc {
	return func() (*brokerProc, error) {
		stdinR, stdinW, err := os.Pipe()
		if err != nil {
			return nil, err
		}
		stdoutR, stdoutW, err := os.Pipe()
		if err != nil {
			closeFiles(stdinR, stdinW)
			return nil, err
		}
		stderrR, stderrW, err := os.Pipe()
		if err != nil {
			closeFiles(stdinR, stdinW, stdoutR, stdoutW)
			return nil, err
		}
		cmd := exec.Command(brokerPath, args...)
		cmd.Dir = filepath.Dir(brokerPath)
		cmd.Stdin, cmd.Stdout, cmd.Stderr = stdinR, stdoutW, stderrW
		configureBroker(cmd)
		if err := cmd.Start(); err != nil {
			closeFiles(stdinR, stdinW, stdoutR, stdoutW, stderrR, stderrW)
			return nil, fmt.Errorf("start %s: %w", brokerPath, err)
		}
		// The child holds its own copies; ours would keep stdout and stderr
		// from reaching EOF when it exits.
		closeFiles(stdinR, stdoutW, stderrW)
		return &brokerProc{
			stdin:  stdinW,
			stdout: stdoutR,
			stderr: stderrR,
			pid:    cmd.Process.Pid,
			wait: func() int {
				_ = cmd.Wait()
				if cmd.ProcessState == nil {
					return -1
				}
				return cmd.ProcessState.ExitCode()
			},
			kill:        func() error { return cmd.Process.Kill() },
			closeOutput: func() { closeFiles(stdoutR, stderrR) },
		}, nil
	}
}

func closeFiles(files ...*os.File) {
	for _, f := range files {
		_ = f.Close()
	}
}
