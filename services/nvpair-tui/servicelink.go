// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"time"

	"nvpair-shared/servicectl"
	"nvpair-tui/rpc"
)

// serviceName is the binary the TUI attaches to. It is resolved next to our
// own executable: the installed bin/ layout puts nvpair-tui beside
// nvpair-service, which in turn runs nvpair-ui-broker from the same directory.
const serviceName = "nvpair-service"

// stopServiceTimeout bounds service/stop. The service answers only once the
// broker has exited, after up to 2 s for the broker's shutdown answer and 18 s
// of grace (nvpair-service's defaultStopGrace, which matches the broker's own
// 15 s teardown budget plus headroom), so this needs margin beyond both.
const stopServiceTimeout = 30 * time.Second

// endpointGoneTimeout bounds the wait, after service/stop answers, for the
// service process to release its endpoint and log files.
const endpointGoneTimeout = 5 * time.Second

// logLineBuffer bounds the service/log lines waiting for the Logs view. The
// view drains promptly; a burst beyond this is dropped rather than stalling the
// connection's read loop, which also carries every response.
const logLineBuffer = 2000

// serviceLink is the TUI's connection to nvpair-service: the JSON-RPC client
// the views use, and the broker's stderr reassembled from service/log
// notifications for the Logs view.
type serviceLink struct {
	conn   io.ReadWriteCloser
	Client *rpc.Client
	// Logs yields one broker stderr line per line, as the broker's stderr pipe
	// did when the TUI owned the broker.
	Logs  io.Reader
	logsW *io.PipeWriter
	lines chan string
}

// resolveServicePath honours an explicit override, otherwise looks for the
// service binary alongside this executable. We deliberately do not fall back to
// PATH: a stray PATH match for a stale dev build is worse than a clear error.
func resolveServicePath(override string) (string, error) {
	if override != "" {
		if _, err := os.Stat(override); err != nil {
			return "", fmt.Errorf("service binary %q not accessible: %w", override, err)
		}
		return filepath.Abs(override)
	}
	bin := serviceName
	if runtime.GOOS == "windows" {
		bin += ".exe"
	}
	exe, err := os.Executable()
	if err != nil {
		return "", fmt.Errorf("locate own executable: %w", err)
	}
	candidate := filepath.Join(filepath.Dir(exe), bin)
	if _, err := os.Stat(candidate); err != nil {
		return "", fmt.Errorf("%s not found next to nvpair-tui (use --service-path to override): %w", bin, err)
	}
	return candidate, nil
}

// connectService attaches to the running service, starting it detached from
// servicePath when it is not running, and starts the JSON-RPC read loop. ctx
// governs the read loop; detach ends the session without stopping anything.
func connectService(ctx context.Context, servicePath string) (*serviceLink, error) {
	conn, err := servicectl.ConnectOrStart(ctx, servicePath, nil)
	if err != nil {
		return nil, err
	}
	return newServiceLink(ctx, conn), nil
}

func newServiceLink(ctx context.Context, conn io.ReadWriteCloser) *serviceLink {
	logsR, logsW := io.Pipe()
	l := &serviceLink{
		conn:   conn,
		Client: rpc.NewClient(conn, conn),
		Logs:   logsR,
		logsW:  logsW,
		lines:  make(chan string, logLineBuffer),
	}
	l.Client.SetNotificationFilter(l.takeLogLine)
	go l.pumpLogs()
	go func() {
		_ = l.Client.Run(ctx)
		close(l.lines)
	}()
	return l
}

// takeLogLine diverts service/log notifications to the Logs view. Called on the
// read loop, so it never blocks.
func (l *serviceLink) takeLogLine(m *rpc.Message) bool {
	if m.Method != "service/log" {
		return false
	}
	var p struct {
		Text string `json:"text"`
	}
	if err := json.Unmarshal(m.Params, &p); err != nil {
		return true
	}
	select {
	case l.lines <- p.Text:
	default:
	}
	return true
}

func (l *serviceLink) pumpLogs() {
	for line := range l.lines {
		if _, err := io.WriteString(l.logsW, line+"\n"); err != nil {
			break
		}
	}
	_ = l.logsW.Close()
}

// detach leaves the service and its broker running and closes this client's
// connection. The service treats a closed connection exactly like a client
// shutdown request.
func (l *serviceLink) detach() {
	_ = l.conn.Close()
	_ = l.logsW.Close()
}

// stopService asks the service to stop its broker cleanly and exit, then waits
// for it to release the endpoint. The broker tears its workers down in its own
// order; nothing here gets ahead of it.
func (l *serviceLink) stopService() error {
	ctx, cancel := context.WithTimeout(context.Background(), stopServiceTimeout)
	defer cancel()
	if _, err := l.Client.Call(ctx, "service/stop", nil); err != nil {
		return fmt.Errorf("service/stop: %w", err)
	}
	waitEndpointGone()
	return nil
}

// stopRunningService implements --stop-service: it stops a running service
// without starting one. It reports whether a service was running.
func stopRunningService() (bool, error) {
	ctx, cancel := context.WithTimeout(context.Background(), stopServiceTimeout)
	defer cancel()
	conn, err := servicectl.Dial(ctx)
	if errors.Is(err, servicectl.ErrNotRunning) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	link := newServiceLink(ctx, conn)
	defer link.detach()
	return true, link.stopService()
}

// waitEndpointGone polls until the service no longer answers, so a caller about
// to delete the data directory does not race the service's own exit.
func waitEndpointGone() {
	deadline := time.Now().Add(endpointGoneTimeout)
	for time.Now().Before(deadline) {
		ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
		conn, err := servicectl.Dial(ctx)
		cancel()
		if err != nil {
			return
		}
		_ = conn.Close()
		time.Sleep(100 * time.Millisecond)
	}
	slog.Warn("nvpair-service still answering after it was asked to stop")
}
