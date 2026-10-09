// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"time"

	"nvpair-shared/jsonrpc"
	"nvpair-shared/servicectl"
)

// ctlStopTimeout covers the broker's full stop grace plus the shutdown request
// and some margin, so `nvpair-service stop` does not give up on a service that
// is still stopping within its budget.
const ctlStopTimeout = defaultShutdownRequestTimeout + defaultStopGrace + 10*time.Second

// runStatus prints service/status from the running service.
func runStatus(stdout, stderr io.Writer) int {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	result, err := callRunningService(ctx, methodStatus)
	if errors.Is(err, servicectl.ErrNotRunning) {
		fmt.Fprintln(stdout, "nvpair-service: not running")
		return 3
	}
	if err != nil {
		fmt.Fprintln(stderr, "nvpair-service:", err)
		return 1
	}
	var pretty bytes.Buffer
	if json.Indent(&pretty, result, "", "  ") != nil {
		pretty.Write(result)
	}
	fmt.Fprintln(stdout, pretty.String())
	return 0
}

// runStop asks the running service to stop its broker and exit, and waits for
// the answer.
func runStop(stdout, stderr io.Writer) int {
	ctx, cancel := context.WithTimeout(context.Background(), ctlStopTimeout)
	defer cancel()
	_, err := callRunningService(ctx, methodStop)
	if errors.Is(err, servicectl.ErrNotRunning) {
		fmt.Fprintln(stdout, "nvpair-service: not running")
		return 0
	}
	if err != nil {
		fmt.Fprintln(stderr, "nvpair-service:", err)
		return 1
	}
	fmt.Fprintln(stdout, "nvpair-service: stopped")
	return 0
}

// ctlCallTimeout bounds `nvpair-service call`. Pausing waits for running work
// to drain, so it is generous.
const ctlCallTimeout = 15 * time.Minute

// runCall sends one JSON-RPC request through the running service, for driving
// the broker from a terminal: `nvpair-service call policy:get`, or
// `nvpair-service call node:set-availability '{"state":"paused"}'`.
func runCall(args []string, stdout, stderr io.Writer) int {
	if len(args) < 1 || len(args) > 2 {
		fmt.Fprintln(stderr, "usage: nvpair-service call <method> [json-params]")
		return 2
	}
	var params json.RawMessage
	if len(args) == 2 {
		if !json.Valid([]byte(args[1])) {
			fmt.Fprintln(stderr, "nvpair-service: params are not valid JSON")
			return 2
		}
		params = json.RawMessage(args[1])
	}
	ctx, cancel := context.WithTimeout(context.Background(), ctlCallTimeout)
	defer cancel()
	result, err := callRunningServiceWith(ctx, args[0], params)
	if errors.Is(err, servicectl.ErrNotRunning) {
		fmt.Fprintln(stderr, "nvpair-service: not running")
		return 3
	}
	if err != nil {
		fmt.Fprintln(stderr, "nvpair-service:", err)
		return 1
	}
	var pretty bytes.Buffer
	if json.Indent(&pretty, result, "", "  ") != nil {
		pretty.Write(result)
	}
	fmt.Fprintln(stdout, pretty.String())
	return 0
}

// callRunningService sends one request to the running service and returns its
// result, skipping the notifications the service sends every client.
func callRunningService(ctx context.Context, method string) (json.RawMessage, error) {
	return callRunningServiceWith(ctx, method, nil)
}

func callRunningServiceWith(ctx context.Context, method string, params json.RawMessage) (json.RawMessage, error) {
	conn, err := servicectl.Dial(ctx)
	if err != nil {
		return nil, err
	}
	defer conn.Close()
	go func() {
		<-ctx.Done()
		_ = conn.Close()
	}()

	req, _ := json.Marshal(&jsonrpc.Message{JSONRPC: "2.0", ID: rawID(1), Method: method, Params: params})
	if _, err := conn.Write(append(req, '\n')); err != nil {
		return nil, err
	}
	sc := bufio.NewScanner(conn)
	sc.Buffer(make([]byte, 0, 64*1024), jsonrpc.WorkerFrameBytes)
	for sc.Scan() {
		var msg jsonrpc.Message
		if json.Unmarshal(sc.Bytes(), &msg) != nil || !msg.IsResponse() || string(*msg.ID) != "1" {
			continue
		}
		if msg.Error != nil {
			return nil, fmt.Errorf("%s: %s", method, msg.Error.Message)
		}
		return msg.Result, nil
	}
	if ctx.Err() != nil {
		return nil, fmt.Errorf("%s: %w", method, ctx.Err())
	}
	return nil, fmt.Errorf("%s: connection closed before the service answered", method)
}
