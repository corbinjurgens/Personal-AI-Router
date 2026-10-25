// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"strings"
)

// Invoked only by nvpair-service's cross-process recovery regression. No worker
// process, listener, app-data path or engine operation is used. Engine stdout
// travels through the real Peer callback and relay to the service's broker
// output reader.
const engineNotificationServiceFixtureArg = "--engine-notification-service-fixture"

func runEngineNotificationServiceFixture() error {
	codec := NewCodecMaxFrame(readWriter{os.Stdin, os.Stdout}, workerFrameBytes)
	b := &Broker{codec: codec, disconnectClient: func() { _ = os.Stdout.Close() }}
	stop := b.startEngineNotificationRelay(context.Background())
	defer stop()
	source := b.activateEngineManagerSource()
	engine, reader := net.Pipe()
	defer engine.Close()
	defer reader.Close()
	peer := NewPeer(NewCodecMaxFrame(reader, workerFrameBytes))
	handled := make(chan string, engineNotificationQueueFrames*3+16)
	pump := make(chan struct{})
	go func() {
		defer close(pump)
		peer.Serve(nil, func(method string, params json.RawMessage) {
			b.forwardEngineNotificationFrom(source, method, params)
			handled <- method
		})
	}()
	defer func() { _ = reader.Close(); <-pump }()
	publisher := NewCodecMaxFrame(engine, workerFrameBytes)
	if err := codec.Notify("app:ready", ReadyParams{Version: "relay-fixture"}); err != nil {
		return err
	}
	afterReset := false
	for {
		msg, err := codec.Read()
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return err
		}
		switch msg.Method {
		case "test:overflow":
			afterReset = true
			payload := map[string]string{"data": strings.Repeat("x", 256<<10)}
			for range engineNotificationQueueFrames*3 + 4 {
				if err := publisher.Notify("engine:pull-progress", payload); err != nil {
					return err
				}
			}
			if err := publisher.Notify(engineModelIdentityMethod, engineManagerSnapshot("ollama")); err != nil {
				return err
			}
			for range engineNotificationQueueFrames*3 + 5 {
				<-handled
			}
			if _, err := fmt.Fprintln(os.Stderr, "engine-drained"); err != nil {
				return err
			}
		case "engine:subscribe":
			b.handleMessage(msg)
		case "engine:models":
			model := "before-reset"
			if afterReset {
				model = "after-reset"
			}
			if err := codec.Respond(msg.ID, map[string]any{"models": []string{model}}); err != nil {
				return err
			}
		case "test:engine-alive":
			if err := publisher.Notify(engineModelIdentityMethod, engineManagerSnapshot("ollama")); err != nil {
				return err
			}
			<-handled
			if err := codec.Respond(msg.ID, map[string]bool{"alive": true}); err != nil {
				return err
			}
		default:
			if msg.ID != nil {
				if err := codec.RespondError(msg.ID, -32601, "fixture method not found"); err != nil {
					return err
				}
			}
		}
	}
}
