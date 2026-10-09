// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"io"
	"log/slog"
	"sync"
	"time"

	"nvpair-shared/jsonrpc"
)

// clientQueueFrames bounds the frames waiting to be written to one client.
//
// Writes go through a queue so a client that stops reading cannot stall the
// broker's stdout pump and with it every other client. A client that falls this
// far behind is disconnected rather than silently missing frames: a gap in a
// JSON-RPC stream is a lost response or a lost state change, and the client
// cannot tell. Reconnecting gets it a fresh app:ready and lets it re-read state.
const clientQueueFrames = 4096

// clientFlushTimeout bounds how long a gracefully closed client's queued
// frames may take to drain before its connection is closed anyway.
const clientFlushTimeout = 2 * time.Second

// client is one attached connection. Frames are queued whole, newline
// included, and written by a single goroutine.
type client struct {
	id   int
	conn io.ReadWriteCloser

	mu      sync.Mutex
	out     chan []byte
	closing bool

	// written is closed when the writer goroutine has finished and the
	// connection is closed.
	written chan struct{}
}

func newClient(id int, conn io.ReadWriteCloser) *client {
	c := &client{
		id:      id,
		conn:    conn,
		out:     make(chan []byte, clientQueueFrames),
		written: make(chan struct{}),
	}
	go c.writeLoop()
	return c
}

func (c *client) writeLoop() {
	defer close(c.written)
	defer c.conn.Close()
	for frame := range c.out {
		if _, err := c.conn.Write(frame); err != nil {
			// The reader notices the closed connection and detaches the
			// client; drain so senders never block on a dead queue.
			_ = c.conn.Close()
			for range c.out {
			}
			return
		}
	}
}

// send queues one frame (which must end in a newline). It never blocks; a full
// queue disconnects the client.
func (c *client) send(frame []byte) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closing {
		return
	}
	select {
	case c.out <- frame:
	default:
		slog.Warn("client is not reading; disconnecting it", "client", c.id, "queued", len(c.out))
		c.closing = true
		close(c.out)
		_ = c.conn.Close()
	}
}

// sendMessage marshals and queues msg.
func (c *client) sendMessage(msg *jsonrpc.Message) {
	data, err := json.Marshal(msg)
	if err != nil {
		slog.Error("marshal frame for client", "client", c.id, "err", err)
		return
	}
	c.send(append(data, '\n'))
}

func (c *client) respond(id *json.RawMessage, result any) {
	raw, err := json.Marshal(result)
	if err != nil {
		c.respondError(id, codeInternal, err.Error())
		return
	}
	c.sendMessage(&jsonrpc.Message{JSONRPC: "2.0", ID: id, Result: raw})
}

func (c *client) respondError(id *json.RawMessage, code int, message string) {
	c.sendMessage(&jsonrpc.Message{JSONRPC: "2.0", ID: id, Error: &jsonrpc.RPCError{Code: code, Message: message}})
}

func (c *client) notify(method string, params any) {
	raw, err := json.Marshal(params)
	if err != nil {
		slog.Error("marshal notification", "method", method, "err", err)
		return
	}
	c.sendMessage(&jsonrpc.Message{JSONRPC: "2.0", Method: method, Params: raw})
}

// closeGraceful stops accepting frames, lets the queue drain, then closes the
// connection. Used after a frame that must reach the client — the answer to its
// own shutdown or service/stop — has been queued.
func (c *client) closeGraceful() {
	c.mu.Lock()
	if !c.closing {
		c.closing = true
		close(c.out)
	}
	c.mu.Unlock()
	go func() {
		select {
		case <-c.written:
		case <-time.After(clientFlushTimeout):
			_ = c.conn.Close()
		}
	}()
}

func (c *client) isClosing() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.closing
}

// closeNow drops anything queued and closes the connection.
func (c *client) closeNow() {
	c.mu.Lock()
	if !c.closing {
		c.closing = true
		close(c.out)
	}
	c.mu.Unlock()
	_ = c.conn.Close()
}
