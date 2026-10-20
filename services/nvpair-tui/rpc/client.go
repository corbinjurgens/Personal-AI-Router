// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package rpc

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strconv"
	"sync"
	"sync/atomic"
	"time"
)

const notifyBuffer = 256

// EventKind identifies a notification or the loss of its transport.
type EventKind uint8

const (
	EventNotification EventKind = iota
	EventDisconnected
)

// Event preserves wire order across transports. Generation identifies the
// transport that produced Message or disconnected with Err.
type Event struct {
	Kind       EventKind
	Generation uint64
	Message    *Message
	Err        error
}

type transport struct {
	codec      *Codec
	close      func()
	generation uint64
	pending    map[int64]chan *Message
	done       chan struct{}
	once       sync.Once
	closeOnce  sync.Once
}

func (t *transport) connected() bool {
	select {
	case <-t.done:
		return false
	default:
		return true
	}
}

// Client multiplexes requests and pushes. NewClient is terminal after one
// transport; NewReconnectingClient keeps the same handle across replacements.
type Client struct {
	mu         sync.Mutex
	current    *transport
	generation uint64
	stopped    bool
	reconnect  bool
	cancel     context.CancelFunc
	dial       func(context.Context) (io.ReadWriteCloser, error)
	retryDelay func(int) time.Duration
	nextID     atomic.Int64
	events     chan Event
	filter     func(*Message) bool
}

func newClient(r io.Reader, w io.Writer, closeTransport func()) *Client {
	t := &transport{
		codec: NewCodec(r, w), close: closeTransport, generation: 1,
		pending: make(map[int64]chan *Message), done: make(chan struct{}),
	}
	return &Client{
		current: t, generation: 1, events: make(chan Event, notifyBuffer),
		retryDelay: reconnectDelay,
	}
}

// NewClient builds a single-transport client. Run must be called once. When
// either stream supports Close, cancellation closes it to interrupt blocked IO.
func NewClient(r io.Reader, w io.Writer) *Client {
	return newClient(r, w, func() {
		if closer, ok := r.(io.Closer); ok {
			_ = closer.Close()
		}
		if closer, ok := w.(io.Closer); ok {
			_ = closer.Close()
		}
	})
}

// NewReconnectingClient owns conn and all later transports returned by dial.
// Initial connect/start remains the caller's responsibility. After loss, dial
// must only attach to an existing service; it must not launch one.
func NewReconnectingClient(conn io.ReadWriteCloser, dial func(context.Context) (io.ReadWriteCloser, error)) *Client {
	c := newClient(conn, conn, func() { _ = conn.Close() })
	c.dial, c.reconnect = dial, true
	return c
}

// Events is the sole ordered event stream. It closes only when Run finally exits.
func (c *Client) Events() <-chan Event {
	return c.events
}

// SetNotificationFilter installs a nonblocking hook before Run. It survives
// transport replacement; returning true consumes the notification.
func (c *Client) SetNotificationFilter(filter func(*Message) bool) { c.filter = filter }

// Generation returns the latest transport generation. IsCurrent additionally
// checks that it is connected, for consumers applying asynchronous work.
func (c *Client) Generation() uint64 {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.generation
}

func (c *Client) IsCurrent(generation uint64) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return !c.stopped && c.current.connected() && c.current.generation == generation
}

// StopReconnecting suppresses reattachment before an explicit service/stop.
// The current transport remains usable for that final request.
func (c *Client) StopReconnecting() {
	c.mu.Lock()
	c.reconnect = false
	disconnected := !c.current.connected()
	c.mu.Unlock()
	if disconnected {
		c.Close()
	}
}

// Close ends the session without stopping the service. It interrupts blocked
// reads/writes and cancels any backoff or dial in progress.
func (c *Client) Close() {
	c.mu.Lock()
	c.stopped = true
	if c.cancel != nil {
		c.cancel()
	}
	t := c.current
	c.mu.Unlock()
	if t != nil {
		c.breakTransport(t)
	}
}

// Run owns publication and replacement. It is safe to call once. No pending
// request crosses a transport boundary and no request is replayed.
func (c *Client) Run(ctx context.Context) error {
	ctx, cancel := context.WithCancel(ctx)
	c.mu.Lock()
	c.cancel = cancel
	stopped, t := c.stopped, c.current
	c.mu.Unlock()
	defer cancel()
	defer close(c.events)
	defer c.Close()
	if stopped {
		return context.Canceled
	}
	interrupt := context.AfterFunc(ctx, c.Close)
	defer interrupt()
	for {
		err := c.readTransport(ctx, t)
		c.breakTransport(t)
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if c.dial != nil && !c.publish(ctx, Event{Kind: EventDisconnected, Generation: t.generation, Err: err}) {
			return ctx.Err()
		}
		c.mu.Lock()
		retry := c.reconnect && !c.stopped
		c.mu.Unlock()
		if !retry {
			if errors.Is(err, io.EOF) {
				return nil
			}
			return err
		}
		conn, err := c.reattach(ctx)
		if err != nil {
			return err
		}
		c.mu.Lock()
		if c.stopped || !c.reconnect || ctx.Err() != nil {
			c.mu.Unlock()
			_ = conn.Close()
			return context.Canceled
		}
		c.generation++
		t = &transport{
			codec: NewCodec(conn, conn), close: func() { _ = conn.Close() },
			generation: c.generation, pending: make(map[int64]chan *Message), done: make(chan struct{}),
		}
		c.current = t
		c.mu.Unlock()
	}
}

func (c *Client) readTransport(ctx context.Context, t *transport) error {
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		msg, err := t.codec.Read()
		if err != nil {
			if errors.Is(err, io.EOF) || errors.Is(err, ErrStreamBroken) {
				return err
			}
			continue // Skip malformed frames, never a finished scanner.
		}
		switch {
		case msg.IsResponse():
			var id int64
			if json.Unmarshal(*msg.ID, &id) != nil {
				continue
			}
			c.mu.Lock()
			ch := t.pending[id]
			delete(t.pending, id)
			c.mu.Unlock()
			if ch != nil {
				ch <- msg
			}
		case msg.IsNotification():
			if c.filter != nil && c.filter(msg) {
				continue
			}
			if !c.publish(ctx, Event{Kind: EventNotification, Generation: t.generation, Message: msg}) {
				return ctx.Err()
			}
		}
	}
}

func (c *Client) publish(ctx context.Context, event Event) bool {
	select {
	case c.events <- event:
		return true
	case <-ctx.Done():
		return false
	}
}

func (c *Client) breakTransport(t *transport) {
	c.mu.Lock()
	t.once.Do(func() {
		close(t.done)
		for id, ch := range t.pending {
			close(ch)
			delete(t.pending, id)
		}
	})
	c.mu.Unlock()
	t.closeOnce.Do(t.close)
}

func reconnectDelay(attempt int) time.Duration {
	delay := 100 * time.Millisecond
	for i := 0; i < attempt && delay < 2*time.Second; i++ {
		delay *= 2
	}
	if delay > 2*time.Second {
		delay = 2 * time.Second
	}
	return delay
}

func (c *Client) reattach(ctx context.Context) (io.ReadWriteCloser, error) {
	for attempt := 0; ; attempt++ {
		timer := time.NewTimer(c.retryDelay(attempt))
		select {
		case <-ctx.Done():
			timer.Stop()
			return nil, ctx.Err()
		case <-timer.C:
		}
		type dialResult struct {
			conn io.ReadWriteCloser
			err  error
		}
		result := make(chan dialResult)
		go func() {
			conn, err := c.dial(ctx)
			select {
			case result <- dialResult{conn, err}:
			case <-ctx.Done():
				if conn != nil {
					_ = conn.Close()
				}
			}
		}()
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case reply := <-result:
			if reply.err == nil {
				return reply.conn, nil
			}
			if reply.conn != nil {
				_ = reply.conn.Close()
			}
		}
	}
}

// Call uses the transport current at invocation. CallGeneration also refuses
// stale queued commands before writing, even if a replacement is connected.
func (c *Client) Call(ctx context.Context, method string, params any) (*Message, error) {
	return c.CallGeneration(ctx, c.Generation(), method, params)
}

func (c *Client) CallGeneration(ctx context.Context, generation uint64, method string, params any) (*Message, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	raw, err := marshalParams(params)
	if err != nil {
		return nil, err
	}
	c.mu.Lock()
	t := c.current
	if c.stopped || !t.connected() || t.generation != generation {
		c.mu.Unlock()
		return nil, fmt.Errorf("connection closed or generation changed before %q was sent", method)
	}
	id := c.nextID.Add(1)
	ch := make(chan *Message, 1)
	t.pending[id] = ch
	c.mu.Unlock()
	defer func() {
		c.mu.Lock()
		delete(t.pending, id)
		c.mu.Unlock()
	}()
	rawID := json.RawMessage(strconv.FormatInt(id, 10))
	if err := t.codec.Write(&Message{JSONRPC: "2.0", ID: &rawID, Method: method, Params: raw}); err != nil {
		c.breakTransport(t)
		return nil, err
	}
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-t.done:
		return nil, fmt.Errorf("connection closed before response to %q", method)
	case msg, ok := <-ch:
		if !ok {
			return nil, fmt.Errorf("connection closed before response to %q", method)
		}
		if msg.Error != nil {
			return msg, msg.Error
		}
		return msg, nil
	}
}

// Notify sends a fire-and-forget notification on the current transport.
func (c *Client) Notify(method string, params any) error {
	raw, err := marshalParams(params)
	if err != nil {
		return err
	}
	c.mu.Lock()
	t, stopped := c.current, c.stopped
	c.mu.Unlock()
	if stopped || !t.connected() {
		return fmt.Errorf("connection closed before %q was sent", method)
	}
	if err := t.codec.Write(&Message{JSONRPC: "2.0", Method: method, Params: raw}); err != nil {
		c.breakTransport(t)
		return err
	}
	return nil
}

func marshalParams(params any) (json.RawMessage, error) {
	if params == nil {
		return nil, nil
	}
	return json.Marshal(params)
}
