// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package rpc

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"nvpair-shared/jsonrpc"
	"nvpair-shared/nodepolicy"
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
	history    *historySession
}

func newClient(codec *Codec, closeTransport func()) *Client {
	t := &transport{
		codec: codec, close: closeTransport, generation: 1,
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
	return newClient(NewCodec(r, w), func() {
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
	c := newClient(NewCodec(conn, conn), func() { _ = conn.Close() })
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
	if c.history != nil {
		go c.history.run(ctx, c)
	}
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
		if c.history != nil {
			t, err = c.attachHistoryTransport(ctx, conn)
			if err != nil {
				_ = conn.Close()
				return err
			}
			continue
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
		msg, err := c.readMessage(t)
		if err != nil {
			if errors.Is(err, io.EOF) || errors.Is(err, ErrStreamBroken) {
				return err
			}
			continue // Skip malformed frames, never a finished scanner.
		}
		if msg == nil {
			continue
		} // History was consumed under its raw grant.
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
	// Run cannot bind a replacement until this first retirement has interrupted
	// and joined this transport's read/apply borrower. A delayed old Write error
	// observes the completed once operation without entering a newer read lease.
	t.closeOnce.Do(func() {
		t.close()
		if c.history != nil {
			c.history.fence(t)
		}
	})
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
	// History values have a synchronous consume/apply owner, never an ordinary
	// Call waiter whose lifetime could outlast the next physical read.
	if c.history != nil && (method == nodepolicy.MethodRoutingGetInitial || method == nodepolicy.MethodRoutingSubscribe || method == nodepolicy.MethodRoutingUnsubscribe) {
		return nil, jsonrpc.ErrHistoryInvalid
	}
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

// HistoryState contains only scalar presentation state. Busy/validation errors
// retain validated rows and Loading. Exhaustion is fixed and history-only.
type HistoryState struct {
	Generation                                                 uint64
	Connected, Loading, Pending, Retry, Quarantined, Exhausted bool
	RefusalDiscardCategory                                     string
	RefusalDiscardCount                                        uint32
}

// historySession owns the single physical H, M offsets, correlation and fixed
// control slots across every transport generation. It never stores raw values
// in Events or Call waiters. The shared schema owns semantic validation.
type historySession struct {
	mu, readMu                                      sync.Mutex
	t                                               *transport
	correlation                                     jsonrpc.HistoryCorrelation
	raw                                             []byte
	reader                                          *bufio.Reader
	readerGeneration                                uint64
	offsets                                         []uint32
	control, refusal                                [jsonrpc.HistoryEnvelopeBytes]byte
	refusalN                                        int
	refusalActive, writeActive                      bool
	wanted, subscribed, pending, loading, exhausted bool
	unsubscribeNeeded                               bool
	pendingMethod                                   string
	retryAt                                         time.Time
	attempt                                         int
	wake                                            chan struct{}
	ready                                           chan uint64
	consume                                         func(uint64, bool, []byte) error
	reset                                           func(uint64)
	refusalDiscardCount                             uint32
}

// NewHistoryClient constructs a history-aware single-transport client. consume
// must synchronously apply or fund-copy projected bytes before returning, retain
// no borrowed bytes. reset fences/clears that same bounded consumer. Neither
// callback may reenter Client. Validation failure leaves streams with the caller.
func NewHistoryClient(r io.Reader, w io.Writer, consume func(uint64, bool, []byte) error, reset func(uint64)) (*Client, error) {
	return newHistoryClient(r, w, func() {
		if closer, ok := r.(io.Closer); ok {
			_ = closer.Close()
		}
		if closer, ok := w.(io.Closer); ok {
			_ = closer.Close()
		}
	}, consume, reset)
}

// NewReconnectingHistoryClient owns conn and later dial results using one fixed
// session reader and backing. dial only attaches to an existing service. The
// consume/reset borrowing contract is identical to NewHistoryClient.
func NewReconnectingHistoryClient(conn io.ReadWriteCloser, dial func(context.Context) (io.ReadWriteCloser, error), consume func(uint64, bool, []byte) error, reset func(uint64)) (*Client, error) {
	if dial == nil {
		return nil, jsonrpc.ErrHistoryInvalid
	}
	c, err := newHistoryClient(conn, conn, func() { _ = conn.Close() }, consume, reset)
	if err != nil {
		return nil, err
	}
	c.dial, c.reconnect = dial, true
	return c, nil
}

func newHistoryClient(r io.Reader, w io.Writer, closeTransport func(), consume func(uint64, bool, []byte) error, reset func(uint64)) (*Client, error) {
	if r == nil || w == nil || consume == nil || reset == nil {
		return nil, jsonrpc.ErrHistoryInvalid
	}
	// No ordinary scanner is constructed, and no new field/referent is added
	// to a transport or Codec retained by an old ordinary writer/caller.
	c := newClient(&Codec{writer: w}, closeTransport)
	h := &historySession{
		reader:  bufio.NewReaderSize(historyEOFReader{}, 4096),
		raw:     make([]byte, 0, jsonrpc.HistoryFrameBytes),
		offsets: make([]uint32, 0, jsonrpc.HistoryInitialBytes/4),
		wake:    make(chan struct{}, 1), ready: make(chan uint64, 1), consume: consume, reset: reset,
	}
	c.history = h
	h.bind(c.current, r)
	return c, nil
}

func (h *historySession) signal() {
	select {
	case h.wake <- struct{}{}:
	default:
	}
}

// changed coalesces to one scalar ready token, including across generations.
// A delayed token is never authority to read/apply a different generation.
func (h *historySession) changed() {
	select {
	case <-h.ready:
	default:
	}
	select {
	case h.ready <- h.t.generation:
	default:
	}
}

func (c *Client) HistoryReady() <-chan uint64 {
	if c.history == nil {
		return nil
	}
	return c.history.ready
}

func (c *Client) HistoryState() HistoryState {
	if c.history == nil {
		return HistoryState{}
	}
	h := c.history
	h.mu.Lock()
	defer h.mu.Unlock()
	state := HistoryState{Generation: h.t.generation, Connected: h.t.connected(), Loading: h.loading, Pending: h.pending, Retry: !h.retryAt.IsZero(), Quarantined: h.correlation.Quarantined(), Exhausted: h.exhausted, RefusalDiscardCount: h.refusalDiscardCount}
	if state.RefusalDiscardCount != 0 {
		state.RefusalDiscardCategory = "history-wrong-direction"
	}
	return state
}

func (c *Client) SetHistorySubscription(wanted bool) {
	if c.history == nil {
		return
	}
	h := c.history
	h.mu.Lock()
	if h.wanted == wanted {
		h.mu.Unlock()
		return
	}
	h.wanted = wanted
	h.retryAt = time.Time{}
	h.attempt = 0
	if wanted {
		h.loading = !h.exhausted
	} else {
		h.correlation.Retire()
		h.pending = false
		h.pendingMethod = ""
		h.loading = false
		h.subscribed = false
	}
	h.changed()
	h.signal()
	h.mu.Unlock()
}

func (h *historySession) bind(t *transport, r io.Reader) {
	h.readMu.Lock()
	defer h.readMu.Unlock()
	h.mu.Lock()
	defer h.mu.Unlock()
	h.reset(t.generation)
	h.bindLocked(t, r)
}

// bindLocked holds readMu and mu. Reader Reset and source/generation admission
// change together, before publication; it allocates no per-generation reader.
func (h *historySession) bindLocked(t *transport, r io.Reader) {
	h.reader.Reset(r)
	h.readerGeneration = t.generation
	h.t = t
	h.correlation = jsonrpc.HistoryCorrelation{}
	// A fresh actual transport is the sole reset of the generated frontier.
	if err := h.correlation.EnableGeneratedIDs(); err != nil {
		h.exhausted = true
		return
	}
	h.subscribed = false
	h.unsubscribeNeeded = false
	h.pending = false
	h.pendingMethod = ""
	h.exhausted = false
	h.retryAt = time.Time{}
	h.attempt = 0
	h.loading = h.wanted
	h.refusalDiscardCount = 0
	// Active output slots are intentionally not reset by generation loss.
	if !h.refusalActive {
		h.refusalN = 0
	}
	h.changed()
	h.signal()
}

func (c *Client) attachHistoryTransport(ctx context.Context, conn io.ReadWriteCloser) (*transport, error) {
	h := c.history
	h.readMu.Lock()
	defer h.readMu.Unlock()
	h.mu.Lock()
	defer h.mu.Unlock()
	c.mu.Lock()
	generation := c.generation + 1
	stopped := c.stopped || !c.reconnect || ctx.Err() != nil
	c.mu.Unlock()
	if stopped {
		return nil, context.Canceled
	}
	t := &transport{codec: &Codec{writer: conn}, close: func() { _ = conn.Close() }, generation: generation, pending: make(map[int64]chan *Message), done: make(chan struct{})}
	// A consumer/output loan can delay reset. Do not hold Client.mu across it.
	// History readers/writers stay excluded until binding and publication finish.
	h.reset(generation)
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.stopped || !c.reconnect || ctx.Err() != nil {
		return nil, context.Canceled
	}
	h.bindLocked(t, conn)
	c.current = t
	c.generation = generation
	return t, nil
}

func (h *historySession) fence(t *transport) {
	// breakTransport has already closed the real transport, interrupting an
	// admitted physical Read. Join that borrower, never an ordinary callback.
	h.readMu.Lock()
	defer h.readMu.Unlock()
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.t != t {
		return
	}
	h.reader.Reset(historyEOFReader{})
	h.readerGeneration = 0
	h.correlation.Retire()
	h.pending = false
	h.pendingMethod = ""
	h.subscribed = false
	h.retryAt = time.Time{}
	h.loading = false
	h.reset(t.generation)
	h.changed()
	h.signal()
}

func historyRetryDelay(attempt int) time.Duration {
	if attempt == 0 {
		return 250 * time.Millisecond
	}
	if attempt == 1 {
		return 500 * time.Millisecond
	}
	return time.Second
}

// failed is called with mu held. No timeout ever completes a read/write loan.
func (h *historySession) failed(err error) {
	h.correlation.Retire()
	h.pending = false
	h.pendingMethod = ""
	if errors.Is(err, jsonrpc.ErrHistoryExhausted) {
		h.exhausted = true
		h.retryAt = time.Time{}
		h.loading = false
	} else if h.wanted && h.t.connected() && !h.exhausted {
		h.loading = true
		if h.retryAt.IsZero() {
			h.retryAt = time.Now().Add(historyRetryDelay(h.attempt))
			if h.attempt < 2 {
				h.attempt++
			}
		}
	}
	h.changed()
	h.signal()
}

// run is one control writer and one reusable retry timer per Client, never per
// request/generation. It keeps its frame until actual Write returns, even when
// quarantine has already retired the logical pending call.
func (h *historySession) run(ctx context.Context, c *Client) {
	timer := time.NewTimer(time.Hour)
	if !timer.Stop() {
		<-timer.C
	}
	defer timer.Stop()
	for {
		h.mu.Lock()
		t := h.t
		var frame []byte
		refusal := false
		if !h.writeActive && t.connected() {
			if h.refusalN != 0 {
				frame = h.refusal[:h.refusalN]
				h.refusalActive = true
				refusal = true
			} else if !h.pending && !h.exhausted && (h.retryAt.IsZero() || !time.Now().Before(h.retryAt)) {
				method := ""
				if h.wanted {
					method = nodepolicy.MethodRoutingSubscribe
					if h.subscribed {
						method = nodepolicy.MethodRoutingGetInitial
					}
					if !h.loading && !h.correlation.Quarantined() {
						method = ""
					}
				} else if h.unsubscribeNeeded {
					method = nodepolicy.MethodRoutingUnsubscribe
				}
				if method != "" {
					id, err := h.correlation.BeginGeneratedMethod(method)
					if err != nil {
						h.failed(err)
					} else {
						frame = append(h.control[:0], `{"jsonrpc":"2.0","id":`...)
						frame = append(frame, id...)
						frame = append(frame, `,"method":"`...)
						frame = append(frame, method...)
						frame = append(frame, `","params":{}}`...)
						frame = append(frame, '\n')
						h.pending = true
						if method == nodepolicy.MethodRoutingSubscribe {
							h.unsubscribeNeeded = true
						}
						if method == nodepolicy.MethodRoutingUnsubscribe {
							h.unsubscribeNeeded = false
						}
						h.pendingMethod = method
						h.retryAt = time.Time{}
					}
				}
			}
		}
		deadline := h.retryAt
		if len(frame) != 0 {
			h.writeActive = true
		}
		h.mu.Unlock()
		if len(frame) != 0 {
			err := t.codec.writeHistoryFrame(frame)
			h.mu.Lock()
			h.writeActive = false
			if refusal {
				h.refusalActive = false
				h.refusalN = 0
			}
			h.signal()
			h.mu.Unlock()
			if err != nil {
				c.breakTransport(t)
			}
			continue
		}
		if !timer.Stop() {
			select {
			case <-timer.C:
			default:
			}
		}
		var due <-chan time.Time
		if !deadline.IsZero() {
			timer.Reset(max(time.Until(deadline), 0))
			due = timer.C
		}
		select {
		case <-ctx.Done():
			return
		case <-h.wake:
		case <-due:
		}
	}
}

func (c *Client) readMessage(t *transport) (*Message, error) {
	h := c.history
	if h == nil {
		return t.codec.Read()
	}
	// A fenced old direct caller must not join a replacement's idle Read.
	if !t.connected() {
		return nil, io.EOF
	}
	h.readMu.Lock()
	defer h.readMu.Unlock()
	h.mu.Lock()
	admitted := h.t == t && t.connected() && h.readerGeneration == t.generation
	h.mu.Unlock()
	if !admitted {
		return nil, io.EOF
	}
	raw, err := readHistoryFrame(h.reader, h.raw)
	if err != nil {
		return nil, err
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.t != t || !t.connected() {
		return nil, io.EOF
	}
	msg, err := guardHistoryFrame(raw, h.offsets, &h.correlation, nodepolicy.RoutingHistoryTransportSchema)
	if err != nil {
		var failure *jsonrpc.HistoryFailure
		if errors.As(err, &failure) {
			if failure.Refusal != nil {
				h.refuse(failure.Refusal)
			} else if failure.FailPending || failure.ScheduleRefetch {
				h.failed(jsonrpc.ErrHistoryInvalid)
			}
			return nil, nil
		}
		return nil, err
	}
	if msg.IsResponse() && msg.ID != nil && h.correlation.MatchesPending(*msg.ID) {
		method := h.pendingMethod
		if msg.Error != nil {
			h.correlation.Retire()
			h.failed(jsonrpc.ErrHistoryInvalid)
		} else if method == nodepolicy.MethodRoutingUnsubscribe {
			h.correlation.Retire()
			h.pending = false
			h.pendingMethod = ""
			h.subscribed = false
			h.unsubscribeNeeded = false
			h.changed()
			h.signal()
		} else if h.wanted && h.consume(t.generation, true, msg.Result) == nil {
			h.correlation.Complete()
			h.pending = false
			h.pendingMethod = ""
			h.subscribed = true
			h.loading = false
			h.retryAt = time.Time{}
			h.attempt = 0
			h.changed()
			h.signal()
		} else {
			h.correlation.Retire()
			h.failed(jsonrpc.ErrHistoryInvalid)
		}
		return nil, nil
	}
	switch msg.Method {
	case nodepolicy.NotifyRoutingChanged:
		// Guarded pushes must reach the bounded merge even during quarantine:
		// a delayed refetch baseline may be older. Only a successful initial
		// completes correlation and clears quarantine/loading above.
		if msg.IsNotification() && h.wanted {
			if err := h.consume(t.generation, false, msg.Params); err != nil {
				h.failed(jsonrpc.ErrHistoryInvalid)
			} else {
				h.changed()
			}
		}
		return nil, nil
	case nodepolicy.MethodRoutingGetInitial, nodepolicy.MethodRoutingSubscribe, nodepolicy.MethodRoutingUnsubscribe,
		nodepolicy.NotifyRoutingRequest, nodepolicy.NotifyRoutingHistoryWindow, nodepolicy.NotifyRoutingHistoryWindowEnd, nodepolicy.NotifyRoutingHistoryAccepted:
		// The TUI is a consumer, never a history server/private worker. Incoming
		// calls do not complete any outbound correlation, even with the same ID.
		if msg.IsRequest() {
			h.refuse(&jsonrpc.HistoryRequestRefusal{ReplyID: *msg.ID, Code: -32601, Message: "method not found"})
		}
		return nil, nil
	}
	// The guard owns ordinary decoded messages independently of raw/output.
	// Detach here before the filter, queue, Call waiter, or any ordinary callback.
	ordinary := &Message{JSONRPC: msg.JSONRPC, ID: msg.ID, Method: msg.Method, Params: msg.Params, Result: msg.Result}
	if msg.Error != nil {
		ordinary.Error = &RPCError{Code: msg.Error.Code, Message: msg.Error.Message, Data: msg.Error.Data}
	}
	return ordinary, nil
}

type historyEOFReader struct{}

func (historyEOFReader) Read([]byte) (int, error) { return 0, io.EOF }

func (h *historySession) refuse(refusal *jsonrpc.HistoryRequestRefusal) {
	// Service-to-TUI history requests are unexpected, not a public RPC server
	// surface. This client direction admits one best-effort fixed refusal.
	// An occupied slot discards the additional unaccepted request before any
	// extra token copy, queue, wait or refetch. Admitted backing stays pinned
	// until the sole writer's actual completion, independently of correlation.
	if h.refusalN != 0 || h.refusalActive {
		if h.refusalDiscardCount < ^uint32(0) {
			h.refusalDiscardCount++
		}
		return
	}
	frame := append(h.refusal[:0], `{"jsonrpc":"2.0","id":`...)
	frame = append(frame, refusal.ReplyID...)
	frame = append(frame, `,"error":{"code":`...)
	frame = strconv.AppendInt(frame, int64(refusal.Code), 10)
	frame = append(frame, `,"message":"`...)
	frame = append(frame, refusal.Message...)
	frame = append(frame, `"}}`...)
	frame = append(frame, '\n')
	h.refusalN = len(frame)
	h.signal()
}
