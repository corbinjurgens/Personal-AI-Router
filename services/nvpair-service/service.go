// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"nvpair-shared/ipc"
	"nvpair-shared/jsonrpc"
	"nvpair-shared/servicectl"
)

// JSON-RPC error codes the service answers with itself.
const (
	codeMethodNotFound = -32601
	codeInternal       = -32603
	// codeUnavailable matches the broker's own "worker not available" code, so
	// clients already handle it.
	codeUnavailable = -32000
)

// Service methods and notifications. Everything under service/ is answered
// here and never reaches the broker.
const (
	methodStatus          = "service/status"
	methodStop            = "service/stop"
	notifyBrokerRestarted = "service/broker-restarted"
	notifyLog             = "service/log"
	servicePrefix         = "service/"

	methodShutdown = "shutdown"
	methodAppReady = "app:ready"

	subscribeSuffix   = ":subscribe"
	unsubscribeSuffix = ":unsubscribe"

	// discoveryBaseline is the snapshot the broker pushes once, on a fresh
	// discovery:subscribe. See baselinesFor.
	discoveryBaseline = "discovery:nodes-changed"
)

// errAlreadyRunning means another nvpair-service already owns the endpoint.
var errAlreadyRunning = errors.New("already running")

// config is everything a Service needs. Tests replace spawn and backoff.
type config struct {
	endpoint string
	version  string
	// logPath is the broker log file; empty disables the file.
	logPath    string
	logMaxSize int64
	spawn      spawnFunc
	// backoff is the pause before restart attempt n (1-based).
	backoff func(attempt int) time.Duration
	// stopGrace is how long a broker gets to exit after its shutdown request
	// and stdin EOF before it is killed.
	stopGrace time.Duration
	// shutdownRequestTimeout bounds the wait for the broker's answer to
	// shutdown, which comes before the stdin close.
	shutdownRequestTimeout time.Duration
}

// Service owns one broker at a time and multiplexes any number of clients onto
// its stdio.
type Service struct {
	cfg       config
	startedAt time.Time
	log       *rotatingLog

	mu      sync.Mutex
	clients map[*client]struct{}
	nextCID int
	pending map[int64]*pendingRequest
	nextID  int64
	broker  *brokerRun
	// readyFrame is the current broker's app:ready, replayed to each client
	// that attaches after it was sent.
	readyFrame []byte
	// baselines holds the latest frame of each notification the broker sends
	// only once per fresh subscription. See baselinesFor.
	baselines map[string][]byte
	// subscribed is the set of *:subscribe methods the current broker has
	// acknowledged. resubscribe carries them across a broker restart.
	subscribed  map[string]bool
	resubscribe []string
	runs        int
	restarts    int
	stopping    bool

	stopCh     chan struct{}
	stopOnce   sync.Once
	stopped    chan struct{}
	exitCh     chan struct{}
	exitOnce   sync.Once
	brokerDone chan struct{}
}

// pendingRequest is a request in flight at the broker under a service id.
type pendingRequest struct {
	// client and origID route a client's request back; internal handles one
	// the service sent itself. Exactly one is set.
	client   *client
	origID   json.RawMessage
	internal func(*jsonrpc.Message)
	method   string
	run      *brokerRun
}

func newService(cfg config) *Service {
	return &Service{
		cfg:        cfg,
		clients:    make(map[*client]struct{}),
		pending:    make(map[int64]*pendingRequest),
		baselines:  make(map[string][]byte),
		subscribed: make(map[string]bool),
		stopCh:     make(chan struct{}),
		stopped:    make(chan struct{}),
		exitCh:     make(chan struct{}),
		brokerDone: make(chan struct{}),
	}
}

// Run claims the endpoint, supervises the broker, and serves clients until ctx
// ends or a client sends service/stop. It returns errAlreadyRunning when another
// service answers on the endpoint.
func (s *Service) Run(ctx context.Context) error {
	release, err := acquireInstanceLock(s.cfg.endpoint)
	if err != nil {
		return err
	}
	defer release()

	if alreadyAnswering(ctx, s.cfg.endpoint) {
		return errAlreadyRunning
	}
	if err := prepareEndpoint(s.cfg.endpoint); err != nil {
		return err
	}
	listener, err := ipc.ListenPrivate(s.cfg.endpoint)
	if err != nil {
		// Lost a race to a service that started between the dial and here
		// (on Windows the first pipe instance wins).
		if alreadyAnswering(ctx, s.cfg.endpoint) {
			return errAlreadyRunning
		}
		return fmt.Errorf("listen on %s: %w", s.cfg.endpoint, err)
	}
	defer listener.Close()

	if s.cfg.logPath != "" {
		l, err := openRotatingLog(s.cfg.logPath, s.cfg.logMaxSize)
		if err != nil {
			slog.Warn("broker log file unavailable; logging to clients only", "path", s.cfg.logPath, "err", err)
		} else {
			s.log = l
			defer l.Close()
		}
	}

	s.startedAt = time.Now()
	slog.Info("service listening", "endpoint", s.cfg.endpoint, "pid", os.Getpid())

	go s.superviseBroker()
	go s.acceptLoop(listener)

	select {
	case <-ctx.Done():
		slog.Info("stop requested by signal")
		s.stop()
	case <-s.exitCh:
	}

	_ = listener.Close()
	s.mu.Lock()
	clients := make([]*client, 0, len(s.clients))
	for c := range s.clients {
		clients = append(clients, c)
	}
	s.clients = make(map[*client]struct{})
	s.mu.Unlock()
	for _, c := range clients {
		c.closeGraceful()
	}
	for _, c := range clients {
		select {
		case <-c.written:
		case <-time.After(clientFlushTimeout):
		}
	}
	slog.Info("service stopped")
	return nil
}

// alreadyAnswering reports whether a service is listening on endpoint.
func alreadyAnswering(ctx context.Context, endpoint string) bool {
	dctx, cancel := context.WithTimeout(ctx, time.Second)
	defer cancel()
	conn, err := servicectl.DialEndpoint(dctx, endpoint)
	if err != nil {
		return false
	}
	_ = conn.Close()
	return true
}

// prepareEndpoint creates the socket's directory and removes a stale socket
// left by a service that did not exit cleanly. It only runs once the instance
// lock is held and nothing answered, so the file cannot belong to a live
// service. Named pipes need neither step.
func prepareEndpoint(endpoint string) error {
	if runtime.GOOS == "windows" && strings.HasPrefix(endpoint, `\\.\pipe\`) {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(endpoint), 0o700); err != nil {
		return fmt.Errorf("create endpoint directory: %w", err)
	}
	if err := os.Remove(endpoint); err != nil && !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("remove stale endpoint %s: %w", endpoint, err)
	}
	return nil
}

func (s *Service) acceptLoop(l net.Listener) {
	for {
		conn, err := l.Accept()
		if err != nil {
			return
		}
		s.attach(conn)
	}
}

// attach registers a connection and replays the current app:ready to it. Both
// happen under s.mu, which broadcasts also hold, so a client sees app:ready
// exactly once and before any later notification.
func (s *Service) attach(conn io.ReadWriteCloser) {
	s.mu.Lock()
	if s.stopping {
		s.mu.Unlock()
		_ = conn.Close()
		return
	}
	s.nextCID++
	c := newClient(s.nextCID, conn)
	s.clients[c] = struct{}{}
	if s.readyFrame != nil {
		c.send(s.readyFrame)
	}
	n := len(s.clients)
	s.mu.Unlock()
	slog.Info("client attached", "client", c.id, "clients", n)
	go s.readClient(c)
}

// detach forgets a client and closes it, gracefully when a final frame is
// queued for it.
func (s *Service) detach(c *client, graceful bool) {
	s.mu.Lock()
	_, ok := s.clients[c]
	delete(s.clients, c)
	n := len(s.clients)
	s.mu.Unlock()
	if graceful {
		c.closeGraceful()
	} else {
		c.closeNow()
	}
	if ok {
		slog.Info("client detached", "client", c.id, "clients", n)
	}
}

func (s *Service) readClient(c *client) {
	sc := bufio.NewScanner(c.conn)
	sc.Buffer(make([]byte, 0, 64*1024), jsonrpc.WorkerFrameBytes)
	for sc.Scan() {
		if !s.handleClientFrame(c, sc.Bytes()) {
			return
		}
	}
	if err := sc.Err(); err != nil && !c.isClosing() {
		slog.Warn("client stream broken; detaching it", "client", c.id, "err", err)
	}
	s.detach(c, false)
}

// handleClientFrame dispatches one frame from a client. It returns false once
// the client has been detached.
func (s *Service) handleClientFrame(c *client, frame []byte) bool {
	var msg jsonrpc.Message
	if err := json.Unmarshal(frame, &msg); err != nil || msg.JSONRPC != "2.0" {
		slog.Debug("ignoring malformed client frame", "client", c.id)
		return true
	}
	switch {
	case msg.IsRequest():
		return s.handleClientRequest(c, &msg)
	case msg.IsNotification():
		switch {
		case msg.Method == methodShutdown:
			// Never forwarded in any form: it would stop the broker every
			// other client is using.
			s.detach(c, true)
			return false
		case strings.HasSuffix(msg.Method, unsubscribeSuffix), strings.HasPrefix(msg.Method, servicePrefix):
			return true
		}
		s.forwardNotification(frame)
	}
	// A response from a client answers nothing: the broker sends no requests
	// upward through the service.
	return true
}

func (s *Service) handleClientRequest(c *client, msg *jsonrpc.Message) bool {
	switch {
	case msg.Method == methodStatus:
		c.respond(msg.ID, s.status())
	case msg.Method == methodStop:
		go s.handleStop(c, msg.ID)
	case msg.Method == methodShutdown:
		// A client's shutdown means "I am leaving", not "stop the broker".
		c.respond(msg.ID, nil)
		s.detach(c, true)
		return false
	case strings.HasSuffix(msg.Method, unsubscribeSuffix):
		// Streams are shared: another client may still depend on this one, so
		// the broker keeps sending it and this client simply ignores it.
		c.respond(msg.ID, map[string]bool{"subscribed": false})
	case strings.HasPrefix(msg.Method, servicePrefix):
		c.respondError(msg.ID, codeMethodNotFound, "method not found: "+msg.Method)
	default:
		s.forwardRequest(c, msg)
	}
	return true
}

// statusResult is the service/status reply.
type statusResult struct {
	PID            int    `json:"pid"`
	BrokerPID      int    `json:"brokerPid"`
	Clients        int    `json:"clients"`
	StartedAt      string `json:"startedAt"`
	Version        string `json:"version"`
	BrokerRestarts int    `json:"brokerRestarts"`
}

func (s *Service) status() statusResult {
	s.mu.Lock()
	defer s.mu.Unlock()
	r := statusResult{
		PID:            os.Getpid(),
		Clients:        len(s.clients),
		StartedAt:      s.startedAt.UTC().Format(time.RFC3339),
		Version:        s.cfg.version,
		BrokerRestarts: s.restarts,
	}
	if s.broker != nil {
		r.BrokerPID = s.broker.proc.pid
	}
	return r
}

func (s *Service) handleStop(c *client, id *json.RawMessage) {
	slog.Info("stop requested by client", "client", c.id)
	s.stop()
	c.respond(id, map[string]bool{"stopped": true})
	s.exitOnce.Do(func() { close(s.exitCh) })
}

// stop shuts the broker down cleanly and waits for supervision to end. Safe to
// call more than once and from several goroutines; every call returns once the
// broker is gone.
func (s *Service) stop() {
	s.stopOnce.Do(func() {
		s.mu.Lock()
		s.stopping = true
		run := s.broker
		s.mu.Unlock()
		close(s.stopCh)
		if run != nil {
			s.shutdownBroker(run)
		}
		<-s.brokerDone
		close(s.stopped)
	})
	<-s.stopped
}

func (s *Service) isStopping() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.stopping
}

// forwardRequest sends a client's request to the broker under a service-unique
// id and remembers how to route the answer back.
func (s *Service) forwardRequest(c *client, msg *jsonrpc.Message) {
	s.mu.Lock()
	run := s.broker
	switch {
	case s.stopping:
		s.mu.Unlock()
		c.respondError(msg.ID, codeUnavailable, "service is stopping")
		return
	case run == nil:
		s.mu.Unlock()
		c.respondError(msg.ID, codeUnavailable, "broker is not running")
		return
	}
	s.nextID++
	id := s.nextID
	s.pending[id] = &pendingRequest{client: c, origID: append(json.RawMessage(nil), *msg.ID...), method: msg.Method, run: run}
	s.mu.Unlock()

	out := *msg
	out.ID = rawID(id)
	if err := run.writeMessage(&out); err != nil {
		s.mu.Lock()
		delete(s.pending, id)
		s.mu.Unlock()
		c.respondError(msg.ID, codeUnavailable, "broker is not accepting requests: "+err.Error())
	}
}

func (s *Service) forwardNotification(frame []byte) {
	s.mu.Lock()
	run := s.broker
	s.mu.Unlock()
	if run == nil {
		return
	}
	_ = run.writeFrame(append(append([]byte(nil), frame...), '\n'))
}

// callBroker sends a request of the service's own to run. The returned channel
// receives the response, or nil if the broker exits first.
func (s *Service) callBroker(run *brokerRun, method string) <-chan *jsonrpc.Message {
	ch := make(chan *jsonrpc.Message, 1)
	s.mu.Lock()
	s.nextID++
	id := s.nextID
	s.pending[id] = &pendingRequest{internal: func(m *jsonrpc.Message) { ch <- m }, method: method, run: run}
	s.mu.Unlock()
	if err := run.writeMessage(&jsonrpc.Message{JSONRPC: "2.0", ID: rawID(id), Method: method}); err != nil {
		s.mu.Lock()
		delete(s.pending, id)
		s.mu.Unlock()
		ch <- nil
	}
	return ch
}

// routeResponse delivers a broker response to whoever asked.
func (s *Service) routeResponse(msg *jsonrpc.Message) {
	id, err := strconv.ParseInt(string(*msg.ID), 10, 64)
	if err != nil {
		slog.Debug("broker response with a foreign id", "id", string(*msg.ID))
		return
	}
	s.mu.Lock()
	p := s.pending[id]
	delete(s.pending, id)
	if p == nil {
		s.mu.Unlock()
		return
	}
	var replay [][]byte
	if strings.HasSuffix(p.method, subscribeSuffix) && msg.Error == nil {
		// The broker pushes a baseline only on a fresh subscription. A client
		// subscribing to a stream another client already opened gets the
		// cached copy instead, after its own acknowledgement as the broker
		// would have sent it.
		if s.subscribed[p.method] && p.client != nil {
			replay = s.baselinesFor(p.method)
		}
		s.subscribed[p.method] = true
	}
	s.mu.Unlock()

	if p.internal != nil {
		p.internal(msg)
		return
	}
	out := *msg
	out.ID = &p.origID
	p.client.sendMessage(&out)
	for _, frame := range replay {
		p.client.send(frame)
	}
}

// baselinesFor returns the cached once-per-subscription frames for a subscribe
// method: discovery's full snapshot, and a "<prefix>:ready" such as an engine
// proxy's. Callers hold s.mu.
func (s *Service) baselinesFor(subscribeMethod string) [][]byte {
	prefix := strings.TrimSuffix(subscribeMethod, subscribeSuffix)
	var out [][]byte
	if prefix == "discovery" {
		if f := s.baselines[discoveryBaseline]; f != nil {
			out = append(out, f)
		}
	}
	if f := s.baselines[prefix+":ready"]; f != nil {
		out = append(out, f)
	}
	return out
}

// isBaseline reports whether a notification is one baselinesFor may replay.
func isBaseline(method string) bool {
	return method == discoveryBaseline || (method != methodAppReady && strings.HasSuffix(method, ":ready"))
}

// handleBrokerNotification caches what late clients need and broadcasts the
// frame to every client.
func (s *Service) handleBrokerNotification(run *brokerRun, method string, frame []byte) {
	var resubscribe []string
	s.mu.Lock()
	switch {
	case method == methodAppReady:
		s.readyFrame = frame
		resubscribe = s.resubscribe
		s.resubscribe = nil
	case isBaseline(method):
		s.baselines[method] = frame
	}
	for c := range s.clients {
		c.send(frame)
	}
	s.mu.Unlock()

	// A restarted broker has no subscriptions. Reopen the ones clients had,
	// so their streams resume without each client noticing the restart.
	for _, m := range resubscribe {
		go func(method string) {
			if resp := <-s.callBroker(run, method); resp == nil || resp.Error != nil {
				slog.Warn("could not restore subscription after broker restart", "method", method)
			}
		}(m)
	}
}

// broadcast sends a service notification to every client.
func (s *Service) broadcast(method string, params any) {
	raw, err := json.Marshal(params)
	if err != nil {
		return
	}
	data, err := json.Marshal(&jsonrpc.Message{JSONRPC: "2.0", Method: method, Params: raw})
	if err != nil {
		return
	}
	frame := append(data, '\n')
	s.mu.Lock()
	for c := range s.clients {
		c.send(frame)
	}
	s.mu.Unlock()
}

func rawID(id int64) *json.RawMessage {
	raw := json.RawMessage(strconv.FormatInt(id, 10))
	return &raw
}
