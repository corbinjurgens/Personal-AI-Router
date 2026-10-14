// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package clustertrust

import (
	"bytes"
	"context"
	"crypto/tls"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/hashicorp/yamux"
	"nvpair-shared/noderec"
)

const (
	ReverseUpgradePath     = "/cluster/reverse"
	ReverseUpgradeProtocol = "nvpair-reverse/1"
	ReverseSessionHeader   = "X-Nvpair-Reverse-Session"
	reverseStreamPrefix    = "PAIR-REVERSE/1 "
	reverseControlHeader   = "PAIR-REVERSE-CONTROL/1\n"
	reverseMaxHeader       = 64
	reverseMaxHello        = 64 * 1024
)

// IsReverseUpgrade checks HTTP/1.1 upgrade headers. The listener must separately
// check HTTP version, the exact current client pin, and the session identifier.
func IsReverseUpgrade(h http.Header) bool {
	if !strings.EqualFold(h.Get("Upgrade"), ReverseUpgradeProtocol) {
		return false
	}
	for _, v := range h.Values("Connection") {
		for _, token := range strings.Split(v, ",") {
			if strings.EqualFold(strings.TrimSpace(token), "upgrade") {
				return true
			}
		}
	}
	return false
}

// ReversePeer includes the owner's admission generation and both certificate
// incarnations. Incarnation is opaque: equality is never a substitute for Validate.
type ReversePeer struct{ UUID, Incarnation string }

type ReverseProtocol uint8

const (
	ReverseTLS ReverseProtocol = iota + 1
	ReversePlain
)

// ReverseTarget is resolved from current local registrations, never stream input.
// Dial must honor ctx and connect only to the named service's loopback listener.
type ReverseTarget struct {
	Protocol ReverseProtocol
	Dial     func(context.Context) (net.Conn, error)
}

type ReverseOptions struct {
	LocalUUID string
	// Callbacks must be bounded and must not call back into this manager.
	Validate       func(ReversePeer) bool
	ResolveService func(context.Context, ReversePeer, noderec.ServiceKey) (ReverseTarget, error)
	// Stream limits count application streams in both directions. Each session
	// also owns one control stream, bounded process-wide by MaxSessions.
	PerSessionStreams int           // default 16
	MaxStreams        int           // default 128
	MaxSessions       int           // default 64
	SetupTimeout      time.Duration // default 5 seconds
}

// ReverseUnavailableError reports failure before handing application bytes to a
// stream. Transport code never retries a stream or an application request.
type ReverseUnavailableError struct {
	PeerUUID string
	Cause    error
}

func (e *ReverseUnavailableError) Error() string {
	return fmt.Sprintf("reverse peer %s unavailable: %v", e.PeerUUID, e.Cause)
}
func (e *ReverseUnavailableError) Unwrap() error { return e.Cause }

type reverseSession struct {
	peer          ReversePeer
	initiator, id string
	session       *yamux.Session
	ctx           context.Context
	cancel        context.CancelFunc
	active        int
	ready         chan struct{}
	control       *yamux.Stream
	cancellations chan uint32
	mu            sync.Mutex
	forwards      map[uint32]context.CancelFunc
	pendingCancel map[uint32]bool
	lastAccepted  uint32
}

// ReverseManager owns authenticated outer connections and their streams. The
// caller publishes trust changes then calls Revalidate (or Revoke) immediately.
// One manager owns all reverse sessions in a process, sharing the stream budget.
type ReverseManager struct {
	opts     ReverseOptions
	mu       sync.Mutex
	sessions map[string]*reverseSession
	active   int
	closed   bool
	wg       sync.WaitGroup
}

func NewReverseManager(opts ReverseOptions) (*ReverseManager, error) {
	if opts.LocalUUID == "" || opts.Validate == nil || opts.ResolveService == nil {
		return nil, errors.New("reverse owner and authority callbacks required")
	}
	if opts.PerSessionStreams == 0 {
		opts.PerSessionStreams = 16
	}
	if opts.MaxStreams == 0 {
		opts.MaxStreams = 128
	}
	if opts.MaxSessions == 0 {
		opts.MaxSessions = 64
	}
	if opts.SetupTimeout == 0 {
		opts.SetupTimeout = 5 * time.Second
	}
	if opts.PerSessionStreams < 1 || opts.MaxStreams < 1 || opts.MaxSessions < 1 || opts.SetupTimeout <= 0 {
		return nil, errors.New("reverse limits must be positive")
	}
	return &ReverseManager{opts: opts, sessions: make(map[string]*reverseSession)}, nil
}

// Register takes ownership of conn, including on rejection. Both ends use the
// initiator's authenticated UUID and same random session ID from the upgrade.
// Lexicographic ordering converges duplicate connections without prohibiting
// either sole reachable dial direction. conn must already have completed pinned
// outer mTLS; a hijacker must preserve its buffered bytes in conn.Read.
func (m *ReverseManager) Register(peer ReversePeer, conn net.Conn, initiator, id string) error {
	if conn == nil {
		return errors.New("nil reverse connection")
	}
	if peer.UUID == "" || peer.UUID == m.opts.LocalUUID || peer.Incarnation == "" ||
		(initiator != peer.UUID && initiator != m.opts.LocalUUID) || !ValidReverseSessionID(id) {
		conn.Close()
		return errors.New("invalid reverse session identity")
	}
	m.mu.Lock()
	if m.closed || !m.opts.Validate(peer) {
		m.mu.Unlock()
		conn.Close()
		return &ReverseUnavailableError{peer.UUID, errors.New("obsolete peer incarnation")}
	}
	old := m.sessions[peer.UUID]
	if old == nil && len(m.sessions) >= m.opts.MaxSessions {
		m.mu.Unlock()
		conn.Close()
		return &ReverseUnavailableError{peer.UUID, errors.New("reverse session capacity")}
	}
	if old != nil && old.peer == peer && !old.session.IsClosed() &&
		(old.initiator < initiator || (old.initiator == initiator && old.id <= id)) {
		m.mu.Unlock()
		conn.Close()
		return &ReverseUnavailableError{peer.UUID, errors.New("duplicate session")}
	}
	if err := conn.SetDeadline(time.Time{}); err != nil {
		m.mu.Unlock()
		conn.Close()
		return err
	}
	cfg := yamux.DefaultConfig()
	cfg.AcceptBacklog = m.opts.PerSessionStreams + 1
	cfg.StreamOpenTimeout = m.opts.SetupTimeout
	cfg.ConnectionWriteTimeout = m.opts.SetupTimeout
	cfg.LogOutput = io.Discard
	conn = newReverseMuxConn(conn, m.opts.SetupTimeout)
	var session *yamux.Session
	var err error
	if initiator == m.opts.LocalUUID {
		session, err = yamux.Client(conn, cfg)
	} else {
		session, err = yamux.Server(conn, cfg)
	}
	if err != nil {
		m.mu.Unlock()
		conn.Close()
		return err
	}
	ctx, cancel := context.WithCancel(context.Background())
	s := &reverseSession{peer: peer, initiator: initiator, id: id, session: session, ctx: ctx, cancel: cancel,
		ready: make(chan struct{}), cancellations: make(chan uint32, m.opts.PerSessionStreams),
		forwards: make(map[uint32]context.CancelFunc), pendingCancel: make(map[uint32]bool)}
	m.sessions[peer.UUID] = s
	m.wg.Add(1)
	m.mu.Unlock()
	if old != nil {
		old.cancel()
		old.session.Close()
	}
	go m.accept(s)
	return nil
}

const (
	reverseMuxHeaderSize  = 12
	reverseMuxData        = 0
	reverseMuxPing        = 2
	reverseMuxSYN         = 1
	reverseMuxACK         = 2
	reverseMuxPingBacklog = 8
)

// reverseMuxConn bounds yamux v0.1.2's peer-ping reply path, which otherwise
// starts a new goroutine per query even when the peer stops reading. Only ping
// queries are consumed here; stream frames and ping responses remain yamux's.
// The existing yamux wire format is unchanged (v0.1.2/spec.md).
type reverseMuxConn struct {
	net.Conn
	timeout      time.Duration
	readHeader   [reverseMuxHeaderSize]byte
	readOffset   int
	readPayload  uint32
	writePayload uint32
	writeGate    chan struct{}
	replies      chan [reverseMuxHeaderSize]byte
	done         chan struct{}
	replyDone    chan struct{}
	once         sync.Once
	closeErr     error
}

func newReverseMuxConn(conn net.Conn, timeout time.Duration) *reverseMuxConn {
	c := &reverseMuxConn{Conn: conn, timeout: timeout, writeGate: make(chan struct{}, 1), replies: make(chan [reverseMuxHeaderSize]byte, reverseMuxPingBacklog), done: make(chan struct{}), replyDone: make(chan struct{})}
	c.writeGate <- struct{}{}
	go c.replyPings()
	return c
}

func (c *reverseMuxConn) shutdown() {
	c.once.Do(func() { close(c.done); c.closeErr = c.Conn.Close() })
}

func (c *reverseMuxConn) Close() error {
	c.shutdown()
	<-c.replyDone
	return c.closeErr
}

// A frame owner retains the token across header/payload calls. Waiting for a
// frame boundary must remain cancellable even if its payload never follows.
// Shutdown never releases another writer's token or inserts a partial frame.
func (c *reverseMuxConn) acquireWrite() error {
	select {
	case <-c.done:
		return net.ErrClosed
	default:
	}
	select {
	case <-c.done:
		return net.ErrClosed
	case <-c.writeGate:
	default:
		timer := time.NewTimer(c.timeout)
		defer timer.Stop()
		select {
		case <-c.done:
			return net.ErrClosed
		case <-timer.C:
			c.shutdown()
			return yamux.ErrConnectionWriteTimeout
		case <-c.writeGate:
		}
	}
	select {
	case <-c.done:
		c.releaseWrite()
		return net.ErrClosed
	default:
		return nil
	}
}

func (c *reverseMuxConn) releaseWrite() { c.writeGate <- struct{}{} }

func (c *reverseMuxConn) Read(b []byte) (int, error) {
	if len(b) == 0 {
		return 0, nil
	}
	for {
		if c.readOffset > 0 {
			n := copy(b, c.readHeader[reverseMuxHeaderSize-c.readOffset:])
			c.readOffset -= n
			return n, nil
		}
		if c.readPayload > 0 {
			if uint64(len(b)) > uint64(c.readPayload) {
				b = b[:c.readPayload]
			}
			n, err := c.Conn.Read(b)
			c.readPayload -= uint32(n)
			return n, err
		}
		if _, err := io.ReadFull(c.Conn, c.readHeader[:]); err != nil {
			return 0, err
		}
		flags := binary.BigEndian.Uint16(c.readHeader[2:4])
		if c.readHeader[0] == 0 && c.readHeader[1] == reverseMuxPing && flags&reverseMuxSYN != 0 {
			if flags != reverseMuxSYN || binary.BigEndian.Uint32(c.readHeader[4:8]) != 0 {
				c.shutdown()
				return 0, errors.New("invalid reverse yamux ping")
			}
			reply := c.readHeader
			binary.BigEndian.PutUint16(reply[2:4], reverseMuxACK)
			select {
			case c.replies <- reply:
				continue
			default:
				c.shutdown()
				return 0, errors.New("reverse yamux ping reply capacity")
			}
		}
		if c.readHeader[1] == reverseMuxData {
			c.readPayload = binary.BigEndian.Uint32(c.readHeader[8:12])
		}
		c.readOffset = reverseMuxHeaderSize
	}
}

// Yamux v0.1.2 has one sender that writes each header separately, then its
// payload. Retain the write token across those calls so a ping reply can only
// enter between complete frames. Payload bytes are never copied or buffered.
func (c *reverseMuxConn) Write(b []byte) (int, error) {
	select {
	case <-c.done:
		return 0, net.ErrClosed
	default:
	}
	if len(b) == 0 {
		return 0, nil
	}
	if c.writePayload == 0 {
		if err := c.acquireWrite(); err != nil {
			return 0, err
		}
		if len(b) != reverseMuxHeaderSize {
			c.releaseWrite()
			c.shutdown()
			return 0, errors.New("invalid reverse yamux outgoing header")
		}
		if err := c.Conn.SetWriteDeadline(time.Now().Add(c.timeout)); err != nil {
			c.releaseWrite()
			c.shutdown()
			return 0, err
		}
		if b[1] == reverseMuxData {
			c.writePayload = binary.BigEndian.Uint32(b[8:12])
		}
	} else if uint64(len(b)) > uint64(c.writePayload) {
		c.writePayload = 0
		c.releaseWrite()
		c.shutdown()
		return 0, errors.New("invalid reverse yamux outgoing payload")
	} else {
		n, err := c.writeAll(b)
		c.writePayload -= uint32(n)
		if err != nil || c.writePayload == 0 {
			c.writePayload = 0
			c.releaseWrite()
		}
		if err != nil {
			c.shutdown()
		}
		return n, err
	}
	n, err := c.writeAll(b)
	if err != nil || c.writePayload == 0 {
		c.writePayload = 0
		c.releaseWrite()
	}
	if err != nil {
		c.shutdown()
	}
	return n, err
}

func (c *reverseMuxConn) writeAll(b []byte) (int, error) {
	written := 0
	for written < len(b) {
		n, err := c.Conn.Write(b[written:])
		written += n
		if err != nil {
			return written, err
		}
		if n == 0 {
			return written, io.ErrNoProgress
		}
	}
	return written, nil
}

func (c *reverseMuxConn) replyPings() {
	defer close(c.replyDone)
	for {
		select {
		case <-c.done:
			return
		case reply := <-c.replies:
			if err := c.acquireWrite(); err != nil {
				return
			}
			err := c.Conn.SetWriteDeadline(time.Now().Add(c.timeout))
			if err == nil {
				_, err = c.writeAll(reply[:])
			}
			c.releaseWrite()
			if err != nil {
				c.shutdown()
				return
			}
		}
	}
}

// ValidReverseSessionID admits the initiator's random 128-bit lowercase hex ID.
// Upgrade handlers use the same check before hijacking the HTTP connection.
func ValidReverseSessionID(id string) bool {
	if len(id) != 32 {
		return false
	}
	for _, c := range id {
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')) {
			return false
		}
	}
	return true
}

func reverseServiceAllowed(service noderec.ServiceKey) bool {
	switch service {
	case noderec.ServiceNodeInfo, noderec.ServiceOllama, noderec.ServiceLMStudio,
		noderec.ServiceLlamaCPP, noderec.ServiceErrors, noderec.ServiceWorkload,
		noderec.ServiceCluster, noderec.ServiceEngineManager, noderec.ServiceEngineControl:
		return true
	}
	return false
}

func (m *ReverseManager) current(s *reverseSession) bool {
	return !m.closed && m.sessions[s.peer.UUID] == s && !s.session.IsClosed() && m.opts.Validate(s.peer)
}

func (m *ReverseManager) Has(peer ReversePeer) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	s := m.sessions[peer.UUID]
	if s == nil || s.peer != peer || !m.current(s) {
		return false
	}
	select {
	case <-s.ready:
		return true
	default:
		return false
	}
}

// Revalidate closes obsolete sessions, including all their active streams.
func (m *ReverseManager) Revalidate() {
	m.mu.Lock()
	var obsolete []*reverseSession
	for uuid, s := range m.sessions {
		if !m.opts.Validate(s.peer) {
			delete(m.sessions, uuid)
			obsolete = append(obsolete, s)
		}
	}
	m.mu.Unlock()
	for _, s := range obsolete {
		s.cancel()
		s.session.Close()
	}
}

func (m *ReverseManager) Revoke(uuid string) {
	m.mu.Lock()
	s := m.sessions[uuid]
	delete(m.sessions, uuid)
	m.mu.Unlock()
	if s != nil {
		s.cancel()
		s.session.Close()
	}
}

func (m *ReverseManager) Close() error {
	m.mu.Lock()
	m.closed = true
	all := m.sessions
	m.sessions = make(map[string]*reverseSession)
	m.mu.Unlock()
	for _, s := range all {
		s.cancel()
		s.session.Close()
	}
	m.wg.Wait()
	return nil
}

func (m *ReverseManager) reserve(s *reverseSession) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	if !m.current(s) || s.active >= m.opts.PerSessionStreams || m.active >= m.opts.MaxStreams {
		return false
	}
	s.active++
	m.active++
	m.wg.Add(1)
	return true
}
func (m *ReverseManager) release(s *reverseSession) {
	m.mu.Lock()
	s.active--
	m.active--
	m.mu.Unlock()
	m.wg.Done()
}

func (m *ReverseManager) accept(s *reverseSession) {
	defer m.wg.Done()
	defer func() {
		s.cancel()
		s.session.Close()
		m.mu.Lock()
		if m.sessions[s.peer.UUID] == s {
			delete(m.sessions, s.peer.UUID)
		}
		m.mu.Unlock()
	}()
	if err := m.setupControl(s); err != nil {
		return
	}
	m.wg.Add(2)
	go m.readControl(s)
	go m.writeControl(s)
	close(s.ready)
	for {
		stream, err := s.session.AcceptStream()
		if err != nil {
			return
		}
		// Yamux Close is a half-close. Closing an over-budget session bounds
		// even peers that keep sending after a rejected stream's FIN.
		if s.session.NumStreams() > m.opts.PerSessionStreams+1 || !m.reserve(s) {
			return
		}
		ctx, cancel := context.WithCancel(s.ctx)
		s.mu.Lock()
		id := stream.StreamID()
		if id > s.lastAccepted {
			s.lastAccepted = id
		}
		s.forwards[id] = cancel
		if s.pendingCancel[id] {
			delete(s.pendingCancel, id)
			cancel()
		}
		s.mu.Unlock()
		go func() {
			defer m.release(s)
			defer abortReverseStream(stream)
			defer cancel()
			defer func() { s.mu.Lock(); delete(s.forwards, id); s.mu.Unlock() }()
			m.serve(ctx, s, stream)
		}()
	}
}

// Yamux v0.1.2 exposes half-close, but no stream reset. One fixed control
// stream per session carries bounded stream-ID cancellation messages, letting
// Close terminate the remote forwarder without interrupting unrelated streams.
func (m *ReverseManager) setupControl(s *reverseSession) error {
	ctx, cancel := context.WithTimeout(s.ctx, m.opts.SetupTimeout)
	defer cancel()
	stop := context.AfterFunc(ctx, func() { s.session.Close() })
	defer stop()
	var stream *yamux.Stream
	var err error
	if s.initiator == m.opts.LocalUUID {
		stream, err = s.session.OpenStream()
	} else {
		stream, err = s.session.AcceptStreamWithContext(ctx)
	}
	if err != nil {
		return err
	}
	if stream.StreamID() != 1 {
		return errors.New("reverse control must be first stream")
	}
	deadline, _ := ctx.Deadline()
	stream.SetDeadline(deadline)
	if s.initiator == m.opts.LocalUUID {
		_, err = io.WriteString(stream, reverseControlHeader)
		if err == nil {
			var ack [1]byte
			_, err = io.ReadFull(stream, ack[:])
			if err == nil && ack[0] != 0 {
				err = errors.New("reverse control refused")
			}
		}
	} else {
		header := make([]byte, len(reverseControlHeader))
		_, err = io.ReadFull(stream, header)
		if err == nil && string(header) != reverseControlHeader {
			err = errors.New("invalid reverse control header")
		}
		if err == nil {
			_, err = stream.Write([]byte{0})
		}
	}
	if err != nil || ctx.Err() != nil {
		if err != nil {
			return err
		}
		return ctx.Err()
	}
	stream.SetDeadline(time.Time{})
	s.control = stream
	return nil
}

func (m *ReverseManager) readControl(s *reverseSession) {
	defer m.wg.Done()
	defer func() { s.cancel(); s.session.Close() }()
	var frame [4]byte
	for {
		if _, err := io.ReadFull(s.control, frame[:1]); err != nil {
			return
		}
		s.control.SetReadDeadline(time.Now().Add(m.opts.SetupTimeout))
		if _, err := io.ReadFull(s.control, frame[1:]); err != nil {
			return
		}
		s.control.SetReadDeadline(time.Time{})
		id := binary.BigEndian.Uint32(frame[:])
		// A peer can cancel only streams it initiated, never our streams or
		// the control channel. Pending early cancellations have a finite cap.
		odd := s.initiator != m.opts.LocalUUID
		if id <= 1 || (id%2 == 1) != odd {
			return
		}
		s.mu.Lock()
		cancel := s.forwards[id]
		if cancel == nil && id > s.lastAccepted {
			if len(s.pendingCancel) >= m.opts.PerSessionStreams {
				s.mu.Unlock()
				return
			}
			s.pendingCancel[id] = true
		}
		s.mu.Unlock()
		if cancel != nil {
			cancel()
		}
	}
}

func (m *ReverseManager) writeControl(s *reverseSession) {
	defer m.wg.Done()
	defer func() { s.cancel(); s.session.Close() }()
	var frame [4]byte
	for {
		select {
		case <-s.ctx.Done():
			return
		case id := <-s.cancellations:
			binary.BigEndian.PutUint32(frame[:], id)
			s.control.SetWriteDeadline(time.Now().Add(m.opts.SetupTimeout))
			if _, err := s.control.Write(frame[:]); err != nil {
				return
			}
		}
	}
}

func (m *ReverseManager) cancelStream(s *reverseSession, id uint32) {
	select {
	case <-s.ctx.Done():
		return
	default:
	}
	select {
	case s.cancellations <- id:
	default:
		s.cancel()
		s.session.Close()
	}
}

func abortReverseStream(stream *yamux.Stream) { stream.SetDeadline(time.Now()); stream.Close() }

func readReverseService(r io.Reader) (noderec.ServiceKey, error) {
	var line []byte
	var b [1]byte
	for len(line) < reverseMaxHeader {
		if _, err := io.ReadFull(r, b[:]); err != nil {
			return "", err
		}
		if b[0] == '\n' {
			text := string(line)
			if !strings.HasPrefix(text, reverseStreamPrefix) {
				break
			}
			service := noderec.ServiceKey(strings.TrimPrefix(text, reverseStreamPrefix))
			if reverseServiceAllowed(service) {
				return service, nil
			}
			break
		}
		line = append(line, b[0])
	}
	return "", errors.New("invalid reverse service header")
}

func (m *ReverseManager) serve(lifetime context.Context, s *reverseSession, stream *yamux.Stream) {
	stopStream := context.AfterFunc(lifetime, func() { abortReverseStream(stream) })
	defer stopStream()
	ctx, cancel := context.WithTimeout(lifetime, m.opts.SetupTimeout)
	defer cancel()
	stream.SetDeadline(time.Now().Add(m.opts.SetupTimeout))
	service, err := readReverseService(stream)
	if err != nil {
		return
	}
	m.mu.Lock()
	current := m.current(s)
	m.mu.Unlock()
	if !current {
		return
	}
	target, err := m.opts.ResolveService(ctx, s.peer, service)
	if err != nil || target.Dial == nil ||
		(target.Protocol != ReverseTLS && !(target.Protocol == ReversePlain && service == noderec.ServiceNodeInfo)) {
		stream.Write([]byte{1})
		return
	}
	m.mu.Lock()
	current = m.current(s)
	m.mu.Unlock()
	if !current {
		return
	}
	// Acknowledge only the service admission, before waiting for ClientHello:
	// the caller must receive its stream before starting inner TLS.
	if _, err := stream.Write([]byte{0}); err != nil {
		return
	}
	var prefix []byte
	if target.Protocol == ReverseTLS {
		prefix, err = readReverseClientHello(ctx, stream)
		if err != nil {
			return
		}
	}
	m.mu.Lock()
	current = m.current(s)
	m.mu.Unlock()
	if !current || ctx.Err() != nil {
		return
	}
	local, err := target.Dial(ctx)
	if err != nil {
		return
	}
	defer local.Close()
	// The setup budget ends here; streaming lifetime follows session/caller
	// cancellation and caller deadlines, with no whole-body timeout.
	stream.SetDeadline(time.Time{})
	stop := context.AfterFunc(lifetime, func() { local.Close(); abortReverseStream(stream) })
	defer stop()
	ForwardReverse(lifetime, stream, local, io.MultiReader(bytes.NewReader(prefix), stream))
}

var errReverseHelloParsed = errors.New("reverse ClientHello parsed")

// Parsing through crypto/tls rejects malformed TLS-looking prefixes and full
// malformed ClientHellos. The probe never writes to the peer or the local
// listener. Every consumed byte is replayed for the real inner mTLS handshake.
func readReverseClientHello(ctx context.Context, conn net.Conn) ([]byte, error) {
	probe := &reverseHelloProbe{Conn: conn, remaining: reverseMaxHello}
	parsed := false
	tlsProbe := tls.Server(probe, &tls.Config{GetConfigForClient: func(*tls.ClientHelloInfo) (*tls.Config, error) {
		parsed = true
		return nil, errReverseHelloParsed
	}})
	err := tlsProbe.HandshakeContext(ctx)
	if !parsed || !errors.Is(err, errReverseHelloParsed) {
		return nil, errors.New("reverse stream requires a valid TLS ClientHello")
	}
	return probe.bytes.Bytes(), nil
}

type reverseHelloProbe struct {
	net.Conn
	bytes     bytes.Buffer
	remaining int
}

func (p *reverseHelloProbe) Read(b []byte) (int, error) {
	if p.remaining == 0 {
		return 0, errors.New("reverse ClientHello too large")
	}
	if len(b) > p.remaining {
		b = b[:p.remaining]
	}
	n, err := p.Conn.Read(b)
	p.bytes.Write(b[:n])
	p.remaining -= n
	return n, err
}
func (p *reverseHelloProbe) Write(b []byte) (int, error) { return len(b), nil }

// ForwardReverse copies with fixed-size io.Copy buffers. EOF half-closes the
// opposite writer so a service can finish its response after request EOF. An
// error or cancellation closes both directions; no interrupted bytes are replayed.
// first may prepend bytes read for protocol validation; nil reads directly from a.
func ForwardReverse(ctx context.Context, a, b net.Conn, first io.Reader) error {
	if first == nil {
		first = a
	}
	stop := context.AfterFunc(ctx, func() { a.SetDeadline(time.Now()); b.SetDeadline(time.Now()); a.Close(); b.Close() })
	defer stop()
	results := make(chan error, 2)
	copyTo := func(dst net.Conn, src io.Reader) {
		_, err := io.Copy(dst, src)
		if err == nil {
			if half, ok := dst.(interface{ CloseWrite() error }); ok {
				err = half.CloseWrite()
			} else if stream, ok := dst.(*yamux.Stream); ok {
				err = stream.Close()
			} else {
				err = dst.Close()
			}
		}
		results <- err
	}
	go copyTo(b, first)
	go copyTo(a, b)
	err := <-results
	if err != nil {
		a.SetDeadline(time.Now())
		b.SetDeadline(time.Now())
		a.Close()
		b.Close()
	}
	second := <-results
	if ctx.Err() != nil {
		return ctx.Err()
	}
	if err != nil {
		return err
	}
	return second
}

// Open admits a named service. Its context and deadline govern the returned
// stream's lifetime, not just setup. The peer independently verifies the inner
// protocol before dialing its local listener.
func (m *ReverseManager) Open(ctx context.Context, peer ReversePeer, service noderec.ServiceKey) (net.Conn, error) {
	unavailable := func(err error) (net.Conn, error) { return nil, &ReverseUnavailableError{peer.UUID, err} }
	if !reverseServiceAllowed(service) {
		return unavailable(errors.New("unknown reverse service"))
	}
	if err := ctx.Err(); err != nil {
		return unavailable(err)
	}
	setupCtx, cancel := context.WithTimeout(ctx, m.opts.SetupTimeout)
	defer cancel()
	m.mu.Lock()
	s := m.sessions[peer.UUID]
	current := s != nil && s.peer == peer && m.current(s)
	m.mu.Unlock()
	if !current {
		return unavailable(errors.New("no current reverse session"))
	}
	select {
	case <-s.ready:
	case <-s.ctx.Done():
		return unavailable(errors.New("reverse session closed"))
	case <-setupCtx.Done():
		return unavailable(setupCtx.Err())
	}
	if s.session.NumStreams() >= m.opts.PerSessionStreams+1 || !m.reserve(s) {
		return unavailable(errors.New("no current reverse session or stream capacity"))
	}
	type result struct {
		stream *yamux.Stream
		err    error
	}
	ready := make(chan result, 1)
	go func() {
		stream, err := s.session.OpenStream()
		if err == nil {
			deadline, _ := setupCtx.Deadline()
			stream.SetDeadline(deadline)
			stop := context.AfterFunc(setupCtx, func() { abortReverseStream(stream) })
			_, err = io.WriteString(stream, reverseStreamPrefix+string(service)+"\n")
			if err == nil {
				var ack [1]byte
				_, err = io.ReadFull(stream, ack[:])
				if err == nil && ack[0] != 0 {
					err = errors.New("reverse service unavailable")
				}
			}
			stop()
		}
		if err != nil || setupCtx.Err() != nil {
			if stream != nil {
				m.cancelStream(s, stream.StreamID())
				abortReverseStream(stream)
			}
			m.release(s)
			if err == nil {
				err = setupCtx.Err()
			}
			ready <- result{err: err}
			return
		}
		ready <- result{stream: stream}
	}()
	select {
	case <-setupCtx.Done():
		// Drain a concurrently successful open and release its reservation.
		go func() {
			r := <-ready
			if r.stream != nil {
				m.cancelStream(s, r.stream.StreamID())
				abortReverseStream(r.stream)
				m.release(s)
			}
		}()
		return unavailable(setupCtx.Err())
	case r := <-ready:
		if r.err != nil {
			return unavailable(r.err)
		}
		deadline, _ := ctx.Deadline()
		r.stream.SetDeadline(deadline)
		c := &reverseConn{Stream: r.stream, release: func() { m.release(s) }, notify: func() { m.cancelStream(s, r.stream.StreamID()) }}
		c.mu.Lock()
		c.stops = []func() bool{context.AfterFunc(ctx, func() { c.Close() }), context.AfterFunc(s.ctx, func() { c.Close() })}
		c.mu.Unlock()
		return c, nil
	}
}

type reverseConn struct {
	*yamux.Stream
	once    sync.Once
	mu      sync.Mutex
	release func()
	stops   []func() bool
	notify  func()
}

func (c *reverseConn) CloseWrite() error { return c.Stream.Close() }
func (c *reverseConn) Close() error {
	c.once.Do(func() {
		c.mu.Lock()
		for _, stop := range c.stops {
			stop()
		}
		c.mu.Unlock()
		c.notify()
		abortReverseStream(c.Stream)
		c.release()
	})
	return nil
}
