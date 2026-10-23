// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"crypto/tls"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/engines"
	"nvpair-shared/noderec"
)

const reverseSetupTimeout = 5 * time.Second
const reverseMaxPrivateConnections = 256
const reverseMaxPrivateCalls = 128

type localServicesSnapshot struct {
	Generation uint64                   `json:"generation"`
	Services   []noderec.RegisterParams `json:"services"`
}

type reverseState struct {
	manager      *clustertrust.ReverseManager
	listener     net.Listener
	mu           sync.RWMutex
	services     map[noderec.ServiceKey]noderec.RegisterParams
	generation   uint64
	initialized  bool
	calls        map[string]*reversePrivateCall
	callChanged  chan struct{}
	privateSlots chan struct{}
	forwardMu    sync.Mutex
	forwarded    map[reverseForwardKey]*reverseForwardMarker
	failures     reverseFailureLog
	refused      map[string]struct{}
	wg           sync.WaitGroup
}

// A peer that stays unreachable retries about once a second, so a repeated
// identical failure is logged once and then as a count each interval.
const (
	reverseFailureLogInterval = 30 * time.Second
	reverseFailureLogMaxKeys  = 256
)

type reverseFailureRun struct {
	text    string
	logged  time.Time
	repeats int
}

// reverseFailureLog decides which failures to log. The zero value is ready.
type reverseFailureLog struct {
	mu       sync.Mutex
	interval time.Duration    // zero means reverseFailureLogInterval
	now      func() time.Time // nil means time.Now
	runs     map[string]*reverseFailureRun
}

// note reports whether the failure for key should be logged now, and how many
// identical failures were held back since the last logged one.
func (l *reverseFailureLog) note(key, text string) (emit bool, repeats int) {
	l.mu.Lock()
	defer l.mu.Unlock()
	now := time.Now()
	if l.now != nil {
		now = l.now()
	}
	interval := l.interval
	if interval == 0 {
		interval = reverseFailureLogInterval
	}
	run := l.runs[key]
	if run == nil || run.text != text {
		if l.runs == nil || (run == nil && len(l.runs) >= reverseFailureLogMaxKeys) {
			l.runs = make(map[string]*reverseFailureRun)
		}
		l.runs[key] = &reverseFailureRun{text: text, logged: now}
		return true, 0
	}
	if now.Sub(run.logged) < interval {
		run.repeats++
		return false, 0
	}
	repeats, run.repeats, run.logged = run.repeats, 0, now
	return true, repeats
}

func (l *reverseFailureLog) clear(key string) {
	l.mu.Lock()
	delete(l.runs, key)
	l.mu.Unlock()
}

// logReverseFailure logs a failed reverse session setup with the peer's node
// ID and, when known, the remote address. key groups repeats: the peer's node
// ID, or the caller's host when the caller is not an authenticated peer.
func (m *Manager) logReverseFailure(direction, key, peer, remote string, err error) {
	emit, repeats := m.reverse.failures.note(direction+" "+key, err.Error())
	if !emit {
		return
	}
	attrs := []any{"peer", peer, "direction", direction, "err", err.Error()}
	if remote != "" {
		attrs = append(attrs, "remote", remote)
	}
	if repeats > 0 {
		attrs = append(attrs, "repeatsSinceLastLog", repeats)
	}
	slog.Warn("reverse session failed", attrs...)
}

func (m *Manager) clearReverseFailure(direction, key string) {
	m.reverse.failures.clear(direction + " " + key)
}

type reverseForwardKey struct {
	source string
	target string
}

type reverseForwardMarker struct {
	conn     net.Conn
	accepted net.Conn
	timer    *time.Timer
	closing  bool
	consumed bool
}

type reverseForwardConn struct {
	net.Conn
	state  *reverseState
	key    reverseForwardKey
	marker *reverseForwardMarker
}

func (c *reverseForwardConn) Close() error {
	c.state.forwardMu.Lock()
	defer c.state.forwardMu.Unlock()
	if c.state.forwarded[c.key] == c.marker {
		// Keep a pending dial reserved until Accept transfers its provenance
		// to the concrete accepted connection, or the setup bound expires.
		c.marker.closing = true
		return c.Conn.SetDeadline(time.Now())
	}
	return c.Conn.Close()
}

func (c *reverseForwardConn) CloseWrite() error {
	half, ok := c.Conn.(interface{ CloseWrite() error })
	if !ok {
		return errors.New("reverse forward connection does not support half-close")
	}
	return half.CloseWrite()
}

type reverseForwardContextKey struct{}

// The caller holds forwardMu across dialing and publication so Accept
// cannot inspect an accepted connection before its marker exists.
func (m *Manager) markReverseForward(conn net.Conn) net.Conn {
	key := reverseForwardKey{source: conn.LocalAddr().String(), target: conn.RemoteAddr().String()}
	marker := &reverseForwardMarker{conn: conn}
	if m.reverse.forwarded == nil {
		m.reverse.forwarded = make(map[reverseForwardKey]*reverseForwardMarker)
	}
	m.reverse.forwarded[key] = marker
	marker.timer = time.AfterFunc(reverseSetupTimeout, func() {
		m.reverse.retireReverseForward(key, marker)
	})
	return &reverseForwardConn{Conn: conn, state: m.reverse, key: key, marker: marker}
}

func (s *reverseState) retireReverseForward(key reverseForwardKey, marker *reverseForwardMarker) {
	s.forwardMu.Lock()
	defer s.forwardMu.Unlock()
	if marker.consumed {
		return
	}
	if s.forwarded[key] == marker {
		delete(s.forwarded, key)
	}
	// Close only this handoff's sockets, never a later tuple owner. An
	// unconsumed accepted connection has not begun its server TLS handshake.
	marker.conn.Close()
	if marker.accepted != nil {
		marker.accepted.Close()
	}
}

type reverseForwardListener struct {
	net.Listener
	manager *Manager
}

func (l reverseForwardListener) Accept() (net.Conn, error) {
	conn, err := l.Listener.Accept()
	if err != nil {
		return nil, err
	}
	return l.manager.acceptReverseForward(conn), nil
}

type reverseAcceptedConn struct {
	net.Conn
	state  *reverseState
	marker *reverseForwardMarker
}

// acceptReverseForward runs before split/TLS wrapping or reads. Tuple matching
// ends here; HTTP receives provenance owned by this accepted connection alone.
func (m *Manager) acceptReverseForward(conn net.Conn) net.Conn {
	if m.reverse == nil {
		return conn
	}
	m.reverse.forwardMu.Lock()
	defer m.reverse.forwardMu.Unlock()
	key := reverseForwardKey{source: conn.RemoteAddr().String(), target: conn.LocalAddr().String()}
	if marker := m.reverse.forwarded[key]; marker != nil {
		delete(m.reverse.forwarded, key)
		marker.accepted = conn
		if marker.closing {
			marker.conn.Close()
		}
		return &reverseAcceptedConn{Conn: conn, state: m.reverse, marker: marker}
	}
	return conn
}

func (m *Manager) reverseConnContext(ctx context.Context, conn net.Conn) context.Context {
	for {
		if accepted, ok := conn.(*reverseAcceptedConn); ok {
			accepted.state.forwardMu.Lock()
			defer accepted.state.forwardMu.Unlock()
			if accepted.marker.consumed {
				return ctx
			}
			accepted.marker.consumed = true
			accepted.marker.timer.Stop()
			return context.WithValue(ctx, reverseForwardContextKey{}, true)
		}
		// Both crypto/tls and splitlisten preserve the accepted connection.
		wrapped, ok := conn.(interface{ NetConn() net.Conn })
		if !ok {
			return ctx
		}
		conn = wrapped.NetConn()
	}
}

func isReverseForward(r *http.Request) bool {
	forwarded, _ := r.Context().Value(reverseForwardContextKey{}).(bool)
	return forwarded
}

type reversePrivateCall struct {
	ctx        context.Context
	cancel     context.CancelFunc
	watchReady chan struct{}
	watched    bool
}

func reverseProtocol(service noderec.ServiceKey) (clustertrust.ReverseProtocol, bool) {
	switch service {
	case noderec.ServiceNodeInfo:
		return clustertrust.ReversePlain, true
	case noderec.ServiceCluster, noderec.ServiceOllama, noderec.ServiceLMStudio,
		noderec.ServiceLlamaCPP, noderec.ServiceEngineManager, noderec.ServiceEngineControl,
		noderec.ServiceErrors, noderec.ServiceWorkload:
		return clustertrust.ReverseTLS, true
	default:
		// A descriptor-backed engine facade is pinned TLS like the built-in
		// ones. Only its syntax is fixed; resolveReverseService still requires
		// the broker's live registration.
		if engines.IsDynamicServiceKey(service) {
			return clustertrust.ReverseTLS, true
		}
		return 0, false
	}
}

func (m *Manager) initReverse() error {
	m.reverse = &reverseState{services: make(map[noderec.ServiceKey]noderec.RegisterParams), calls: make(map[string]*reversePrivateCall), callChanged: make(chan struct{}), privateSlots: make(chan struct{}, reverseMaxPrivateConnections)}
	manager, err := clustertrust.NewReverseManager(clustertrust.ReverseOptions{
		LocalUUID: m.identity.NodeUUID, Validate: func(peer clustertrust.ReversePeer) bool {
			current, ok := m.reversePeer(peer.UUID)
			return ok && current == peer
		}, ResolveService: m.resolveReverseService,
	})
	if err != nil {
		return err
	}
	m.reverse.manager = manager
	return nil
}

// No rosterMu is acquired here: callbacks can run while a composition writer
// owns it. Each mutation publishes its state before synchronous revalidation.
func (m *Manager) reversePeer(uuid string) (clustertrust.ReversePeer, bool) {
	return m.reversePeerCertificate(uuid, nil)
}

func (m *Manager) reversePeerCertificate(uuid string, authenticatedDER []byte) (clustertrust.ReversePeer, bool) {
	if m.teardownPending.Load() || uuid == m.identity.NodeUUID {
		return clustertrust.ReversePeer{}, false
	}
	cid, localEpoch := m.currentAdmission()
	id, _ := m.clusterIdentity()
	if cid == "" || id != cid || localEpoch == 0 {
		return clustertrust.ReversePeer{}, false
	}
	node, ok := m.memberByNodeID(uuid)
	if !ok || node.NodeUUID != uuid || node.State != stateMember || node.ClusterID != cid || node.AdmissionEpoch == 0 {
		return clustertrust.ReversePeer{}, false
	}
	pin, ok := m.trust.Get(uuid)
	if !ok || pin.ClusterID != cid || pin.AdmissionEpoch != node.AdmissionEpoch {
		return clustertrust.ReversePeer{}, false
	}
	der, ok := m.trust.DER(uuid)
	if !ok || (authenticatedDER != nil && !bytes.Equal(der, authenticatedDER)) {
		return clustertrust.ReversePeer{}, false
	}
	// JSON gives unambiguous field boundaries; both certs and admissions bind
	// sessions to the current local and remote incarnation.
	identity, _ := json.Marshal([]any{cid, localEpoch, node.AdmissionEpoch, sha256.Sum256(der), sha256.Sum256(m.identity.Cert.Certificate[0])})
	return clustertrust.ReversePeer{UUID: uuid, Incarnation: hex.EncodeToString(identity)}, true
}

func (m *Manager) revalidateReverse() {
	if m.reverse != nil && m.reverse.manager != nil {
		m.reverse.manager.Revalidate()
	}
}

func (m *Manager) applyLocalServices(params json.RawMessage) error {
	if !m.localServicesAllowed {
		return errors.New("local services require broker stdio")
	}
	if len(params) > maxBodyBytes {
		return errors.New("local service snapshot too large")
	}
	var wire struct {
		Generation *uint64                   `json:"generation"`
		Services   *[]noderec.RegisterParams `json:"services"`
	}
	decoder := json.NewDecoder(bytes.NewReader(params))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&wire); err != nil {
		return err
	}
	if decoder.Decode(&struct{}{}) != io.EOF {
		return errors.New("trailing local service snapshot data")
	}
	if wire.Generation == nil || wire.Services == nil {
		return errors.New("incomplete local service snapshot")
	}
	snapshot := localServicesSnapshot{Generation: *wire.Generation, Services: *wire.Services}
	m.reverse.mu.Lock()
	defer m.reverse.mu.Unlock()
	if m.reverse.initialized && snapshot.Generation < m.reverse.generation {
		return nil
	}
	services := make(map[noderec.ServiceKey]noderec.RegisterParams)
	for _, service := range snapshot.Services {
		if _, ok := reverseProtocol(service.Service); !ok || service.Port < 1 || service.Port > 65535 {
			return errors.New("invalid local service registration")
		}
		if _, ok := services[service.Service]; ok {
			return errors.New("duplicate local service registration")
		}
		service.TXT = append([]string(nil), service.TXT...)
		services[service.Service] = service
	}
	if m.reverse.initialized {
		if snapshot.Generation == m.reverse.generation {
			if !reflect.DeepEqual(services, m.reverse.services) {
				return errors.New("conflicting local service generation")
			}
			return nil
		}
	}
	m.reverse.services, m.reverse.generation, m.reverse.initialized = services, snapshot.Generation, true
	return nil
}

func (m *Manager) resolveReverseService(ctx context.Context, peer clustertrust.ReversePeer, key noderec.ServiceKey) (clustertrust.ReverseTarget, error) {
	current, ok := m.reversePeer(peer.UUID)
	if !ok || current != peer {
		return clustertrust.ReverseTarget{}, errors.New("obsolete peer")
	}
	protocol, ok := reverseProtocol(key)
	if !ok {
		return clustertrust.ReverseTarget{}, errors.New("unknown service")
	}
	m.reverse.mu.RLock()
	service, registered := m.reverse.services[key]
	generation := m.reverse.generation
	m.reverse.mu.RUnlock()
	if !registered {
		return clustertrust.ReverseTarget{}, errors.New("service not registered")
	}
	return clustertrust.ReverseTarget{Protocol: protocol, Dial: func(ctx context.Context) (net.Conn, error) {
		current, ok := m.reversePeer(peer.UUID)
		if !ok || current != peer {
			return nil, errors.New("obsolete peer")
		}
		m.reverse.mu.RLock()
		defer m.reverse.mu.RUnlock()
		if m.reverse.generation != generation || m.reverse.services[key].Port != service.Port {
			return nil, errors.New("service registration changed")
		}
		if key == noderec.ServiceCluster {
			m.reverse.forwardMu.Lock()
			defer m.reverse.forwardMu.Unlock()
		}
		conn, err := (&net.Dialer{}).DialContext(ctx, "tcp", net.JoinHostPort("127.0.0.1", fmt.Sprint(service.Port)))
		if err != nil {
			return nil, err
		}
		if key == noderec.ServiceCluster {
			return m.markReverseForward(conn), nil
		}
		return conn, nil
	}}, nil
}

type reverseBufferedConn struct {
	net.Conn
	reader io.Reader
}

func (c *reverseBufferedConn) Read(p []byte) (int, error) { return c.reader.Read(p) }

func (m *Manager) handleReverseUpgrade(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet || r.ProtoMajor != 1 || r.ProtoMinor != 1 || !clustertrust.IsReverseUpgrade(r.Header) || !clustertrust.ValidReverseSessionID(r.Header.Get(clustertrust.ReverseSessionHeader)) || r.ContentLength != 0 || len(r.TransferEncoding) != 0 {
		http.Error(w, "invalid reverse upgrade", http.StatusBadRequest)
		return
	}
	uuid, trusted := m.verifyClientPin(r)
	var authenticatedDER []byte
	if trusted {
		authenticatedDER = r.TLS.PeerCertificates[0].Raw
	}
	peer, admitted := m.reversePeerCertificate(uuid, authenticatedDER)
	if !trusted || !admitted {
		if trusted {
			m.logReverseFailure("inbound", uuid, uuid, r.RemoteAddr, errors.New("peer is not admitted to the current cluster"))
		} else {
			host, _, _ := net.SplitHostPort(r.RemoteAddr)
			m.logReverseFailure("inbound", host, "", r.RemoteAddr, errors.New("client certificate is not a pinned member"))
		}
		http.Error(w, "forbidden", http.StatusForbidden)
		return
	}
	hijacker, ok := w.(http.Hijacker)
	if !ok {
		http.Error(w, "upgrade unavailable", http.StatusInternalServerError)
		return
	}
	conn, buffered, err := hijacker.Hijack()
	if err != nil {
		return
	}
	conn.SetDeadline(time.Now().Add(reverseSetupTimeout))
	if _, err = fmt.Fprintf(buffered, "HTTP/1.1 101 Switching Protocols\r\nConnection: Upgrade\r\nUpgrade: %s\r\n\r\n", clustertrust.ReverseUpgradeProtocol); err == nil {
		err = buffered.Flush()
	}
	if err != nil {
		conn.Close()
		return
	}
	err = m.reverse.manager.Register(peer, &reverseBufferedConn{conn, buffered.Reader}, uuid, r.Header.Get(clustertrust.ReverseSessionHeader))
	switch {
	case err == nil:
		m.clearReverseFailure("inbound", uuid)
		slog.Info("reverse session established", "peer", uuid, "direction", "inbound", "remote", r.RemoteAddr)
	case errors.Is(err, clustertrust.ErrReverseDuplicate):
		m.clearReverseFailure("inbound", uuid)
	default:
		m.logReverseFailure("inbound", uuid, uuid, r.RemoteAddr, err)
	}
}

func (m *Manager) dialReverseSession(ctx context.Context, node ClusterNode) error {
	peer, ok := m.reversePeer(node.NodeUUID)
	if !ok {
		return errors.New("peer not admitted")
	}
	der, ok := m.trust.DER(peer.UUID)
	if !ok {
		return errors.New("peer not pinned")
	}
	var id [16]byte
	if _, err := rand.Read(id[:]); err != nil {
		return err
	}
	var failures []string
	for _, addr := range m.resolvePeerAddrs(node) {
		setup, cancel := context.WithTimeout(ctx, reverseSetupTimeout)
		conn, err := (&net.Dialer{}).DialContext(setup, "tcp", addr)
		if err != nil {
			cancel()
			failures = append(failures, reverseAttemptFailure(addr, "dial", err))
			continue
		}
		stop := context.AfterFunc(setup, func() { conn.Close() })
		deadline, _ := setup.Deadline()
		conn.SetDeadline(deadline)
		tlsConn := tls.Client(conn, clustertrust.ClientTLSConfig(m.identity.Cert, der))
		stage := "TLS handshake"
		err = tlsConn.HandshakeContext(setup)
		if err == nil {
			stage = "upgrade request"
			_, err = fmt.Fprintf(tlsConn, "GET %s HTTP/1.1\r\nHost: %s\r\nConnection: Upgrade\r\nUpgrade: %s\r\n%s: %s\r\n\r\n", clustertrust.ReverseUpgradePath, addr, clustertrust.ReverseUpgradeProtocol, clustertrust.ReverseSessionHeader, hex.EncodeToString(id[:]))
		}
		reader := bufio.NewReader(io.LimitReader(tlsConn, 8192))
		// Header parsing has a hard byte bound; the preserved post-header reader
		// switches back to the connection without limiting application traffic.
		var response *http.Response
		if err == nil {
			stage = "upgrade response"
			response, err = http.ReadResponse(reader, &http.Request{Method: http.MethodGet})
			if err != nil {
				err = responseReadFailure(err)
			}
		}
		if err == nil && (response.StatusCode != http.StatusSwitchingProtocols || !clustertrust.IsReverseUpgrade(response.Header)) {
			stage = "upgrade"
			err = fmt.Errorf("rejected with HTTP status %d", response.StatusCode)
		}
		stopped := stop()
		expired := setup.Err()
		cancel()
		if err == nil && stopped && expired == nil {
			remaining := make([]byte, reader.Buffered())
			stage = "handoff"
			_, err = io.ReadFull(reader, remaining)
			if err == nil {
				err = m.reverse.manager.Register(peer, &reverseBufferedConn{tlsConn, io.MultiReader(bytes.NewReader(remaining), tlsConn)}, m.identity.NodeUUID, hex.EncodeToString(id[:]))
				if err == nil {
					slog.Info("reverse session established", "peer", peer.UUID, "direction", "outbound", "addr", addr)
				}
				return err
			}
		}
		tlsConn.Close()
		if err == nil {
			stage, err = "setup", errors.New("setup budget expired")
		}
		failures = append(failures, reverseAttemptFailure(addr, stage, err))
	}
	if len(failures) == 0 {
		return errors.New("peer has no known address")
	}
	return errors.New(strings.Join(failures, "; "))
}

// responseReadFailure keeps a transport error but replaces a parser error with
// a fixed description: the parser quotes the peer's response bytes, and an
// unauthenticated listener must not choose what reaches the log.
func responseReadFailure(err error) error {
	var netErr net.Error
	switch {
	case errors.Is(err, io.EOF), errors.Is(err, io.ErrUnexpectedEOF):
		return errors.New("connection closed before a complete response")
	case errors.As(err, &netErr):
		return err
	}
	return errors.New("malformed HTTP response")
}

// reverseAttemptFailure describes one failed address attempt by address, stage
// and cause. A network error is reduced to its cause: the connection's own
// local and remote endpoints would make every retry's text differ.
func reverseAttemptFailure(addr, stage string, err error) string {
	reason := err.Error()
	var netErr *net.OpError
	if errors.As(err, &netErr) && netErr.Err != nil {
		reason = netErr.Err.Error()
	}
	return fmt.Sprintf("%s %s: %s", addr, stage, reason)
}

// attemptReverseSession dials one session and logs a failure through the
// repeat limiter. A duplicate means a session to the peer already exists.
func (m *Manager) attemptReverseSession(ctx context.Context, node ClusterNode) {
	err := m.dialReverseSession(ctx, node)
	switch {
	case err == nil || errors.Is(err, clustertrust.ErrReverseDuplicate):
		m.clearReverseFailure("outbound", node.NodeUUID)
	case ctx.Err() == nil:
		m.logReverseFailure("outbound", node.NodeUUID, node.NodeUUID, "", err)
	}
}

func (m *Manager) reverseConnectLoop(ctx context.Context) {
	defer m.reverse.wg.Done()
	// Each peer gets an independent retry worker: a blocked address cannot
	// postpone the only reachable direction for another member.
	workers := make(map[string]bool)
	slots := make(chan struct{}, 64)
	var workersMu sync.Mutex
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	for {
		for _, node := range m.snapshotNodes() {
			peer, ok := m.reversePeer(node.NodeUUID)
			if !ok || m.reverse.manager.Has(peer) {
				continue
			}
			workersMu.Lock()
			running := workers[node.NodeUUID]
			workers[node.NodeUUID] = true
			workersMu.Unlock()
			if running {
				continue
			}
			select {
			case slots <- struct{}{}:
			default:
				workersMu.Lock()
				delete(workers, node.NodeUUID)
				workersMu.Unlock()
				continue
			}
			m.reverse.wg.Add(1)
			go func(node ClusterNode) {
				defer m.reverse.wg.Done()
				defer func() { <-slots }()
				defer func() { workersMu.Lock(); delete(workers, node.NodeUUID); workersMu.Unlock() }()
				m.attemptReverseSession(ctx, node)
			}(node)
		}
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
	}
}

func (m *Manager) startReversePrivate() error {
	if m.reverse.listener != nil {
		return nil
	}
	ln, err := clustertrust.ListenReversePrivate(clustertrust.ReverseDialPath(filepath.Dir(m.clusterDir)))
	if err != nil {
		return fmt.Errorf("listen private reverse dial: %w", err)
	}
	m.reverse.listener = ln
	return nil
}

func (m *Manager) runReversePrivate(ctx context.Context) {
	defer m.reverse.wg.Done()
	stop := context.AfterFunc(ctx, func() { m.reverse.listener.Close() })
	defer stop()
	slots := m.reverse.privateSlots
	for {
		conn, err := m.reverse.listener.Accept()
		if err != nil {
			return
		}
		select {
		case slots <- struct{}{}:
		default:
			conn.Close()
			continue
		}
		m.reverse.wg.Add(1)
		go func() { defer m.reverse.wg.Done(); defer func() { <-slots }(); m.serveReversePrivate(ctx, conn) }()
	}
}

func (m *Manager) serveReversePrivate(ctx context.Context, conn net.Conn) {
	defer conn.Close()
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	stop := context.AfterFunc(ctx, func() { conn.Close() })
	defer stop()
	conn.SetDeadline(time.Now().Add(reverseSetupTimeout))
	reader := bufio.NewReaderSize(conn, clustertrust.ReverseDialMaxHeader)
	header, err := reader.ReadSlice('\n')
	if err != nil || len(header) > clustertrust.ReverseDialMaxHeader {
		return
	}
	var request clustertrust.ReverseDialRequest
	decoder := json.NewDecoder(bytes.NewReader(header))
	decoder.DisallowUnknownFields()
	if decoder.Decode(&request) != nil {
		return
	}
	if decoder.Decode(&struct{}{}) != io.EOF {
		return
	}
	if !clustertrust.ValidReverseSessionID(request.CallID) {
		return
	}
	if request.Watch {
		if request.PeerUUID != "" || request.Service != "" || request.DeadlineUnixNano != 0 {
			return
		}
		m.serveReverseWatch(ctx, conn, reader, request.CallID)
		return
	}
	if request.DeadlineUnixNano != 0 {
		var deadlineCancel context.CancelFunc
		ctx, deadlineCancel = context.WithDeadline(ctx, time.Unix(0, request.DeadlineUnixNano))
		defer deadlineCancel()
		stopDeadline := context.AfterFunc(ctx, func() { conn.Close() })
		defer stopDeadline()
		parentCancel := cancel
		cancel = func() { deadlineCancel(); parentCancel() }
	}
	m.reverse.mu.Lock()
	if _, exists := m.reverse.calls[request.CallID]; exists {
		m.reverse.mu.Unlock()
		return
	}
	if len(m.reverse.calls) >= reverseMaxPrivateCalls {
		// The call never registers, so its lifecycle connection learns the
		// refusal from this record instead of waiting out the setup budget. IDs
		// are random, so a record whose watch never arrives is harmless; the
		// table is bounded.
		if m.reverse.refused == nil || len(m.reverse.refused) >= reverseMaxPrivateConnections {
			m.reverse.refused = make(map[string]struct{})
		}
		m.reverse.refused[request.CallID] = struct{}{}
		close(m.reverse.callChanged)
		m.reverse.callChanged = make(chan struct{})
		m.reverse.mu.Unlock()
		conn.Write([]byte{byte(clustertrust.ReverseRefusedCapacity)})
		return
	}
	call := &reversePrivateCall{ctx: ctx, cancel: cancel, watchReady: make(chan struct{})}
	m.reverse.calls[request.CallID] = call
	close(m.reverse.callChanged)
	m.reverse.callChanged = make(chan struct{})
	m.reverse.mu.Unlock()
	defer func() { m.reverse.mu.Lock(); delete(m.reverse.calls, request.CallID); m.reverse.mu.Unlock() }()
	setup, setupCancel := context.WithTimeout(ctx, reverseSetupTimeout)
	defer setupCancel()
	select {
	case <-call.watchReady:
	case <-setup.Done():
		return
	}
	setupCancel()
	peer, ok := m.reversePeer(request.PeerUUID)
	if !ok {
		conn.Write([]byte{byte(clustertrust.ReverseRefusedAdmission)})
		return
	}
	stream, err := m.reverse.manager.Open(ctx, peer, request.Service)
	if err != nil {
		conn.Write([]byte{byte(clustertrust.ReverseRefusalFor(err))})
		return
	}
	defer stream.Close()
	if _, err = conn.Write([]byte{0}); err != nil {
		return
	}
	conn.SetDeadline(time.Time{})
	clustertrust.ForwardReverse(ctx, conn, stream, reader)
}

func (m *Manager) serveReverseWatch(ctx context.Context, conn net.Conn, reader io.Reader, id string) {
	setup, cancel := context.WithTimeout(ctx, reverseSetupTimeout)
	defer cancel()
	// Observe caller EOF during attachment as well as during application work.
	// One reader is bounded by the admitted connection slot and joined on exit.
	readDone := make(chan struct{})
	m.reverse.wg.Add(1)
	go func() { defer m.reverse.wg.Done(); defer close(readDone); var invalid [1]byte; reader.Read(invalid[:]) }()
	var call *reversePrivateCall
	for {
		select {
		case <-readDone:
			return
		default:
		}
		m.reverse.mu.Lock()
		call = m.reverse.calls[id]
		changed := m.reverse.callChanged
		if call != nil {
			if call.watched {
				m.reverse.mu.Unlock()
				return
			}
			call.watched = true
		}
		_, refused := m.reverse.refused[id]
		delete(m.reverse.refused, id)
		m.reverse.mu.Unlock()
		if call == nil && refused {
			conn.Write([]byte{byte(clustertrust.ReverseRefusedCapacity)})
			return
		}
		if call != nil {
			break
		}
		select {
		case <-changed:
		case <-readDone:
			return
		case <-setup.Done():
			return
		}
	}
	defer call.cancel()
	stop := context.AfterFunc(call.ctx, func() { conn.Close() })
	defer stop()
	if call.ctx.Err() != nil {
		return
	}
	select {
	case <-readDone:
		return
	default:
	}
	if _, err := conn.Write([]byte{0}); err != nil {
		return
	}
	close(call.watchReady)
	cancel()
	conn.SetDeadline(time.Time{})
	<-readDone // EOF or unexpected data both revoke this call.
}
