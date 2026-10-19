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
	"net"
	"net/http"
	"path/filepath"
	"reflect"
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
	wg           sync.WaitGroup
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
	m.reverse.manager.Register(peer, &reverseBufferedConn{conn, buffered.Reader}, uuid, r.Header.Get(clustertrust.ReverseSessionHeader))
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
	for _, addr := range m.resolvePeerAddrs(node) {
		setup, cancel := context.WithTimeout(ctx, reverseSetupTimeout)
		conn, err := (&net.Dialer{}).DialContext(setup, "tcp", addr)
		if err != nil {
			cancel()
			continue
		}
		stop := context.AfterFunc(setup, func() { conn.Close() })
		deadline, _ := setup.Deadline()
		conn.SetDeadline(deadline)
		tlsConn := tls.Client(conn, clustertrust.ClientTLSConfig(m.identity.Cert, der))
		err = tlsConn.HandshakeContext(setup)
		if err == nil {
			_, err = fmt.Fprintf(tlsConn, "GET %s HTTP/1.1\r\nHost: %s\r\nConnection: Upgrade\r\nUpgrade: %s\r\n%s: %s\r\n\r\n", clustertrust.ReverseUpgradePath, addr, clustertrust.ReverseUpgradeProtocol, clustertrust.ReverseSessionHeader, hex.EncodeToString(id[:]))
		}
		reader := bufio.NewReader(io.LimitReader(tlsConn, 8192))
		// Header parsing has a hard byte bound; the preserved post-header reader
		// switches back to the connection without limiting application traffic.
		var response *http.Response
		if err == nil {
			response, err = http.ReadResponse(reader, &http.Request{Method: http.MethodGet})
		}
		if err == nil && (response.StatusCode != http.StatusSwitchingProtocols || !clustertrust.IsReverseUpgrade(response.Header)) {
			err = errors.New("reverse upgrade rejected")
		}
		stopped := stop()
		expired := setup.Err()
		cancel()
		if err == nil && stopped && expired == nil {
			remaining := make([]byte, reader.Buffered())
			_, err = io.ReadFull(reader, remaining)
			if err == nil {
				return m.reverse.manager.Register(peer, &reverseBufferedConn{tlsConn, io.MultiReader(bytes.NewReader(remaining), tlsConn)}, m.identity.NodeUUID, hex.EncodeToString(id[:]))
			}
		}
		tlsConn.Close()
	}
	return errors.New("no direct reverse upgrade endpoint")
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
				m.dialReverseSession(ctx, node)
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
		m.reverse.mu.Unlock()
		conn.Write([]byte{1})
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
		conn.Write([]byte{1})
		return
	}
	stream, err := m.reverse.manager.Open(ctx, peer, request.Service)
	if err != nil {
		conn.Write([]byte{1})
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
		m.reverse.mu.Unlock()
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
