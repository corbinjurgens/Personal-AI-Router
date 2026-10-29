// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package clustertrust

import (
	"bytes"
	"context"
	"crypto/sha256"
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"nvpair-shared/noderec"
)

// PeerServiceRoute is supplied by the current admission and endpoint authority.
// Incarnation must include admission epochs, even when re-admission keeps the
// same certificate. Address is the original HTTP URL's host:port; Generation
// changes when the endpoint or its protocol registration changes.
type PeerServiceRoute struct {
	Incarnation string
	Address     string
	Protocol    ReverseProtocol
	Generation  uint64
}

type PeerServiceClientOptions struct {
	PeerClientOptions
	BaseConfigDir string
	ResolveRoute  func(string, noderec.ServiceKey) (PeerServiceRoute, bool)
	DirectTimeout time.Duration
	ProbeInterval time.Duration
	// ReuseConnections opts into pooling for receiver-deduplicated POSTs with
	// a nonempty, known-length body. Other request shapes keep fresh connections.
	// GetBody is withheld from net/http during dispatch, even with idempotency
	// headers, so a stale connection cannot trigger an internal mutation replay.
	ReuseConnections bool
	// Optional transport seams retain the same cancellation/lifetime contract
	// as net.Dialer.DialContext and DialReverse respectively.
	DialContext func(context.Context, string, string) (net.Conn, error)
	DialReverse func(context.Context, string, string, noderec.ServiceKey) (net.Conn, error)
}

// PeerServiceClient is opt-in. Classic PeerClientPool callers keep their pooled
// transport. This adapter defaults to a fresh HTTP/1 connection per request: net/http
// may replay a mutation on a reused connection, including a request carrying an
// Idempotency-Key even when its GetBody is nil. Fresh connections make its
// shouldRetryRequest gate refuse replay, while preserving request headers/body.
// ReuseConnections explicitly enables pooling for eligible POSTs whose receiver
// owns deduplication. No idle connection survives an observed authority change;
// active bodies drain normally.
type PeerServiceClient struct {
	mesh    *Mesh
	opts    PeerServiceClientOptions
	mu      sync.Mutex
	entries map[peerServiceKey]*peerServiceHealth
	idle    int
}

// ErrPeerSlotOccupied reports that a request bounded by WithPeerSlotWait found
// every slot for its peer and service in use until the bound elapsed. Nothing
// was dialed or sent, and the request holds no slot.
var ErrPeerSlotOccupied = errors.New("peer service slots are occupied")

type peerSlotWaitKey struct{}
type peerSlotAcquiredKey struct{}

// WithPeerSlotWait bounds how long a PeerServiceClient request made with ctx
// waits for one of the peerMaxConnsPerHost slots shared by its peer and
// service. A wait of zero or less takes a free slot or fails at once. When the
// bound elapses first, the request fails with ErrPeerSlotOccupied; when ctx
// ends first, it fails with ctx's error. Without this option a request waits
// until a slot frees or ctx ends.
func WithPeerSlotWait(ctx context.Context, wait time.Duration) context.Context {
	return context.WithValue(ctx, peerSlotWaitKey{}, wait)
}

// WithPeerSlotAcquired runs prepare synchronously after slot acquisition and
// before any connection is opened or request bytes are sent. It receives a
// private request clone and the actual acquisition duration, and may update
// headers or refuse dispatch. A refusal releases the slot and closes the body.
func WithPeerSlotAcquired(ctx context.Context, prepare func(*http.Request, time.Duration) error) context.Context {
	return context.WithValue(ctx, peerSlotAcquiredKey{}, prepare)
}

// acquirePeerSlot reserves one of slots for the caller, who must release it by
// receiving from slots.
func acquirePeerSlot(ctx context.Context, slots chan struct{}) error {
	var expired <-chan time.Time
	if wait, bounded := ctx.Value(peerSlotWaitKey{}).(time.Duration); bounded {
		select {
		case slots <- struct{}{}:
			return nil
		default:
		}
		if wait <= 0 {
			return ErrPeerSlotOccupied
		}
		timer := time.NewTimer(wait)
		defer timer.Stop()
		expired = timer.C
	}
	select {
	case slots <- struct{}{}:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	case <-expired:
		return ErrPeerSlotOccupied
	}
}

type peerServiceKey struct {
	uuid    string
	service noderec.ServiceKey
}

type peerServiceIdentity struct {
	route      PeerServiceRoute
	peer, self [32]byte
}

type peerServiceHealth struct {
	identity        peerServiceIdentity
	nextProbe       time.Time
	probing         bool
	probeGeneration uint64
	slots           chan struct{}
	idle            []*peerServiceReuse
	retired         bool
}

func NewPeerServiceClient(mesh *Mesh, opts PeerServiceClientOptions) *PeerServiceClient {
	if opts.DirectTimeout <= 0 {
		opts.DirectTimeout = time.Second
	}
	if opts.ProbeInterval <= 0 {
		opts.ProbeInterval = 30 * time.Second
	}
	if opts.DialContext == nil {
		opts.DialContext = (&net.Dialer{KeepAlive: peerKeepAlive}).DialContext
	}
	if opts.DialReverse == nil {
		opts.DialReverse = DialReverse
	}
	return &PeerServiceClient{mesh: mesh, opts: opts, entries: make(map[peerServiceKey]*peerServiceHealth)}
}

// Client binds the logical destination, without changing the caller's URL,
// Host, TLS ServerName, headers or body. Authorization is resolved on every
// RoundTrip and again before handing an established connection to HTTP.
func (p *PeerServiceClient) Client(uuid string, service noderec.ServiceKey) *http.Client {
	return &http.Client{Timeout: p.opts.Timeout, Transport: &peerServiceTransport{p, peerServiceKey{uuid, service}},
		// Redirects may replay a 307/308 mutation; return them for the caller to
		// decide, just like any other exposed response.
		CheckRedirect: func(next *http.Request, _ []*http.Request) error {
			// net/http may already have recreated the body through GetBody.
			// ErrUseLastResponse leaves this unsent body to the callback.
			if next.Body != nil {
				next.Body.Close()
			}
			return http.ErrUseLastResponse
		},
	}
}

type peerServiceTransport struct {
	owner *PeerServiceClient
	key   peerServiceKey
}

func (p *PeerServiceClient) authority(key peerServiceKey) (peerServiceIdentity, *tls.Config, error) {
	fail := func() (peerServiceIdentity, *tls.Config, error) {
		return peerServiceIdentity{}, nil, &ReverseUnavailableError{PeerUUID: key.uuid, Cause: errors.New("peer service is not currently admitted")}
	}
	if p.opts.ResolveRoute == nil || !reverseServiceAllowed(key.service) {
		return fail()
	}
	route, ok := p.opts.ResolveRoute(key.uuid, key.service)
	if !ok || route.Incarnation == "" || route.Address == "" {
		return fail()
	}
	if route.Protocol != ReverseTLS && !(route.Protocol == ReversePlain && key.service == noderec.ServiceNodeInfo) {
		return fail()
	}
	var der []byte
	if p.opts.ResolvePin != nil {
		der, ok = p.opts.ResolvePin(key.uuid)
	} else {
		der, ok = p.mesh.trustedDER(key.uuid)
	}
	id, local := p.mesh.identity()
	if !ok || !local || len(der) == 0 || len(selfLeafDER(id)) == 0 {
		return fail()
	}
	return peerServiceIdentity{route, sha256.Sum256(der), sha256.Sum256(selfLeafDER(id))}, ClientTLSConfig(id.Cert, bytes.Clone(der)), nil
}

func (p *PeerServiceClient) current(key peerServiceKey, identity peerServiceIdentity) error {
	now, _, err := p.authority(key)
	if err != nil || now != identity {
		p.mu.Lock()
		if h := p.entries[key]; h != nil && h.identity == identity {
			p.retireLocked(h)
		}
		p.mu.Unlock()
		if err != nil {
			return err
		}
		return &ReverseUnavailableError{PeerUUID: key.uuid, Cause: errors.New("peer service authority changed before dispatch")}
	}
	return nil
}

func (p *PeerServiceClient) health(key peerServiceKey, identity peerServiceIdentity) *peerServiceHealth {
	p.mu.Lock()
	defer p.mu.Unlock()
	h := p.entries[key]
	if h == nil {
		h = &peerServiceHealth{identity: identity, slots: make(chan struct{}, peerMaxConnsPerHost)}
		p.entries[key] = h
	} else if h.identity != identity || h.retired {
		// Health belongs to this identity, but active responses from older
		// identities still count against the logical peer/service limit.
		p.retireLocked(h)
		h = &peerServiceHealth{identity: identity, slots: h.slots}
		p.entries[key] = h
	}
	return h
}

func (p *PeerServiceClient) direct(h *peerServiceHealth) (bool, uint64) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if h.nextProbe.IsZero() {
		return true, 0
	}
	if h.probing || time.Now().Before(h.nextProbe) {
		return false, 0
	}
	h.probing = true
	h.probeGeneration++
	return true, h.probeGeneration
}

func (p *PeerServiceClient) releaseProbe(h *peerServiceHealth, generation uint64) {
	p.mu.Lock()
	defer p.mu.Unlock()
	// Setup may already have completed this probe and another caller may have
	// reserved its successor. Cancellation releases only this request's probe.
	if h.probeGeneration == generation {
		h.probing = false
	}
}

func (p *PeerServiceClient) outcome(h *peerServiceHealth, healthy bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	h.probing = false
	if healthy {
		h.nextProbe = time.Time{}
	} else {
		h.nextProbe = time.Now().Add(p.opts.ProbeInterval)
	}
}

func (p *PeerServiceClient) budget(ctx context.Context, plain bool) time.Duration {
	d := p.opts.DirectTimeout
	if deadline, ok := ctx.Deadline(); ok {
		remaining := time.Until(deadline) / 2
		// Plain GET also bounds waiting for headers after its TCP dial.
		if plain {
			remaining /= 2
		}
		if remaining < d {
			d = remaining
		}
	}
	if d <= 0 {
		return time.Nanosecond
	}
	return d
}

func (t *peerServiceTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	handedToHTTP := false
	defer func() {
		if !handedToHTTP && req.Body != nil {
			req.Body.Close()
		}
	}()
	p, key := t.owner, t.key
	if err := reverseDialContextError(req.Context()); err != nil {
		return nil, err
	}
	identity, cfg, err := p.authority(key)
	if err != nil {
		p.mu.Lock()
		if h := p.entries[key]; h != nil {
			p.retireLocked(h)
		}
		p.mu.Unlock()
		return nil, err
	}
	h := p.health(key, identity)
	plain := identity.route.Protocol == ReversePlain
	scheme := "https"
	if plain {
		scheme = "http"
	}
	if req.URL.Scheme != scheme || req.URL.Host != identity.route.Address {
		return nil, &ReverseUnavailableError{PeerUUID: key.uuid, Cause: errors.New("URL does not match current peer service endpoint")}
	}
	started := time.Now()
	if err := acquirePeerSlot(req.Context(), h.slots); err != nil {
		return nil, err
	}
	waited := time.Since(started)
	release := func() { <-h.slots }
	if err := reverseDialContextError(req.Context()); err != nil {
		release()
		return nil, err
	}
	if prepare, ok := req.Context().Value(peerSlotAcquiredKey{}).(func(*http.Request, time.Duration) error); ok {
		req = req.Clone(req.Context())
		if err := prepare(req, waited); err != nil {
			release()
			return nil, err
		}
		if err := reverseDialContextError(req.Context()); err != nil {
			release()
			return nil, err
		}
	}
	if err := p.current(key, identity); err != nil {
		release()
		return nil, err
	}
	direct, probe := p.direct(h)
	if probe != 0 {
		defer p.releaseProbe(h, probe)
	}
	budget := p.budget(req.Context(), plain)
	if p.opts.ReuseConnections && req.Method == http.MethodPost && req.Body != nil && req.Body != http.NoBody && req.ContentLength > 0 {
		response, err := p.reusableRoundTrip(req, key, identity, cfg, h, direct, budget)
		handedToHTTP = true // reusableRoundTrip owns the request body, including setup failures.
		return response, err
	}
	p.mu.Lock()
	p.trimIdleLocked(h)
	p.mu.Unlock()
	var conn net.Conn
	var usedDirect atomic.Bool
	var replayed bool
	var tlsState atomic.Pointer[tls.ConnectionState]
	if plain {
		var connectedDirect bool
		conn, connectedDirect, err = p.connect(req, req.Context(), key, identity, cfg, h, direct, true, budget)
		usedDirect.Store(connectedDirect)
		if err != nil {
			release()
			return nil, err
		}
	}
	newTransport := func(conn net.Conn, directConnection bool) *http.Transport {
		tr := &http.Transport{DisableKeepAlives: true, TLSClientConfig: cfg,
			TLSNextProto:          map[string]func(string, *tls.Conn) http.RoundTripper{},
			ResponseHeaderTimeout: p.opts.ResponseHeaderTimeout,
		}
		if plain && directConnection && (tr.ResponseHeaderTimeout <= 0 || tr.ResponseHeaderTimeout > budget) {
			tr.ResponseHeaderTimeout = budget
		}
		dial := func(context.Context, string, string) (net.Conn, error) {
			if err := reverseDialContextError(req.Context()); err != nil {
				return nil, err
			}
			if err := p.current(key, identity); err != nil {
				return nil, err
			}
			return p.applicationConn(req, key, identity, conn), nil
		}
		if plain {
			tr.DialContext = dial
		} else {
			tr.DialTLSContext = func(context.Context, string, string) (net.Conn, error) {
				// DialTLSContext bypasses Transport's TLS configuration and
				// handshake timeout. Complete both explicitly in this hook.
				secure, connectedDirect, err := p.connect(req, req.Context(), key, identity, cfg, h, direct, true, budget)
				usedDirect.Store(connectedDirect)
				if err != nil {
					return nil, err
				}
				guarded := p.applicationConn(req, key, identity, secure)
				state := guarded.ConnectionState()
				tlsState.Store(&state)
				return guarded, nil
			}
		}
		return tr
	}
	tr := newTransport(conn, usedDirect.Load())
	handedToHTTP = true
	response, err := tr.RoundTrip(req)
	if err != nil && usedDirect.Load() {
		if req.Context().Err() == nil {
			p.outcome(h, false)
		}
		// This is the only HTTP replay: a bodyless node-info GET failed
		// before any response was exposed. Mutations and TLS calls never replay.
		if plain && req.Method == http.MethodGet && (req.Body == nil || req.Body == http.NoBody) && req.Context().Err() == nil {
			directErr := err
			replayed = true
			tr.CloseIdleConnections()
			conn.Close()
			var connectedDirect bool
			conn, connectedDirect, err = p.connect(req, req.Context(), key, identity, cfg, h, false, false, budget)
			usedDirect.Store(connectedDirect)
			if err == nil {
				tr = newTransport(conn, false)
				response, err = tr.RoundTrip(req)
			}
			if err != nil {
				err = fmt.Errorf("direct: %w; reverse: %w", directErr, routeCause(err))
			}
		}
	}
	if err != nil {
		if conn != nil {
			conn.Close()
		}
		tr.CloseIdleConnections()
		release()
		if req.Context().Err() != nil {
			return nil, req.Context().Err()
		}
		return nil, unavailableOver(key.uuid, routeOf(usedDirect.Load(), replayed), err)
	}
	if usedDirect.Load() {
		p.outcome(h, true)
	}
	// Preserve TLS response metadata on Go versions whose Transport only
	// recognizes a concrete *tls.Conn from DialTLSContext.
	if !plain && response.TLS == nil {
		response.TLS = tlsState.Load()
	}
	response.Body = &peerServiceBody{ReadCloser: response.Body, ctx: req.Context(), finish: func() { tr.CloseIdleConnections(); release() },
		failed: func() {},
	}
	return response, nil
}

// Each pooled transport owns exactly one established connection. It cannot
// redial behind the caller's back; the next caller owns any replacement setup.
type peerServiceReuse struct {
	conn   *peerServiceAuthorizedConn
	tr     *http.Transport
	direct bool
	timer  *time.Timer
}

func (c *peerServiceReuse) close() {
	c.tr.CloseIdleConnections()
	c.conn.Close()
}

func (p *PeerServiceClient) removeIdleLocked(h *peerServiceHealth, index int) *peerServiceReuse {
	c := h.idle[index]
	h.idle = append(h.idle[:index], h.idle[index+1:]...)
	p.idle--
	c.timer.Stop()
	return c
}

func (p *PeerServiceClient) retireLocked(h *peerServiceHealth) {
	h.retired = true
	for len(h.idle) != 0 {
		p.removeIdleLocked(h, len(h.idle)-1).close()
	}
}

func (p *PeerServiceClient) trimIdleLocked(h *peerServiceHealth) {
	for len(h.idle)+len(h.slots) > peerMaxConnsPerHost {
		p.removeIdleLocked(h, len(h.idle)-1).close()
	}
}

// CloseIdle retires pooled idle connections without interrupting active bodies.
func (p *PeerServiceClient) CloseIdle() {
	p.mu.Lock()
	defer p.mu.Unlock()
	for _, h := range p.entries {
		p.retireLocked(h)
	}
}

// DropUnauthorized retires pooled idle connections of every peer service whose
// current authority no longer matches the one it was built under: the route is
// gone, or the pin, local certificate, or route changed. Services still
// authorized keep their idle connections and health. Like DropUnpinned, it
// resolves authority with p.mu NOT held, because the resolvers take other locks.
func (p *PeerServiceClient) DropUnauthorized() {
	p.mu.Lock()
	pooled := make(map[peerServiceKey]peerServiceIdentity, len(p.entries))
	for key, h := range p.entries {
		pooled[key] = h.identity
	}
	p.mu.Unlock()
	for key, identity := range pooled {
		_ = p.current(key, identity) // retires the entry when authority moved
	}
}

func (p *PeerServiceClient) reusableRoundTrip(req *http.Request, key peerServiceKey, identity peerServiceIdentity, cfg *tls.Config, h *peerServiceHealth, direct bool, budget time.Duration) (*http.Response, error) {
	release := func() { <-h.slots }
	p.mu.Lock()
	var c *peerServiceReuse
	for i := len(h.idle) - 1; i >= 0; i-- {
		if h.idle[i].direct == direct {
			c = p.removeIdleLocked(h, i)
			break
		}
	}
	// Idle connections plus reserved active requests share the logical bound,
	// including while direct probes coexist with reverse responses.
	p.trimIdleLocked(h)
	p.mu.Unlock()
	if c == nil {
		// A successful reverse stream must outlive this request's deadline. Its
		// private lifetime ends when the transport closes, not when the caller
		// finishes a response. Setup still observes the caller's cancellation.
		lifetime, cancel := context.WithCancel(context.WithoutCancel(req.Context()))
		stop := context.AfterFunc(req.Context(), cancel)
		conn, usedDirect, err := p.connect(req.WithContext(lifetime), req.Context(), key, identity, cfg, h, direct, true, budget)
		stop()
		if err != nil || req.Context().Err() != nil {
			cancel()
			if conn != nil {
				conn.Close()
			}
			req.Body.Close()
			release()
			if req.Context().Err() != nil {
				return nil, req.Context().Err()
			}
			return nil, err
		}
		guarded := &peerServiceAuthorizedConn{Conn: conn, authorize: func() error { return p.current(key, identity) }, cancel: cancel}
		tr := &http.Transport{
			TLSClientConfig: cfg, TLSNextProto: map[string]func(string, *tls.Conn) http.RoundTripper{},
			ResponseHeaderTimeout: p.opts.ResponseHeaderTimeout,
			MaxConnsPerHost:       1, MaxIdleConns: 1, MaxIdleConnsPerHost: 1,
		}
		var handedOff atomic.Bool
		dial := func(context.Context, string, string) (net.Conn, error) {
			if !handedOff.CompareAndSwap(false, true) {
				return nil, errors.New("pooled peer service connection is no longer available")
			}
			if err := p.current(key, identity); err != nil {
				return nil, err
			}
			return guarded, nil
		}
		if identity.route.Protocol == ReversePlain {
			tr.DialContext = dial
			if usedDirect && (tr.ResponseHeaderTimeout <= 0 || tr.ResponseHeaderTimeout > budget) {
				tr.ResponseHeaderTimeout = budget
			}
		} else {
			tr.DialTLSContext = dial
		}
		c = &peerServiceReuse{conn: guarded, tr: tr, direct: usedDirect}
	}
	// A positive outgoing length and nil GetBody jointly disable both
	// shouldRetryRequest paths: nothingWrittenError and isReplayable. Preserve
	// the original GetBody for http.Client's redirect handling and body ownership.
	dispatch := req.Clone(req.Context())
	dispatch.GetBody = nil
	response, err := c.tr.RoundTrip(dispatch)
	if err != nil {
		if c.direct && req.Context().Err() == nil {
			p.outcome(h, false)
		}
		c.close()
		release()
		if req.Context().Err() != nil {
			return nil, req.Context().Err()
		}
		return nil, unavailableOver(key.uuid, routeOf(c.direct, false), err)
	}
	if c.direct {
		p.outcome(h, true)
	}
	if identity.route.Protocol == ReverseTLS && response.TLS == nil {
		state := c.conn.ConnectionState()
		response.TLS = &state
	}
	body := &peerServiceBody{ReadCloser: response.Body, ctx: req.Context(), failed: func() {}}
	body.finish = func() {
		current := p.current(key, identity) == nil
		p.mu.Lock()
		defer p.mu.Unlock()
		release()
		if body.complete.Load() && !response.Close && current && req.Context().Err() == nil && p.entries[key] == h && !h.retired && p.idle < peerMaxIdleConns && len(h.idle)+len(h.slots) < peerMaxConnsPerHost {
			h.idle = append(h.idle, c)
			p.idle++
			var timer *time.Timer
			timer = time.AfterFunc(PeerIdleTimeout, func() {
				p.mu.Lock()
				defer p.mu.Unlock()
				for i, idle := range h.idle {
					if idle == c && c.timer == timer {
						p.removeIdleLocked(h, i).close()
						break
					}
				}
			})
			c.timer = timer
		} else {
			c.close()
		}
	}
	response.Body = body
	return response, nil
}

// Every application write on a reusable connection checks its frozen authority,
// including the first write of each subsequent request. HTTP/1.1 serializes
// writes; request cancellation belongs to Transport, which closes only this conn.
type peerServiceAuthorizedConn struct {
	net.Conn
	authorize func() error
	cancel    context.CancelFunc
}

func (c *peerServiceAuthorizedConn) Write(data []byte) (int, error) {
	if err := c.authorize(); err != nil {
		c.Close()
		return 0, err
	}
	return c.Conn.Write(data)
}

func (c *peerServiceAuthorizedConn) Close() error {
	c.cancel()
	return c.Conn.Close()
}

func (c *peerServiceAuthorizedConn) ConnectionState() tls.ConnectionState {
	if secure, ok := c.Conn.(interface{ ConnectionState() tls.ConnectionState }); ok {
		return secure.ConnectionState()
	}
	return tls.ConnectionState{}
}

// Setup completes before HTTP owns any request bytes. In particular, a TCP
// connection that resets or stalls on its first TLS data is a direct failure.
// setupContext remains the caller's context when a reusable connection has a
// detached lifetime, so cancellation cannot race a direct-to-reverse fallback.
func (p *PeerServiceClient) connect(req *http.Request, setupContext context.Context, key peerServiceKey, identity peerServiceIdentity, cfg *tls.Config, h *peerServiceHealth, direct, retryDirect bool, budget time.Duration) (net.Conn, bool, error) {
	if err := reverseDialContextError(setupContext); err != nil {
		return nil, false, err
	}
	finish := func(conn net.Conn, setup context.Context) (net.Conn, error) {
		if identity.route.Protocol == ReverseTLS {
			config := cfg.Clone()
			config.ServerName = req.URL.Hostname()
			secure := tls.Client(conn, config)
			if err := secure.HandshakeContext(setup); err != nil {
				conn.Close()
				return nil, err
			}
			conn = secure
		}
		if err := reverseDialContextError(setup); err != nil {
			conn.Close()
			return nil, err
		}
		if err := p.current(key, identity); err != nil {
			conn.Close()
			return nil, err
		}
		return conn, nil
	}
	tryDirect := func() (net.Conn, error) {
		setup, cancel := context.WithTimeout(setupContext, budget)
		conn, err := p.opts.DialContext(setup, "tcp", identity.route.Address)
		if err == nil {
			conn, err = finish(conn, setup)
		}
		cancel()
		if err == nil {
			p.outcome(h, true)
			return conn, nil
		}
		if setupContext.Err() == nil {
			p.outcome(h, false)
		} else {
			return nil, setupContext.Err()
		}
		return nil, err
	}
	var directErr error
	if direct {
		conn, err := tryDirect()
		if err == nil {
			return conn, true, nil
		}
		directErr = err
		if setupContext.Err() != nil {
			return nil, false, setupContext.Err()
		}
	}
	if err := p.current(key, identity); err != nil {
		return nil, false, err
	}
	if err := reverseDialContextError(setupContext); err != nil {
		return nil, false, err
	}
	// The private context owns the full lifetime; only inner TLS uses this
	// separate setup timer, so its successful stream survives setup cancellation.
	conn, err := p.opts.DialReverse(req.Context(), p.opts.BaseConfigDir, key.uuid, key.service)
	if err == nil {
		setup, cancel := context.WithTimeout(setupContext, 5*time.Second)
		conn, err = finish(conn, setup)
		cancel()
	}
	if err != nil {
		// A refusal is authoritative at dial time, including when a session
		// disappeared after backoff was selected. No request was dispatched.
		if !direct && retryDirect && errors.Is(err, ErrReverseNoSession) && reverseDialContextError(setupContext) == nil {
			if currentErr := p.current(key, identity); currentErr != nil {
				return nil, false, currentErr
			}
			conn, directErr := tryDirect()
			if directErr == nil {
				return conn, true, nil
			}
			if setupContext.Err() != nil {
				return nil, false, setupContext.Err()
			}
			return nil, false, &ReverseUnavailableError{PeerUUID: key.uuid, Route: ReverseRoute("reverse, then direct"),
				Cause: fmt.Errorf("reverse: %w; direct: %w", routeCause(err), directErr)}
		}
		if directErr == nil {
			return nil, false, unavailableOver(key.uuid, RouteReverse, err)
		}
		return nil, false, &ReverseUnavailableError{PeerUUID: key.uuid, Route: RouteDirectThenReverse,
			Cause: fmt.Errorf("direct: %w; reverse: %w", directErr, routeCause(err))}
	}
	return conn, false, nil
}

// routeOf names the transport a connection used; fellBack means a direct
// attempt failed first and the reverse session was tried after it.
func routeOf(direct, fellBack bool) ReverseRoute {
	switch {
	case fellBack:
		return RouteDirectThenReverse
	case direct:
		return RouteDirect
	}
	return RouteReverse
}

// unavailableOver wraps err once. A failure that already names its route, such
// as one returned by connect, passes through unchanged.
func unavailableOver(peerUUID string, route ReverseRoute, err error) error {
	if unavailable, ok := err.(*ReverseUnavailableError); ok {
		return unavailable
	}
	return &ReverseUnavailableError{PeerUUID: peerUUID, Route: route, Cause: err}
}

// routeCause strips a lower layer's ReverseUnavailableError wrapper, so the
// failure is wrapped once at the layer that knows the whole route.
func routeCause(err error) error {
	if unavailable, ok := err.(*ReverseUnavailableError); ok {
		return unavailable.Cause
	}
	return err
}

// TLS setup writes are complete before this wrapper is installed. Its first
// application Write is the request-dispatch authorization boundary, separate
// from the earlier connection-handoff check.
func (p *PeerServiceClient) applicationConn(req *http.Request, key peerServiceKey, identity peerServiceIdentity, conn net.Conn) *peerServiceConn {
	return &peerServiceConn{Conn: conn, authorize: func() error {
		if err := reverseDialContextError(req.Context()); err != nil {
			return err
		}
		return p.current(key, identity)
	}}
}

type peerServiceConn struct {
	net.Conn
	once      sync.Once
	authorize func() error
	err       error
}

func (c *peerServiceConn) Write(data []byte) (int, error) {
	c.once.Do(func() {
		c.err = c.authorize()
		if c.err != nil {
			c.Conn.Close()
		}
	})
	if c.err != nil {
		return 0, c.err
	}
	return c.Conn.Write(data)
}

func (c *peerServiceConn) ConnectionState() tls.ConnectionState {
	if secure, ok := c.Conn.(interface{ ConnectionState() tls.ConnectionState }); ok {
		return secure.ConnectionState()
	}
	return tls.ConnectionState{}
}

type peerServiceBody struct {
	io.ReadCloser
	ctx      context.Context
	once     sync.Once
	finish   func()
	failed   func()
	complete atomic.Bool
}

func (b *peerServiceBody) Read(data []byte) (int, error) {
	n, err := b.ReadCloser.Read(data)
	if err != nil {
		if contextErr := reverseDialContextError(b.ctx); contextErr != nil {
			err = contextErr
		}
		if err != io.EOF {
			b.failed()
		} else {
			b.complete.Store(true)
		}
		b.once.Do(b.finish)
	}
	return n, err
}

func (b *peerServiceBody) Close() error {
	err := b.ReadCloser.Close()
	b.once.Do(b.finish)
	return err
}
