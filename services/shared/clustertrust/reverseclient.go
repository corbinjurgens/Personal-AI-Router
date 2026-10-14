// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package clustertrust

import (
	"bytes"
	"context"
	"crypto/sha256"
	"crypto/tls"
	"errors"
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
	// Optional transport seams retain the same cancellation/lifetime contract
	// as net.Dialer.DialContext and DialReverse respectively.
	DialContext func(context.Context, string, string) (net.Conn, error)
	DialReverse func(context.Context, string, string, noderec.ServiceKey) (net.Conn, error)
}

// PeerServiceClient is opt-in. Classic PeerClientPool callers keep their pooled
// transport. This adapter uses a fresh HTTP/1 connection per request: net/http
// may replay a mutation on a reused connection, including a request carrying an
// Idempotency-Key even when its GetBody is nil. Fresh connections make its
// shouldRetryRequest gate refuse replay, while preserving request headers/body.
// No idle connection survives a route or authority change; active bodies drain
// normally. The cost is one pinned TLS handshake per request until a transport
// with explicit dispatch ownership can safely pool these operations.
type PeerServiceClient struct {
	mesh    *Mesh
	opts    PeerServiceClientOptions
	mu      sync.Mutex
	entries map[peerServiceKey]*peerServiceHealth
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
	identity  peerServiceIdentity
	nextProbe time.Time
	probing   bool
	slots     chan struct{}
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
	if err != nil {
		return err
	}
	if now != identity {
		return &ReverseUnavailableError{PeerUUID: key.uuid, Cause: errors.New("peer service authority changed during dial")}
	}
	return nil
}

func (p *PeerServiceClient) health(key peerServiceKey, identity peerServiceIdentity) *peerServiceHealth {
	p.mu.Lock()
	defer p.mu.Unlock()
	h := p.entries[key]
	if h == nil || h.identity != identity {
		h = &peerServiceHealth{identity: identity, slots: make(chan struct{}, peerMaxConnsPerHost)}
		p.entries[key] = h
	}
	return h
}

func (p *PeerServiceClient) direct(h *peerServiceHealth) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	if h.nextProbe.IsZero() {
		return true
	}
	if h.probing || time.Now().Before(h.nextProbe) {
		return false
	}
	h.probing = true
	return true
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
		return nil, err
	}
	plain := identity.route.Protocol == ReversePlain
	scheme := "https"
	if plain {
		scheme = "http"
	}
	if req.URL.Scheme != scheme || req.URL.Host != identity.route.Address {
		return nil, &ReverseUnavailableError{PeerUUID: key.uuid, Cause: errors.New("URL does not match current peer service endpoint")}
	}
	h := p.health(key, identity)
	select {
	case h.slots <- struct{}{}:
	case <-req.Context().Done():
		return nil, req.Context().Err()
	}
	release := func() { <-h.slots }
	if err := reverseDialContextError(req.Context()); err != nil {
		release()
		return nil, err
	}
	if err := p.current(key, identity); err != nil {
		release()
		return nil, err
	}
	direct := p.direct(h)
	budget := p.budget(req.Context(), plain)
	var conn net.Conn
	var usedDirect atomic.Bool
	var tlsState atomic.Pointer[tls.ConnectionState]
	if plain {
		var connectedDirect bool
		conn, connectedDirect, err = p.connect(req, key, identity, cfg, h, direct, budget)
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
				secure, connectedDirect, err := p.connect(req, key, identity, cfg, h, direct, budget)
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
			tr.CloseIdleConnections()
			conn.Close()
			var connectedDirect bool
			conn, connectedDirect, err = p.connect(req, key, identity, cfg, h, false, budget)
			usedDirect.Store(connectedDirect)
			if err == nil {
				tr = newTransport(conn, false)
				response, err = tr.RoundTrip(req)
			}
			if err != nil {
				err = errors.Join(directErr, err)
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
		return nil, &ReverseUnavailableError{PeerUUID: key.uuid, Cause: err}
	}
	// Preserve TLS response metadata on Go versions whose Transport only
	// recognizes a concrete *tls.Conn from DialTLSContext.
	if !plain && response.TLS == nil {
		response.TLS = tlsState.Load()
	}
	response.Body = &peerServiceBody{ReadCloser: response.Body, ctx: req.Context(), finish: func() { tr.CloseIdleConnections(); release() },
		failed: func() {
			if usedDirect.Load() && reverseDialContextError(req.Context()) == nil {
				p.outcome(h, false)
			}
		},
	}
	return response, nil
}

// Setup completes before HTTP owns any request bytes. In particular, a TCP
// connection that resets or stalls on its first TLS data is a direct failure.
func (p *PeerServiceClient) connect(req *http.Request, key peerServiceKey, identity peerServiceIdentity, cfg *tls.Config, h *peerServiceHealth, direct bool, budget time.Duration) (net.Conn, bool, error) {
	if err := reverseDialContextError(req.Context()); err != nil {
		if direct {
			p.mu.Lock()
			h.probing = false
			p.mu.Unlock()
		}
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
	var directErr error
	if direct {
		setup, cancel := context.WithTimeout(req.Context(), budget)
		conn, err := p.opts.DialContext(setup, "tcp", identity.route.Address)
		if err == nil {
			conn, err = finish(conn, setup)
		}
		cancel()
		if err == nil {
			p.outcome(h, true)
			return conn, true, nil
		}
		directErr = err
		if req.Context().Err() == nil {
			p.outcome(h, false)
		} else {
			p.mu.Lock()
			h.probing = false
			p.mu.Unlock()
			return nil, false, req.Context().Err()
		}
	}
	if err := p.current(key, identity); err != nil {
		return nil, false, err
	}
	if err := reverseDialContextError(req.Context()); err != nil {
		return nil, false, err
	}
	// The private context owns the full lifetime; only inner TLS uses this
	// separate setup timer, so its successful stream survives setup cancellation.
	conn, err := p.opts.DialReverse(req.Context(), p.opts.BaseConfigDir, key.uuid, key.service)
	if err == nil {
		setup, cancel := context.WithTimeout(req.Context(), 5*time.Second)
		conn, err = finish(conn, setup)
		cancel()
	}
	if err != nil {
		return nil, false, &ReverseUnavailableError{PeerUUID: key.uuid, Cause: errors.Join(directErr, err)}
	}
	return conn, false, nil
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
	ctx    context.Context
	once   sync.Once
	finish func()
	failed func()
}

func (b *peerServiceBody) Read(data []byte) (int, error) {
	n, err := b.ReadCloser.Read(data)
	if err != nil {
		if contextErr := reverseDialContextError(b.ctx); contextErr != nil {
			err = contextErr
		}
		if err != io.EOF {
			b.failed()
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
