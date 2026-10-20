// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"crypto/tls"
	"errors"
	"io"
	"net"
	"net/http"
	"net/netip"
	"net/url"
	"strings"
	"sync"
	"time"
	"unicode"

	"nvpair-shared/nodepolicy"
)

// publicBudget must acquire without waiting. Its release must be idempotent.
// The caller owns the process-wide pool; no unlimited production implementation exists.
type publicBudget func(int64) (release func(), err error)

type publicClient struct {
	endpoint string
	bounds   nodepolicy.Transport
	client   *http.Client
	resolve  func(context.Context, string) ([]netip.Addr, error)
	dial     func(context.Context, string, string) (net.Conn, error)
}

// newPublicClient is deliberately independent of ingress and cluster clients.
func newPublicClient(endpoint string, bounds nodepolicy.Transport) (*publicClient, error) {
	if err := nodepolicy.CheckEndpoint(endpoint); err != nil {
		return nil, errors.New("public endpoint refused")
	}
	u, err := url.Parse(endpoint)
	if err != nil || u.Scheme != "https" || u.Hostname() == "" || u.User != nil || u.RawQuery != "" || u.ForceQuery || u.Fragment != "" || strings.Contains(endpoint, "#") {
		return nil, errors.New("public endpoint refused")
	}
	if bounds.ConnectTimeoutSeconds < 1 || bounds.ConnectTimeoutSeconds > 60 || bounds.HeaderTimeoutSeconds < 1 || bounds.HeaderTimeoutSeconds > 600 || bounds.IdleTimeoutSeconds < 1 || bounds.IdleTimeoutSeconds > 600 || bounds.MaxResponseBytes < 1 || bounds.MaxResponseBytes > 64<<20 || bounds.MaxErrorBodyBytes < 1 || bounds.MaxErrorBodyBytes > 1<<20 || bounds.MaxEventBytes < 1 || bounds.MaxEventBytes > bounds.MaxResponseBytes {
		return nil, errors.New("public transport bounds refused")
	}
	connect := time.Duration(bounds.ConnectTimeoutSeconds) * time.Second
	c := &publicClient{endpoint: endpoint, bounds: bounds,
		dial: (&net.Dialer{Timeout: connect}).DialContext}
	// LookupNetIP needs an explicit network; keep the seam hostname-only.
	c.resolve = func(ctx context.Context, host string) ([]netip.Addr, error) {
		return net.DefaultResolver.LookupNetIP(ctx, "ip", host)
	}
	transport := &http.Transport{
		Proxy:                  nil,
		DialContext:            c.dialPublic,
		TLSClientConfig:        &tls.Config{MinVersion: tls.VersionTLS12},
		TLSHandshakeTimeout:    connect,
		ResponseHeaderTimeout:  time.Duration(bounds.HeaderTimeoutSeconds) * time.Second,
		IdleConnTimeout:        time.Duration(bounds.IdleTimeoutSeconds) * time.Second,
		MaxResponseHeaderBytes: 64 << 10,
		DisableCompression:     true,
		DisableKeepAlives:      true,
		ForceAttemptHTTP2:      false,
		TLSNextProto:           make(map[string]func(string, *tls.Conn) http.RoundTripper),
	}
	c.client = &http.Client{Transport: transport, CheckRedirect: func(*http.Request, []*http.Request) error {
		return errors.New("public redirect refused")
	}}
	return c, nil
}

var publicDeniedPrefixes = []netip.Prefix{
	netip.MustParsePrefix("0.0.0.0/8"), netip.MustParsePrefix("10.0.0.0/8"),
	netip.MustParsePrefix("100.64.0.0/10"), netip.MustParsePrefix("127.0.0.0/8"),
	netip.MustParsePrefix("169.254.0.0/16"), netip.MustParsePrefix("172.16.0.0/12"),
	netip.MustParsePrefix("192.0.0.0/24"), netip.MustParsePrefix("192.0.2.0/24"),
	netip.MustParsePrefix("192.88.99.0/24"), netip.MustParsePrefix("192.168.0.0/16"),
	netip.MustParsePrefix("198.18.0.0/15"), netip.MustParsePrefix("198.51.100.0/24"),
	netip.MustParsePrefix("203.0.113.0/24"), netip.MustParsePrefix("224.0.0.0/3"),
	netip.MustParsePrefix("2001::/23"), netip.MustParsePrefix("2001:db8::/32"),
	netip.MustParsePrefix("2002::/16"), netip.MustParsePrefix("3fff::/20"),
}

func publicAddress(addr netip.Addr) bool {
	addr = addr.Unmap()
	if !addr.IsValid() || addr.Zone() != "" || !addr.IsGlobalUnicast() || addr.IsPrivate() || (addr.Is6() && !netip.MustParsePrefix("2000::/3").Contains(addr)) {
		return false
	}
	for _, prefix := range publicDeniedPrefixes {
		if prefix.Contains(addr) {
			return false
		}
	}
	return true
}

func (c *publicClient) dialPublic(ctx context.Context, network, authority string) (net.Conn, error) {
	u, _ := url.Parse(c.endpoint) // constructor validated this immutable string
	host, port, err := net.SplitHostPort(authority)
	if err != nil || host != u.Hostname() || network != "tcp" {
		return nil, errors.New("public dial refused")
	}
	dialCtx, cancel := context.WithTimeout(ctx, time.Duration(c.bounds.ConnectTimeoutSeconds)*time.Second)
	defer cancel()
	addresses, err := c.resolve(dialCtx, host)
	if err != nil || len(addresses) == 0 {
		return nil, errors.New("public DNS refused")
	}
	for _, addr := range addresses {
		if !publicAddress(addr) {
			return nil, errors.New("public address refused")
		}
	}
	// One numeric dial, never a second hostname resolution or automatic retry.
	return c.dial(dialCtx, network, net.JoinHostPort(addresses[0].Unmap().String(), port))
}

type publicResponse struct {
	Status      int
	RetryAfter  string
	ContentType string
	HeaderAt    time.Time
	Body        *publicBody
	timing      *publicBodyTiming
}

type publicBodyTiming struct {
	mu        sync.Mutex
	firstByte time.Time
}

// FirstBodyAt remains available after an error body has been discarded too.
func (r *publicResponse) FirstBodyAt() time.Time {
	r.timing.mu.Lock()
	defer r.timing.mu.Unlock()
	return r.timing.firstByte
}

// post makes one attempt. Bounded classification metadata survives body errors.
// A returned body owns its reservation until Close; consumers must defer Close,
// including parser failures and panic unwinding.
var errPublicFinalGuard = errors.New("public final pre-send guard refused")

func (c *publicClient) post(ctx context.Context, prepared io.Reader, authorization string, stream bool, budget publicBudget, guard func(context.Context) error) (*publicResponse, error) {
	if authorization == "" || strings.ContainsFunc(authorization, unicode.IsControl) {
		return nil, errors.New("public credential refused")
	}
	if budget == nil || ctx.Err() != nil {
		return nil, errors.New("public attempt refused")
	}
	// Reserve worst-case raw, decoded and copied headroom before any dispatch.
	limit := max(c.bounds.MaxResponseBytes, c.bounds.MaxErrorBodyBytes)
	release, err := budget(3 * (limit + 1))
	if err != nil || release == nil {
		return nil, errors.New("public response budget refused")
	}
	var once sync.Once
	free := func() { once.Do(release) }
	owned := false
	defer func() {
		if !owned {
			free()
		}
	}()
	requestCtx, cancel := context.WithCancel(ctx)
	defer func() {
		if !owned {
			cancel()
		}
	}()
	req, err := http.NewRequestWithContext(requestCtx, http.MethodPost, c.endpoint, prepared)
	if err != nil {
		return nil, errors.New("public request refused")
	}
	req.Header = http.Header{
		"Content-Type":    {"application/json"},
		"Accept":          {"application/json"},
		"Accept-Encoding": {"identity"},
		"Authorization":   {"Bearer " + authorization},
	}
	if stream {
		req.Header.Set("Accept", "text/event-stream")
	}
	// The callback is the definitive permission/lease acceptance. All fallible
	// construction and headroom acquisition precedes it; no work may block
	// between its successful return and Do. Release its locks before IO.
	if guard == nil || ctx.Err() != nil || guard(ctx) != nil {
		return nil, errPublicFinalGuard
	}
	resp, err := c.client.Do(req)
	if err != nil {
		return nil, errors.New("public connection failed")
	}
	timing := &publicBodyTiming{}
	result := &publicResponse{Status: resp.StatusCode, HeaderAt: time.Now(), ContentType: req.Header.Get("Accept"), timing: timing}
	// Retry-After is classification data only, never downstream header material.
	if value := resp.Header.Get("Retry-After"); len(value) <= 128 {
		result.RetryAfter = value
	}
	if encodings := resp.Header.Values("Content-Encoding"); len(encodings) > 1 || (len(encodings) == 1 && !strings.EqualFold(encodings[0], "identity")) {
		resp.Body.Close()
		return result, errors.New("public encoding refused")
	}
	cap := c.bounds.MaxResponseBytes
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		cap = c.bounds.MaxErrorBodyBytes
	}
	if resp.ContentLength > cap {
		resp.Body.Close()
		return result, errors.New("public response too large")
	}
	b := &publicBody{source: resp.Body, remaining: cap, cancel: cancel, release: free, idle: time.Duration(c.bounds.IdleTimeoutSeconds) * time.Second, timing: timing}
	b.mu.Lock()
	b.timer = time.AfterFunc(b.idle, func() { b.Close() })
	b.stopContext = context.AfterFunc(requestCtx, func() { b.Close() })
	b.mu.Unlock()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		defer b.Close()
		if _, err := io.Copy(io.Discard, b); err != nil {
			return result, errors.New("public error body failed")
		}
		return result, nil
	}
	result.Body = b
	owned = true
	return result, nil
}

type publicBody struct {
	source      io.ReadCloser
	remaining   int64
	cancel      context.CancelFunc
	release     func()
	idle        time.Duration
	mu          sync.Mutex
	timer       *time.Timer
	stopContext func() bool
	closed      bool
	closeOnce   sync.Once
	timing      *publicBodyTiming
}

func (b *publicBody) Read(p []byte) (int, error) {
	defer func() {
		if failure := recover(); failure != nil {
			b.Close()
			panic(failure)
		}
	}()
	if len(p) == 0 {
		return 0, nil
	}
	// Probe at most one extra byte, without exposing the overflow byte.
	if int64(len(p)) > b.remaining+1 {
		p = p[:b.remaining+1]
	}
	n, err := b.source.Read(p)
	if n > 0 {
		b.timing.mu.Lock()
		if b.timing.firstByte.IsZero() {
			b.timing.firstByte = time.Now()
		}
		b.timing.mu.Unlock()
	}
	b.mu.Lock()
	if n > 0 {
		if !b.closed {
			b.timer.Reset(b.idle)
		}
	}
	closed := b.closed
	b.mu.Unlock()
	if int64(n) > b.remaining {
		b.Close()
		return 0, errors.New("public response too large")
	}
	b.remaining -= int64(n)
	if closed {
		return 0, errors.New("public body interrupted")
	}
	if err != nil && err != io.EOF {
		b.Close()
		return n, errors.New("public body failed")
	}
	if err == io.EOF {
		b.mu.Lock()
		b.timer.Stop()
		b.mu.Unlock()
	}
	// EOF does not release headroom while the caller still holds raw/parsed data.
	return n, err
}

func (b *publicBody) Close() error {
	b.closeOnce.Do(func() {
		b.mu.Lock()
		b.closed = true
		if b.timer != nil {
			b.timer.Stop()
		}
		if b.stopContext != nil {
			b.stopContext()
		}
		b.mu.Unlock()
		defer b.release()
		b.cancel()
		b.source.Close()
	})
	return nil
}
