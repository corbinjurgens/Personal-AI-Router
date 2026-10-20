// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"compress/gzip"
	"context"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/http/httptest"
	"net/netip"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/nodepolicy"
)

func publicTestBounds() nodepolicy.Transport {
	return nodepolicy.Transport{ConnectTimeoutSeconds: 1, HeaderTimeoutSeconds: 1, IdleTimeoutSeconds: 1, MaxResponseBytes: 64, MaxErrorBodyBytes: 16, MaxEventBytes: 32}
}

type publicTestBudget struct {
	used    atomic.Int64
	ceiling int64
}

func (b *publicTestBudget) acquire(n int64) (func(), error) {
	if n <= 0 {
		return nil, errors.New("invalid reservation")
	}
	if b.used.Add(n) > b.ceiling {
		b.used.Add(-n)
		return nil, errors.New("capacity")
	}
	var once sync.Once
	return func() { once.Do(func() { b.used.Add(-n) }) }, nil
}

func publicTestServer(t *testing.T, handler http.HandlerFunc) *httptest.Server {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	s := httptest.NewUnstartedServer(handler)
	s.Listener.Close()
	s.Listener = ln
	s.Config.ErrorLog = log.New(io.Discard, "", 0)
	s.StartTLS()
	t.Cleanup(s.Close)
	return s
}

func publicTestClient(t *testing.T, s *httptest.Server) *publicClient {
	t.Helper()
	c, err := newPublicClient("https://example.com/v1/chat/completions", publicTestBounds())
	if err != nil {
		t.Fatal(err)
	}
	c.resolve = func(context.Context, string) ([]netip.Addr, error) {
		return []netip.Addr{netip.MustParseAddr("8.8.8.8")}, nil
	}
	c.dial = func(ctx context.Context, network, address string) (net.Conn, error) {
		if address != "8.8.8.8:443" {
			t.Errorf("forbidden or unpinned logical dial: %s", address)
			return nil, fmt.Errorf("dial was not pinned: %s", address)
		}
		return (&net.Dialer{}).DialContext(ctx, network, s.Listener.Addr().String())
	}
	roots := x509.NewCertPool()
	roots.AddCert(s.Certificate())
	c.client.Transport.(*http.Transport).TLSClientConfig.RootCAs = roots
	return c
}

func publicTestPost(c *publicClient, ctx context.Context, budget publicBudget) (*publicResponse, error) {
	return c.post(ctx, strings.NewReader(`{"model":"synthetic"}`), "selected-synthetic-key", false, budget, publicTestGuard)
}

func publicTestGuard(context.Context) error { return nil }

func publicTestRead(t *testing.T, c *publicClient) ([]byte, error) {
	t.Helper()
	b := &publicTestBudget{ceiling: 4096}
	t.Cleanup(func() {
		if n := b.used.Load(); n != 0 {
			t.Errorf("budget leaked: %d", n)
		}
	})
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Second)
	defer cancel()
	r, err := publicTestPost(c, ctx, b.acquire)
	if err != nil {
		return nil, err
	}
	if r.Body == nil {
		return nil, nil
	}
	defer r.Body.Close()
	return io.ReadAll(r.Body)
}

func TestPublicTransportEndpoint(t *testing.T) {
	for _, suffix := range []string{"?", "?unknown=synthetic", "#", "#fragment"} {
		t.Run(suffix, func(t *testing.T) {
			if err := nodepolicy.CheckEndpoint("https://example.com/chat" + suffix); err == nil {
				t.Fatal("canonical validator accepted forbidden URL")
			}
			if _, err := newPublicClient("https://example.com/chat"+suffix, publicTestBounds()); err == nil {
				t.Fatal("forbidden URL accepted")
			}
		})
	}
	for _, endpoint := range []string{"https://key@example.com/chat", "http://example.com/chat", "https://127.0.0.1/chat", "https://api.local/chat", "https://EXAMPLE.com/chat", "https://example.com/a/../chat", "https://example.com/%63hat"} {
		t.Run(endpoint, func(t *testing.T) {
			if err := nodepolicy.CheckEndpoint(endpoint); err == nil {
				t.Fatal("canonical validator accepted forbidden URL")
			}
			if _, err := newPublicClient(endpoint, publicTestBounds()); err == nil {
				t.Fatal("forbidden URL accepted")
			}
		})
	}
	if _, err := newPublicClient("https://example.com/v1/chat", publicTestBounds()); err != nil {
		t.Fatal(err)
	}
}

func TestPublicTransportAddress(t *testing.T) {
	for _, address := range []string{"0.0.0.0", "10.1.2.3", "100.64.0.1", "127.0.0.1", "169.254.169.254", "172.16.0.1", "192.168.1.1", "192.0.0.1", "192.0.2.1", "192.88.99.1", "198.18.0.1", "198.51.100.1", "203.0.113.1", "224.0.0.1", "240.0.0.1", "255.255.255.255", "::", "::1", "::ffff:127.0.0.1", "::ffff:192.0.2.1", "fe80::1", "fc00::1", "ff02::1", "2001:db8::1", "2001:2::1", "2002:0808:0808::1", "64:ff9b::808:808", "3fff::1"} {
		t.Run(address, func(t *testing.T) {
			c, err := newPublicClient("https://example.com/chat", publicTestBounds())
			if err != nil {
				t.Fatal(err)
			}
			c.resolve = func(context.Context, string) ([]netip.Addr, error) {
				return []netip.Addr{netip.MustParseAddr("8.8.8.8"), netip.MustParseAddr(address)}, nil
			}
			calls := 0
			c.dial = func(context.Context, string, string) (net.Conn, error) {
				calls++
				return nil, errors.New("synthetic dial")
			}
			if _, err := c.dialPublic(context.Background(), "tcp", "example.com:443"); err == nil || calls != 0 {
				t.Fatalf("forbidden answer reached dial: %d, %v", calls, err)
			}
		})
	}
	t.Run("empty answers", func(t *testing.T) {
		c, err := newPublicClient("https://example.com/chat", publicTestBounds())
		if err != nil {
			t.Fatal(err)
		}
		c.resolve = func(context.Context, string) ([]netip.Addr, error) { return nil, nil }
		c.dial = func(context.Context, string, string) (net.Conn, error) {
			t.Error("empty DNS reached dial")
			return nil, errors.New("dial")
		}
		if _, err := c.dialPublic(context.Background(), "tcp", "example.com:443"); err == nil {
			t.Fatal("empty DNS accepted")
		}
	})
	t.Run("numeric pin and rebinding between attempts", func(t *testing.T) {
		var calls atomic.Int64
		s := publicTestServer(t, func(w http.ResponseWriter, r *http.Request) {
			calls.Add(1)
			if r.Host != "example.com" || r.TLS.ServerName != "example.com" {
				t.Error("logical authority lost")
			}
			fmt.Fprint(w, "ok")
		})
		c := publicTestClient(t, s)
		var resolves atomic.Int64
		c.resolve = func(context.Context, string) ([]netip.Addr, error) {
			if resolves.Add(1) == 1 {
				return []netip.Addr{netip.MustParseAddr("::ffff:8.8.8.8")}, nil
			}
			return []netip.Addr{netip.MustParseAddr("127.0.0.1")}, nil
		}
		if body, err := publicTestRead(t, c); err != nil || string(body) != "ok" {
			t.Fatalf("pinned POST failed: %q %v", body, err)
		}
		if _, err := publicTestRead(t, c); err == nil {
			t.Fatal("rebound destination accepted")
		}
		if calls.Load() != 1 || resolves.Load() != 2 {
			t.Fatalf("calls=%d resolves=%d", calls.Load(), resolves.Load())
		}
	})
}

func TestPublicTransportTLS(t *testing.T) {
	for _, mode := range []string{"trusted", "system untrusted", "wrong hostname"} {
		t.Run(mode, func(t *testing.T) {
			var calls atomic.Int64
			s := publicTestServer(t, func(w http.ResponseWriter, r *http.Request) { calls.Add(1); fmt.Fprint(w, "ok") })
			c := publicTestClient(t, s)
			switch mode {
			case "system untrusted":
				c.client.Transport.(*http.Transport).TLSClientConfig.RootCAs = nil
			case "wrong hostname":
				c.endpoint = "https://wrong.example/chat"
			}
			_, err := publicTestRead(t, c)
			if mode == "trusted" {
				if err != nil || calls.Load() != 1 {
					t.Fatalf("trusted TLS failed: %v", err)
				}
			} else if err == nil || calls.Load() != 0 {
				t.Fatalf("invalid TLS reached provider: %v calls=%d", err, calls.Load())
			}
		})
	}
}

func TestPublicTransportHeaders(t *testing.T) {
	for _, mode := range []string{"success", "SSE", "error"} {
		t.Run(mode, func(t *testing.T) {
			s := publicTestServer(t, func(w http.ResponseWriter, r *http.Request) {
				wantAccept := "application/json"
				if mode == "SSE" {
					wantAccept = "text/event-stream"
				}
				if r.Method != http.MethodPost || r.URL.String() != "/v1/chat/completions" || r.Header.Get("Authorization") != "Bearer selected-synthetic-key" || r.Header.Get("Accept") != wantAccept || r.Header.Get("Accept-Encoding") != "identity" || r.Header.Get("Content-Type") != "application/json" {
					t.Error("prepared request changed")
				}
				for key := range r.Header {
					switch key {
					case "Authorization", "Accept", "Accept-Encoding", "Content-Type", "User-Agent", "Connection", "Content-Length":
					default:
						t.Errorf("unexpected outgoing header %s", key)
					}
				}
				for _, key := range []string{"Authorization", "Set-Cookie", "X-PAIR-Node", "Forwarded", "Access-Control-Allow-Origin", "Content-Type"} {
					w.Header().Set(key, "synthetic-forbidden")
				}
				w.Header().Set("Trailer", "X-PAIR-Secret")
				w.Header().Set("Retry-After", "2")
				if mode == "error" {
					w.WriteHeader(http.StatusBadRequest)
				}
				fmt.Fprint(w, "body-sentinel")
				w.Header().Set("X-PAIR-Secret", "synthetic-trailer")
			})
			c := publicTestClient(t, s)
			b := &publicTestBudget{ceiling: 4096}
			r, err := c.post(context.Background(), strings.NewReader("{}"), "selected-synthetic-key", mode == "SSE", b.acquire, publicTestGuard)
			if err != nil {
				t.Fatal(err)
			}
			wantType := "application/json"
			if mode == "SSE" {
				wantType = "text/event-stream"
			}
			if r.ContentType != wantType || r.RetryAfter != "2" {
				t.Fatalf("unsafe response metadata: %+v", r)
			}
			if mode == "error" {
				if r.Body != nil {
					t.Fatal("raw error body exposed")
				}
				// Windows clock resolution can make both reads equal, so require a
				// recorded time that is not before the headers rather than a later one.
				if r.FirstBodyAt().IsZero() || r.FirstBodyAt().Before(r.HeaderAt) {
					t.Fatal("discarded error body timing lost")
				}
			} else {
				if _, err := io.ReadAll(r.Body); err != nil {
					t.Fatal(err)
				}
				r.Body.Close()
			}
			if b.used.Load() != 0 {
				t.Fatal("budget leak")
			}
			// The result type has no raw provider header/trailer/request field.
			if reflect.TypeOf(*r).NumField() != 6 {
				t.Fatal("unexpected public response material")
			}
		})
	}
	t.Run("credential control characters", func(t *testing.T) {
		c, err := newPublicClient("https://example.com/chat", publicTestBounds())
		if err != nil {
			t.Fatal(err)
		}
		for _, key := range []string{"", "synthetic\r\nCookie: injected", "synthetic\t", "synthetic\x7f"} {
			if _, err := c.post(context.Background(), strings.NewReader("{}"), key, false, func(int64) (func(), error) {
				t.Error("credential refusal happened after budget")
				return func() {}, nil
			}, publicTestGuard); err == nil || strings.Contains(err.Error(), "synthetic") {
				t.Fatalf("unsafe credential error: %v", err)
			}
		}
	})
	t.Run("sanitized connection error", func(t *testing.T) {
		c, err := newPublicClient("https://example.com/chat", publicTestBounds())
		if err != nil {
			t.Fatal(err)
		}
		c.client.Transport = publicTestRoundTrip(func(*http.Request) (*http.Response, error) {
			return nil, errors.New("selected-synthetic-key body-sentinel")
		})
		b := &publicTestBudget{ceiling: 4096}
		_, err = publicTestPost(c, context.Background(), b.acquire)
		if err == nil || strings.Contains(err.Error(), "synthetic") || strings.Contains(err.Error(), "example.com") || strings.Contains(err.Error(), "body-sentinel") {
			t.Fatalf("unsafe error: %v", err)
		}
	})
}

func TestPublicTransportBounds(t *testing.T) {
	t.Run("declared length refused before body read", func(t *testing.T) {
		reader := &publicCountingReader{Reader: strings.NewReader(strings.Repeat("x", 65))}
		c, err := newPublicClient("https://example.com/chat", publicTestBounds())
		if err != nil {
			t.Fatal(err)
		}
		c.client.Transport = publicTestRoundTrip(func(*http.Request) (*http.Response, error) {
			return &http.Response{StatusCode: http.StatusOK, ContentLength: 65, Header: make(http.Header), Body: io.NopCloser(reader)}, nil
		})
		budget := &publicTestBudget{ceiling: 4096}
		r, err := publicTestPost(c, context.Background(), budget.acquire)
		if r != nil && r.Body != nil {
			defer r.Body.Close()
		}
		if err == nil || reader.read != 0 {
			t.Fatalf("declared overflow reached body: read=%d err=%v", reader.read, err)
		}
		if budget.used.Load() != 0 {
			t.Fatal("declared overflow leaked budget")
		}
	})
	for _, mode := range []string{"exact success", "declared overflow", "chunked overflow", "error overflow", "gzip", "deflate", "br", "duplicate encoding", "header overflow", "slow headers"} {
		t.Run(mode, func(t *testing.T) {
			s := publicTestServer(t, func(w http.ResponseWriter, r *http.Request) {
				switch mode {
				case "declared overflow":
					w.Header().Set("Content-Length", "65")
				case "chunked overflow":
					w.(http.Flusher).Flush()
				case "error overflow":
					w.WriteHeader(http.StatusInternalServerError)
				case "gzip", "deflate", "br":
					w.Header().Set("Content-Encoding", mode)
				case "duplicate encoding":
					w.Header().Add("Content-Encoding", "identity")
					w.Header().Add("Content-Encoding", "gzip")
				case "header overflow":
					w.Header().Set("X-Large", strings.Repeat("x", 66<<10))
				case "slow headers":
					select {
					case <-r.Context().Done():
						return
					case <-time.After(2 * time.Second):
					}
				}
				n := 65
				if mode == "gzip" {
					var compressed bytes.Buffer
					gz := gzip.NewWriter(&compressed)
					if _, err := gz.Write(bytes.Repeat([]byte("x"), 4096)); err != nil {
						t.Error(err)
						return
					}
					if err := gz.Close(); err != nil {
						t.Error(err)
						return
					}
					if compressed.Len() >= 64 {
						t.Error("compressed fixture exceeds wire cap")
						return
					}
					if _, err := w.Write(compressed.Bytes()); err != nil {
						t.Error(err)
					}
					return
				}
				if mode == "deflate" || mode == "br" || mode == "duplicate encoding" {
					n = 8
				}
				if mode == "exact success" {
					n = 64
				}
				fmt.Fprint(w, strings.Repeat("x", n))
			})
			body, err := publicTestRead(t, publicTestClient(t, s))
			if mode == "exact success" {
				if err != nil || len(body) != 64 {
					t.Fatalf("exact bound failed: len=%d err=%v", len(body), err)
				}
			} else if err == nil {
				t.Fatal("unbounded response accepted")
			}
		})
	}
	t.Run("false small length and max plus one", func(t *testing.T) {
		// A misleading RoundTripper cannot bypass the reader's independent count.
		reader := &publicCountingReader{Reader: strings.NewReader(strings.Repeat("x", 80))}
		c, err := newPublicClient("https://example.com/chat", publicTestBounds())
		if err != nil {
			t.Fatal(err)
		}
		c.client.Transport = publicTestRoundTrip(func(*http.Request) (*http.Response, error) {
			return &http.Response{StatusCode: http.StatusOK, ContentLength: 1, Header: make(http.Header), Body: io.NopCloser(reader)}, nil
		})
		if _, err := publicTestRead(t, c); err == nil || reader.read != 65 {
			t.Fatalf("overflow read=%d err=%v", reader.read, err)
		}
	})
}

type publicTestRoundTrip func(*http.Request) (*http.Response, error)

func (f publicTestRoundTrip) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

type publicCountingReader struct {
	*strings.Reader
	read int
}

func (r *publicCountingReader) Read(p []byte) (int, error) {
	n, err := r.Reader.Read(p)
	r.read += n
	return n, err
}

type publicErrorTestBody struct {
	io.Reader
	interrupted bool
	closes      atomic.Int64
}

func (b *publicErrorTestBody) Read(p []byte) (int, error) {
	if b.interrupted {
		return 0, io.ErrUnexpectedEOF
	}
	return b.Reader.Read(p)
}

func (b *publicErrorTestBody) Close() error {
	b.closes.Add(1)
	return nil
}

func TestPublicTransportErrorBodyBounds(t *testing.T) {
	for _, tc := range []struct {
		name          string
		bodyBytes     int
		contentLength int64
		wantError     bool
	}{
		{name: "exact error bound", bodyBytes: 16, contentLength: 16},
		{name: "declared overflow", bodyBytes: 17, contentLength: 17, wantError: true},
		{name: "unknown length overflow", bodyBytes: 17, contentLength: -1, wantError: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c, err := newPublicClient("https://example.com/chat", publicTestBounds())
			if err != nil {
				t.Fatal(err)
			}
			c.client.Transport = publicTestRoundTrip(func(*http.Request) (*http.Response, error) {
				return &http.Response{
					StatusCode:    http.StatusInternalServerError,
					ContentLength: tc.contentLength,
					Header:        http.Header{"Retry-After": {"5"}},
					Body:          io.NopCloser(strings.NewReader(strings.Repeat("x", tc.bodyBytes))),
				}, nil
			})
			budget := &publicTestBudget{ceiling: 4096}
			response, err := publicTestPost(c, context.Background(), budget.acquire)
			if tc.wantError {
				if err == nil {
					t.Fatal("error body exceeded its cap but was accepted")
				}
			} else if err != nil || response == nil || response.Body != nil {
				t.Fatalf("exact error bound failed: response=%+v err=%v", response, err)
			}
			if response == nil || response.Status != http.StatusInternalServerError || response.RetryAfter != "5" || response.HeaderAt.IsZero() || response.Body != nil {
				t.Fatalf("bounded classification metadata lost: response=%+v err=%v", response, err)
			}
			if budget.used.Load() != 0 {
				t.Fatalf("error body leaked budget: %d", budget.used.Load())
			}
		})
	}
}

func TestPublicTransportCancel(t *testing.T) {
	for _, mode := range []string{"idle", "cancel", "before send", "bytes reset idle"} {
		t.Run(mode, func(t *testing.T) {
			var calls atomic.Int64
			s := publicTestServer(t, func(w http.ResponseWriter, r *http.Request) {
				calls.Add(1)
				w.(http.Flusher).Flush()
				if mode == "bytes reset idle" {
					for i := 0; i < 3; i++ {
						select {
						case <-r.Context().Done():
							return
						case <-time.After(400 * time.Millisecond):
						}
						fmt.Fprint(w, "x")
						w.(http.Flusher).Flush()
					}
					return
				}
				select {
				case <-r.Context().Done():
				case <-time.After(2500 * time.Millisecond):
				}
			})
			c := publicTestClient(t, s)
			b := &publicTestBudget{ceiling: 4096}
			ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
			defer cancel()
			if mode == "before send" {
				cancel()
			}
			r, err := publicTestPost(c, ctx, b.acquire)
			if mode == "before send" {
				if err == nil || calls.Load() != 0 || b.used.Load() != 0 {
					t.Fatal("canceled attempt dispatched")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if r.FirstBodyAt() != (time.Time{}) || r.HeaderAt.IsZero() {
				t.Fatal("headers confused with body")
			}
			if mode == "cancel" {
				cancel()
			}
			start := time.Now()
			body, err := io.ReadAll(r.Body)
			allowed := 1800 * time.Millisecond
			if mode == "cancel" {
				allowed = 500 * time.Millisecond
			}
			if mode == "bytes reset idle" {
				if err != nil || string(body) != "xxx" || r.FirstBodyAt().IsZero() || r.FirstBodyAt().Before(r.HeaderAt) {
					t.Fatalf("idle was not reset: %q %v", body, err)
				}
			} else if err == nil || time.Since(start) > allowed {
				t.Fatalf("cancellation failed promptly: %v", err)
			}
			r.Body.Close()
			if b.used.Load() != 0 {
				t.Fatal("cancellation leaked budget")
			}
		})
	}
}

func TestPublicTransportBudget(t *testing.T) {
	var calls atomic.Int64
	s := publicTestServer(t, func(w http.ResponseWriter, r *http.Request) { calls.Add(1); fmt.Fprint(w, strings.Repeat("x", 64)) })
	c := publicTestClient(t, s)
	t.Run("missing and refused before dispatch", func(t *testing.T) {
		for _, budget := range []publicBudget{nil, (&publicTestBudget{ceiling: 1}).acquire} {
			if _, err := publicTestPost(c, context.Background(), budget); err == nil {
				t.Fatal("missing capacity accepted")
			}
		}
		if calls.Load() != 0 {
			t.Fatal("budget refusal dispatched")
		}
	})
	t.Run("concurrent headroom and repeated close", func(t *testing.T) {
		b := &publicTestBudget{ceiling: 390}
		responses := make(chan *publicResponse, 2)
		var wg sync.WaitGroup
		for i := 0; i < 2; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				r, err := publicTestPost(c, context.Background(), b.acquire)
				if err != nil {
					t.Error(err)
					return
				}
				responses <- r
			}()
		}
		wg.Wait()
		close(responses)
		if b.used.Load() != 390 {
			t.Fatalf("raw/parse/copy headroom=%d", b.used.Load())
		}
		if _, err := publicTestPost(c, context.Background(), b.acquire); err == nil {
			t.Fatal("overcommitted capacity")
		}
		for r := range responses {
			if _, err := io.ReadAll(r.Body); err != nil {
				t.Error(err)
			}
			r.Body.Close()
			r.Body.Close()
		}
		if b.used.Load() != 0 {
			t.Fatal("budget leaked after close")
		}
	})
	t.Run("parser abort and panic unwinding", func(t *testing.T) {
		b := &publicTestBudget{ceiling: 4096}
		func() {
			defer func() {
				if recover() == nil {
					t.Error("missing synthetic panic")
				}
			}()
			r, err := publicTestPost(c, context.Background(), b.acquire)
			if err != nil {
				t.Fatal(err)
			}
			defer r.Body.Close()
			panic("synthetic parser panic")
		}()
		if b.used.Load() != 0 {
			t.Fatal("panic leaked budget")
		}
	})
}

func TestPublicTransportRedirectProxy(t *testing.T) {
	t.Run("no second hop", func(t *testing.T) {
		var calls atomic.Int64
		s := publicTestServer(t, func(w http.ResponseWriter, r *http.Request) {
			calls.Add(1)
			if r.URL.Path == "/second" {
				fmt.Fprint(w, "second")
				return
			}
			w.Header().Set("Location", "https://example.com/second")
			w.WriteHeader(http.StatusTemporaryRedirect)
		})
		if _, err := publicTestRead(t, publicTestClient(t, s)); err == nil || calls.Load() != 1 {
			t.Fatalf("redirect followed: err=%v calls=%d", err, calls.Load())
		}
	})
	t.Run("ambient proxy ignored", func(t *testing.T) {
		var proxyCalls atomic.Int64
		ln, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			t.Fatal(err)
		}
		sentinel := &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { proxyCalls.Add(1); w.WriteHeader(http.StatusBadGateway) })}
		go sentinel.Serve(ln)
		defer sentinel.Close()
		for _, name := range []string{"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"} {
			// The logical proxy is public too, so dial-address checks cannot mask
			// accidental ambient proxy use. Both physical fixtures stay loopback.
			t.Setenv(name, "http://example.com:443")
		}
		t.Setenv("NO_PROXY", "")
		t.Setenv("no_proxy", "")
		s := publicTestServer(t, func(w http.ResponseWriter, r *http.Request) { fmt.Fprint(w, "ok") })
		c := publicTestClient(t, s)
		c.dial = func(ctx context.Context, network, address string) (net.Conn, error) {
			if address != "8.8.8.8:443" {
				return nil, errors.New("dial was not pinned")
			}
			physical := s.Listener.Addr().String()
			if c.client.Transport.(*http.Transport).Proxy != nil {
				physical = ln.Addr().String()
			}
			return (&net.Dialer{}).DialContext(ctx, network, physical)
		}
		if _, err := publicTestRead(t, c); err != nil || proxyCalls.Load() != 0 {
			t.Fatalf("ambient proxy used: %v calls=%d", err, proxyCalls.Load())
		}
	})
	t.Run("fresh finite transport", func(t *testing.T) {
		c, err := newPublicClient("https://example.com/chat", publicTestBounds())
		if err != nil {
			t.Fatal(err)
		}
		tr := c.client.Transport.(*http.Transport)
		if tr == http.DefaultTransport || c.client.Jar != nil || !tr.DisableKeepAlives || !tr.DisableCompression || tr.ForceAttemptHTTP2 || tr.MaxResponseHeaderBytes != 64<<10 || tr.TLSHandshakeTimeout <= 0 || tr.ResponseHeaderTimeout <= 0 || tr.IdleConnTimeout <= 0 || tr.TLSClientConfig.InsecureSkipVerify || len(tr.TLSClientConfig.Certificates) != 0 || tr.TLSClientConfig.RootCAs != nil || tr.TLSClientConfig.GetClientCertificate != nil || tr.TLSClientConfig.MinVersion != tls.VersionTLS12 {
			t.Fatal("public transport boundary weakened")
		}
	})
}
