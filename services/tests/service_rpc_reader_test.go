// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"sync"
	"testing"
	"time"

	"nvpair-shared/jsonrpc"
)

type serviceRPCReader struct {
	conn           io.ReadWriteCloser
	mu             sync.Mutex
	frames         []*jsonrpc.Message
	bytes          int
	err            error
	readErr        error
	budgetErr      error
	wake           chan struct{}
	done           chan struct{}
	closeOnce      sync.Once
	closing        bool
	closeErr       error
	readerCloseErr error
}

var servicePlatformCloseError = func(error) bool { return false }

func serviceExpectedReadError(err error, closing bool) bool {
	if err == io.EOF {
		return true
	}
	return closing && (errors.Is(err, net.ErrClosed) || errors.Is(err, os.ErrClosed) || errors.Is(err, io.ErrClosedPipe) || servicePlatformCloseError(err))
}

func newServiceRPCReader(conn io.ReadWriteCloser) *serviceRPCReader {
	c := &serviceRPCReader{conn: conn, wake: make(chan struct{}, 1), done: make(chan struct{})}
	go func() {
		defer close(c.done)
		codec := jsonrpc.NewCodecMaxFrame(conn, jsonrpc.WorkerFrameBytes)
		for {
			m, err := codec.Read()
			c.mu.Lock()
			if err != nil {
				if c.err == nil {
					c.err = err
				}
				if !serviceExpectedReadError(err, c.closing) {
					c.readErr = fmt.Errorf("fixture client reader: %w", err)
				}
				c.mu.Unlock()
				c.signal()
				return
			}
			size := serviceFrameBytes(m)
			if len(c.frames) >= 4096 || c.bytes+size > 16*1024*1024 {
				c.budgetErr = fmt.Errorf("fixture client retained-frame budget exceeded")
				c.err = c.budgetErr
			} else {
				c.frames = append(c.frames, m)
				c.bytes += size
			}
			c.mu.Unlock()
			c.signal()
		}
	}()
	return c
}

func (c *serviceRPCReader) signal() {
	select {
	case c.wake <- struct{}{}:
	default:
	}
}

func (c *serviceRPCReader) closeConnection() {
	c.closeOnce.Do(func() {
		c.mu.Lock()
		c.closing = true
		c.mu.Unlock()
		err := c.conn.Close()
		c.mu.Lock()
		if err != nil {
			c.closeErr = fmt.Errorf("close fixture client: %w", err)
		}
		c.mu.Unlock()
	})
}

func (c *serviceRPCReader) close() error {
	c.closeConnection()
	select {
	case <-c.done:
	case <-time.After(2 * time.Second):
		c.mu.Lock()
		if c.readerCloseErr == nil {
			c.readerCloseErr = errors.New("fixture client reader did not exit")
		}
		c.mu.Unlock()
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return errors.Join(c.closeErr, c.readerCloseErr, c.budgetErr, c.readErr)
}

func serviceFrameBytes(m *jsonrpc.Message) int {
	size := 128 + len(m.JSONRPC) + len(m.Params) + len(m.Result) + len(m.Method)
	if m.ID != nil {
		size += len(*m.ID)
	}
	if m.Error != nil {
		size += len(m.Error.Message) + len(m.Error.Data)
	}
	return size
}

func TestServiceRPCReaderPreservesMalformedFrames(t *testing.T) {
	test := func(name, frame string) {
		t.Run(name, func(t *testing.T) {
			conn, peer := net.Pipe()
			c := newServiceRPCReader(conn)
			t.Cleanup(func() { _ = c.close(); _ = peer.Close() })
			if err := peer.SetWriteDeadline(time.Now().Add(time.Second)); err != nil {
				t.Fatal(err)
			}
			if _, err := io.WriteString(peer, frame); err != nil {
				t.Fatal(err)
			}
			if err := peer.Close(); err != nil {
				t.Fatal(err)
			}
			waitServiceRPCReader(t, c)
			for _, operation := range []string{"first close", "repeated close"} {
				var decodeErr *jsonrpc.DecodeError
				if err := c.close(); !errors.As(err, &decodeErr) {
					t.Fatalf("%s erased malformed-frame error: %v", operation, err)
				}
			}
		})
	}
	test("malformed newline frame", "{\n")
	test("truncated final frame", `{"jsonrpc":`)
}

func TestServiceRPCReaderAcceptsNormalShutdown(t *testing.T) {
	t.Run("remote EOF", func(t *testing.T) {
		conn, peer := net.Pipe()
		c := newServiceRPCReader(conn)
		t.Cleanup(func() { _ = c.close(); _ = peer.Close() })
		if err := peer.Close(); err != nil {
			t.Fatal(err)
		}
		waitServiceRPCReader(t, c)
		if err := c.close(); err != nil {
			t.Fatalf("normal remote EOF is a cleanup failure: %v", err)
		}
	})
	t.Run("intentional local close", func(t *testing.T) {
		conn, peer := net.Pipe()
		c := newServiceRPCReader(conn)
		t.Cleanup(func() { _ = peer.Close() })
		if err := c.close(); err != nil {
			t.Fatalf("intentional local close is a cleanup failure: %v", err)
		}
	})
}

func TestServiceRPCReaderPreservesUnexpectedTransportErrors(t *testing.T) {
	t.Run("closed transport before local close", func(t *testing.T) {
		c := newServiceRPCReader(&serviceReaderFaultConn{err: net.ErrClosed})
		waitServiceRPCReader(t, c)
		if err := c.close(); !errors.Is(err, net.ErrClosed) {
			t.Fatalf("unexpected closed transport was suppressed: %v", err)
		}
	})
	t.Run("unknown error during intentional close", func(t *testing.T) {
		injected := errors.New("unexpected transport failure")
		c := newServiceRPCReader(&serviceReaderFaultConn{err: injected, release: make(chan struct{})})
		if err := c.close(); !errors.Is(err, injected) {
			t.Fatalf("local close erased unexpected transport error: %v", err)
		}
	})
}

type serviceReaderFaultConn struct {
	err     error
	release chan struct{}
	once    sync.Once
}

func (c *serviceReaderFaultConn) Read([]byte) (int, error) {
	if c.release != nil {
		<-c.release
	}
	return 0, c.err
}
func (c *serviceReaderFaultConn) Write(p []byte) (int, error) { return len(p), nil }
func (c *serviceReaderFaultConn) Close() error {
	c.once.Do(func() {
		if c.release != nil {
			close(c.release)
		}
	})
	return nil
}

func waitServiceRPCReader(t *testing.T, c *serviceRPCReader) {
	t.Helper()
	select {
	case <-c.done:
	case <-time.After(time.Second):
		t.Fatal("in-memory RPC reader did not exit")
	}
}
