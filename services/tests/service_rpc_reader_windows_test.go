//go:build windows

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"errors"
	"testing"

	"github.com/Microsoft/go-winio"
)

func init() {
	servicePlatformCloseError = func(err error) bool { return errors.Is(err, winio.ErrFileClosed) }
}

func TestServiceRPCReaderWindowsCloseErrors(t *testing.T) {
	t.Run("intentional named pipe close", func(t *testing.T) {
		c := newServiceRPCReader(&serviceReaderFaultConn{err: winio.ErrFileClosed, release: make(chan struct{})})
		if err := c.close(); err != nil {
			t.Fatalf("intentional named pipe close is a cleanup failure: %v", err)
		}
	})
	t.Run("named pipe closed before local close", func(t *testing.T) {
		c := newServiceRPCReader(&serviceReaderFaultConn{err: winio.ErrFileClosed})
		waitServiceRPCReader(t, c)
		if err := c.close(); !errors.Is(err, winio.ErrFileClosed) {
			t.Fatalf("unexpected named pipe close was suppressed: %v", err)
		}
	})
}
