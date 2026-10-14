// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package clustertrust

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"path/filepath"
	"runtime"
	"sync"
	"time"

	"nvpair-shared/noderec"
)

const ReverseDialMaxHeader = 512

// ReverseDialRequest is private IPC framing; it is never an HTTP operation.
type ReverseDialRequest struct {
	PeerUUID         string             `json:"peerUuid"`
	Service          noderec.ServiceKey `json:"service"`
	CallID           string             `json:"callId"`
	Watch            bool               `json:"watch,omitempty"`
	DeadlineUnixNano int64              `json:"deadlineUnixNano,omitempty"`
}

// ReverseDialPath is derived from the same base config directory as the broker.
func ReverseDialPath(base string) string {
	base, _ = filepath.Abs(base)
	if runtime.GOOS == "windows" {
		return fmt.Sprintf(`\\.\pipe\nvpair-reverse-%x`, sha256.Sum256([]byte(filepath.Clean(base))))
	}
	return filepath.Join(base, "cluster", "reverse.sock")
}

// Socket deadlines can fire before the context timer publishes Err. Both
// timers use this deadline, so reaching it already determines the context
// outcome without waiting for that timer's goroutine.
func reverseDialContextError(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if deadline, ok := ctx.Deadline(); ok && !time.Now().Before(deadline) {
		return context.DeadlineExceeded
	}
	return nil
}

// DialReverse returns a service byte stream whose lifetime follows ctx. The
// current cluster-manager alone decides whether this peer/service is admitted.
func DialReverse(ctx context.Context, base, peerUUID string, service noderec.ServiceKey) (net.Conn, error) {
	fail := func(err error) (net.Conn, error) { return nil, &ReverseUnavailableError{peerUUID, err} }
	if peerUUID == "" || !reverseServiceAllowed(service) {
		return fail(fmt.Errorf("invalid private dial target"))
	}
	var id [16]byte
	if _, err := rand.Read(id[:]); err != nil {
		return fail(err)
	}
	callID := hex.EncodeToString(id[:])
	request := ReverseDialRequest{PeerUUID: peerUUID, Service: service, CallID: callID}
	if deadline, ok := ctx.Deadline(); ok {
		request.DeadlineUnixNano = deadline.UnixNano()
	}
	header, err := json.Marshal(request)
	if err != nil || len(header)+1 > ReverseDialMaxHeader {
		return fail(fmt.Errorf("private dial header too large"))
	}
	setup, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	conn, err := dialReversePrivate(setup, ReverseDialPath(base))
	if err != nil {
		if contextErr := reverseDialContextError(setup); contextErr != nil {
			err = contextErr
		}
		return fail(err)
	}
	stopData := context.AfterFunc(setup, func() { conn.Close() })
	defer stopData()
	deadline, _ := setup.Deadline()
	conn.SetDeadline(deadline)
	if _, err = conn.Write(append(header, '\n')); err != nil {
		conn.Close()
		if contextErr := reverseDialContextError(setup); contextErr != nil {
			err = contextErr
		}
		return fail(err)
	}
	// Attach cancellation before admitting any application stream. Keeping
	// this connection avoids another accept/header slot when Close is called.
	lifecycle, err := dialReversePrivate(setup, ReverseDialPath(base))
	if err != nil {
		conn.Close()
		if contextErr := reverseDialContextError(setup); contextErr != nil {
			err = contextErr
		}
		return fail(err)
	}
	stopWatch := context.AfterFunc(setup, func() { lifecycle.Close() })
	defer stopWatch()
	lifecycle.SetDeadline(deadline)
	watchHeader, err := json.Marshal(ReverseDialRequest{CallID: callID, Watch: true})
	if err == nil {
		_, err = lifecycle.Write(append(watchHeader, '\n'))
	}
	readReply := func(c net.Conn) error {
		var reply [1]byte
		if _, err := io.ReadFull(c, reply[:]); err != nil {
			return err
		}
		if reply[0] != 0 {
			return fmt.Errorf("private service dial refused")
		}
		return nil
	}
	if err == nil {
		err = readReply(lifecycle)
	}
	if err == nil {
		err = readReply(conn)
	}
	dataStopped, watchStopped := stopData(), stopWatch()
	if contextErr := reverseDialContextError(setup); contextErr != nil {
		err = contextErr
	}
	if err != nil || !dataStopped || !watchStopped {
		lifecycle.Close()
		conn.Close()
		return fail(err)
	}
	conn.SetDeadline(time.Time{})
	lifecycle.SetDeadline(time.Time{})
	if deadline, ok := ctx.Deadline(); ok {
		conn.SetDeadline(deadline)
	}
	c := &reverseDialConn{Conn: conn, lifecycle: lifecycle}
	c.stop = context.AfterFunc(ctx, func() { c.close() })
	return c, nil
}

type reverseDialConn struct {
	net.Conn
	stop      func() bool
	lifecycle net.Conn
	once      sync.Once
}

func (c *reverseDialConn) close()       { c.once.Do(func() { c.lifecycle.Close(); c.Conn.Close() }) }
func (c *reverseDialConn) Close() error { c.stop(); c.close(); return nil }
func (c *reverseDialConn) CloseWrite() error {
	if half, ok := c.Conn.(interface{ CloseWrite() error }); ok {
		return half.CloseWrite()
	}
	return c.Close()
}
