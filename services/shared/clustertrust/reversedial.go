// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package clustertrust

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
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

// ReverseRefusal is the one-byte reply to a private dial request: zero admits
// the stream, any other value refuses it and names why. The cluster-manager
// that writes it and every DialReverse caller that reads it are the same
// release, so there is no compatibility reading of other values.
type ReverseRefusal byte

const (
	ReverseAdmitted         ReverseRefusal = 0
	ReverseRefusedNoSession ReverseRefusal = 1 // no current reverse session to the peer
	ReverseRefusedCapacity  ReverseRefusal = 2 // call, session or stream limit reached
	ReverseRefusedAdmission ReverseRefusal = 3 // the peer is not admitted to this cluster
	ReverseRefusedRemote    ReverseRefusal = 4 // the peer's service refused the stream
	ReverseRefusedFailed    ReverseRefusal = 5 // opening the stream failed for another reason
)

// ReverseRefusalFor maps the reverse manager's failure to the refusal byte.
func ReverseRefusalFor(err error) ReverseRefusal {
	switch {
	case errors.Is(err, ErrReverseNoSession):
		return ReverseRefusedNoSession
	case errors.Is(err, ErrReverseCapacity):
		return ReverseRefusedCapacity
	case errors.Is(err, ErrReverseAdmission):
		return ReverseRefusedAdmission
	case errors.Is(err, ErrReverseRemoteRefused):
		return ReverseRefusedRemote
	}
	return ReverseRefusedFailed
}

func (r ReverseRefusal) cause() error {
	switch r {
	case ReverseRefusedNoSession:
		return ErrReverseNoSession
	case ReverseRefusedCapacity:
		return ErrReverseCapacity
	case ReverseRefusedAdmission:
		return ErrReverseAdmission
	case ReverseRefusedRemote:
		return ErrReverseRemoteRefused
	case ReverseRefusedFailed:
		return errors.New("reverse stream open failed")
	}
	return fmt.Errorf("unrecognized refusal code %d", byte(r))
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
	fail := func(err error) (net.Conn, error) { return nil, &ReverseUnavailableError{peerUUID, RouteReverse, err} }
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
			if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
				// The listener drops a connection without a reply when its
				// connection slots are full, and when it shuts down.
				return fmt.Errorf("private service closed the connection without a reply: %w", err)
			}
			return err
		}
		if refusal := ReverseRefusal(reply[0]); refusal != ReverseAdmitted {
			return fmt.Errorf("private service dial refused: %w", refusal.cause())
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
