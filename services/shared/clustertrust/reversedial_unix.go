// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !windows

package clustertrust

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"sync"
	"syscall"

	"nvpair-shared/ipc"
)

func dialReversePrivate(ctx context.Context, path string) (net.Conn, error) {
	return (&net.Dialer{}).DialContext(ctx, "unix", path)
}

// ListenReversePrivate binds the owner-only service gateway before goroutines.
func ListenReversePrivate(path string) (net.Listener, error) {
	return listenReversePrivate(path, nil)
}

func listenReversePrivate(path string, checkpoint func(string)) (net.Listener, error) {
	path = filepath.Clean(path)
	check := func(stage string) {
		if checkpoint != nil {
			checkpoint(stage)
		}
	}
	if len(path) >= len(syscall.RawSockaddrUnix{}.Path) {
		return nil, &net.OpError{Op: "listen", Net: "unix", Addr: &net.UnixAddr{Name: path, Net: "unix"}, Err: syscall.EINVAL}
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return nil, fmt.Errorf("create reverse endpoint directory: %w", err)
	}
	lock, err := os.OpenFile(path+".lock", os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, fmt.Errorf("open reverse endpoint lock: %w", err)
	}
	fail := func(err error) (net.Listener, error) {
		_ = lock.Close()
		return nil, err
	}
	// Only the lifetime lock owner may remove a pathname left by a crash.
	// Keep the lock file itself: unlinking it would allow competing lock inodes.
	if err := syscall.Flock(int(lock.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		return fail(fmt.Errorf("lock reverse endpoint: %w", err))
	}
	var previous reversePrivateRecord
	recordErr := json.NewDecoder(io.LimitReader(lock, 256)).Decode(&previous)
	identity, err := reversePrivateSocketIdentity(path)
	if err == nil {
		// A refused connection can also mean a live listener has a full backlog.
		// Reclaim only the exact socket recorded by a former lock owner.
		if recordErr != nil {
			return fail(fmt.Errorf("reverse endpoint has no valid ownership record: %w", recordErr))
		}
		if previous.reversePrivateIdentity != identity {
			return fail(fmt.Errorf("reverse endpoint ownership record does not match socket"))
		}
		if err := os.Remove(path); err != nil {
			return fail(fmt.Errorf("remove stale reverse endpoint: %w", err))
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return fail(err)
	}
	if recordErr == nil && previous.Stage != "" {
		if filepath.Base(previous.Stage) != previous.Stage || previous.Stage == "." || previous.Stage == ".." || previous.Stage == filepath.Base(path) {
			return fail(fmt.Errorf("invalid reverse endpoint staging record"))
		}
		if _, err := reversePrivateRemoveOwned(filepath.Join(filepath.Dir(path), previous.Stage), previous.reversePrivateIdentity); err != nil {
			return fail(fmt.Errorf("remove stale reverse staging socket: %w", err))
		}
	}
	check("before-bind")
	stage, listener, err := reversePrivateBindStage(path)
	if err != nil {
		return fail(err)
	}
	listener.(*net.UnixListener).SetUnlinkOnClose(false)
	check("bound")
	identity, err = reversePrivateSocketIdentity(stage)
	if err != nil {
		_ = listener.Close()
		return fail(err)
	}
	owned := &reversePrivateListener{Listener: listener, lock: lock, path: path, stage: stage, identity: identity}
	failOwned := func(err error) (net.Listener, error) {
		return nil, errors.Join(err, owned.Close())
	}
	if err := lock.Truncate(0); err != nil {
		return failOwned(fmt.Errorf("clear reverse endpoint ownership record: %w", err))
	}
	if _, err := lock.Seek(0, io.SeekStart); err != nil {
		return failOwned(err)
	}
	check("recording")
	record := reversePrivateRecord{reversePrivateIdentity: identity, Stage: filepath.Base(stage)}
	if err := json.NewEncoder(lock).Encode(record); err != nil {
		return failOwned(fmt.Errorf("record reverse endpoint ownership: %w", err))
	}
	if err := lock.Sync(); err != nil {
		return failOwned(fmt.Errorf("sync reverse endpoint ownership: %w", err))
	}
	check("recorded")
	// Linking is create-only: a concurrently appearing canonical socket wins.
	// Its pathname is never visible before this inode's record has been synced.
	if err := os.Link(stage, path); err != nil {
		return failOwned(fmt.Errorf("publish reverse endpoint: %w", err))
	}
	check("published")
	if _, err := reversePrivateRemoveOwned(stage, identity); err != nil {
		return failOwned(fmt.Errorf("remove published reverse staging socket: %w", err))
	}
	return owned, nil
}

func reversePrivateBindStage(path string) (string, net.Listener, error) {
	// A staging basename no longer than the canonical basename preserves every
	// supported canonical Unix pathname length, including the platform limit.
	nameLen := min(len(filepath.Base(path)), 12)
	for attempt := 0; attempt < 64; attempt++ {
		var random [6]byte
		if _, err := rand.Read(random[:]); err != nil {
			return "", nil, err
		}
		stage := filepath.Join(filepath.Dir(path), hex.EncodeToString(random[:])[:nameLen])
		if stage == path {
			continue
		}
		listener, err := ipc.ListenPrivate(stage)
		if errors.Is(err, syscall.EADDRINUSE) {
			continue
		}
		return stage, listener, err
	}
	return "", nil, fmt.Errorf("no unused reverse staging socket pathname")
}

type reversePrivateIdentity struct {
	Version int    `json:"version"`
	Device  uint64 `json:"device"`
	Inode   uint64 `json:"inode"`
}

type reversePrivateRecord struct {
	reversePrivateIdentity
	Stage string `json:"stage,omitempty"`
}

func reversePrivateSocketIdentity(path string) (reversePrivateIdentity, error) {
	info, err := os.Lstat(path)
	if err != nil {
		return reversePrivateIdentity{}, err
	}
	stat, ok := info.Sys().(*syscall.Stat_t)
	if !ok || info.Mode()&os.ModeSocket == 0 {
		return reversePrivateIdentity{}, fmt.Errorf("reverse endpoint is not a Unix socket")
	}
	return reversePrivateIdentity{Version: 1, Device: uint64(stat.Dev), Inode: uint64(stat.Ino)}, nil
}

func reversePrivateRemoveOwned(path string, identity reversePrivateIdentity) (bool, error) {
	current, err := reversePrivateSocketIdentity(path)
	if errors.Is(err, os.ErrNotExist) {
		return true, nil
	}
	if err != nil {
		return false, err
	}
	if current != identity {
		return false, nil
	}
	if err := os.Remove(path); err != nil {
		return false, err
	}
	_, err = os.Lstat(path)
	if errors.Is(err, os.ErrNotExist) {
		return true, nil
	}
	return false, err
}

type reversePrivateListener struct {
	net.Listener
	lock      *os.File
	path      string
	stage     string
	identity  reversePrivateIdentity
	closeOnce sync.Once
	closeErr  error
}

func (l *reversePrivateListener) Addr() net.Addr {
	return &net.UnixAddr{Name: l.path, Net: "unix"}
}

func (l *reversePrivateListener) Close() error {
	l.closeOnce.Do(func() {
		closeErr := l.Listener.Close()
		canonicalAbsent, canonicalErr := reversePrivateRemoveOwned(l.path, l.identity)
		stageAbsent, stageErr := reversePrivateRemoveOwned(l.stage, l.identity)
		var recordErr error
		if canonicalAbsent && stageAbsent {
			recordErr = errors.Join(l.lock.Truncate(0), l.lock.Sync())
		}
		l.closeErr = errors.Join(closeErr, canonicalErr, stageErr, recordErr, l.lock.Close())
	})
	return l.closeErr
}
