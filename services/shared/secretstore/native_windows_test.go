// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build windows

package secretstore

import (
	"context"
	"errors"
	"testing"

	"github.com/danieljoos/wincred"
)

func TestWindowsWritesExactTargetWithLocalPersistence(t *testing.T) {
	w := &windowsAPI{write: func(c *wincred.GenericCredential) error {
		if c.TargetName != servicePrefix+"purpose/id" || c.UserName != "id" || c.Persist != wincred.PersistLocalMachine {
			t.Fatal("wrong target, username or persistence")
		}
		wantValue(t, c.CredentialBlob, []byte("fake"), nil)
		return nil
	}}
	wantError(t, w.set(t.Context(), servicePrefix+"purpose", "id", []byte("fake")), nil)
}

func TestWindowsOnlyExplicitMissingIsAbsent(t *testing.T) {
	for _, test := range []struct {
		name              string
		native, errorWant error
	}{
		{"absent", wincred.ErrElementNotFound, ErrNotFound},
		{"unavailable", errors.New("native diagnostic"), ErrUnavailable},
		{"invalid parameter", wincred.ErrInvalidParameter, ErrUnavailable},
	} {
		t.Run(test.name, func(t *testing.T) {
			w := &windowsAPI{
				read: func(target string) (*wincred.GenericCredential, error) {
					if target != servicePrefix+"purpose/id" {
						t.Fatal("wrong read target")
					}
					return nil, test.native
				},
				remove: func(c *wincred.GenericCredential) error {
					if c.TargetName != servicePrefix+"purpose/id" {
						t.Fatal("wrong delete target")
					}
					return test.native
				},
				write: func(*wincred.GenericCredential) error { return test.native },
			}
			_, err := w.get(t.Context(), servicePrefix+"purpose", "id")
			wantError(t, err, test.errorWant)
			wantError(t, w.delete(t.Context(), servicePrefix+"purpose", "id"), test.errorWant)
			wantError(t, w.set(t.Context(), servicePrefix+"purpose", "id", []byte("fake")), ErrUnavailable)
		})
	}
}

func TestWindowsRejectsMalformedRead(t *testing.T) {
	w := &windowsAPI{read: func(string) (*wincred.GenericCredential, error) { return nil, nil }}
	_, err := w.get(t.Context(), "service", "account")
	wantError(t, err, ErrUnavailable)
}

func TestWindowsSynchronousWriteIsNotRecanceled(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	calls := 0
	w := &windowsAPI{write: func(*wincred.GenericCredential) error { calls++; cancel(); return nil }}
	s := &Store{backend: w}
	wantError(t, s.Set(ctx, "purpose", "id", []byte("fake")), nil)
	if calls != 1 {
		t.Fatal("durable write repeated")
	}
}

func TestWindowsCanceledOperationMakesNoNativeCalls(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	w := &windowsAPI{}
	wantError(t, w.set(ctx, "service", "account", []byte("fake")), context.Canceled)
	_, err := w.get(ctx, "service", "account")
	wantError(t, err, context.Canceled)
	wantError(t, w.delete(ctx, "service", "account"), context.Canceled)
}
