// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package secretstore stores host-local credentials in the current user's OS store.
package secretstore

import (
	"bytes"
	"context"
	"errors"
	"sync"
	"unicode/utf8"
)

var (
	ErrInvalidName   = errors.New("invalid secret name")
	ErrInvalidSecret = errors.New("invalid secret value")
	ErrNotFound      = errors.New("secret not found")
	ErrUnavailable   = errors.New("secret store unavailable")
)

const servicePrefix = "com.nvidia.pair.secretstore."
const maxSecretBytes = 512

type backend interface {
	set(context.Context, string, string, []byte) error
	get(context.Context, string, string) ([]byte, error)
	delete(context.Context, string, string) error
}

// Store serializes its operations through native write completion. It holds no
// credential cache. A Store must not be copied after first use; its zero value is
// usable. Cross-process writers must be coordinated by the owning service.
type Store struct {
	mu      sync.Mutex
	backend backend
	open    func() (backend, error)
}

// New does not contact the OS store. Failed initialization is retried by the
// next operation, but an operation with an unknown durable outcome is not retried.
func New() *Store { return &Store{open: newNative} }

func validName(s string) bool {
	if len(s) == 0 || len(s) > 64 || s[0] < 'a' || s[0] > 'z' {
		return false
	}
	for _, c := range []byte(s[1:]) {
		if (c < 'a' || c > 'z') && (c < '0' || c > '9') && c != '-' {
			return false
		}
	}
	return true
}

func validSecret(v []byte) bool {
	return len(v) > 0 && len(v) <= maxSecretBytes && utf8.Valid(v)
}

func (s *Store) prepare(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if s.backend == nil {
		open := s.open
		if open == nil {
			open = newNative
		}
		b, err := open()
		if err != nil || b == nil {
			return ErrUnavailable
		}
		s.backend = b
	}
	return ctx.Err()
}

// Set replaces a credential. Values must be nonempty UTF-8 of at most 512
// bytes. The input is copied; callers must drop their own references after use.
func (s *Store) Set(ctx context.Context, purpose, id string, value []byte) error {
	if !validName(purpose) || !validName(id) {
		return ErrInvalidName
	}
	if !validSecret(value) {
		return ErrInvalidSecret
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	value = bytes.Clone(value)
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.prepare(ctx); err != nil {
		return err
	}
	return sanitize(s.backend.set(ctx, servicePrefix+purpose, id, value))
}

// Get returns an independent copy. Malformed stored values are unavailable.
func (s *Store) Get(ctx context.Context, purpose, id string) ([]byte, error) {
	if !validName(purpose) || !validName(id) {
		return nil, ErrInvalidName
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.prepare(ctx); err != nil {
		return nil, err
	}
	value, err := s.backend.get(ctx, servicePrefix+purpose, id)
	if err != nil {
		if err == ErrNotFound {
			return nil, ErrNotFound
		}
		return nil, sanitize(err)
	}
	if !validSecret(value) {
		return nil, ErrUnavailable
	}
	return bytes.Clone(value), nil
}

// Delete succeeds if the exact entry is already absent. Clear uses Delete,
// rather than storing an empty value.
func (s *Store) Delete(ctx context.Context, purpose, id string) error {
	if !validName(purpose) || !validName(id) {
		return ErrInvalidName
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.prepare(ctx); err != nil {
		return err
	}
	err := s.backend.delete(ctx, servicePrefix+purpose, id)
	if err == ErrNotFound {
		return nil
	}
	return sanitize(err)
}

func sanitize(err error) error {
	switch err {
	case nil, context.Canceled, context.DeadlineExceeded:
		return err
	default:
		return ErrUnavailable
	}
}
