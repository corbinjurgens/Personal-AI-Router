// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package secretstore

import (
	"bytes"
	"context"
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

type fakeStore struct {
	values map[string][]byte
	err    error
	calls  int
}

func (f *fakeStore) set(_ context.Context, service, account string, value []byte) error {
	f.calls++
	if f.err != nil {
		return f.err
	}
	f.values[service+"/"+account] = value
	return nil
}
func (f *fakeStore) get(_ context.Context, service, account string) ([]byte, error) {
	f.calls++
	if f.err != nil {
		return nil, f.err
	}
	value, ok := f.values[service+"/"+account]
	if !ok {
		return nil, ErrNotFound
	}
	return value, nil
}
func (f *fakeStore) delete(_ context.Context, service, account string) error {
	f.calls++
	if f.err != nil {
		return f.err
	}
	if _, ok := f.values[service+"/"+account]; !ok {
		return ErrNotFound
	}
	delete(f.values, service+"/"+account)
	return nil
}
func fakeStoreForTest() (*Store, *fakeStore) {
	f := &fakeStore{values: make(map[string][]byte)}
	return &Store{backend: f}, f
}
func wantError(t *testing.T, got, want error) {
	t.Helper()
	if !errors.Is(got, want) {
		t.Fatalf("error = %v, want %v", got, want)
	}
}
func wantValue(t *testing.T, got, want []byte, err error) {
	t.Helper()
	if err != nil {
		t.Fatalf("read failed: %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatal("credential mismatch (values omitted)")
	}
}

func TestStoreValidatesNamesBeforeInitialization(t *testing.T) {
	for _, name := range []string{"", "A", "1a", "-a", "a_", "a/b", "a.b", "a b", "a\x00", "é", strings.Repeat("a", 65)} {
		t.Run("rejected name "+strings.ReplaceAll(name, "\x00", "nul"), func(t *testing.T) {
			s := &Store{open: func() (backend, error) { t.Fatal("invalid input accessed store"); return nil, nil }}
			for _, pair := range [][2]string{{name, "id"}, {"purpose", name}} {
				wantError(t, s.Set(t.Context(), pair[0], pair[1], []byte("fake")), ErrInvalidName)
				_, err := s.Get(t.Context(), pair[0], pair[1])
				wantError(t, err, ErrInvalidName)
				wantError(t, s.Delete(t.Context(), pair[0], pair[1]), ErrInvalidName)
			}
		})
	}
}

func TestStoreValidatesSecretBoundaries(t *testing.T) {
	test := func(name string, value []byte, want error) {
		t.Run(name, func(t *testing.T) {
			s, f := fakeStoreForTest()
			wantError(t, s.Set(t.Context(), "engine-api-key", "engine", value), want)
			if want != nil && f.calls != 0 {
				t.Fatal("invalid value accessed store")
			}
		})
	}
	test("nil", nil, ErrInvalidSecret)
	test("empty", []byte{}, ErrInvalidSecret)
	test("invalid UTF-8", []byte{0xff}, ErrInvalidSecret)
	test("oversized bytes", bytes.Repeat([]byte("x"), 513), ErrInvalidSecret)
	test("multibyte boundary", bytes.Repeat([]byte("é"), 256), nil)
	test("multibyte overflow", bytes.Repeat([]byte("é"), 257), ErrInvalidSecret)
	test("one byte", []byte("x"), nil)
	test("UTF-8 NUL allowed", []byte{0}, nil)
	s, _ := fakeStoreForTest()
	wantError(t, s.Set(t.Context(), "a"+strings.Repeat("-", 63), "z"+strings.Repeat("9", 63), []byte("fake")), nil)
}

func TestStoreCopiesCallerAndReturnedBuffers(t *testing.T) {
	s, _ := fakeStoreForTest()
	value := []byte("fake-value")
	wantError(t, s.Set(t.Context(), "purpose", "id", value), nil)
	value[0] = 'X'
	got, err := s.Get(t.Context(), "purpose", "id")
	wantValue(t, got, []byte("fake-value"), err)
	got[0] = 'Y'
	got, err = s.Get(t.Context(), "purpose", "id")
	wantValue(t, got, []byte("fake-value"), err)
}

func TestStoreSeparatesPurposeAndAccount(t *testing.T) {
	s, f := fakeStoreForTest()
	for _, pair := range [][2]string{{"alpha", "one"}, {"alpha", "two"}, {"beta", "one"}} {
		wantError(t, s.Set(t.Context(), pair[0], pair[1], []byte("fake")), nil)
		if _, ok := f.values[servicePrefix+pair[0]+"/"+pair[1]]; !ok {
			t.Fatal("wrong namespace or account")
		}
	}
	if len(f.values) != 3 {
		t.Fatal("names collided")
	}
}

func TestStoreOnlyExplicitMissingIsAbsent(t *testing.T) {
	s, f := fakeStoreForTest()
	_, err := s.Get(t.Context(), "purpose", "id")
	wantError(t, err, ErrNotFound)
	wantError(t, s.Delete(t.Context(), "purpose", "id"), nil)
	for _, nativeErr := range []error{ErrUnavailable, errors.New("native diagnostic"), errors.New("secret not found"), wrappedMissing{}} {
		f.err = nativeErr
		_, err = s.Get(t.Context(), "purpose", "id")
		wantError(t, err, ErrUnavailable)
		wantError(t, s.Delete(t.Context(), "purpose", "id"), ErrUnavailable)
		wantError(t, s.Set(t.Context(), "purpose", "id", []byte("fake")), ErrUnavailable)
	}
}

// Even a native error wrapping a missing marker is not the explicit result.
type wrappedMissing struct{}

func (wrappedMissing) Error() string { return "native diagnostic" }
func (wrappedMissing) Unwrap() error { return ErrNotFound }

func TestStoreRejectsMalformedStoredValues(t *testing.T) {
	for _, value := range [][]byte{nil, {}, {0xff}, bytes.Repeat([]byte("x"), 513)} {
		s, f := fakeStoreForTest()
		f.values[servicePrefix+"purpose/id"] = value
		got, err := s.Get(t.Context(), "purpose", "id")
		wantError(t, err, ErrUnavailable)
		if got != nil {
			t.Fatal("malformed value returned")
		}
	}
}

func TestStoreFailedMutationsRetainDurableValue(t *testing.T) {
	for _, operation := range []string{"replace", "delete"} {
		t.Run(operation, func(t *testing.T) {
			s, f := fakeStoreForTest()
			wantError(t, s.Set(t.Context(), "purpose", "id", []byte("previous-fake")), nil)
			f.err = errors.New("native diagnostic")
			if operation == "replace" {
				wantError(t, s.Set(t.Context(), "purpose", "id", []byte("next-fake")), ErrUnavailable)
			} else {
				wantError(t, s.Delete(t.Context(), "purpose", "id"), ErrUnavailable)
			}
			f.err = nil
			got, err := s.Get(t.Context(), "purpose", "id")
			wantValue(t, got, []byte("previous-fake"), err)
			if f.calls != 3 {
				t.Fatal("mutation was retried")
			}
		})
	}
}

func TestStoreLazyInitializationRetriesFailure(t *testing.T) {
	s := New()
	if s.backend != nil {
		t.Fatal("New initialized native store")
	}
	f := &fakeStore{values: make(map[string][]byte)}
	calls := 0
	s.open = func() (backend, error) {
		calls++
		if calls == 1 {
			return nil, errors.New("native startup diagnostic")
		}
		return f, nil
	}
	wantError(t, s.Set(t.Context(), "purpose", "id", []byte("fake")), ErrUnavailable)
	wantError(t, s.Set(t.Context(), "purpose", "id", []byte("fake")), nil)
	wantError(t, s.Delete(t.Context(), "purpose", "id"), nil)
	if calls != 2 {
		t.Fatal("initialization did not retry once and retain successful adapter")
	}
}

func TestStoreCanceledContextDoesNotInitialize(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	s := &Store{open: func() (backend, error) { t.Fatal("canceled operation initialized adapter"); return nil, nil }}
	wantError(t, s.Set(ctx, "purpose", "id", []byte("fake")), context.Canceled)
	_, err := s.Get(ctx, "purpose", "id")
	wantError(t, err, context.Canceled)
	wantError(t, s.Delete(ctx, "purpose", "id"), context.Canceled)
}

type blockingStore struct {
	entered chan struct{}
	release chan struct{}
	active  atomic.Int32
	overlap atomic.Bool
}

func (b *blockingStore) set(context.Context, string, string, []byte) error {
	if b.active.Add(1) != 1 {
		b.overlap.Store(true)
	}
	b.entered <- struct{}{}
	<-b.release
	b.active.Add(-1)
	return nil
}
func (b *blockingStore) get(context.Context, string, string) ([]byte, error) {
	if b.active.Add(1) != 1 {
		b.overlap.Store(true)
	}
	b.active.Add(-1)
	return []byte("fake"), nil
}
func (b *blockingStore) delete(context.Context, string, string) error {
	if b.active.Add(1) != 1 {
		b.overlap.Store(true)
	}
	b.active.Add(-1)
	return nil
}
func TestStoreSerializesThroughDurableCompletion(t *testing.T) {
	b := &blockingStore{entered: make(chan struct{}), release: make(chan struct{})}
	s := &Store{backend: b}
	var wg sync.WaitGroup
	wg.Add(1)
	go func() { defer wg.Done(); wantError(t, s.Set(t.Context(), "purpose", "id", []byte("fake")), nil) }()
	<-b.entered
	for range 20 {
		wg.Add(2)
		go func() { defer wg.Done(); _, err := s.Get(t.Context(), "purpose", "id"); wantError(t, err, nil) }()
		go func() { defer wg.Done(); wantError(t, s.Delete(t.Context(), "purpose", "id"), nil) }()
	}
	close(b.release)
	wg.Wait()
	if b.overlap.Load() {
		t.Fatal("operations overlapped native write completion")
	}
}

func TestStoreSentinelTexts(t *testing.T) {
	for _, test := range []struct {
		err  error
		text string
	}{
		{ErrInvalidName, "invalid secret name"}, {ErrInvalidSecret, "invalid secret value"},
		{ErrNotFound, "secret not found"}, {ErrUnavailable, "secret store unavailable"},
	} {
		if test.err.Error() != test.text {
			t.Fatal("public sentinel text changed")
		}
	}
}
