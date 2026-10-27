// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"sync"
	"testing"

	"nvpair-shared/secretstore"
)

type credentialFake struct {
	lock                  sync.Mutex
	value                 string
	getErr                error
	writeErr              error
	gets                  int
	writes                int
	failedWriteKeepsValue bool
	purposes              []string
	ids                   []string
}

type blockingCredentialBackend struct {
	credentialFake
	entered     chan struct{}
	release     chan struct{}
	mu          sync.Mutex
	blockDelete bool
}

func (f *blockingCredentialBackend) Set(ctx context.Context, purpose, id string, v []byte) error {
	if !f.blockDelete {
		close(f.entered)
		<-f.release
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.credentialFake.Set(ctx, purpose, id, v)
}

func (f *blockingCredentialBackend) Delete(ctx context.Context, purpose, id string) error {
	if f.blockDelete {
		close(f.entered)
		<-f.release
	}
	return f.credentialFake.Delete(ctx, purpose, id)
}

func (f *blockingCredentialBackend) Get(ctx context.Context, purpose, id string) ([]byte, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.credentialFake.Get(ctx, purpose, id)
}

func TestCredentialStoreWaitsForDurableWrite(t *testing.T) {
	ctx := context.Background()
	f := &blockingCredentialBackend{credentialFake: credentialFake{value: "old-key"}, entered: make(chan struct{}), release: make(chan struct{})}
	s := credentialStore{backend: f}
	if _, err := s.get(ctx, "lmstudio"); err != nil {
		t.Fatal(err)
	}
	written := make(chan error, 1)
	go func() { written <- s.set(ctx, "lmstudio", "new-key") }()
	<-f.entered
	type result struct {
		key string
		err error
	}
	read := make(chan result, 1)
	go func() { key, err := s.get(ctx, "lmstudio"); read <- result{key, err} }()
	if s.mu.TryLock() {
		s.mu.Unlock()
		t.Fatal("write released credential mutex before durable completion")
	}
	select {
	case <-read:
		t.Fatal("read completed before durable write")
	default:
	}
	close(f.release)
	if err := <-written; err != nil {
		t.Fatal(err)
	}
	if got := <-read; got.err != nil || got.key != "new-key" {
		t.Fatal("read did not observe completed write")
	}
}

func TestCredentialStoreClearReconcilesAndRestartLoads(t *testing.T) {
	ctx := context.Background()
	f := &credentialFake{value: "old-key", writeErr: errors.New("transport failed")}
	s := credentialStore{backend: f}
	if _, err := s.get(ctx, "lmstudio"); err != nil {
		t.Fatal(err)
	}
	if err := s.set(ctx, "lmstudio", ""); !errors.Is(err, errCredentialUnavailable) {
		t.Fatal("ambiguous clear did not fail safely")
	}
	if key, err := s.get(ctx, "lmstudio"); err != nil || key != "" {
		t.Fatal("clear was not reconciled")
	}
	f.writeErr = nil
	if err := s.set(ctx, "lmstudio", "new-key"); err != nil {
		t.Fatal(err)
	}
	restarted := credentialStore{backend: f}
	if key, err := restarted.get(ctx, "lmstudio"); err != nil || key != "new-key" {
		t.Fatal("restart did not load durable value")
	}
}

func (f *credentialFake) Get(_ context.Context, purpose, id string) ([]byte, error) {
	f.lock.Lock()
	defer f.lock.Unlock()
	f.purposes = append(f.purposes, purpose)
	f.ids = append(f.ids, id)
	f.gets++
	if f.getErr != nil {
		return nil, f.getErr
	}
	if f.value == "" {
		return nil, secretstore.ErrNotFound
	}
	return []byte(f.value), nil
}
func (f *credentialFake) Set(_ context.Context, purpose, id string, v []byte) error {
	f.lock.Lock()
	defer f.lock.Unlock()
	f.purposes = append(f.purposes, purpose)
	f.ids = append(f.ids, id)
	f.writes++
	if f.writeErr == nil || !f.failedWriteKeepsValue {
		f.value = string(v)
	}
	return f.writeErr
}
func (f *credentialFake) Delete(_ context.Context, purpose, id string) error {
	f.lock.Lock()
	defer f.lock.Unlock()
	f.purposes = append(f.purposes, purpose)
	f.ids = append(f.ids, id)
	f.writes++
	if f.writeErr == nil || !f.failedWriteKeepsValue {
		f.value = ""
	}
	return f.writeErr
}

func TestCredentialStoreFailedMutationsRetainConfirmedCache(t *testing.T) {
	for _, replacement := range []string{"replacement", ""} {
		name := "replace"
		if replacement == "" {
			name = "clear"
		}
		t.Run(name, func(t *testing.T) {
			f := &credentialFake{value: "confirmed", failedWriteKeepsValue: true}
			s := credentialStore{backend: f}
			if _, err := s.get(t.Context(), "lmstudio"); err != nil {
				t.Fatal(err)
			}
			f.writeErr = errors.New("native diagnostic must not escape")
			if err := s.set(t.Context(), "lmstudio", replacement); err != errCredentialUnavailable {
				t.Fatal("mutation did not return fixed unavailable error")
			}
			entry := s.entries["lmstudio"]
			if !entry.known || !entry.refresh || entry.key != "confirmed" {
				t.Fatal("failed mutation changed confirmed cache")
			}
			f.getErr = errors.New("locked")
			if key, err := s.get(t.Context(), "lmstudio"); key != "" || err != errCredentialUnavailable {
				t.Fatal("failed refresh exposed stale key")
			}
			f.getErr = nil
			if key, err := s.get(t.Context(), "lmstudio"); err != nil || key != "confirmed" {
				t.Fatal("retry did not restore confirmed eligibility")
			}
		})
	}
}

func TestCredentialStoreUnknownFailureAndUnavailableRetry(t *testing.T) {
	f := &credentialFake{getErr: errors.New("locked"), writeErr: errors.New("lost reply")}
	s := credentialStore{backend: f}
	if err := s.set(t.Context(), "lmstudio", "new"); err != errCredentialUnavailable {
		t.Fatal("unknown mutation succeeded")
	}
	if s.entries["lmstudio"].known {
		t.Fatal("failed mutation claimed known cache")
	}
	if key, err := s.get(t.Context(), "lmstudio"); key != "" || err != errCredentialUnavailable {
		t.Fatal("unavailable load claimed absence")
	}
	f.getErr = nil
	if key, err := s.get(t.Context(), "lmstudio"); err != nil || key != "new" {
		t.Fatal("retry did not reconcile durable state")
	}
	if f.writes != 1 {
		t.Fatal("ambiguous write repeated")
	}
}

func TestCredentialStoreLazyLifetimeAndTeardown(t *testing.T) {
	f := &credentialFake{value: "confirmed"}
	created := 0
	s := credentialStore{factory: func() credentialBackend { created++; return f }}
	if created != 0 {
		t.Fatal("eager factory")
	}
	if key, err := s.get(t.Context(), "lmstudio"); err != nil || key != "confirmed" {
		t.Fatal("load failed")
	}
	f.value = "external-edit"
	if key, err := s.get(t.Context(), "lmstudio"); err != nil || key != "confirmed" || f.gets != 1 || created != 1 {
		t.Fatal("known cache did not last for executor lifetime")
	}
	s.close()
	if len(s.entries) != 0 || s.backend != nil || s.factory != nil {
		t.Fatal("teardown retained references")
	}
	if key, err := s.get(t.Context(), "lmstudio"); key != "" || err != errCredentialUnavailable {
		t.Fatal("teardown allowed a reload")
	}
	if err := s.set(t.Context(), "lmstudio", "new"); err != errCredentialUnavailable || f.writes != 0 {
		t.Fatal("teardown wrote native state")
	}
}

func TestCredentialStoreKnownAbsenceAndMalformedValue(t *testing.T) {
	f := &credentialFake{}
	s := credentialStore{backend: f}
	for range 2 {
		if key, err := s.get(t.Context(), "lmstudio"); err != nil || key != "" {
			t.Fatal("missing was not known absent")
		}
	}
	if f.gets != 1 {
		t.Fatal("known absence reloaded")
	}
	f.value = "invalid stored value"
	restarted := credentialStore{backend: f}
	if key, err := restarted.get(t.Context(), "lmstudio"); key != "" || err != errCredentialUnavailable {
		t.Fatal("malformed stored value accepted")
	}
}

func TestCredentialStoreSerializesSetThenClear(t *testing.T) {
	f := &blockingCredentialBackend{entered: make(chan struct{}), release: make(chan struct{})}
	s := credentialStore{backend: f}
	setDone, clearDone := make(chan error, 1), make(chan error, 1)
	go func() { setDone <- s.set(t.Context(), "lmstudio", "new") }()
	<-f.entered
	clearStarted := make(chan struct{})
	go func() { close(clearStarted); clearDone <- s.set(t.Context(), "lmstudio", "") }()
	<-clearStarted
	if s.mu.TryLock() {
		s.mu.Unlock()
		t.Fatal("set did not retain mutation lock")
	}
	close(f.release)
	if err := <-setDone; err != nil {
		t.Fatal(err)
	}
	if err := <-clearDone; err != nil {
		t.Fatal(err)
	}
	if key, err := s.get(t.Context(), "lmstudio"); err != nil || key != "" {
		t.Fatal("clear did not follow durable set")
	}
	if f.writes != 2 {
		t.Fatal("mutation count changed")
	}
}

func TestCredentialStoreSerializesClearThenSet(t *testing.T) {
	f := &blockingCredentialBackend{credentialFake: credentialFake{value: "confirmed"}, entered: make(chan struct{}), release: make(chan struct{}), blockDelete: true}
	s := credentialStore{backend: f}
	if _, err := s.get(t.Context(), "lmstudio"); err != nil {
		t.Fatal(err)
	}
	cleared, replaced := make(chan error, 1), make(chan error, 1)
	go func() { cleared <- s.set(t.Context(), "lmstudio", "") }()
	<-f.entered
	started := make(chan struct{})
	go func() { close(started); replaced <- s.set(t.Context(), "lmstudio", "replacement") }()
	<-started
	if s.mu.TryLock() {
		s.mu.Unlock()
		t.Fatal("clear released credential mutex before durable completion")
	}
	close(f.release)
	if err := <-cleared; err != nil {
		t.Fatal(err)
	}
	if err := <-replaced; err != nil {
		t.Fatal(err)
	}
	if key, err := s.get(t.Context(), "lmstudio"); err != nil || key != "replacement" || f.writes != 2 {
		t.Fatal("replace did not follow durable clear")
	}
}

func TestCredentialStorePurposeAndEngineIdentity(t *testing.T) {
	f := &credentialFake{}
	s := credentialStore{backend: f}
	if _, err := s.get(t.Context(), "lmstudio"); err != nil {
		t.Fatal(err)
	}
	if err := s.set(t.Context(), "lmstudio", "new"); err != nil {
		t.Fatal(err)
	}
	if err := s.set(t.Context(), "lmstudio", ""); err != nil {
		t.Fatal(err)
	}
	if len(f.purposes) != 3 || len(f.ids) != 3 {
		t.Fatal("native operation count changed")
	}
	for i, purpose := range f.purposes {
		if purpose != "engine-api-key" || f.ids[i] != "lmstudio" {
			t.Fatal("native namespace is not fixed")
		}
	}
}

func TestCredentialStoreNewExecutorUsesLazyPrivateFactory(t *testing.T) {
	previous := newCredentialBackend
	defer func() { newCredentialBackend = previous }()
	var stores []*credentialFake
	newCredentialBackend = func() credentialBackend { fake := &credentialFake{}; stores = append(stores, fake); return fake }
	m := testEngineManifest(fakeEngineBin)
	reg := NewRegistry()
	reg.engines[m.Engine] = m
	one := NewExecutor(reg, NewReporter(nil), func(string, any) {}, t.TempDir())
	two := NewExecutor(reg, NewReporter(nil), func(string, any) {}, t.TempDir())
	for _, ex := range []*Executor{one, two} {
		st, err := ex.state(m.Engine)
		if err != nil {
			t.Fatal(err)
		}
		st.plat.Runtime.UserCredential = true
	}
	if len(stores) != 0 {
		t.Fatal("executor eagerly constructed native adapter")
	}
	for _, ex := range []*Executor{one, two} {
		if _, err := ex.credentialState(t.Context(), m.Engine); err != nil {
			t.Fatal(err)
		}
	}
	if len(stores) != 2 || stores[0] == stores[1] {
		t.Fatal("executors did not own distinct stores")
	}
	if _, err := one.setCredential(t.Context(), m.Engine, "new"); err != nil {
		t.Fatal(err)
	}
	if state, err := two.credentialState(t.Context(), m.Engine); err != nil || state.Configured {
		t.Fatal("executor stores shared mutable state")
	}
}

func TestCredentialStoreAmbiguousWriteReconciles(t *testing.T) {
	ctx := context.Background()
	f := &credentialFake{value: "old-key"}
	s := credentialStore{backend: f}
	if got, err := s.get(ctx, "lmstudio"); err != nil || got != "old-key" {
		t.Fatal("initial read failed")
	}
	f.writeErr = errors.New("transport failed")
	if err := s.set(ctx, "lmstudio", "new-key"); !errors.Is(err, errCredentialUnavailable) {
		t.Fatal("ambiguous write did not fail safely")
	}
	f.getErr = errors.New("unavailable")
	if got, err := s.get(ctx, "lmstudio"); got != "" || !errors.Is(err, errCredentialUnavailable) {
		t.Fatal("failed refresh served a credential")
	}
	f.getErr = nil
	if got, err := s.get(ctx, "lmstudio"); err != nil || got != "new-key" {
		t.Fatal("refresh did not reconcile durable result")
	}
	if f.writes != 1 {
		t.Fatal("ambiguous write was repeated")
	}
}

func TestCredentialStoreValidationDoesNotCallBackend(t *testing.T) {
	f := &credentialFake{}
	s := credentialStore{backend: f}
	if err := s.set(context.Background(), "../invalid", "key"); !errors.Is(err, errInvalidCredential) {
		t.Fatal("invalid engine accepted")
	}
	if err := s.set(context.Background(), "lmstudio", "invalid key"); !errors.Is(err, errInvalidCredential) {
		t.Fatal("invalid credential accepted")
	}
	if _, err := s.get(context.Background(), "INVALID"); !errors.Is(err, errInvalidCredential) {
		t.Fatal("invalid engine accepted")
	}
	if f.gets != 0 || f.writes != 0 {
		t.Fatal("validation called backend")
	}
}
