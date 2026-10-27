// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"net/http"
	"strconv"
	"strings"
	"sync"

	"nvpair-shared/engines"
	"nvpair-shared/secretstore"
)

const credentialPurpose = "engine-api-key"

type credentialBackend interface {
	Set(context.Context, string, string, []byte) error
	Get(context.Context, string, string) ([]byte, error)
	Delete(context.Context, string, string) error
}

var newCredentialBackend = func() credentialBackend { return secretstore.New() }

var errCredentialUnavailable = errors.New("Credential store is unavailable. Unlock the operating system credential store and try again.")
var errInvalidCredential = errors.New("Invalid engine or credential.")

type credentialEntry struct {
	key     string
	known   bool
	refresh bool
}

type credentialStore struct {
	mu      sync.Mutex
	backend credentialBackend
	factory func() credentialBackend
	entries map[string]credentialEntry
	closed  bool
}

// Native construction and all I/O happen under the credential lock, never st.mu.
func (s *credentialStore) backendLocked() credentialBackend {
	if s.backend == nil && s.factory != nil {
		s.backend = s.factory()
	}
	return s.backend
}

func (s *credentialStore) close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.closed = true
	s.entries = nil
	s.backend = nil
	s.factory = nil
}

func validCredentialEngine(id string) bool {
	if len(id) == 0 || len(id) > 64 || id[0] < 'a' || id[0] > 'z' {
		return false
	}
	for i := 1; i < len(id); i++ {
		c := id[i]
		if (c < 'a' || c > 'z') && (c < '0' || c > '9') && c != '-' {
			return false
		}
	}
	return true
}

func (s *credentialStore) get(ctx context.Context, id string) (string, error) {
	if !validCredentialEngine(id) {
		return "", errInvalidCredential
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return "", errCredentialUnavailable
	}
	entry := s.entries[id]
	if entry.known && !entry.refresh {
		return entry.key, nil
	}
	backend := s.backendLocked()
	if backend == nil {
		return "", errCredentialUnavailable
	}
	value, err := backend.Get(ctx, credentialPurpose, id)
	if errors.Is(err, secretstore.ErrNotFound) {
		entry = credentialEntry{known: true}
	} else if err != nil || len(value) == 0 || !engines.ValidCredential(string(value)) {
		return "", errCredentialUnavailable
	} else {
		entry = credentialEntry{key: string(value), known: true}
	}
	if s.entries == nil {
		s.entries = make(map[string]credentialEntry)
	}
	s.entries[id] = entry
	return entry.key, nil
}

func (s *credentialStore) set(ctx context.Context, id, key string) error {
	if !validCredentialEngine(id) || !engines.ValidCredential(key) {
		return errInvalidCredential
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return errCredentialUnavailable
	}
	backend := s.backendLocked()
	if backend == nil {
		return errCredentialUnavailable
	}
	var err error
	if key == "" {
		err = backend.Delete(ctx, credentialPurpose, id)
	} else {
		err = backend.Set(ctx, credentialPurpose, id, []byte(key))
	}
	if s.entries == nil {
		s.entries = make(map[string]credentialEntry)
	}
	if err != nil {
		entry := s.entries[id]
		entry.refresh = true
		s.entries[id] = entry
		return errCredentialUnavailable
	}
	s.entries[id] = credentialEntry{key: key, known: true}
	return nil
}

type candidateCredential struct {
	key  string
	port int
}

func (e *Executor) credentialForCandidate(ctx context.Context, st *engineState, effectivePort int) (candidateCredential, error) {
	c := candidateCredential{port: effectivePort}
	st.mu.Lock()
	rt := st.plat.Runtime
	owned := st.proc != nil && st.port == effectivePort
	id := st.manifest.Engine
	if owned && rt.servingLaunch != nil {
		c.key = rt.servingLaunch.Credential
	}
	st.mu.Unlock()
	if c.key == "" && rt.UserCredential {
		key, err := e.credentials.get(ctx, id)
		if err != nil {
			return candidateCredential{}, err
		}
		c.key = key
	}
	// Unauthenticated requests retain their existing port and transport
	// semantics. Endpoint eligibility matters only when attaching a key.
	if c.key != "" && (effectivePort < 1 || effectivePort > 65535 || !engines.ValidCredential(c.key)) {
		return candidateCredential{}, errCredentialRequest
	}
	return c, nil
}

// Native I/O can outlive a stop or a new adoption on the same port. Running
// users must still refer to the run and owned preparation they observed before
// selection. Cache mutations alone do not invalidate an in-flight snapshot.
func (e *Executor) credentialForRunning(ctx context.Context, st *engineState) (candidateCredential, bool, error) {
	st.mu.Lock()
	running, port, runID := st.running, st.port, st.runID
	proc, launch, userCredential := st.proc, st.plat.Runtime.servingLaunch, st.plat.Runtime.UserCredential
	st.mu.Unlock()
	if !running {
		return candidateCredential{}, false, nil
	}
	c, err := e.credentialForCandidate(ctx, st, port)
	if err != nil {
		return candidateCredential{}, false, err
	}
	st.mu.Lock()
	eligible := st.running && st.port == port && st.runID == runID && st.proc == proc && st.plat.Runtime.servingLaunch == launch && st.plat.Runtime.UserCredential == userCredential
	st.mu.Unlock()
	if !eligible {
		return candidateCredential{}, false, nil
	}
	return c, true, nil
}

func (e *Executor) credentialState(ctx context.Context, engine string) (engines.CredentialState, error) {
	if !validCredentialEngine(engine) {
		return engines.CredentialState{}, errInvalidCredential
	}
	st, err := e.state(engine)
	if err != nil {
		if _, known := e.reg.Get(engine); known {
			return engines.CredentialState{}, nil
		}
		return engines.CredentialState{}, errInvalidCredential
	}
	st.mu.Lock()
	supported := st.plat.Runtime.UserCredential
	st.mu.Unlock()
	if !supported {
		return engines.CredentialState{}, nil
	}
	key, err := e.credentials.get(ctx, engine)
	if err != nil {
		return engines.CredentialState{}, err
	}
	return engines.CredentialState{Supported: true, Configured: key != ""}, nil
}

func (e *Executor) setCredential(ctx context.Context, engine, key string) (engines.CredentialState, error) {
	if !validCredentialEngine(engine) || !engines.ValidCredential(key) {
		return engines.CredentialState{}, errInvalidCredential
	}
	st, err := e.state(engine)
	if err != nil {
		return engines.CredentialState{}, errInvalidCredential
	}
	st.mu.Lock()
	supported := st.plat.Runtime.UserCredential
	st.mu.Unlock()
	if !supported {
		return engines.CredentialState{}, errInvalidCredential
	}
	if err := e.credentials.set(ctx, engine, key); err != nil {
		return engines.CredentialState{}, err
	}
	return engines.CredentialState{Supported: true, Configured: key != ""}, nil
}

var errCredentialRequest = errors.New("Cannot send the engine API key: invalid local endpoint or HTTP transport.")
var errCredentialHTTP = errors.New("Authenticated engine HTTP request failed.")

// Bind a key to the initial numeric loopback endpoint. Clone the transport so
// proxy and redirect policy cannot forward it elsewhere, including via env.
func credentialHTTPClient(client *http.Client, req *http.Request, c candidateCredential) (*http.Client, error) {
	if c.key == "" {
		return client, nil
	}
	u := req.URL
	port, err := strconv.Atoi(u.Port())
	if !engines.ValidCredential(c.key) || c.port < 1 || c.port > 65535 || err != nil || port != c.port || u.Scheme != "http" || u.Hostname() != "127.0.0.1" || u.User != nil || u.Fragment != "" || u.Opaque != "" {
		return nil, errCredentialRequest
	}
	transport := client.Transport
	if transport == nil {
		transport = http.DefaultTransport
	}
	tr, ok := transport.(*http.Transport)
	if !ok {
		return nil, errCredentialRequest
	}
	copyClient := *client
	copyTransport := tr.Clone()
	copyTransport.Proxy = nil
	// Do not keep a fresh clone's idle pool alive beyond this operation.
	copyTransport.DisableKeepAlives = true
	copyClient.Transport = copyTransport
	copyClient.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	req.Header.Set("Authorization", "Bearer "+c.key)
	return &copyClient, nil
}

func redactCredentialJSON(value any, key string) any {
	switch v := value.(type) {
	case string:
		return strings.ReplaceAll(v, key, "[redacted]")
	case []any:
		for i := range v {
			v[i] = redactCredentialJSON(v[i], key)
		}
	case map[string]any:
		clean := make(map[string]any, len(v))
		for name, field := range v {
			clean[strings.ReplaceAll(name, key, "[redacted]")] = redactCredentialJSON(field, key)
		}
		return clean
	}
	return value
}
