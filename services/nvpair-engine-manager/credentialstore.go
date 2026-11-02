// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"net/http"
	"strconv"
	"strings"

	"nvpair-shared/engines"
)

type candidateCredential struct {
	key  string
	port int
}

// credentialForCandidate returns the manager-generated serving credential when
// this executor owns the process serving effectivePort. Other engines are
// reached without a key.
func (e *Executor) credentialForCandidate(_ context.Context, st *engineState, effectivePort int) (candidateCredential, error) {
	c := candidateCredential{port: effectivePort}
	st.mu.Lock()
	if st.proc != nil && st.port == effectivePort && st.plat.Runtime.servingLaunch != nil {
		c.key = st.plat.Runtime.servingLaunch.Credential
	}
	st.mu.Unlock()
	// Unauthenticated requests retain their existing port and transport
	// semantics. Endpoint eligibility matters only when attaching a key.
	if c.key != "" && (effectivePort < 1 || effectivePort > 65535 || !engines.ValidCredential(c.key)) {
		return candidateCredential{}, errCredentialRequest
	}
	return c, nil
}

func (e *Executor) credentialForRunning(ctx context.Context, st *engineState) (candidateCredential, bool, error) {
	st.mu.Lock()
	running, port, runID := st.running, st.port, st.runID
	proc, launch := st.proc, st.plat.Runtime.servingLaunch
	st.mu.Unlock()
	if !running {
		return candidateCredential{}, false, nil
	}
	c, err := e.credentialForCandidate(ctx, st, port)
	if err != nil {
		return candidateCredential{}, false, err
	}
	st.mu.Lock()
	eligible := st.running && st.port == port && st.runID == runID && st.proc == proc && st.plat.Runtime.servingLaunch == launch
	st.mu.Unlock()
	if !eligible {
		return candidateCredential{}, false, nil
	}
	return c, true, nil
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
