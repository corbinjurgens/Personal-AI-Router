// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package secretenv filters the reserved provider-secret namespace out of a
// child process environment. Only the client-to-service and service-to-broker
// launches carry the whole namespace; the broker-to-proxy launch retains the
// configured credential references and every other PAIR-owned child loses the
// namespace entirely.
//
// This is application launch filtering, not OS isolation: a child started by a
// credential-bearing parent through some other path still inherits it.
//
// Nothing here logs or returns environment values in errors.
package secretenv

import (
	"fmt"
	"runtime"
	"strings"

	"nvpair-shared/nodepolicy"
)

// Prefix is the reserved provider-secret namespace. Membership is decided
// case-insensitively on every platform, so a case variant of a reserved name
// is never forwarded by accident.
const Prefix = "NVPAIR_PROVIDER_SECRET_"

// Remove returns a copy of env, in "KEY=value" form, without any variable in
// the reserved namespace. Every other entry is kept in its original order.
func Remove(env []string) []string {
	out, _ := filter(env, nil, runtime.GOOS == "windows")
	return out
}

// Retain returns a copy of env without reserved-namespace variables other than
// refs. Every other entry is kept in its original order.
//
// A retained name matches refs exactly, or case-insensitively on Windows where
// environment names are case-insensitive. Retain fails when a ref is not a
// canonical credential reference, or when more than one entry matches the same
// ref, because the child would then see an ambiguous value.
func Retain(env []string, refs []nodepolicy.CredentialRef) ([]string, error) {
	return filter(env, refs, runtime.GOOS == "windows")
}

func filter(env []string, refs []nodepolicy.CredentialRef, foldCase bool) ([]string, error) {
	keep := make(map[string]bool, len(refs))
	for _, ref := range refs {
		if _, err := nodepolicy.ParseCredentialRef(string(ref)); err != nil {
			return nil, fmt.Errorf("secretenv: %w", err)
		}
		keep[string(ref)] = false
	}

	out := make([]string, 0, len(env))
	for _, kv := range env {
		key := envKey(kv)
		upper := strings.ToUpper(key)
		if !strings.HasPrefix(upper, Prefix) {
			out = append(out, kv)
			continue
		}
		name := key
		if foldCase {
			name = upper
		}
		seen, retained := keep[name]
		if !retained {
			continue
		}
		if seen {
			return nil, fmt.Errorf("secretenv: ambiguous duplicate environment entries for %s", name)
		}
		keep[name] = true
		out = append(out, kv)
	}
	return out, nil
}

// envKey returns the name part of a "KEY=value" entry. A leading '=' belongs to
// the name, as in Windows per-drive entries such as "=C:=C:\". An entry without
// '=' is all name.
func envKey(kv string) string {
	start := 0
	if strings.HasPrefix(kv, "=") {
		start = 1
	}
	if i := strings.IndexByte(kv[start:], '='); i >= 0 {
		return kv[:start+i]
	}
	return kv
}
