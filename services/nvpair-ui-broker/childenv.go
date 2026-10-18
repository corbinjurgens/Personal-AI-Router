// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"fmt"
	"os"

	"nvpair-shared/nodepolicy"
	"nvpair-shared/secretenv"
)

// The broker inherits the reserved provider-secret namespace from the service
// because it hands configured credentials on to the proxy. Every child it
// starts gets its environment from one of the two builders below, so no launch
// site decides that on its own.

// workerEnv is the environment of every broker child except the proxy: the
// broker's own without the reserved provider-secret namespace.
func workerEnv() []string {
	return secretenv.Remove(os.Environ())
}

// proxyEnv is the proxy's environment, built from env: the reserved namespace
// is reduced to the configured credential references refs. An ambiguous
// duplicate of a reference fails, so the proxy is not started with a value it
// could not tell apart; the error names the reference, never a value.
func proxyEnv(env []string, refs []nodepolicy.CredentialRef) ([]string, error) {
	out, err := secretenv.Retain(env, refs)
	if err != nil {
		return nil, fmt.Errorf("proxy environment: %w", err)
	}
	return out, nil
}
