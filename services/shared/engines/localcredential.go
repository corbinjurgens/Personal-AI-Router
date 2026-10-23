// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package engines

// LocalCredentialMethod is the broker's request to engine-manager for the
// native API key of a running local engine. It is deliberately outside the
// engine: namespace, which the broker relays for ordinary clients, and the
// broker also refuses it from clients by name. The key stays on the serving
// host: it never enters descriptors, discovery, peer payloads or logs.
const LocalCredentialMethod = "internal:local-engine-credential"

// LocalCredential answers LocalCredentialMethod. Credential is empty when the
// engine has no key or no owned running launch; Port is the port that launch
// serves on, so the key is only ever paired with that loopback endpoint.
type LocalCredential struct {
	Credential string `json:"credential,omitempty"`
	Port       int    `json:"port,omitempty"`
}
