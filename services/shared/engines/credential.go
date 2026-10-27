// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package engines

// These methods are for local clients only. Neither returns a key; the separate
// LocalCredentialMethod is reserved for the broker.
const (
	SetCredentialMethod   = "engine:set-credential"
	CredentialStateMethod = "engine:credential-state"
	MaxCredentialBytes    = 512
)

// SetCredentialParams selects an engine and its credential. Empty clears it.
type SetCredentialParams struct {
	Engine     string `json:"engine"`
	Credential string `json:"credential"`
}

// CredentialParams selects an engine for a credential state request.
type CredentialParams struct {
	Engine string `json:"engine"`
}

// CredentialState reports whether an engine supports and has a credential.
type CredentialState struct {
	Supported  bool `json:"supported"`
	Configured bool `json:"configured"`
}

// ValidCredential accepts an empty credential or up to 512 printable ASCII bytes.
func ValidCredential(s string) bool {
	if len(s) > MaxCredentialBytes {
		return false
	}
	for i := 0; i < len(s); i++ {
		if s[i] < 0x21 || s[i] > 0x7e {
			return false
		}
	}
	return true
}
