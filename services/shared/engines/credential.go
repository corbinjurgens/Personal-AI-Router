// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package engines

// MaxCredentialBytes bounds the manager-generated serving credential.
const MaxCredentialBytes = 512

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
