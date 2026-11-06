// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nodepolicy

import (
	"encoding/json"
	"errors"
	"regexp"
	"strconv"
	"strings"
)

// NanoUSD is an exact money amount in nano-USD (10^-9 USD), held in its
// canonical decimal form. It travels as a JSON string, never a number, so a
// JavaScript consumer cannot round it. Build values with ParseNanoUSD; the zero
// value is invalid and fails Int64 and JSON marshaling.
type NanoUSD string

// nanoUSDCanonical is "0" or a nonzero leading digit followed by digits.
var nanoUSDCanonical = regexp.MustCompile(`^(0|[1-9][0-9]*)$`)

// ParseNanoUSD accepts only canonical unsigned decimal digits within the signed
// 64-bit range. It rejects signs, whitespace, exponents, fractions, leading
// zeros and overflow.
func ParseNanoUSD(s string) (NanoUSD, error) {
	if !nanoUSDCanonical.MatchString(s) {
		return "", errors.New("money must be canonical nano-USD digits: 0 or a nonzero leading digit, with no sign, space, exponent or fraction")
	}
	if _, err := strconv.ParseInt(s, 10, 64); err != nil {
		return "", errors.New("money exceeds the signed 64-bit nano-USD range")
	}
	return NanoUSD(s), nil
}

// Int64 returns the exact amount.
func (m NanoUSD) Int64() (int64, error) {
	if _, err := ParseNanoUSD(string(m)); err != nil {
		return 0, err
	}
	return strconv.ParseInt(string(m), 10, 64)
}

// MarshalJSON writes the amount as a JSON string.
func (m NanoUSD) MarshalJSON() ([]byte, error) {
	if _, err := ParseNanoUSD(string(m)); err != nil {
		return nil, err
	}
	return json.Marshal(string(m))
}

// UnmarshalJSON requires a JSON string holding canonical digits. JSON numbers
// and null are rejected.
func (m *NanoUSD) UnmarshalJSON(data []byte) error {
	if len(data) == 0 || data[0] != '"' {
		return errors.New("money must be a JSON string")
	}
	var s string
	if err := json.Unmarshal(data, &s); err != nil {
		return err
	}
	v, err := ParseNanoUSD(s)
	if err != nil {
		return err
	}
	*m = v
	return nil
}

// credentialRef is the reserved provider-secret reference namespace. Matching
// is exact: no case folding, no trimming.
var credentialRef = regexp.MustCompile(`^NVPAIR_PROVIDER_SECRET_[A-Z0-9_]+$`)

// CredentialRef names a provider secret in the launch environment. It is a
// reference only, never the secret value.
type CredentialRef string

// ParseCredentialRef accepts only NVPAIR_PROVIDER_SECRET_ followed by one or
// more uppercase ASCII letters, digits or underscores. The error does not echo
// the input.
func ParseCredentialRef(s string) (CredentialRef, error) {
	if !credentialRef.MatchString(s) {
		return "", errors.New("credential reference must be NVPAIR_PROVIDER_SECRET_ followed by uppercase letters, digits or underscores")
	}
	return CredentialRef(s), nil
}

// AliasMode is the destination restriction a hybrid alias names.
type AliasMode string

const (
	// AliasBare uses the configured defaults for the tier.
	AliasBare AliasMode = "bare"
	// AliasAPI always excludes local destinations.
	AliasAPI AliasMode = "api"
	// AliasLocal always excludes API destinations.
	AliasLocal AliasMode = "local"
)

var roleNamePattern = regexp.MustCompile(`^[a-z0-9][a-z0-9._-]{0,63}$`)

// ValidateRoleName implements the plain-name grammar of owner decision D1a.
func ValidateRoleName(name string) error {
	if len(name) > MaxRoleNameBytes || !roleNamePattern.MatchString(name) {
		return errors.New("role name must match ^[a-z0-9][a-z0-9._-]{0,63}$")
	}
	return nil
}

// ParseAlias accepts a plain role name or its :api or :local override.
func ParseAlias(name string) (tier string, mode AliasMode, ok bool) {
	tier, variable, hasColon := strings.Cut(name, ":")
	mode = AliasBare
	if hasColon {
		switch variable {
		case "api":
			mode = AliasAPI
		case "local":
			mode = AliasLocal
		default:
			return "", "", false
		}
	}
	if ValidateRoleName(tier) != nil {
		return "", "", false
	}
	return tier, mode, true
}

// IsReservedAlias reports whether name is a role's hybrid override.
func IsReservedAlias(name string) bool {
	_, mode, ok := ParseAlias(name)
	return ok && mode != AliasBare
}

// IsReservedProfileKey reserves every variable after a valid plain role name.
func IsReservedProfileKey(name string) bool {
	base, _, hasColon := strings.Cut(name, ":")
	return hasColon && ValidateRoleName(base) == nil
}
