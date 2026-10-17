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

const (
	aliasAPISuffix   = "-api"
	aliasLocalSuffix = "-local"
)

// ParseAlias splits a hybrid alias into its tier and mode. The reserved names
// are each tier in TierOrder, bare or with the -api or -local suffix. Matching
// is exact: no case folding, no trimming and no Ollama :latest handling. ok is
// false, with empty tier and mode, for any other name.
func ParseAlias(name string) (tier string, mode AliasMode, ok bool) {
	tier, mode = name, AliasBare
	switch {
	case strings.HasSuffix(name, aliasAPISuffix):
		tier, mode = strings.TrimSuffix(name, aliasAPISuffix), AliasAPI
	case strings.HasSuffix(name, aliasLocalSuffix):
		tier, mode = strings.TrimSuffix(name, aliasLocalSuffix), AliasLocal
	}
	if !IsTier(tier) {
		return "", "", false
	}
	return tier, mode, true
}

// IsReservedAlias reports whether name is one of the nine reserved hybrid
// aliases. Policy validation and the proxy's alias handling are meant to share
// it, so they cannot disagree about what is reserved.
func IsReservedAlias(name string) bool {
	_, _, ok := ParseAlias(name)
	return ok
}
