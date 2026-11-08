// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nodepolicy

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strconv"
	"strings"
)

// MethodSetPeerSizing is the broker's private, process-scoped replacement of
// the peer sizing projection a local proxy may use to size tier candidates.
const MethodSetPeerSizing = "node/set-peer-sizing"

// Peer sizing bounds reuse existing limits: the remote policy response body
// limit, the routing inventory bounds and the peer-performance freshness window.
const (
	PeerSizingFreshnessMs   = 90000
	MaxPeerSizingPeers      = 256
	MaxPeerSizingProfiles   = 1024
	MaxPeerSizingSnapshot   = 256 << 10
	peerSizingIdentityBytes = 128
)

// ActiveRequestPolicy attests, on a peer policy read, that the destination's
// live proxy run has acknowledged exactly this saved policy revision. Nil means
// activation is pending, failed, unacknowledged or the proxy was replaced.
type ActiveRequestPolicy struct {
	Revision   string `json:"revision"`
	ProxyRunID string `json:"proxyRunId"`
}

// PeerSizingProfile is one advertised shared profile's request recipe. An
// empty, null or {} RequestOptions is a known no-op recipe; an absent profile
// is unknown.
type PeerSizingProfile struct {
	Key                string             `json:"key"`
	Engine             string             `json:"engine"`
	Model              string             `json:"model"`
	RequestOptions     json.RawMessage    `json:"requestOptions,omitempty"`
	RequestOptionsMode RequestOptionsMode `json:"requestOptionsMode,omitempty"`
}

// PeerSizing is one admitted peer's attested, bounded-age sizing observation.
type PeerSizing struct {
	Host                 string              `json:"host"`
	AdmissionIdentity    string              `json:"admissionIdentity"`
	AdmissionIncarnation string              `json:"admissionIncarnation"`
	PolicyRevision       string              `json:"policyRevision"`
	ProxyRunID           string              `json:"proxyRunId"`
	AgeMs                int64               `json:"ageMs"`
	Profiles             []PeerSizingProfile `json:"profiles"`
}

// PeerSizingSnapshot is a full replacement, never a delta. An empty Peers list
// explicitly withdraws every observation.
type PeerSizingSnapshot struct {
	Revision uint64       `json:"revision"`
	Peers    []PeerSizing `json:"peers"`
}

func checkSizingIdentity(name, value string) error {
	if value == "" || len(value) > peerSizingIdentityBytes {
		return fmt.Errorf("%s must be 1-%d bytes", name, peerSizingIdentityBytes)
	}
	if !strictID.MatchString(value) {
		return fmt.Errorf("%s must be an ASCII identity", name)
	}
	return nil
}

// CanonicalAdmissionEpoch reports whether text is a positive canonical decimal
// admission epoch.
func CanonicalAdmissionEpoch(text string) (uint64, bool) {
	epoch, err := strconv.ParseUint(text, 10, 64)
	return epoch, err == nil && epoch > 0 && strconv.FormatUint(epoch, 10) == text
}

func (a ActiveRequestPolicy) Validate() error {
	if err := validateRevision(a.Revision); err != nil {
		return err
	}
	return checkSizingIdentity("proxyRunId", a.ProxyRunID)
}

func (p PeerSizingProfile) Validate() error {
	if err := ValidateProfileKey(p.Key); err != nil {
		return err
	}
	if err := checkModelRef(p.Engine, p.Model, 0, nil); err != nil {
		return err
	}
	return checkRequestOptions(p.RequestOptions, p.RequestOptionsMode)
}

func (s PeerSizing) Validate() error {
	if err := checkSizingIdentity("host", s.Host); err != nil {
		return err
	}
	if err := checkSizingIdentity("admissionIdentity", s.AdmissionIdentity); err != nil {
		return err
	}
	if _, ok := CanonicalAdmissionEpoch(s.AdmissionIncarnation); !ok {
		return errors.New("admissionIncarnation must be a positive canonical decimal epoch")
	}
	if err := validateRevision(s.PolicyRevision); err != nil {
		return err
	}
	if err := checkSizingIdentity("proxyRunId", s.ProxyRunID); err != nil {
		return err
	}
	if s.AgeMs < 0 || s.AgeMs >= PeerSizingFreshnessMs {
		return fmt.Errorf("ageMs must be in [0, %d)", PeerSizingFreshnessMs)
	}
	if s.Profiles == nil {
		return errors.New("profiles is required")
	}
	if len(s.Profiles) > MaxPeerSizingProfiles {
		return fmt.Errorf("profiles exceed %d", MaxPeerSizingProfiles)
	}
	keys := make(map[string]bool, len(s.Profiles))
	for i, p := range s.Profiles {
		if err := p.Validate(); err != nil {
			return fmt.Errorf("profiles[%d]: %w", i, err)
		}
		if keys[p.Key] {
			return fmt.Errorf("profiles[%d]: duplicate key %q", i, p.Key)
		}
		keys[p.Key] = true
	}
	return nil
}

func (s PeerSizingSnapshot) Validate() error {
	if s.Revision == 0 {
		return errors.New("revision must be positive")
	}
	if s.Peers == nil {
		return errors.New("peers is required")
	}
	if len(s.Peers) > MaxPeerSizingPeers {
		return fmt.Errorf("peers exceed %d", MaxPeerSizingPeers)
	}
	hosts := make(map[string]bool, len(s.Peers))
	for i, p := range s.Peers {
		if err := p.Validate(); err != nil {
			return fmt.Errorf("peers[%d]: %w", i, err)
		}
		if hosts[p.Host] {
			return fmt.Errorf("peers[%d]: duplicate host", i)
		}
		hosts[p.Host] = true
	}
	encoded, err := json.Marshal(s)
	if err != nil {
		return err
	}
	if len(encoded) > MaxPeerSizingSnapshot {
		return fmt.Errorf("snapshot exceeds %d bytes", MaxPeerSizingSnapshot)
	}
	return nil
}

// decodeSizingEnvelope decodes one closed JSON object: unknown, duplicate and
// case-folded duplicate fields are refused before the standard decoder, which
// matches names case-insensitively, can merge them. Field values keep their
// own semantics, so requestOptions object contents are not constrained here.
func decodeSizingEnvelope(data []byte, fields []string, into any) error {
	dec := json.NewDecoder(bytes.NewReader(data))
	if token, err := dec.Token(); err != nil || token != json.Delim('{') {
		return errors.New("expected a JSON object")
	}
	seen := make(map[string]bool, len(fields))
	for dec.More() {
		token, err := dec.Token()
		name, ok := token.(string)
		if err != nil || !ok {
			return errors.New("invalid object member")
		}
		known := false
		for _, field := range fields {
			if name == field {
				known = true
				break
			}
		}
		if !known {
			return fmt.Errorf("unknown field %q", name)
		}
		for previous := range seen {
			if strings.EqualFold(previous, name) {
				return fmt.Errorf("duplicate field %q", name)
			}
		}
		seen[name] = true
		var skip json.RawMessage
		if err := dec.Decode(&skip); err != nil {
			return err
		}
	}
	if _, err := dec.Token(); err != nil {
		return err
	}
	if _, err := dec.Token(); err != io.EOF {
		return errors.New("trailing data after object")
	}
	return json.Unmarshal(data, into)
}

func (a *ActiveRequestPolicy) UnmarshalJSON(data []byte) error {
	type wire ActiveRequestPolicy
	var next wire
	if err := decodeSizingEnvelope(data, []string{"revision", "proxyRunId"}, &next); err != nil {
		return err
	}
	*a = ActiveRequestPolicy(next)
	return a.Validate()
}

func (p *PeerSizingProfile) UnmarshalJSON(data []byte) error {
	type wire PeerSizingProfile
	var next wire
	if err := decodeSizingEnvelope(data, []string{"key", "engine", "model", "requestOptions", "requestOptionsMode"}, &next); err != nil {
		return err
	}
	*p = PeerSizingProfile(next)
	return p.Validate()
}

func (s *PeerSizing) UnmarshalJSON(data []byte) error {
	type wire PeerSizing
	var next wire
	if err := decodeSizingEnvelope(data, []string{"host", "admissionIdentity", "admissionIncarnation", "policyRevision", "proxyRunId", "ageMs", "profiles"}, &next); err != nil {
		return err
	}
	*s = PeerSizing(next)
	return s.Validate()
}

func (s *PeerSizingSnapshot) UnmarshalJSON(data []byte) error {
	type wire PeerSizingSnapshot
	var next wire
	if err := decodeSizingEnvelope(data, []string{"revision", "peers"}, &next); err != nil {
		return err
	}
	*s = PeerSizingSnapshot(next)
	return s.Validate()
}

// Clone detaches option bytes so caches and senders own their snapshot.
func (s PeerSizingSnapshot) Clone() PeerSizingSnapshot {
	out := PeerSizingSnapshot{Revision: s.Revision, Peers: make([]PeerSizing, len(s.Peers))}
	for i, peer := range s.Peers {
		peer.Profiles = clonePeerSizingProfiles(peer.Profiles)
		out.Peers[i] = peer
	}
	return out
}

func clonePeerSizingProfiles(profiles []PeerSizingProfile) []PeerSizingProfile {
	out := make([]PeerSizingProfile, len(profiles))
	for i, p := range profiles {
		p.RequestOptions = bytes.Clone(p.RequestOptions)
		out[i] = p
	}
	return out
}
