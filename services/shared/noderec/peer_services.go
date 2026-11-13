// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package noderec

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net/netip"
	"regexp"
	"strings"

	"nvpair-shared/engines"
)

const PeerServicesMaxBytes = 8192

// PeerStateMaxLine bounds one peer-state stream line: a descriptor plus the
// frame's roster hash and field names.
const PeerStateMaxLine = PeerServicesMaxBytes + 128

// PeerService describes presence and port only. Transport remains local policy.
type PeerService struct {
	Service ServiceKey `json:"service"`
	Port    int        `json:"port"`
}

// PeerServicesDescriptor is a complete initialized self snapshot served over cl.
// It carries no membership, trust, liveness, protocol or forwarding authority,
// and no name: a node's own name reaches its peers only on the cluster roster.
type PeerServicesDescriptor struct {
	HostUUID  string        `json:"hostUuid"`
	Addresses []string      `json:"addresses"`
	Services  []PeerService `json:"services"`
}

// PeerServicesClaim is private worker IPC. Incarnation is captured before the
// authenticated read; Sequence is receiver-owned, never a remote generation.
type PeerServicesClaim struct {
	Descriptor  PeerServicesDescriptor `json:"descriptor"`
	Incarnation string                 `json:"incarnation"`
	Sequence    uint64                 `json:"sequence"`
	Source      string                 `json:"source"`
	ObservedAt  int64                  `json:"observedAt"`
}

// PeerStateFrame is one non-keepalive line of a cluster manager's peer-state
// stream: the node's current descriptor and the hash of its canonical roster.
type PeerStateFrame struct {
	Descriptor PeerServicesDescriptor `json:"descriptor"`
	RosterHash string                 `json:"rosterHash"`
}

var rosterHashPattern = regexp.MustCompile(`^[0-9a-f]{64}$`)

// DecodePeerStateFrame accepts exactly one bounded frame per line, rejecting
// unknown fields, trailing data, a descriptor for another subject and a roster
// hash that is not 64 lowercase hex characters. Callers handle the null
// keepalive before decoding; a null line is an error here.
func DecodePeerStateFrame(line []byte, expectedUUID string) (PeerStateFrame, error) {
	var frame PeerStateFrame
	if len(line) > PeerStateMaxLine {
		return frame, errors.New("peer-state frame too large")
	}
	if bytes.Equal(bytes.TrimSpace(line), []byte("null")) {
		return frame, errors.New("peer-state keepalive is not a frame")
	}
	decoder := json.NewDecoder(bytes.NewReader(line))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&frame); err != nil {
		return PeerStateFrame{}, err
	}
	if decoder.Decode(&struct{}{}) != io.EOF {
		return PeerStateFrame{}, errors.New("trailing peer-state data")
	}
	if err := ValidatePeerServicesDescriptor(frame.Descriptor, expectedUUID); err != nil {
		return PeerStateFrame{}, err
	}
	if !rosterHashPattern.MatchString(frame.RosterHash) {
		return PeerStateFrame{}, errors.New("invalid peer-state roster hash")
	}
	return frame, nil
}

func ValidatePeerServicesDescriptor(d PeerServicesDescriptor, expectedUUID string) error {
	if expectedUUID == "" || len(expectedUUID) > 128 || d.HostUUID != expectedUUID || strings.ContainsAny(expectedUUID, "/\\:\x00") {
		return errors.New("descriptor subject mismatch")
	}
	if len(d.Addresses) == 0 || len(d.Addresses) > MaxAdvertisedIPs {
		return errors.New("invalid descriptor address count")
	}
	seenAddresses := make(map[netip.Addr]bool)
	for _, address := range d.Addresses {
		ip, err := netip.ParseAddr(address)
		if err != nil || ip.Zone() != "" || !ip.IsGlobalUnicast() || ip.IsLoopback() || ip.IsUnspecified() || seenAddresses[ip.Unmap()] {
			return errors.New("invalid descriptor address")
		}
		seenAddresses[ip.Unmap()] = true
	}
	if d.Services == nil || len(d.Services) > len(serviceKeyOrder)+engines.MaxDynamicEngines {
		return errors.New("incomplete or oversized descriptor services")
	}
	seen := make(map[ServiceKey]bool)
	dynamic := 0
	for _, service := range d.Services {
		// A dynamic engine key is admitted by syntax and count only. It is
		// carried on this authenticated path, never in TXT, and a transport to
		// it still needs the peer's live registration.
		known := engines.IsDynamicServiceKey(service.Service)
		if known {
			dynamic++
		}
		for _, key := range serviceKeyOrder {
			if service.Service == key {
				known = true
				break
			}
		}
		if !known || dynamic > engines.MaxDynamicEngines || seen[service.Service] || service.Port < 1 || service.Port > 65535 {
			return errors.New("invalid or duplicate descriptor service")
		}
		seen[service.Service] = true
	}
	return nil
}
