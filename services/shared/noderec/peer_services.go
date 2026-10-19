// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package noderec

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net/netip"
	"strings"
	"unicode"
	"unicode/utf8"

	"nvpair-shared/engines"
)

const PeerServicesMaxBytes = 8192

// PeerService describes presence and port only. Transport remains local policy.
type PeerService struct {
	Service ServiceKey `json:"service"`
	Port    int        `json:"port"`
}

// PeerServicesDescriptor is a complete initialized self snapshot served over cl.
// It carries no membership, trust, liveness, protocol or forwarding authority.
type PeerServicesDescriptor struct {
	HostUUID  string        `json:"hostUuid"`
	Name      string        `json:"name"`
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

func ValidatePeerServicesDescriptor(d PeerServicesDescriptor, expectedUUID string) error {
	if expectedUUID == "" || len(expectedUUID) > 128 || d.HostUUID != expectedUUID || strings.ContainsAny(expectedUUID, "/\\:\x00") {
		return errors.New("descriptor subject mismatch")
	}
	if len(d.Name) == 0 || len(d.Name) > 255 || !utf8.ValidString(d.Name) || strings.IndexFunc(d.Name, unicode.IsControl) >= 0 {
		return errors.New("invalid descriptor name")
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

// DecodePeerServices accepts exactly one bounded complete document, including a
// successful EOF read. A transport error after the JSON value still rejects it.
func DecodePeerServices(r io.Reader, expectedUUID string) (PeerServicesDescriptor, error) {
	var d PeerServicesDescriptor
	body, err := io.ReadAll(io.LimitReader(r, PeerServicesMaxBytes+1))
	if err != nil {
		return d, err
	}
	if len(body) > PeerServicesMaxBytes {
		return d, errors.New("descriptor too large")
	}
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&d); err != nil {
		return d, err
	}
	if decoder.Decode(&struct{}{}) != io.EOF {
		return d, errors.New("trailing descriptor data")
	}
	if err := ValidatePeerServicesDescriptor(d, expectedUUID); err != nil {
		return PeerServicesDescriptor{}, err
	}
	return d, nil
}
