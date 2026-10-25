// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"slices"
	"sort"
	"sync"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/netpick"
	"nvpair-shared/noderec"
)

const peerServicesPath = "/v1/cluster/services"
const peerServicesEvent = "cluster:peer-services"
const peerServicesRefreshEvery = 2 * time.Second

type peerServicesState struct {
	mu       sync.Mutex
	self     noderec.DirectoryNode
	source   string
	sequence uint64
	inflight map[string]bool
}

func (m *Manager) setSelfServicesIdentity(nodes []noderec.DirectoryNode) {
	var self noderec.DirectoryNode
	for _, node := range nodes {
		if node.HostUUID == m.identity.NodeUUID {
			self = node
			break
		}
	}
	m.serviceRefresh.mu.Lock()
	m.serviceRefresh.self = self
	m.serviceRefresh.mu.Unlock()
}

func descriptorAddresses(scanner, overlay []string) []string {
	var addresses []string
	seen := make(map[string]bool)
	for _, address := range scanner {
		if !seen[address] && len(addresses) < noderec.MaxAdvertisedIPs {
			addresses = append(addresses, address)
			seen[address] = true
		}
	}
	if len(addresses) == 0 {
		return nil
	}
	orderedOverlay := append([]string(nil), overlay...)
	sort.Strings(orderedOverlay)
	for _, address := range orderedOverlay {
		if seen[address] {
			continue
		}
		// A full scanner list reserves one final position for an overlay hint.
		if len(addresses) == noderec.MaxAdvertisedIPs {
			addresses[len(addresses)-1] = address
			break
		}
		addresses = append(addresses, address)
		seen[address] = true
		if len(addresses) == noderec.MaxAdvertisedIPs {
			break
		}
	}
	return addresses
}

func (m *Manager) selfServicesDescriptor() (noderec.PeerServicesDescriptor, error) {
	m.serviceRefresh.mu.Lock()
	self := m.serviceRefresh.self
	m.serviceRefresh.mu.Unlock()
	d := noderec.PeerServicesDescriptor{HostUUID: self.HostUUID, Name: self.Name, Addresses: self.CandidateIPs(), Services: []noderec.PeerService{}}
	m.reverse.mu.RLock()
	initialized := m.reverse.initialized
	for _, service := range m.reverse.services {
		d.Services = append(d.Services, noderec.PeerService{Service: service.Service, Port: service.Port})
	}
	m.reverse.mu.RUnlock()
	if !initialized {
		return d, errors.New("local registrations not initialized")
	}
	sort.Slice(d.Services, func(i, j int) bool { return d.Services[i].Service < d.Services[j].Service })
	// Overlay hints cannot replace missing or invalid scanner identity data.
	if err := noderec.ValidatePeerServicesDescriptor(d, m.identity.NodeUUID); err != nil {
		return d, err
	}
	d.Addresses = descriptorAddresses(d.Addresses, netpick.SharedOverlayAddresses())
	return d, noderec.ValidatePeerServicesDescriptor(d, m.identity.NodeUUID)
}

func (m *Manager) handlePeerServices(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if r.ContentLength != 0 || len(r.TransferEncoding) != 0 || r.URL.RawQuery != "" {
		http.Error(w, "bodyless read required", http.StatusBadRequest)
		return
	}
	uuid, ok := m.verifyClientPin(r)
	if !ok {
		http.Error(w, "forbidden", http.StatusForbidden)
		return
	}
	peer, ok := m.reversePeerCertificate(uuid, r.TLS.PeerCertificates[0].Raw)
	if !ok {
		http.Error(w, "admission required", http.StatusForbidden)
		return
	}
	d, err := m.selfServicesDescriptor()
	if err != nil {
		http.Error(w, "self services unavailable", http.StatusServiceUnavailable)
		return
	}
	current, ok := m.reversePeer(uuid)
	if !ok || current != peer {
		http.Error(w, "admission changed", http.StatusForbidden)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	_ = json.NewEncoder(w).Encode(d)
}

// There is at most one complete fetch/publish per peer, and four network reads
// globally. Remote process generations are deliberately not used for ordering.
func (m *Manager) peerServicesLoop(ctx context.Context) {
	var source [16]byte
	if _, err := rand.Read(source[:]); err != nil {
		return
	}
	m.serviceRefresh.mu.Lock()
	m.serviceRefresh.source = hex.EncodeToString(source[:])
	m.serviceRefresh.inflight = make(map[string]bool)
	m.serviceRefresh.mu.Unlock()
	slots := make(chan struct{}, 4)
	var work sync.WaitGroup
	defer work.Wait()
	ticker := time.NewTicker(peerServicesRefreshEvery)
	defer ticker.Stop()
	for {
		m.memMu.Lock()
		var peers []string
		for uuid, member := range m.members {
			if member.State == stateMember && uuid != m.identity.NodeUUID {
				peers = append(peers, uuid)
			}
		}
		m.memMu.Unlock()
		sort.Strings(peers)
		for _, uuid := range peers {
			m.serviceRefresh.mu.Lock()
			busy := m.serviceRefresh.inflight[uuid]
			if !busy {
				m.serviceRefresh.inflight[uuid] = true
			}
			m.serviceRefresh.mu.Unlock()
			if busy {
				continue
			}
			select {
			case slots <- struct{}{}:
				work.Add(1)
				go func(uuid string) {
					defer work.Done()
					defer func() {
						<-slots
						m.serviceRefresh.mu.Lock()
						delete(m.serviceRefresh.inflight, uuid)
						m.serviceRefresh.mu.Unlock()
					}()
					_ = m.refreshPeerServices(ctx, uuid)
				}(uuid)
			case <-ctx.Done():
				m.serviceRefresh.mu.Lock()
				delete(m.serviceRefresh.inflight, uuid)
				m.serviceRefresh.mu.Unlock()
				return
			}
		}
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
	}
}

func (m *Manager) refreshPeerServices(ctx context.Context, uuid string) error {
	m.mesh.Refresh()
	incarnation, ok := m.peerServiceIncarnation(uuid)
	if !ok {
		return errors.New("no current admission")
	}
	peer, ok := m.reversePeer(uuid)
	if !ok {
		return errors.New("no current member")
	}
	member, ok := m.memberByNodeID(uuid)
	if !ok {
		return errors.New("member unavailable")
	}
	addresses := m.resolvePeerAddrs(*member)
	ctx, cancel := context.WithTimeout(ctx, 8*time.Second)
	defer cancel()
	if err := ctx.Err(); err != nil {
		return err
	}
	addresses = confirmedFirst(addresses, m.peerAddrs.ChooseWithin(ctx, uuid, addresses))
	deadline, _ := ctx.Deadline()
	for index, address := range addresses {
		if err := ctx.Err(); err != nil {
			return err
		}
		if !m.selectPeerServiceAddress(uuid, address) {
			continue
		}
		client, err := m.peerClient(uuid)
		if err != nil {
			return err
		}
		// Divide the remaining parent time among ALL remaining endpoints. This
		// context owns setup, reverse fallback, headers, complete body and EOF.
		budget := time.Until(deadline) / time.Duration(len(addresses)-index)
		attempt, attemptCancel := context.WithTimeout(ctx, budget)
		req, err := http.NewRequestWithContext(attempt, http.MethodGet, "https://"+address+peerServicesPath, nil)
		if err != nil {
			attemptCancel()
			return err
		}
		response, err := client.Do(req)
		var descriptor noderec.PeerServicesDescriptor
		retry := true
		if err == nil {
			if response.StatusCode != http.StatusOK {
				err = errors.New("peer services HTTP refusal")
			} else {
				// Read the real stream through EOF under the same attempt budget.
				// Keeping transport errors separate from validation means a complete
				// invalid document stays terminal rather than masquerading as failover.
				var payload []byte
				payload, err = io.ReadAll(io.LimitReader(response.Body, noderec.PeerServicesMaxBytes+1))
				if err == nil {
					descriptor, err = noderec.DecodePeerServices(bytes.NewReader(payload), uuid)
					retry = errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF)
				}
			}
		}
		if response != nil {
			if closeErr := response.Body.Close(); err == nil && closeErr != nil {
				err = closeErr
				retry = true
			}
		}
		attemptErr := attempt.Err()
		attemptCancel()
		if parentErr := ctx.Err(); parentErr != nil {
			return parentErr
		}
		if attemptErr != nil {
			err = attemptErr
			retry = true
		}
		if err != nil {
			if !retry {
				return err
			}
			m.peerAddrs.Forget(uuid)
			continue
		}
		if err := m.commitPeerServiceHints(descriptor, incarnation, peer); err != nil {
			return err
		}
		m.serviceRefresh.mu.Lock()
		m.serviceRefresh.sequence++
		claim := noderec.PeerServicesClaim{Descriptor: descriptor, Incarnation: incarnation, Sequence: m.serviceRefresh.sequence, Source: m.serviceRefresh.source, ObservedAt: time.Now().UnixMilli()}
		err = m.codec.Notify(peerServicesEvent, claim)
		m.serviceRefresh.mu.Unlock()
		return err
	}
	return errors.New("peer services unavailable")
}

// commitPeerServiceHints consumes a fully validated descriptor. Identity and
// reverse-peer authentication finish under composition ownership before
// storage. The incarnation check reads members.json, so it runs under storage
// ownership and cannot overlap a replacement; storage then rechecks only member
// fields and owns the complete speculative update/save/rollback.
func (m *Manager) commitPeerServiceHints(descriptor noderec.PeerServicesDescriptor, incarnation string, peer clustertrust.ReversePeer) error {
	m.rosterMu.Lock()
	defer m.rosterMu.Unlock()
	m.mesh.Refresh()
	currentPeer, memberValid := m.reversePeer(descriptor.HostUUID)

	m.membersStoreMu.Lock()
	defer m.membersStoreMu.Unlock()
	current, valid := m.mesh.PeerServiceIncarnation(descriptor.HostUUID)
	if !valid || current != incarnation || !memberValid || currentPeer != peer {
		return errors.New("admission changed during descriptor read")
	}
	authenticated, ok := m.memberByNodeID(descriptor.HostUUID)
	if !ok {
		return errors.New("member unavailable during descriptor commit")
	}

	m.memMu.Lock()
	member := m.members[descriptor.HostUUID]
	if member == nil || member.NodeUUID != authenticated.NodeUUID || member.State != stateMember ||
		member.State != authenticated.State || member.ClusterID != authenticated.ClusterID || member.AdmissionEpoch != authenticated.AdmissionEpoch {
		m.memMu.Unlock()
		return errors.New("member changed during descriptor commit")
	}
	if slices.Equal(member.AdvertisedAddresses, descriptor.Addresses) {
		m.memMu.Unlock()
		return nil
	}
	previous := slices.Clone(member.AdvertisedAddresses)
	member.AdvertisedAddresses = slices.Clone(descriptor.Addresses)
	m.memMu.Unlock()
	if err := m.persistMembersStoreLocked(); err != nil {
		m.memMu.Lock()
		// Restore only this transaction's hints, preserving ordinary address/name
		// updates that do not acquire storage until their persistence call.
		if current := m.members[descriptor.HostUUID]; current == member {
			current.AdvertisedAddresses = previous
		}
		m.memMu.Unlock()
		return err
	}
	return nil
}
