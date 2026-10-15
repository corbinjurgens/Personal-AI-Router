// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"net/http"
	"sort"
	"sync"
	"time"

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
	incarnation, ok := m.mesh.PeerServiceIncarnation(uuid)
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
	if len(addresses) > 8 {
		addresses = addresses[:8]
	}
	ctx, cancel := context.WithTimeout(ctx, 8*time.Second)
	defer cancel()
	for _, address := range addresses {
		if !m.selectPeerServiceAddress(uuid, address) {
			continue
		}
		client, err := m.peerClient(uuid)
		if err != nil {
			return err
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, "https://"+address+peerServicesPath, nil)
		if err != nil {
			return err
		}
		response, err := client.Do(req)
		if err != nil {
			continue
		}
		if response.StatusCode != http.StatusOK {
			response.Body.Close()
			continue
		}
		descriptor, err := noderec.DecodePeerServices(response.Body, uuid)
		closeErr := response.Body.Close()
		if err != nil {
			return err
		}
		if closeErr != nil {
			return closeErr
		}
		m.mesh.Refresh()
		current, valid := m.mesh.PeerServiceIncarnation(uuid)
		currentPeer, memberValid := m.reversePeer(uuid)
		if !valid || current != incarnation || !memberValid || currentPeer != peer {
			return errors.New("admission changed during descriptor read")
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
