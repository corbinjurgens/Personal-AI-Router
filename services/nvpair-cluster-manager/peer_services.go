// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"crypto/rand"
	"encoding/hex"
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

const peerServicesEvent = "cluster:peer-services"

// peerStatePath serves this node's peer-state stream to paired members.
const peerStatePath = "/v1/cluster/peer-state"

// peerStateKeepalive is the interval between null keepalive lines on a served
// peer-state stream. A var so tests can shorten it.
var peerStateKeepalive = 10 * time.Second

// peerStateWriteTimeout bounds each frame write on a served stream.
const peerStateWriteTimeout = 5 * time.Second

// peerStateIdle drops a subscribed stream after this long without a line.
const peerStateIdle = 25 * time.Second

// peerStateResponseHeaderTimeout bounds the wait for a stream's response
// headers. The stream client has no total timeout: http.Client.Timeout covers
// the body and would end every stream.
const peerStateResponseHeaderTimeout = 8 * time.Second

// peerStateConnectTimeout bounds one address's connect and response headers,
// so a stalled address cannot starve the next. A var so tests can shorten it.
var peerStateConnectTimeout = 8 * time.Second

// peerStateMinHealthy is how long a stream that delivered a frame must have
// run, counted from the start of its connect, for its end to reset the peer's
// reconnect backoff.
const peerStateMinHealthy = 5 * time.Second

// peerStateEvaluateEvery re-evaluates the member set without a signal;
// peerStateDueEvery checks reconnect backoff from in-memory state only.
const (
	peerStateEvaluateEvery = 10 * time.Second
	peerStateDueEvery      = time.Second
)

type peerServicesState struct {
	mu       sync.Mutex
	self     noderec.DirectoryNode
	source   string
	sequence uint64
	// reconcileInflight and reconcileDirty coalesce stream-requested roster
	// reconciles per peer (requestPeerReconcile).
	reconcileInflight map[string]bool
	reconcileDirty    map[string]bool
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
	m.signalPeerState()
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
	d := noderec.PeerServicesDescriptor{HostUUID: self.HostUUID, Addresses: self.CandidateIPs(), Services: []noderec.PeerService{}}
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

// handlePeerState serves this node's peer-state stream: the latest frame at
// once, every published change, and a null keepalive every peerStateKeepalive.
// Pin, admission and publisher availability are rechecked before every line.
func (m *Manager) handlePeerState(w http.ResponseWriter, r *http.Request) {
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
	if !m.peerState.available() {
		http.Error(w, "self services unavailable", http.StatusServiceUnavailable)
		return
	}
	current, ok := m.reversePeer(uuid)
	if !ok || current != peer {
		http.Error(w, "admission changed", http.StatusForbidden)
		return
	}
	updates, unsubscribe, ok := m.peerState.subscribe()
	if !ok {
		http.Error(w, "too many peer-state streams", http.StatusServiceUnavailable)
		return
	}
	defer unsubscribe()
	// Only an accepted stream is reachability evidence: counting a 503 would
	// let two not-yet-ready members reset and wake each other's backoff.
	m.notePeerReachable(uuid)
	w.Header().Set("Cache-Control", "no-store")
	clustertrust.ServeSnapshotStream(w, r, clustertrust.SnapshotStreamServer[noderec.PeerStateFrame]{
		Keepalive:    peerStateKeepalive,
		WriteTimeout: peerStateWriteTimeout,
		Updates:      updates,
		Authorize: func() bool {
			if pinned, ok := m.verifyClientPin(r); !ok || pinned != uuid {
				return false
			}
			current, ok := m.reversePeer(uuid)
			return ok && current == peer && m.peerState.available()
		},
	})
}

// peerStateStream is one subscription owned by peerStateLoop.
type peerStateStream struct {
	cancel      context.CancelFunc
	incarnation string
	stopped     bool // cancelled by the supervisor; its end leaves backoff alone
}

// peerStateResult is what a stream goroutine reports to peerStateLoop.
type peerStateResult struct {
	uuid        string
	incarnation string
	addrs       []string
	gotFrame    bool
	forbidden   bool
	open        time.Duration
}

// peerReachableSet holds members just shown reachable, until peerStateLoop
// takes them.
type peerReachableSet struct {
	mu    sync.Mutex
	uuids map[string]bool
}

// notePeerReachable records evidence that uuid is reachable again (a reverse
// session, an accepted inbound peer-state stream, or an accepted roster
// exchange). peerStateLoop then resets that member's reconnect backoff and
// redials it at once, so the 30 s cap applies only while a member stays
// unreachable.
func (m *Manager) notePeerReachable(uuid string) {
	m.peerReachable.mu.Lock()
	if m.peerReachable.uuids == nil {
		m.peerReachable.uuids = make(map[string]bool)
	}
	m.peerReachable.uuids[uuid] = true
	m.peerReachable.mu.Unlock()
	select {
	case m.peerReachableWake <- struct{}{}:
	default:
	}
}

func (m *Manager) takeReachablePeers() []string {
	m.peerReachable.mu.Lock()
	defer m.peerReachable.mu.Unlock()
	uuids := make([]string, 0, len(m.peerReachable.uuids))
	for uuid := range m.peerReachable.uuids {
		uuids = append(uuids, uuid)
	}
	clear(m.peerReachable.uuids)
	return uuids
}

// newPeerStateBackoff builds peerStateLoop's reconnect backoff. A var so tests
// can start from a long delay.
var newPeerStateBackoff = newPeerBackoff

// signalPeerStateMembers asks peerStateLoop to re-evaluate the member set now.
func (m *Manager) signalPeerStateMembers() {
	select {
	case m.peerStateMembers <- struct{}{}:
	default:
	}
}

// peerStateLoop keeps one peer-state subscription open to every other
// current member. It owns the active set and the reconnect backoff; stream
// goroutines only report their result. Remote process generations are never
// used for ordering.
func (m *Manager) peerStateLoop(ctx context.Context) {
	var source [16]byte
	if _, err := rand.Read(source[:]); err != nil {
		return
	}
	m.serviceRefresh.mu.Lock()
	m.serviceRefresh.source = hex.EncodeToString(source[:])
	m.serviceRefresh.mu.Unlock()

	backoff := newPeerStateBackoff()
	active := make(map[string]peerStateStream)
	members := make(map[string]string) // uuid -> incarnation at the last evaluation
	// streamed records, per member, the incarnation under which its stream
	// delivered a frame. Only such a member's later 403 requests a full pass:
	// a member still completing pairing refuses us until it pins this node,
	// and a pass then would prune it as a peer that dropped us.
	streamed := make(map[string]string)
	done := make(chan peerStateResult)
	var work sync.WaitGroup
	defer work.Wait()

	start := func() {
		now := time.Now()
		keep := make(map[string]bool, len(members))
		for uuid, incarnation := range members {
			keep[uuid] = true
			if _, running := active[uuid]; running {
				continue
			}
			member, ok := m.memberByUUID(uuid)
			if !ok {
				continue
			}
			addrs := m.resolvePeerAddrs(*member)
			if !backoff.due(uuid, addrs, now) {
				continue
			}
			streamCtx, cancel := context.WithCancel(ctx)
			active[uuid] = peerStateStream{cancel: cancel, incarnation: incarnation}
			work.Add(1)
			go func() {
				defer work.Done()
				opened := time.Now()
				gotFrame, err := m.runPeerStateStream(streamCtx, uuid, incarnation)
				cancel()
				result := peerStateResult{uuid: uuid, incarnation: incarnation, addrs: addrs, gotFrame: gotFrame,
					forbidden: errors.Is(err, errPeerStateForbidden), open: time.Since(opened)}
				select {
				case done <- result:
				case <-ctx.Done():
				}
			}()
		}
		backoff.retain(keep)
	}
	evaluate := func() {
		m.mesh.Refresh()
		clear(members)
		m.memMu.Lock()
		var peers []string
		for uuid, member := range m.members {
			if member.State == stateMember && uuid != m.identity.NodeUUID {
				peers = append(peers, uuid)
			}
		}
		m.memMu.Unlock()
		for _, uuid := range peers {
			if incarnation, ok := m.peerServiceIncarnation(uuid); ok {
				members[uuid] = incarnation
			}
		}
		for uuid := range streamed {
			if _, ok := members[uuid]; !ok {
				delete(streamed, uuid)
			}
		}
		for uuid, stream := range active {
			if incarnation, ok := members[uuid]; !ok || incarnation != stream.incarnation {
				if !stream.stopped {
					stream.cancel()
					stream.stopped = true
					active[uuid] = stream
				}
			}
		}
		start()
	}

	evaluateTicker := time.NewTicker(peerStateEvaluateEvery)
	defer evaluateTicker.Stop()
	dueTicker := time.NewTicker(peerStateDueEvery)
	defer dueTicker.Stop()
	evaluate()
	for {
		select {
		case <-ctx.Done():
			return
		case result := <-done:
			stream := active[result.uuid]
			delete(active, result.uuid)
			if result.gotFrame {
				streamed[result.uuid] = result.incarnation
			} else if result.forbidden && streamed[result.uuid] == result.incarnation {
				delete(streamed, result.uuid)
				m.requestReconcilePass()
			}
			if stream.stopped {
				continue
			}
			if result.gotFrame && result.open >= peerStateMinHealthy {
				backoff.reset(result.uuid)
			} else {
				backoff.failed(result.uuid, result.addrs, time.Now())
			}
		case <-evaluateTicker.C:
			evaluate()
		case <-m.peerStateMembers:
			evaluate()
		case <-dueTicker.C:
			start()
		case <-m.peerReachableWake:
			for _, uuid := range m.takeReachablePeers() {
				backoff.reset(uuid)
			}
			start()
		}
	}
}

// errPeerStateForbidden reports a 403 on a stream connect. peerStateLoop
// answers it with one full reconcile pass when the member had accepted this
// node before.
var errPeerStateForbidden = errors.New("peer-state stream refused")

// runPeerStateStream opens one peer-state stream to uuid and reads it until it
// ends. It walks the member's addresses, confirmed first: a transport error or
// an unexpected status forgets the chosen address and tries the next, while
// 403, 503 and any opened stream end the walk. gotFrame reports whether the
// stream delivered at least one frame. The request context carries no
// deadline; the adapter's response-header timeout and the idle watchdog bound
// the stream instead.
func (m *Manager) runPeerStateStream(ctx context.Context, uuid, incarnation string) (gotFrame bool, err error) {
	peer, ok := m.reversePeer(uuid)
	if !ok {
		return false, errors.New("no current member")
	}
	member, ok := m.memberByUUID(uuid)
	if !ok {
		return false, errors.New("member unavailable")
	}
	if current, ok := m.peerServiceIncarnation(uuid); !ok || current != incarnation {
		return false, errors.New("admission changed")
	}
	if err := ctx.Err(); err != nil {
		return false, err
	}
	addresses := m.resolvePeerAddrs(*member)
	choose, cancelChoose := context.WithTimeout(ctx, 8*time.Second)
	addresses = confirmedFirst(addresses, m.peerAddrs.ChooseWithin(choose, uuid, addresses))
	cancelChoose()
	for _, address := range addresses {
		if err := ctx.Err(); err != nil {
			return false, err
		}
		if !m.selectPeerServiceAddress(uuid, address) {
			continue
		}
		client, err := m.peerServiceClient(uuid, m.peerStateStreams)
		if err != nil {
			return false, err
		}
		streamCtx, cancel := context.WithCancel(ctx)
		req, err := http.NewRequestWithContext(streamCtx, http.MethodGet, "https://"+address+peerStatePath, nil)
		if err != nil {
			cancel()
			return false, err
		}
		// A cancel timer rather than a context deadline: DialReverse copies a
		// deadline onto the yamux stream, which would end an opened stream.
		// It bounds setup, including a stale reverse dial, and headers.
		watchdog := time.AfterFunc(peerStateConnectTimeout, cancel)
		response, err := client.Do(req)
		if !watchdog.Stop() && err == nil {
			response.Body.Close()
			err = errors.New("peer-state connect timed out")
		}
		if err != nil {
			cancel()
			if parentErr := ctx.Err(); parentErr != nil {
				return false, parentErr
			}
			m.peerAddrs.Forget(uuid)
			continue
		}
		switch response.StatusCode {
		case http.StatusOK:
		case http.StatusForbidden:
			response.Body.Close()
			cancel()
			return false, errPeerStateForbidden
		case http.StatusServiceUnavailable:
			response.Body.Close()
			cancel()
			return false, errors.New("peer-state not ready")
		default:
			response.Body.Close()
			cancel()
			m.peerAddrs.Forget(uuid)
			continue
		}
		gotFrame, err := m.readPeerStateStream(streamCtx, cancel, response.Body, uuid, incarnation, peer)
		response.Body.Close()
		return gotFrame, err
	}
	return false, errors.New("peer state unavailable")
}

// readPeerStateStream commits every frame's descriptor, re-emits the
// peer-services claim on every line (keepalives included, once a frame has
// arrived), and requests a roster reconcile on the first frame and whenever the
// roster hash changes.
func (m *Manager) readPeerStateStream(ctx context.Context, cancel context.CancelFunc, body io.Reader, uuid, incarnation string, peer clustertrust.ReversePeer) (bool, error) {
	var descriptor noderec.PeerServicesDescriptor
	var rosterHash string
	gotFrame := false
	err := clustertrust.ReadSnapshotStream(ctx, cancel, body, clustertrust.SnapshotStreamClient[noderec.PeerStateFrame]{
		Idle:    peerStateIdle,
		MaxLine: noderec.PeerStateMaxLine,
		Current: func() bool {
			current, ok := m.peerServiceIncarnation(uuid)
			if !ok || current != incarnation {
				return false
			}
			currentPeer, ok := m.reversePeer(uuid)
			return ok && currentPeer == peer
		},
		Decode: func(line []byte) (noderec.PeerStateFrame, error) {
			return noderec.DecodePeerStateFrame(line, uuid)
		},
		OnFrame: func(frame noderec.PeerStateFrame) error {
			if err := m.commitPeerServiceHints(frame.Descriptor, incarnation, peer); err != nil {
				return err
			}
			descriptor = frame.Descriptor
			if err := m.emitPeerServicesClaim(descriptor, incarnation); err != nil {
				return err
			}
			if !gotFrame || frame.RosterHash != rosterHash {
				rosterHash = frame.RosterHash
				m.requestPeerReconcile(uuid)
			}
			gotFrame = true
			return nil
		},
		OnKeepalive: func() {
			if gotFrame {
				_ = m.emitPeerServicesClaim(descriptor, incarnation)
			}
		},
	})
	return gotFrame, err
}

// emitPeerServicesClaim publishes descriptor as a fresh cluster:peer-services
// claim with the next receiver-owned sequence.
func (m *Manager) emitPeerServicesClaim(descriptor noderec.PeerServicesDescriptor, incarnation string) error {
	m.serviceRefresh.mu.Lock()
	defer m.serviceRefresh.mu.Unlock()
	m.serviceRefresh.sequence++
	claim := noderec.PeerServicesClaim{Descriptor: descriptor, Incarnation: incarnation, Sequence: m.serviceRefresh.sequence, Source: m.serviceRefresh.source, ObservedAt: time.Now().UnixMilli()}
	return m.codec.Notify(peerServicesEvent, claim)
}

// reconcileMember runs one roster exchange with uuid at its current addresses.
func (m *Manager) reconcileMember(uuid string) {
	if member, ok := m.memberByUUID(uuid); ok {
		m.reconcileWith(m.resolvePeerAddrs(*member), uuid)
	}
}

// requestPeerReconcile runs one roster reconcile (m.reconcilePeer) with uuid, coalescing: a
// request while one is in flight makes it run exactly once more afterwards.
// The reconcile itself runs without serviceRefresh.mu.
func (m *Manager) requestPeerReconcile(uuid string) {
	m.serviceRefresh.mu.Lock()
	if m.serviceRefresh.reconcileInflight[uuid] {
		if m.serviceRefresh.reconcileDirty == nil {
			m.serviceRefresh.reconcileDirty = make(map[string]bool)
		}
		m.serviceRefresh.reconcileDirty[uuid] = true
		m.serviceRefresh.mu.Unlock()
		return
	}
	if m.serviceRefresh.reconcileInflight == nil {
		m.serviceRefresh.reconcileInflight = make(map[string]bool)
	}
	m.serviceRefresh.reconcileInflight[uuid] = true
	m.serviceRefresh.mu.Unlock()
	go func() {
		for {
			m.reconcilePeer(uuid)
			m.serviceRefresh.mu.Lock()
			if m.serviceRefresh.reconcileDirty[uuid] {
				delete(m.serviceRefresh.reconcileDirty, uuid)
				m.serviceRefresh.mu.Unlock()
				continue
			}
			delete(m.serviceRefresh.reconcileInflight, uuid)
			m.serviceRefresh.mu.Unlock()
			return
		}
	}()
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
	authenticated, ok := m.memberByUUID(descriptor.HostUUID)
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
