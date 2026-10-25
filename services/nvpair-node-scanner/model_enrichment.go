// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/json"
	"io"
	"log/slog"
	"net"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"nvpair-shared/jsonrpc"
	"nvpair-shared/modeldownload"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
)

// Model inventory is relayed over the ordinary worker JSON-RPC boundary. Bound
// its HTTP envelope before decoding too, including unknown/oversize extensions.
// Discovery retains its own, smaller frame limit.
const modelsResponseBytes = jsonrpc.WorkerFrameBytes

// enrichModelsAt decorates a node that advertises engine-manager (em) with its
// model list, fetched at the em port (/v1/models) on host. The list moved off the
// size-limited mDNS TXT onto HTTP so it can grow unbounded and be owned by
// engine-manager (which already owns engine state). A peer is fetched over cluster
// mTLS pinned to the principal it advertises, and skipped when this node holds no
// pin for it — which models a machine holds is cluster data. A transient failure
// reuses the last successful list rather than blanking it; a node advertising no em
// has no model list (nothing is carried in the record). Self is dialed over
// loopback (see enrichAt), which stays plaintext, peers at their advertised address.
func (d *daemon) enrichModelsAt(node *noderec.DirectoryNode, host string) {
	d.enrichModelsCandidates(node, []string{host})
}

// enrichModelsCandidates is enrichModelsAt with address failover. For a peer,
// fetchModels performs the pinned mTLS handshake, so a shared address belonging
// to another machine is rejected and the next published address is tried.
func (d *daemon) enrichModelsCandidates(node *noderec.DirectoryNode, hosts []string) {
	em, ok := node.Services[noderec.ServiceEngineManager]
	if !ok || len(hosts) == 0 {
		d.infoMu.Lock()
		observation := d.lastDownloads[node.HostUUID].stale()
		d.infoMu.Unlock()
		node.DownloadState, node.DownloadStateCurrent = observation.snapshot, observation.current
		return
	}
	var generation uint64
	if d.dir != nil {
		generation = d.dir.routeGeneration(node.HostUUID)
	}
	ask := func(host string) (modelInventory, bool) {
		return d.fetchModelsForPeer(node.HostUUID, host, em.Port, node.ClusterUUID, true)
	}
	key := hostKey{hostUUID: node.HostUUID, service: noderec.ServiceEngineManager}
	_, inventory, fetched := askRemembered(&d.enrichHosts, key, hosts, ask)
	d.peerServicesMu.Lock()
	defer d.peerServicesMu.Unlock()
	if d.dir != nil && d.dir.authenticated(node.HostUUID) {
		return
	}
	if d.dir != nil && d.dir.routeGeneration(node.HostUUID) != generation {
		return
	}
	if fetched {
		d.infoMu.Lock()
		accepted := d.cacheInventoryLocked(node.HostUUID, inventory)
		d.infoMu.Unlock()
		if accepted {
			inventory.applyTo(node)
			return
		}
	}
	d.infoMu.Lock()
	_, had := d.lastModels[node.HostUUID]
	if !fetched {
		if observation, exists := d.lastDownloads[node.HostUUID]; exists {
			d.lastDownloads[node.HostUUID] = observation.stale()
		}
	}
	cached := modelInventory{
		models:              d.lastModels[node.HostUUID],
		byEngine:            d.lastModelsByEngine[node.HostUUID],
		loadedByEngine:      d.lastLoadedByEngine[node.HostUUID],
		policy:              d.lastPolicy[node.HostUUID],
		downloadObservation: d.lastDownloads[node.HostUUID].stale(),
	}
	if fetched {
		// A lower revision is ignored, including its claim about freshness.
		cached.downloadObservation = d.lastDownloads[node.HostUUID].clone()
	}
	d.infoMu.Unlock()
	if had || cached.snapshot != nil || cached.current != nil {
		cached.applyTo(node)
	}
}

// cacheInventoryLocked records a successful /v1/models answer as the node's
// last-good inventory. Caller holds infoMu.
func (d *daemon) cacheInventoryLocked(hostUUID string, inventory modelInventory) bool {
	previous := d.lastDownloads[hostUUID]
	if !inventory.accepts(previous) {
		return false
	}
	if d.lastModels == nil {
		d.lastModels = make(map[string][]string)
	}
	if d.lastModelsByEngine == nil {
		d.lastModelsByEngine = make(map[string]map[string][]string)
	}
	if d.lastLoadedByEngine == nil {
		d.lastLoadedByEngine = make(map[string]map[string][]string)
	}
	if d.lastPolicy == nil {
		d.lastPolicy = make(map[string]*nodepolicy.Advertisement)
	}
	if d.lastDownloads == nil {
		d.lastDownloads = make(map[string]downloadObservation)
	}
	d.lastModels[hostUUID] = inventory.models
	d.lastModelsByEngine[hostUUID] = inventory.byEngine
	d.lastLoadedByEngine[hostUUID] = inventory.loadedByEngine
	d.lastPolicy[hostUUID] = inventory.policy
	observation := inventory.downloadObservation.clone()
	if observation.snapshot == nil && previous.snapshot != nil {
		observation = previous.stale()
	}
	d.lastDownloads[hostUUID] = observation
	return true
}

// fetchModels does one GET of a node's /v1/models, returning the
// flat model-name union, the per-engine attribution (engine name -> models),
// the per-engine loaded (in-memory) set, the node's policy advertisement, and
// whether it succeeded. Shape:
// {"models":["name",...],"modelsByEngine":{"ollama":[...]},"loadedByEngine":{"ollama":[...]},"policy":{...}}.
// modelsByEngine, loadedByEngine and policy are additive (omitempty on the
// wire); a peer that doesn't send them yields nil, and the flat union still
// enriches the node. A present engine key with [] is a successful empty
// inventory; an absent policy means the node declares nothing.
// A peer's model inventory is cluster data, so a peer is fetched over cluster
// mTLS keyed on the cluster principal it advertises (clusterUUID, from its
// cluster-uuid= TXT): this node must hold a pin for that principal, or there is
// nothing to fetch and nothing is attempted. THIS node's own engine-manager is
// dialed over loopback (both callers force that for self), which stays plain and
// therefore keeps working when this node belongs to no cluster — a standalone
// machine still has to show its own models.
func (d *daemon) fetchModels(ip string, port int, clusterUUID string) (modelInventory, bool) {
	return d.fetchModelsForPeer("", ip, port, clusterUUID, true)
}

func (d *daemon) fetchPeerModels(hostUUID, ip string, port int, clusterUUID string) (modelInventory, bool) {
	return d.fetchModelsForPeer(hostUUID, ip, port, clusterUUID, false)
}

func (d *daemon) fetchModelsForPeer(hostUUID, ip string, port int, clusterUUID string, discovering bool) (modelInventory, bool) {
	d.infoMu.Lock()
	baseline := d.lastDownloads[hostUUID]
	var sourceBaseline string
	if baseline.snapshot != nil {
		sourceBaseline = baseline.snapshot.SourceID
	}
	d.infoMu.Unlock()
	client, scheme, current, ok := d.modelsClient(hostUUID, ip, port, clusterUUID, discovering)
	if !ok {
		slog.Debug("model enrichment skipped: peer is not a pinned cluster member",
			"ip", ip, "clusterUuid", clusterUUID)
		return modelInventory{}, false
	}
	url := scheme + "://" + net.JoinHostPort(ip, strconv.Itoa(port)) + "/v1/models"
	resp, err := client.Get(url)
	if err != nil {
		slog.Debug("model enrichment failed", "ip", ip, "url", url, "err", err)
		return modelInventory{}, false
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return modelInventory{}, false
	}
	raw, err := io.ReadAll(io.LimitReader(resp.Body, modelsResponseBytes+1))
	if err != nil || len(raw) > modelsResponseBytes {
		return modelInventory{}, false
	}
	var body struct {
		Models         []json.RawMessage            `json:"models"`
		ModelsByEngine map[string][]json.RawMessage `json:"modelsByEngine"`
		LoadedByEngine map[string][]json.RawMessage `json:"loadedByEngine"`
		Policy         json.RawMessage              `json:"policy"`
		DownloadState  json.RawMessage              `json:"downloadState"`
	}
	if err := json.Unmarshal(raw, &body); err != nil {
		return modelInventory{}, false
	}
	members, object := objectMembers(raw)
	if !object {
		return modelInventory{}, false
	}
	seenDownload, invalidDownload := false, false
	for _, member := range members {
		if strings.EqualFold(member.key, "downloadState") {
			invalidDownload = invalidDownload || seenDownload || member.key != "downloadState"
			seenDownload = true
		}
	}
	models, err := decodeInventoryModelNames(body.Models)
	if err != nil {
		return modelInventory{}, false
	}
	byEngine, err := decodeInventoryEngineModels(body.ModelsByEngine)
	if err != nil {
		return modelInventory{}, false
	}
	loadedByEngine, err := decodeInventoryEngineModels(body.LoadedByEngine)
	if err != nil {
		return modelInventory{}, false
	}
	// Decoder errors can quote the peer's field names, so only a fixed
	// category is logged.
	policy, rejected, err := decodeInventoryPolicy(body.Policy)
	if err != nil {
		slog.Debug("model enrichment: invalid policy advertisement", "ip", ip)
		return modelInventory{}, false
	}
	if rejected != "" {
		slog.Debug("model enrichment: ignoring part of a policy advertisement", "ip", ip, "rejected", rejected)
	}
	if !current() {
		return modelInventory{}, false
	}
	var observation downloadObservation
	if invalidDownload || len(body.DownloadState) > 0 && !bytes.Equal(bytes.TrimSpace(body.DownloadState), []byte("null")) {
		snapshot, err := modeldownload.Parse(body.DownloadState)
		valid := err == nil && !invalidDownload
		observation.current = &valid
		if valid {
			observation.snapshot = &snapshot
		} else {
			// Routing consumers read canonical names without interpreting this
			// extension. Invalid lifecycle evidence cannot publish ready names.
			models = []string{}
			if byEngine == nil {
				byEngine = make(map[string][]string)
			}
			for engine := range byEngine {
				byEngine[engine] = []string{}
			}
		}
	}
	return modelInventory{models: models, byEngine: byEngine, loadedByEngine: loadedByEngine, policy: policy,
		downloadObservation: observation, sourceBaseline: sourceBaseline}, true
}

// Keep model tokens raw until the shared validator has seen their original
// encoding. A typed string decoder alone would repair malformed UTF-8. Invalid
// identifiers are removed individually; unrelated type errors still fail the
// document, retaining the caller's last-good inventory.
func decodeInventoryModelNames(raw []json.RawMessage) ([]string, error) {
	if raw == nil {
		return nil, nil
	}
	models := make([]string, 0, len(raw))
	for _, entry := range raw {
		valid := nodepolicy.ValidateModelIdentifierJSON(entry) == nil
		var model string
		if err := json.Unmarshal(entry, &model); err != nil {
			return nil, err
		}
		if valid {
			models = append(models, model)
		}
	}
	return models, nil
}

func decodeInventoryEngineModels(raw map[string][]json.RawMessage) (map[string][]string, error) {
	if raw == nil {
		return nil, nil
	}
	models := make(map[string][]string, len(raw))
	for engine, entries := range raw {
		clean, err := decodeInventoryModelNames(entries)
		if err != nil {
			return nil, err
		}
		models[engine] = clean
	}
	return models, nil
}

// decodeInventoryPolicy reads the policy object of a /v1/models document; nil
// when it is absent or null, which means the node declares nothing. Its strict
// admission capability only makes the node eligible for hybrid routing, so a
// problem confined to that capability never costs the node its model
// inventory, which concrete routing needs:
//
//   - A strictAdmission key in another letter case rejects the whole
//     advertisement (the strict contract's rule against aliases): the node is
//     kept with its last-good inventory under the existing freshness rules.
//   - A malformed strictAdmission (invalid version 1 contents or envelope, or
//     null) rejects only the capability: the advertisement is kept without it.
//
// A capability of an unsupported version decodes as such and is kept for the
// consumers to exclude. Any other invalid advertisement fails the document, as
// it did before the capability existed. rejected names what was dropped, for
// the log: rejectedAdvertisement, rejectedCapability or empty.
//
// The capability is removed from the document's own bytes, keeping every other
// member in its order, duplicates included, so the remaining fields decode
// exactly as the shared decoder would decode them without it.
func decodeInventoryPolicy(raw json.RawMessage) (adv *nodepolicy.Advertisement, rejected string, err error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, "", nil
	}
	var decoded nodepolicy.Advertisement
	err = json.Unmarshal(raw, &decoded)
	if err == nil {
		return &decoded, "", nil
	}
	members, ok := objectMembers(raw)
	if !ok {
		return nil, "", err
	}
	var rest bytes.Buffer
	rest.WriteByte('{')
	found := false
	for _, m := range members {
		if m.key == strictAdmissionKey {
			found = true
			continue
		}
		if strings.EqualFold(m.key, strictAdmissionKey) {
			return nil, rejectedAdvertisement, err
		}
		key, keyErr := json.Marshal(m.key)
		if keyErr != nil {
			return nil, "", err
		}
		if rest.Len() > 1 {
			rest.WriteByte(',')
		}
		rest.Write(key)
		rest.WriteByte(':')
		rest.Write(m.value)
	}
	rest.WriteByte('}')
	if !found {
		return nil, "", err
	}
	decoded = nodepolicy.Advertisement{}
	if json.Unmarshal(rest.Bytes(), &decoded) != nil {
		return nil, "", err
	}
	return &decoded, rejectedCapability, nil
}

const (
	// strictAdmissionKey is the exact name of the capability in the advertisement.
	strictAdmissionKey    = "strictAdmission"
	rejectedAdvertisement = "advertisement with an aliased strictAdmission key"
	rejectedCapability    = "invalid strictAdmission capability"
)

// objectMember is one member of a JSON object, in document order.
type objectMember struct {
	key   string
	value json.RawMessage
}

// objectMembers lists the members of the JSON object raw in document order,
// duplicates included; ok is false when raw is not one well-formed object.
func objectMembers(raw []byte) (members []objectMember, ok bool) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	if tok, err := dec.Token(); err != nil || tok != json.Delim('{') {
		return nil, false
	}
	for dec.More() {
		tok, err := dec.Token()
		if err != nil {
			return nil, false
		}
		key, isKey := tok.(string)
		if !isKey {
			return nil, false
		}
		var value json.RawMessage
		if err := dec.Decode(&value); err != nil {
			return nil, false
		}
		members = append(members, objectMember{key: key, value: value})
	}
	if tok, err := dec.Token(); err != nil || tok != json.Delim('}') {
		return nil, false
	}
	return members, true
}

// modelsClient picks the transport for a model fetch: the plain client over
// loopback (this node's own engine-manager, which serves loopback in plaintext in
// every membership state), or a per-peer pinned mTLS client for a LAN peer. ok is
// false when a peer holds no pin here, which is the client-side cluster gate —
// there is deliberately no plaintext fallback for a peer.
func (d *daemon) modelsClient(hostUUID, ip string, port int, clusterUUID string, discovering bool) (*http.Client, string, func() bool, bool) {
	if isLoopbackHost(ip) {
		return d.modelsHTTP, "http", func() bool { return true }, d.modelsHTTP != nil
	}
	client, current, ok := d.peerServiceClient(hostUUID, clusterUUID, ip, port, noderec.ServiceEngineManager, discovering)
	return client, "https", current, ok
}

// refreshModelsOnce snapshots every directory node advertising engine-manager
// (em) with a usable IP and refreshes each concurrently, bounded by
// modelsRefreshConcurrency so an unexpectedly large network can't spawn an
// unbounded burst of fetches. Running the sweep to completion (wg.Wait) before
// the loop reads the next tick makes overlapping sweeps impossible, and each
// per-node goroutine uses modelsHTTP's own timeout, so one slow peer never
// consumes another's budget.
func (d *daemon) refreshModelsOnce() {
	sem := make(chan struct{}, modelsRefreshConcurrency)
	var wg sync.WaitGroup
	for _, n := range d.dir.snapshot(noderec.ServiceEngineManager) {
		em, ok := n.Services[noderec.ServiceEngineManager]
		if !ok || n.IP == "" {
			continue
		}
		wg.Add(1)
		sem <- struct{}{}
		hosts := append([]string(nil), n.CandidateIPs()...)
		go func(hostUUID, guardIP string, hosts []string, port int, clusterUUID string) {
			defer wg.Done()
			defer func() { <-sem }()
			d.refreshNodeModelsCandidates(hostUUID, guardIP, hosts, port, clusterUUID)
		}(n.HostUUID, n.IP, hosts, em.Port, n.ClusterUUID)
	}
	wg.Wait()
}

// refreshNodeModels fetches one peer's /v1/models and, on success, applies it to
// the directory under the removed/re-addressed guard (directory.applyModels).
// A failed fetch retains the last successful inventory and operation details,
// but withdraws download freshness and emits that change once. On a successful
// change it refreshes the enrichment caches
// (so a later browse-driven fallback can't revert the directory to a stale
// value) and emits exactly one discovery:node-updated carrying the full node.
// Returns whether it emitted an update (used by tests).
//
// The fetch dials our OWN engine-manager over loopback, never the advertised
// LAN ip=: this loop — not the registry-driven publishSelf — is what converges
// model-only changes (a pull/delete, a late engine start) that don't move the
// registry, so on the Windows self-LAN firewall block this daemon guards
// against, a LAN dial would make self's model list freeze at its
// startup value. The stored ip is still passed to applyModels as the
// stale-address guard, so an in-flight result can't overwrite a re-addressed
// node. Peers are still dialed at their advertised address.
func (d *daemon) refreshNodeModels(hostUUID, ip string, emPort int, clusterUUID string) bool {
	return d.refreshNodeModelsCandidates(hostUUID, ip, []string{ip}, emPort, clusterUUID)
}

// refreshNodeModelsCandidates is refreshNodeModels with address failover. The
// canonical IP remains the stale-result guard even when a later address is the
// one that answered.
//
// Which address is asked is askRemembered's business — the one that answered last
// sweep, alone — and it shares that memory with the enrichment path, so both
// converge on one address per peer instead of each rediscovering it.
func (d *daemon) refreshNodeModelsCandidates(hostUUID, guardIP string, hosts []string, emPort int, clusterUUID string) bool {
	generation := d.dir.routeGeneration(hostUUID)
	if d.reg != nil && hostUUID == d.reg.record().HostUUID {
		hosts = []string{loopbackHost}
	}
	ask := func(host string) (modelInventory, bool) {
		return d.fetchPeerModels(hostUUID, host, emPort, clusterUUID)
	}
	key := hostKey{hostUUID: hostUUID, service: noderec.ServiceEngineManager}
	_, inventory, fetched := askRemembered(&d.enrichHosts, key, hosts, ask)
	d.peerServicesMu.Lock()
	defer d.peerServicesMu.Unlock()
	if d.dir.routeGeneration(hostUUID) != generation {
		return false
	}
	if d.dir.authenticated(hostUUID) {
		d.dir.mu.RLock()
		claim := d.dir.peerServices[hostUUID]
		d.dir.mu.RUnlock()
		incarnation, admitted := d.mesh.PeerServiceIncarnation(hostUUID)
		if !admitted || incarnation != claim.Incarnation || !peerServicesFresh(claim, time.Now()) {
			fetched = false
		}
	}
	if !fetched {
		node, changed, valid := d.dir.markDownloadsStale(hostUUID, guardIP, emPort)
		if !valid {
			return false
		}
		d.infoMu.Lock()
		if observation, ok := d.lastDownloads[hostUUID]; ok {
			d.lastDownloads[hostUUID] = observation.stale()
		}
		d.infoMu.Unlock()
		if changed {
			d.emit(noderec.NotifyNodeUpdated, node)
		}
		return changed
	}
	// Refresh the enrichment cache BEFORE touching the directory: the caches
	// (infoMu) and the directory (dir.mu) are guarded by different locks, so if
	// the directory were updated first, a concurrent onBrowse->enrichModelsAt whose
	// own fetch fails would fall back to a still-stale lastModels and upsert the
	// old inventory, reverting this update. Writing the cache first means that
	// fallback reads the fresh value. The only side effect — caching a value for
	// a node applyModels then finds removed — is benign (it is the last-known-good
	// value the cache exists to preserve, and forget() clears it on eviction).
	d.infoMu.Lock()
	accepted := d.cacheInventoryLocked(hostUUID, inventory)
	d.infoMu.Unlock()
	if !accepted {
		return false
	}

	node, changed, valid := d.dir.applyModels(hostUUID, guardIP, emPort, inventory)
	if !valid {
		return false
	}
	if changed {
		d.emit(noderec.NotifyNodeUpdated, node)
	}
	return changed
}
