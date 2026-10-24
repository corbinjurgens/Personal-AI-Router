// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// Discovery is the proxy's routing-target set. Post-cutover the proxy
// no longer runs its own _nvpair-ollama browse: routing targets are pushed down
// from the broker's discovery relay (discovery:nodes snapshots for the ol
// service) into the subscribed overlay, merged with user-added manual nodes. The
// proxy is
// itself advertised — as an ol service — by the node-scanner daemon's single
// _nvpair-node record, keyed off the engine port the broker's poller registers.
//
// The routable Node projection (IP / withPrimaryIP) and the manual-node
// overlay live here; request-path reachability (TCP-probe + failover) lives in
// proxy.go and http_handler.go.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"slices"
	"sync"

	"nvpair-shared/discovery"
	"nvpair-shared/netpick"
	"nvpair-shared/nodepolicy"
)

// uuidFromTXT extracts a node's stable uuid= from its TXT records. Kept as a
// thin re-export of the shared helper (it was triplicated across the two proxies
// and the scanner before consolidation).
var uuidFromTXT = discovery.UUIDFromTXT

// Node is the proxy's routable view of a node. It adds a canonical dialable IP
// field over the discovered node shape.
type Node struct {
	ID        string   `json:"id"`
	Host      string   `json:"host"`
	Port      int      `json:"port"`
	Addresses []string `json:"addresses"`
	TXT       []string `json:"txt"`
	// Models is the latest model inventory carried by the broker's discovery
	// snapshot. Model-bearing inference is eligible only when this list
	// advertises the requested model; an empty list stays in discovery but is
	// not an inference candidate until a later inventory update.
	Models []string `json:"models,omitempty"`
	// Loaded is the subset of Models the node reports resident in memory for
	// this engine (DirectoryNode.LoadedByEngine). Tier routing prefers a warm
	// node. Internal routing metadata like ClusterUUID: it changes with every
	// load and unload, so it is neither part of the outward node contract nor
	// of nodeEqual, which would otherwise announce node/updated for each one.
	Loaded []string `json:"-"`
	// IP is the single canonical LAN address a consumer should dial/display for
	// this node, resolved via the shared netpick ranker: the node's
	// own ip= TXT if present, else the best-scored advertised IPv4. It is
	// stamped onto outbound node/* notifications only (see withPrimaryIP) so a
	// downstream consumer agrees on the same address the proxy routes to.
	IP string `json:"ip,omitempty"`
	// ClusterUUID is the relay peer's cluster principal (its mTLS cert UUID),
	// carried from the discovery DirectoryNode. Non-empty only for a clustered
	// peer; it is the key used to pin the peer's server cert when dialing its
	// promoted proxy over cluster mTLS. Whether we actually hold that pin is
	// resolved against the live mesh at routing time (resolveCandidates), never
	// cached here: a cached answer goes stale the moment a peer is paired or
	// removed. Internal routing metadata, not part of the proxy's outward node
	// contract.
	ClusterUUID string `json:"-"`
	// Roles and PolicyRevision are the node's own policy advertisement
	// (DirectoryNode.Policy): what it declares it serves per tier, and the
	// revision of the policy that says so. Nil and empty when the node
	// publishes none, which means it declares nothing. Internal routing
	// metadata like Loaded, so a role edit does not announce node/updated.
	Roles          map[string]nodepolicy.AdvertisedRole `json:"-"`
	ServedModels   nodepolicy.ServedModels              `json:"-"`
	PolicyRevision string                               `json:"-"`
	// PausedBy is an advisory policy hint fetched over authenticated LAN
	// transport, with the scanner's existing plaintext loopback exception.
	PausedBy nodepolicy.PauseOrigin `json:"-"`
	// StrictAdmission is the strict admission capability from the same
	// advertisement, nil when the node publishes none. It is shared with the
	// decoded advertisement and never modified. Read it through strictModel.
	StrictAdmission *nodepolicy.StrictAdmissionCapability `json:"-"`
}

// Manual entries arrive as raw JSON too: guard model tokens before decoding
// can repair their UTF-8. Directory snapshots use the same filter at ingress.
func (n *Node) UnmarshalJSON(data []byte) error {
	clean, err := filterModelInventoryJSON(data, "models")
	if err != nil {
		return err
	}
	type wire Node
	var next wire
	if err := json.Unmarshal(clean, &next); err != nil {
		return err
	}
	*n = Node(next).filteredModels()
	return nil
}

func filterModelNames(models []string) []string {
	if models == nil {
		return nil
	}
	clean := make([]string, 0, len(models))
	for _, model := range models {
		if nodepolicy.ValidateModelIdentifier(model) == nil {
			clean = append(clean, model)
		}
	}
	return clean
}

func (n Node) filteredModels() Node {
	n.Models = filterModelNames(n.Models)
	n.Loaded = filterModelNames(n.Loaded)
	if n.Roles != nil {
		roles := make(map[string]nodepolicy.AdvertisedRole, len(n.Roles))
		for tier, role := range n.Roles {
			if nodepolicy.ValidateModelIdentifier(role.Model) == nil {
				roles[tier] = role
			}
		}
		n.Roles = roles
	}
	if n.ServedModels != nil {
		served := make(nodepolicy.ServedModels, 0, len(n.ServedModels))
		for _, ref := range n.ServedModels {
			if nodepolicy.ValidateModelIdentifier(ref.Model) == nil {
				served = append(served, ref)
			}
		}
		n.ServedModels = served
	}
	if n.StrictAdmission != nil && n.StrictAdmission.Version == nodepolicy.StrictAdmissionVersion {
		models := n.StrictAdmission.Models
		for i, entry := range models {
			if nodepolicy.ValidateModelIdentifier(entry.Model) == nil {
				continue
			}
			// Clean advertisements retain their original pointer and backing.
			// Clone only after finding an entry that actually needs removal.
			capability := *n.StrictAdmission
			capability.Models = make([]nodepolicy.StrictAdmissionModel, 0, len(models)-1)
			capability.Models = append(capability.Models, models[:i]...)
			for _, remaining := range models[i+1:] {
				if nodepolicy.ValidateModelIdentifier(remaining.Model) == nil {
					capability.Models = append(capability.Models, remaining)
				}
			}
			n.StrictAdmission = &capability
			break
		}
	}
	return n
}

// Keep raw valid siblings and null/empty distinctions. Only model containers
// are inspected; unrelated fields retain their normal decoder behavior.
func filterModelNamesJSON(raw []byte) ([]byte, error) {
	var entries []json.RawMessage
	if err := json.Unmarshal(raw, &entries); err != nil {
		return nil, fmt.Errorf("invalid model inventory")
	}
	if entries == nil {
		return raw, nil
	}
	clean := make([]json.RawMessage, 0, len(entries))
	for _, entry := range entries {
		if nodepolicy.ValidateModelIdentifierJSON(entry) == nil {
			clean = append(clean, entry)
		}
	}
	return json.Marshal(clean)
}

// Rewrite only recognized values, in document order, so folded/repeated
// enclosing members keep encoding/json's existing merge behavior.
func filterModelInventoryJSON(data []byte, modelFields ...string) ([]byte, error) {
	if !ownedJSONValid(data) || !isJSONObject(data) {
		return nil, fmt.Errorf("invalid model inventory")
	}
	var out []byte
	last := 0
	var filterErr error
	visitObject(data, func(key []byte, start, end int) {
		if filterErr != nil {
			return
		}
		field := ""
		for _, name := range modelFields {
			if jsonKeyEqual(key, name, true) {
				field = name
				break
			}
		}
		raw := data[start:end]
		var clean []byte
		var err error
		switch field {
		case "models":
			clean, err = filterModelNamesJSON(raw)
		case "modelsByEngine", "loadedByEngine":
			var byEngine map[string]json.RawMessage
			if err = json.Unmarshal(raw, &byEngine); err == nil && byEngine != nil {
				for engine, models := range byEngine {
					byEngine[engine], err = filterModelNamesJSON(models)
					if err != nil {
						break
					}
				}
				if err == nil {
					clean, err = json.Marshal(byEngine)
				}
			}
		case "policy":
			if !bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
				clean, err = nodepolicy.FilterAdvertisementModels(raw)
			}
		}
		if err != nil {
			filterErr = fmt.Errorf("invalid model inventory")
			return
		}
		if clean != nil {
			out = append(out, data[last:start]...)
			out = append(out, clean...)
			last = end
		}
	})
	if filterErr != nil {
		return nil, filterErr
	}
	return append(out, data[last:]...), nil
}

func filterDiscoveryModelsJSON(data []byte) ([]byte, error) {
	if !ownedJSONValid(data) || !isJSONObject(data) {
		return nil, fmt.Errorf("invalid discovery snapshot")
	}
	var out []byte
	last := 0
	var filterErr error
	visitObject(data, func(key []byte, start, end int) {
		if filterErr != nil || !jsonKeyEqual(key, "nodes", true) {
			return
		}
		var nodes []json.RawMessage
		if err := json.Unmarshal(data[start:end], &nodes); err != nil {
			filterErr = err
			return
		}
		for i, node := range nodes {
			if bytes.Equal(bytes.TrimSpace(node), []byte("null")) {
				continue
			}
			var err error
			nodes[i], err = filterModelInventoryJSON(node, "models", "modelsByEngine", "loadedByEngine", "policy")
			if err != nil {
				filterErr = err
				return
			}
		}
		clean, err := json.Marshal(nodes)
		if err != nil {
			filterErr = err
			return
		}
		out = append(out, data[last:start]...)
		out = append(out, clean...)
		last = end
	})
	if filterErr != nil {
		return nil, fmt.Errorf("invalid discovery snapshot")
	}
	return append(out, data[last:]...), nil
}

// strictModel returns the strict admission entry n advertises for engine and
// model. model must already be named the way that engine's facade names it,
// which is how the destination publishes it. Only a valid capability of the
// supported version counts: a node that advertises none (an older build, or one
// not ready now) or one of an unsupported version is never a hybrid local
// candidate. Concrete routing does not consult it. The entry is an eligibility
// hint; the destination revalidates every strict request.
func (n Node) strictModel(engine, model string) (nodepolicy.StrictAdmissionModel, bool) {
	if nodepolicy.ValidateModelIdentifier(model) != nil || n.StrictAdmission.State() != nodepolicy.StrictCapabilityValid {
		return nodepolicy.StrictAdmissionModel{}, false
	}
	for _, m := range n.StrictAdmission.Models {
		if m.Engine == engine && m.Model == model {
			return m, true
		}
	}
	return nodepolicy.StrictAdmissionModel{}, false
}

// withPrimaryIP returns a copy of the node with IP resolved by the shared ranker
// (netpick.Primary over its TXT + Addresses).
func (n Node) withPrimaryIP() Node {
	n.IP = netpick.Primary(n.TXT, n.Addresses)
	return n
}

// Discovery holds the proxy's routing targets: the relay-fed subscribed overlay
// and the user-added manual overlay.
type Discovery struct {
	mu          sync.RWMutex
	manualNodes map[string]Node
	// subscribedNodes are routing targets pushed down by the broker's discovery
	// relay (discovery:nodes snapshots for the ol service), keyed by node ID (the
	// directory instance name).
	subscribedNodes map[string]Node
}

func NewDiscovery() *Discovery {
	return &Discovery{
		manualNodes:     make(map[string]Node),
		subscribedNodes: make(map[string]Node),
	}
}

// Nodes returns the merged subscribed + manual node set (manual entries that
// aren't also present via the relay are appended).
func (d *Discovery) Nodes() []Node {
	d.mu.RLock()
	defer d.mu.RUnlock()
	out := make([]Node, 0, len(d.manualNodes)+len(d.subscribedNodes))
	seen := make(map[string]struct{}, len(d.subscribedNodes))
	for id, n := range d.subscribedNodes {
		out = append(out, n)
		seen[id] = struct{}{}
	}
	for id, n := range d.manualNodes {
		if _, exists := seen[id]; !exists {
			out = append(out, n)
		}
	}
	return out
}

// Has reports whether a node id is still present in either overlay. Used to
// watch an in-flight request's target: a node that has dropped out of discovery
// is not coming back to answer the attempt already sent to it.
func (d *Discovery) Has(id string) bool {
	d.mu.RLock()
	defer d.mu.RUnlock()
	if _, ok := d.subscribedNodes[id]; ok {
		return true
	}
	_, ok := d.manualNodes[id]
	return ok
}

// SetSubscribed replaces the relay-fed routing overlay with the given set and
// reports what changed versus the previous set (keyed by node ID): nodes newly
// present, nodes whose routable details changed, and nodes that dropped out. The
// broker pushes the full filtered snapshot on every change, so the overlay is
// replaced wholesale; the returned diff lets the caller emit node/discovered|
// updated|removed so a consumer (the UI) learns which peers currently run this
// engine — the signal the proxy's own _nvpair-ollama browse provided before
// discovery was consolidated onto the relay. Manual nodes are a separate overlay
// and are untouched.
func (d *Discovery) SetSubscribed(nodes []Node) (discovered, updated, removed []Node) {
	d.mu.Lock()
	defer d.mu.Unlock()
	next := make(map[string]Node, len(nodes))
	for _, n := range nodes {
		n = n.filteredModels()
		next[n.ID] = n
		switch prev, ok := d.subscribedNodes[n.ID]; {
		case !ok:
			discovered = append(discovered, n)
		case !nodeEqual(prev, n):
			updated = append(updated, n)
		}
	}
	for id, prev := range d.subscribedNodes {
		if _, ok := next[id]; !ok {
			removed = append(removed, prev)
		}
	}
	d.subscribedNodes = next
	return discovered, updated, removed
}

// nodeEqual reports whether two routable Nodes carry the same routing/display
// identity — the fields a consumer dials or renders. A change in any of them
// warrants a node/updated. Loaded, ClusterUUID, Roles, PolicyRevision, PausedBy
// and StrictAdmission are internal routing metadata and deliberately not compared.
func nodeEqual(a, b Node) bool {
	return a.ID == b.ID && a.Host == b.Host && a.Port == b.Port && a.IP == b.IP &&
		slices.Equal(a.Addresses, b.Addresses) && slices.Equal(a.TXT, b.TXT) &&
		slices.Equal(a.Models, b.Models)
}

func (d *Discovery) AddManual(node Node) (added bool) {
	node = node.filteredModels()
	d.mu.Lock()
	defer d.mu.Unlock()
	_, exists := d.manualNodes[node.ID]
	d.manualNodes[node.ID] = node
	return !exists
}

func (d *Discovery) RemoveManual(id string) (removed bool) {
	d.mu.Lock()
	defer d.mu.Unlock()
	_, exists := d.manualNodes[id]
	if exists {
		delete(d.manualNodes, id)
	}
	return exists
}

func (d *Discovery) IsManual(id string) bool {
	d.mu.RLock()
	defer d.mu.RUnlock()
	_, exists := d.manualNodes[id]
	return exists
}
