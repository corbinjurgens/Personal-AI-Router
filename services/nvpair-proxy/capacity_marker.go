// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"unicode"

	"nvpair-shared/appdir"
	"nvpair-shared/durablefile"
	"nvpair-shared/engines"
	"nvpair-shared/nodepolicy"
)

// Markers contain capacity identities only, never request content. Recovered
// markers count as in-flight work until explicit reconciliation, without TTLs.
type capacityMarker struct {
	ID     uint64 `json:"id"`
	Engine string `json:"engine"`
	Model  string `json:"model"`
}

type capacityDocument struct {
	Version int              `json:"version"`
	Slots   []capacityMarker `json:"slots"`
}

const maxStrictCapacitySlots = 1048576

// Internal spaces are valid model identifiers; controls are rejected uniformly.
func (a *admissionController) validStrictModelKey(k admKey) bool {
	return engines.ValidRoutingID(k.engine) && k.model != "" && !nodepolicy.IsReservedAlias(k.model) &&
		strings.IndexFunc(k.model, unicode.IsControl) < 0 && a.normalize(k.engine, k.model) == k.model
}

func (a *admissionController) initStrictCapacity() {
	path, err := appdir.Path("proxy-strict-capacity.json")
	a.strict = strictAdmissionState{
		path: path, storageErr: err, markers: make(map[uint64]capacityMarker),
		models: make(map[admKey]nodeStrictModel), generation: make(map[admKey]uint64), now: timeNow,
		timer: newStrictTimer, write: durablefile.WriteFile, read: os.ReadFile,
	}
	if err == nil {
		a.loadStrictCapacity()
	}
}

func (a *admissionController) loadStrictCapacity() {
	b, err := a.strict.read(a.strict.path)
	if os.IsNotExist(err) {
		return
	}
	if err != nil {
		a.strict.storageErr = err
		return
	}
	var doc struct {
		Version *int             `json:"version"`
		Slots   []capacityMarker `json:"slots"`
	}
	if err = json.Unmarshal(b, &doc); err != nil || doc.Version == nil || *doc.Version != 1 || doc.Slots == nil || len(doc.Slots) > maxStrictCapacitySlots {
		a.strict.storageErr = fmt.Errorf("invalid strict capacity document: %v", err)
		return
	}
	markers := make(map[uint64]capacityMarker, len(doc.Slots))
	for _, m := range doc.Slots {
		if m.ID == 0 || !a.validStrictModelKey(admKey{engine: m.Engine, model: m.Model}) || markers[m.ID].ID != 0 {
			a.strict.storageErr = errors.New("invalid strict capacity slot")
			return
		}
		markers[m.ID] = m
	}
	// Adopt only after the entire document has passed validation.
	for _, m := range markers {
		a.strict.markers[m.ID] = m
		k := admKey{engine: m.Engine, model: a.normalize(m.Engine, m.Model)}
		a.inflight[k]++
		a.active++
		a.activeByEngine[k.engine]++
		if m.ID > a.nextTicket {
			a.nextTicket = m.ID
		}
	}
}

// Caller holds the controller mutex. Always preserve memory on failure. A
// replacement error means bytes landed, but is still a dispatch durability
// failure: never send inference on an unconfirmed fsync.
func (a *admissionController) persistStrictCapacityLocked() error {
	if len(a.strict.markers) > maxStrictCapacitySlots {
		return errors.New("strict capacity slot limit exceeded")
	}
	doc := capacityDocument{Version: 1, Slots: make([]capacityMarker, 0, len(a.strict.markers))}
	for _, m := range a.strict.markers {
		if !a.validStrictModelKey(admKey{engine: m.Engine, model: m.Model}) {
			return errors.New("invalid strict capacity model key")
		}
		doc.Slots = append(doc.Slots, m)
	}
	b, err := json.Marshal(doc)
	if err != nil {
		return err
	}
	if err = os.MkdirAll(filepath.Dir(a.strict.path), 0700); err != nil {
		return err
	}
	err = a.strict.write(a.strict.path, b, 0600)
	if errors.Is(err, durablefile.ErrReplaced) {
		slog.Warn("strict capacity replacement landed with uncertain durability")
	}
	return err
}

// reconcileStrictCapacity is the destination-owned evidence seam. A1b/A2 must
// provide authoritative evidence; names, time and health checks are insufficient.
func (a *admissionController) reconcileStrictCapacity(engine, model string, evidence strictTerminalEvidence) error {
	if evidence != strictAuthoritativeIdle && evidence != strictVerifiedReplacement {
		return errors.New("authoritative capacity evidence required")
	}
	a.mu.Lock()
	defer func() { st := a.noteChangeLocked(); a.mu.Unlock(); a.sendState(st) }()
	if a.strict.storageErr != nil {
		return a.strict.storageErr
	}
	k := admKey{engine: engine, model: a.normalize(engine, model)}
	removed := make(map[uint64]capacityMarker)
	for id, m := range a.strict.markers {
		if m.Engine == k.engine && a.normalize(m.Engine, m.Model) == k.model {
			// Active HTTP owners cannot be reconciled out from under a ticket.
			if _, live := a.tickets[id]; !live {
				removed[id] = m
				delete(a.strict.markers, id)
			}
		}
	}
	if len(removed) == 0 {
		return nil
	}
	if err := a.persistStrictCapacityLocked(); err != nil {
		for id, m := range removed {
			a.strict.markers[id] = m
		}
		return err
	}
	for range removed {
		a.releaseStrictSlotLocked(k)
	}
	a.signalLocked()
	return nil
}
