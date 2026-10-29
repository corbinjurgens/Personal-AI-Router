// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"maps"
	"math"
	"sort"
	"strconv"
	"strings"
	"sync"

	"nvpair-shared/engines"
	"nvpair-shared/modeldownload"
	"nvpair-shared/nodepolicy"
)

type modelContentWriterKey struct{}

func (e *Executor) acquireModelRead(engine, model string, owned bool) (func(), error) {
	manifest, ok := e.reg.Get(engine)
	if !ok {
		return nil, errDownloadBusy
	}
	key, err := downloadKey(engine, model, manifest.ModelNamingOrDefault())
	if err != nil {
		return nil, err
	}
	return e.downloads.acquireRead(key, owned)
}

func (e *Executor) acquireLoadRead(ctx context.Context, st *engineState, action string, params json.RawMessage) (func(), error) {
	binding, load := st.manifest.operationBinding(engines.OperationLoad)
	if !load || action != binding.Action {
		return func() {}, nil
	}
	var fields map[string]json.RawMessage
	if len(params) > 0 && json.Unmarshal(params, &fields) != nil {
		return nil, errDownloadBusy
	}
	owned, _ := ctx.Value(modelContentWriterKey{}).(string)
	if _, present := fields[binding.modelParam()]; !present {
		// Conventional actions may intentionally omit a model. Preserve their
		// opaque payload, fencing all content in that engine for this read.
		return e.downloads.acquireRead(downloadContentKey{engine: st.manifest.Engine}, owned == st.manifest.Engine)
	}
	var model string
	if json.Unmarshal(fields[binding.modelParam()], &model) != nil {
		return nil, errDownloadBusy
	}
	return e.acquireModelRead(st.manifest.Engine, model, owned == st.manifest.Engine)
}

// A Delete resolves only this key, following a successful native mutation and
// a same-run unmasked absence observation. Ordinary canonical inventory is
// deliberately unusable here: the marker itself masks that inventory.
func (e *Executor) resolveDeletedDownload(ctx context.Context, st *engineState, field string, params, result json.RawMessage) {
	var fields map[string]json.RawMessage
	var model string
	if json.Unmarshal(params, &fields) != nil || json.Unmarshal(fields[field], &model) != nil {
		return
	}
	key, err := downloadKey(st.manifest.Engine, model, st.manifest.ModelNamingOrDefault())
	if err != nil {
		return
	}
	e.downloads.mu.Lock()
	marker, exists := e.downloads.unsafe[key]
	e.downloads.mu.Unlock()
	if !exists {
		return
	}
	var outcome struct {
		Success *bool  `json:"success"`
		Error   string `json:"error"`
	}
	if json.Unmarshal(result, &outcome) == nil && (outcome.Error != "" || outcome.Success != nil && !*outcome.Success) {
		return
	}
	run := e.engineRunID(st.manifest.Engine)
	if run == "" {
		return
	}
	list, ok := st.manifest.Actions["list_models"]
	if !ok || list.Result == nil {
		return
	}
	var raw json.RawMessage
	if pull, ok := st.manifest.Actions[pullModelAction]; ok && pull.ProgressProtocol == pullProgressProtocolLlamaCPPModelsSSE {
		raw, err = e.reloadLlamaModelsResult(ctx, st)
	} else {
		raw, err = e.Action(ctx, st.manifest.Engine, "list_models", nil)
	}
	if err != nil || e.engineRunID(st.manifest.Engine) != run {
		return
	}
	if !downloadAbsent(raw, list.Result, key, st.manifest.ModelNamingOrDefault()) {
		return
	}
	if e.downloads.resolveMarker(key, marker.ID) {
		e.pokeLoaded()
	}
}

// Absence requires identifying every native row, including rows excluded by a
// display/residency filter. A malformed row may be the affected model.
func downloadAbsent(raw json.RawMessage, spec *ActionResult, key downloadContentKey, naming string) bool {
	array, ok := downloadIdentityField(raw, spec.Array)
	if !ok || bytes.Equal(bytes.TrimSpace(array), []byte("null")) {
		return false
	}
	var rows []json.RawMessage
	if json.Unmarshal(array, &rows) != nil {
		return false
	}
	for _, row := range rows {
		identity, ok := downloadIdentityField(row, spec.Field)
		var name string
		if !ok || nodepolicy.ValidateModelIdentifierJSON(identity) != nil || json.Unmarshal(identity, &name) != nil {
			return false
		}
		observed, err := downloadKey(key.engine, name, naming)
		if err != nil || observed == key {
			return false
		}
	}
	return true
}

// Preserve additional native fields, but never collapse ambiguous identity
// fields (including case variants) into an absence proof.
func downloadIdentityField(raw json.RawMessage, field string) (json.RawMessage, bool) {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	if token, err := decoder.Token(); err != nil || token != json.Delim('{') {
		return nil, false
	}
	var found json.RawMessage
	for decoder.More() {
		token, err := decoder.Token()
		name, ok := token.(string)
		if err != nil || !ok {
			return nil, false
		}
		var value json.RawMessage
		if decoder.Decode(&value) != nil {
			return nil, false
		}
		if strings.EqualFold(name, field) {
			if name != field || found != nil {
				return nil, false
			}
			found = value
		}
	}
	if _, err := decoder.Token(); err != nil {
		return nil, false
	}
	if _, err := decoder.Token(); err != io.EOF {
		return nil, false
	}
	return found, found != nil
}

var (
	errDownloadBusy     = errors.New("model content is busy")
	errDownloadCapacity = errors.New("model download capacity exhausted")
	errDownloadRecovery = errors.New("model download recovery required")
)

type modelDownloadAttempt struct {
	key    downloadContentKey
	naming string
	entry  modeldownload.Entry
	// Marker publications and terminal retirement, reserved before admission.
	authorityHeld uint64
}

// modelDownloads is a bounded content gate, not an execution owner. Callers
// release mu before notifications, I/O and any other owner's wait. journalMu
// orders capture/save/adopt across engines and is acquired outside mu.
type modelDownloads struct {
	mu                  sync.Mutex
	journalMu           sync.Mutex
	source              string
	revision            uint64
	heldRevisions       uint64
	authorityGeneration uint64
	authorityHeld       uint64
	authorityUncertain  bool
	// Current-run copy additions reserve removal in the authority generation,
	// independently of the public download revision. Loaded markers retain provenance.
	copyRetirements      map[string]bool
	unsafeOrigins        map[downloadContentKey]string
	recoveryRequired     bool
	copyRecoveryRequired bool
	copiesComplete       bool
	active               map[downloadContentKey]*modelDownloadAttempt
	unsafe               map[downloadContentKey]modelDownloadMarker
	copies               map[string]modelCopyMarker
	outcomes             map[downloadContentKey]modeldownload.Entry
	recent               []downloadContentKey
	writers              map[string]bool
	readers              map[string]int
	store                modelDownloadStore
}

func newModelDownloads(source, baseDir string, naming map[string]string) *modelDownloads {
	d := &modelDownloads{source: source, active: make(map[downloadContentKey]*modelDownloadAttempt), unsafe: make(map[downloadContentKey]modelDownloadMarker), outcomes: make(map[downloadContentKey]modeldownload.Entry), writers: make(map[string]bool), readers: make(map[string]int), store: newModelDownloadStore(baseDir), copyRetirements: make(map[string]bool), unsafeOrigins: make(map[downloadContentKey]string)}
	markers, copies, copiesComplete, copyRecoveryRequired, err := d.store.loadJournal()
	d.copies = copies
	d.copiesComplete = copiesComplete
	d.copyRecoveryRequired = copyRecoveryRequired
	if err != nil {
		d.recoveryRequired = true
		d.copyRecoveryRequired = true
		return d
	}
	d.unsafe = markers
	for key, marker := range markers {
		if naming[key.engine] != marker.Naming {
			d.recoveryRequired = true
		}
		d.outcomes[key] = modeldownload.Entry{ID: marker.ID, Engine: marker.Engine, Model: marker.Model, Phase: "unknown", BlocksReadiness: true, Reason: "interrupted"}
	}
	for _, marker := range copies {
		if marker.Phase == "uncertain" || naming[marker.Engine] != marker.Naming {
			d.copyRecoveryRequired = true
		}
	}
	return d
}

// Caller holds mu. Reserve all mandatory future transitions before admitting
// work. Progress notifications never consume authority retirement capacity.
func (d *modelDownloads) reserveAuthorityLocked(steps uint64) bool {
	remaining := math.MaxUint64 - d.authorityGeneration
	if remaining < d.authorityHeld || remaining-d.authorityHeld < steps {
		return false
	}
	d.authorityHeld += steps
	return true
}

func (d *modelDownloads) advanceAuthorityLocked() {
	d.authorityHeld--
	d.authorityGeneration++
}

func (d *modelDownloads) advanceLocked() error {
	if d.revision == math.MaxUint64 {
		return errDownloadCapacity
	}
	d.revision++
	return nil
}

func (d *modelDownloads) reserve(engine, model, naming string) (*modelDownloadAttempt, error) {
	key, err := downloadKey(engine, model, naming)
	if err != nil {
		return nil, err
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.recoveryRequired {
		return nil, errDownloadRecovery
	}
	if d.active[key] != nil {
		return nil, errDownloadBusy
	}
	// Keep one terminal revision for every accepted request plus this one's
	// initial admission and in-progress transition. Progress cannot consume it.
	if math.MaxUint64-d.revision < uint64(len(d.active)*2+3)+d.heldRevisions {
		return nil, errDownloadCapacity
	}
	slots := len(d.unsafe) + len(d.copies)
	for k := range d.active {
		if _, ok := d.unsafe[k]; !ok {
			slots++
		}
	}
	if _, retained := d.unsafe[key]; !retained && slots >= modeldownload.MaxUnsafeEntries {
		return nil, errDownloadCapacity
	}
	if !d.reserveAuthorityLocked(5) {
		return nil, errDownloadCapacity
	}
	if err := d.advanceLocked(); err != nil {
		return nil, err
	}
	d.removeRecentLocked(key)
	a := &modelDownloadAttempt{key: key, naming: naming, entry: modeldownload.Entry{ID: newOpID(), Engine: engine, Model: model, Phase: "queued", BlocksReadiness: true}, authorityHeld: 4}
	d.active[key] = a
	d.advanceAuthorityLocked()
	d.outcomes[key] = a.entry
	return a, nil
}

func (d *modelDownloads) removeRecentLocked(key downloadContentKey) {
	for i, k := range d.recent {
		if k == key {
			d.recent = append(d.recent[:i], d.recent[i+1:]...)
			break
		}
	}
	delete(d.outcomes, key)
}

func (d *modelDownloads) update(a *modelDownloadAttempt, phase string, percent *int) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.active[a.key] != a {
		return errDownloadBusy
	}
	if a.entry.Phase == phase && equalDownloadPercent(a.entry.Percent, percent) {
		return nil
	}
	reserved := d.reservedRevisionsLocked()
	if a.entry.Phase == modeldownload.PhaseQueued && phase == modeldownload.PhaseInProgress {
		// This transition consumes its own queued-to-in-progress reservation.
		reserved--
	}
	if math.MaxUint64-d.revision <= reserved {
		return errDownloadCapacity
	}
	if a.entry.Phase != phase {
		if a.authorityHeld <= 1 {
			if !d.reserveAuthorityLocked(1) {
				return errDownloadCapacity
			}
			a.authorityHeld++
		}
		a.authorityHeld--
		d.advanceAuthorityLocked()
	}
	if err := d.advanceLocked(); err != nil {
		return err
	}
	a.entry.Phase, a.entry.Percent = phase, cloneDownloadPercent(percent)
	d.outcomes[a.key] = a.entry
	return nil
}

func (d *modelDownloads) reservedRevisionsLocked() uint64 {
	reserved := uint64(len(d.active)) + d.heldRevisions
	for _, attempt := range d.active {
		if attempt.entry.Phase == modeldownload.PhaseQueued {
			reserved++
		}
	}
	return reserved
}

func cloneDownloadPercent(p *int) *int {
	if p == nil {
		return nil
	}
	v := *p
	return &v
}
func equalDownloadPercent(a, b *int) bool {
	return a == nil && b == nil || a != nil && b != nil && *a == *b
}

func (d *modelDownloads) finish(a *modelDownloadAttempt, phase, reason, persistence string) error {
	_, err := d.finishSnapshot(a, phase, reason, persistence)
	return err
}

func (d *modelDownloads) finishSnapshot(a *modelDownloadAttempt, phase, reason, persistence string) (modeldownload.Snapshot, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.active[a.key] != a {
		return modeldownload.Snapshot{}, errDownloadBusy
	}
	if err := d.advanceLocked(); err != nil {
		return modeldownload.Snapshot{}, err
	}
	_, blocked := d.unsafe[a.key]
	a.entry.Phase, a.entry.Reason, a.entry.Persistence = phase, reason, persistence
	a.entry.BlocksReadiness, a.entry.Percent = blocked, nil
	delete(d.active, a.key)
	d.advanceAuthorityLocked()
	d.authorityHeld -= a.authorityHeld - 1
	a.authorityHeld = 0
	d.outcomes[a.key] = a.entry
	if !blocked {
		d.recent = append(d.recent, a.key)
		if len(d.recent) > modeldownload.MaxRecentEntries {
			delete(d.outcomes, d.recent[0])
			d.recent = d.recent[1:]
		}
	}
	return d.snapshotLocked(), nil
}

func (d *modelDownloads) snapshot() modeldownload.Snapshot {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.snapshotLocked()
}

func (d *modelDownloads) snapshotLocked() modeldownload.Snapshot {
	s := modeldownload.Snapshot{Version: 1, SourceID: d.source, Revision: strconv.FormatUint(d.revision, 10), RecoveryRequired: d.recoveryRequired, Entries: make([]modeldownload.Entry, 0, len(d.outcomes))}
	for _, e := range d.outcomes {
		e.Percent = cloneDownloadPercent(e.Percent)
		s.Entries = append(s.Entries, e)
	}
	sort.Slice(s.Entries, func(i, j int) bool {
		if s.Entries[i].Engine != s.Entries[j].Engine {
			return s.Entries[i].Engine < s.Entries[j].Engine
		}
		return s.Entries[i].Model < s.Entries[j].Model
	})
	return s
}

// changeMarker adopts exactly the bytes which landed. In particular, a landed
// removal is adopted even if its directory sync failed. A landed addition
// remains unsafe and its caller must refuse the engine effect on any error.
func (d *modelDownloads) changeMarker(a *modelDownloadAttempt, remove bool) (uncertain bool, err error) {
	d.journalMu.Lock()
	defer d.journalMu.Unlock()
	d.mu.Lock()
	if d.recoveryRequired {
		d.mu.Unlock()
		return false, errDownloadRecovery
	}
	if d.active[a.key] != a {
		d.mu.Unlock()
		return false, errDownloadBusy
	}
	// Keep the terminal step untouched. Retries beyond the normal add/remove
	// pair obtain extra capacity before any journal effect.
	if a.authorityHeld <= 1 {
		if !d.reserveAuthorityLocked(1) {
			d.mu.Unlock()
			return false, errDownloadCapacity
		}
		a.authorityHeld++
	}
	next := make(map[downloadContentKey]modelDownloadMarker, len(d.unsafe)+1)
	for key, marker := range d.unsafe {
		next[key] = marker
	}
	if remove {
		delete(next, a.key)
	} else {
		next[a.key] = modelDownloadMarker{ID: a.entry.ID, Engine: a.key.engine, Model: a.entry.Model, ContentKey: a.key.model, Naming: a.naming}
	}
	copies, copiesComplete, copyRecoveryRequired := maps.Clone(d.copies), d.copiesComplete, d.copyRecoveryRequired
	d.mu.Unlock()
	landed, uncertain, err := d.store.save(next, copies, copiesComplete, copyRecoveryRequired)
	if landed {
		d.mu.Lock()
		d.unsafe = next
		if remove {
			delete(d.unsafeOrigins, a.key)
		} else {
			if d.unsafeOrigins == nil {
				d.unsafeOrigins = make(map[downloadContentKey]string)
			}
			d.unsafeOrigins[a.key] = d.source
		}
		d.authorityUncertain = d.authorityUncertain || uncertain
		a.authorityHeld--
		d.advanceAuthorityLocked()
		d.mu.Unlock()
	}
	return uncertain, err
}

// resolveMarker reserves its publication revision before any durable removal.
// Progress and admission preserve that revision while journal I/O is outside
// the gate; the landed journal and readiness outcome are then adopted together.
func (d *modelDownloads) resolveMarker(key downloadContentKey, id string) bool {
	d.journalMu.Lock()
	defer d.journalMu.Unlock()
	d.mu.Lock()
	marker, exists := d.unsafe[key]
	if d.recoveryRequired || !exists || marker.ID != id || d.active[key] != nil || math.MaxUint64-d.revision <= d.reservedRevisionsLocked() {
		d.mu.Unlock()
		return false
	}
	if !d.reserveAuthorityLocked(1) {
		d.mu.Unlock()
		return false
	}
	d.heldRevisions++
	next := make(map[downloadContentKey]modelDownloadMarker, len(d.unsafe))
	for k, value := range d.unsafe {
		if k != key {
			next[k] = value
		}
	}
	copies, copiesComplete, copyRecoveryRequired := maps.Clone(d.copies), d.copiesComplete, d.copyRecoveryRequired
	d.mu.Unlock()
	landed, uncertain, _ := d.store.save(next, copies, copiesComplete, copyRecoveryRequired)
	d.mu.Lock()
	defer d.mu.Unlock()
	d.heldRevisions--
	if !landed {
		d.authorityHeld--
		return false
	}
	d.unsafe = next
	delete(d.unsafeOrigins, key)
	d.authorityUncertain = d.authorityUncertain || uncertain
	d.advanceAuthorityLocked()
	if d.active[key] == nil {
		entry := d.outcomes[key]
		entry.BlocksReadiness = false
		if uncertain {
			entry.Persistence = "uncertain"
		}
		d.removeRecentLocked(key)
		d.outcomes[key] = entry
		d.recent = append(d.recent, key)
		if len(d.recent) > modeldownload.MaxRecentEntries {
			delete(d.outcomes, d.recent[0])
			d.recent = d.recent[1:]
		}
	}
	// Capacity was reserved above and cannot be consumed by other transitions.
	d.revision++
	return true
}

// The Manager has already bound one live claimant to this normalized tuple.
// Selecting retained evidence performs no journal write or provenance adoption.
func (d *modelDownloads) retainedCopyMarker(marker modelCopyMarker) (modelCopyMarker, bool, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.recoveryRequired || d.copyRecoveryRequired || marker.ManagerIncarnation != d.source {
		return marker, false, errDownloadRecovery
	}
	var matching modelCopyMarker
	found := false
	for _, retained := range d.copies {
		sameContent := retained.Role == marker.Role && retained.SourceNodeID == marker.SourceNodeID && retained.DestinationNodeID == marker.DestinationNodeID && retained.Engine == marker.Engine && retained.ContentKey == marker.ContentKey
		if !sameContent {
			if retained.OpID == marker.OpID {
				return marker, false, errDownloadBusy
			}
			continue
		}
		matching, found = retained, true
	}
	if !found {
		return marker, false, nil
	}
	if matching.ManagerIncarnation != d.source || matching.Phase != "unsafe" || matching.OwnerNodeID != marker.OwnerNodeID || matching.Naming != marker.Naming {
		return marker, false, errDownloadRecovery
	}
	// A known original must pin the retry before effects. An alias-only retry
	// could otherwise leave another export's partials under this marker.
	if matching.ExpectedExportID != "" && matching.ExpectedExportID != marker.ExpectedExportID {
		return marker, false, copyConflict("copy retry differs from retained export identity")
	}
	// Reuse the original provenance without rewriting its operation or
	// incarnation. Only a matching positive terminal may retire it.
	return matching, true, nil
}

// The same journalMu orders every copy and download writer. Capture both
// collections only after taking it, and release mu before the durable write.
// Retained copies are not outcomes and are never evicted by live retirement.
func (d *modelDownloads) changeCopyMarker(marker modelCopyMarker, remove bool) error {
	if marker.validate() != nil {
		return errDownloadPersistence
	}
	d.journalMu.Lock()
	defer d.journalMu.Unlock()
	d.mu.Lock()
	if d.recoveryRequired || d.copyRecoveryRequired || marker.ManagerIncarnation != d.source {
		d.mu.Unlock()
		return errDownloadRecovery
	}
	current, exists := d.copies[marker.OpID]
	if remove && (!exists || current != marker) || !remove && exists {
		d.mu.Unlock()
		return errDownloadBusy
	}
	reservedRetirement := remove && d.copyRetirements[marker.OpID]
	steps := uint64(1)
	if !remove {
		steps = 2
	}
	if !reservedRetirement {
		if !d.reserveAuthorityLocked(steps) {
			d.mu.Unlock()
			return errDownloadCapacity
		}
	}
	next, downloads, copiesComplete, copyRecoveryRequired := maps.Clone(d.copies), maps.Clone(d.unsafe), d.copiesComplete, d.copyRecoveryRequired
	if next == nil {
		next = make(map[string]modelCopyMarker)
	}
	if remove {
		delete(next, marker.OpID)
	} else {
		next[marker.OpID] = marker
	}
	d.mu.Unlock()
	landed, uncertain, err := d.store.save(downloads, next, copiesComplete, copyRecoveryRequired)
	d.mu.Lock()
	defer d.mu.Unlock()
	if landed {
		d.advanceAuthorityLocked()
		d.copies = next
		if remove {
			delete(d.copyRetirements, marker.OpID)
		} else {
			if d.copyRetirements == nil {
				d.copyRetirements = make(map[string]bool)
			}
			d.copyRetirements[marker.OpID] = true
		}
		d.authorityUncertain = d.authorityUncertain || uncertain
		if uncertain {
			d.copyRecoveryRequired = true
			if !remove {
				marker.Phase = "uncertain"
				d.copies[marker.OpID] = marker
			}
		}
	} else if !reservedRetirement {
		d.authorityHeld -= steps
	}
	return err
}

func (d *modelDownloads) blockedLocked(key downloadContentKey) bool {
	if key.model == "" {
		for k := range d.active {
			if k.engine == key.engine {
				return true
			}
		}
		for k := range d.unsafe {
			if k.engine == key.engine {
				return true
			}
		}
	}
	_, unsafe := d.unsafe[key]
	return d.recoveryRequired || d.active[key] != nil || unsafe
}

func (d *modelDownloads) blocked(key downloadContentKey) bool {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.blockedLocked(key)
}

func (d *modelDownloads) acquireRead(key downloadContentKey, ownedWriter bool) (func(), error) {
	d.mu.Lock()
	if d.recoveryRequired || !ownedWriter && (d.blockedLocked(key) || d.writers[key.engine]) {
		d.mu.Unlock()
		return nil, errDownloadBusy
	}
	if !d.reserveAuthorityLocked(2) {
		d.mu.Unlock()
		return nil, errDownloadCapacity
	}
	d.readers[key.engine]++
	d.advanceAuthorityLocked()
	d.mu.Unlock()
	var once sync.Once
	return func() {
		once.Do(func() {
			d.mu.Lock()
			d.readers[key.engine]--
			if d.readers[key.engine] == 0 {
				delete(d.readers, key.engine)
			}
			d.advanceAuthorityLocked()
			d.mu.Unlock()
		})
	}, nil
}

func (d *modelDownloads) acquireWrite(engine string) (func(), error) {
	d.mu.Lock()
	if d.recoveryRequired || d.writers[engine] || d.readers[engine] != 0 {
		d.mu.Unlock()
		return nil, errDownloadBusy
	}
	if !d.reserveAuthorityLocked(2) {
		d.mu.Unlock()
		return nil, errDownloadCapacity
	}
	d.writers[engine] = true
	d.advanceAuthorityLocked()
	d.mu.Unlock()
	var once sync.Once
	return func() {
		once.Do(func() {
			d.mu.Lock()
			delete(d.writers, engine)
			d.advanceAuthorityLocked()
			d.mu.Unlock()
		})
	}, nil
}
