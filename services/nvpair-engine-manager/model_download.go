// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"errors"
	"math"
	"sort"
	"strconv"
	"sync"

	"nvpair-shared/modeldownload"
)

var (
	errDownloadBusy     = errors.New("model content is busy")
	errDownloadCapacity = errors.New("model download capacity exhausted")
	errDownloadRecovery = errors.New("model download recovery required")
)

type modelDownloadAttempt struct {
	key    downloadContentKey
	naming string
	entry  modeldownload.Entry
}

// modelDownloads is a bounded content gate, not an execution owner. Callers
// release mu before notifications, I/O and any other owner's wait. journalMu
// orders capture/save/adopt across engines and is acquired outside mu.
type modelDownloads struct {
	mu               sync.Mutex
	journalMu        sync.Mutex
	source           string
	revision         uint64
	recoveryRequired bool
	active           map[downloadContentKey]*modelDownloadAttempt
	unsafe           map[downloadContentKey]modelDownloadMarker
	outcomes         map[downloadContentKey]modeldownload.Entry
	recent           []downloadContentKey
	writers          map[string]bool
	readers          map[string]int
	store            modelDownloadStore
}

func newModelDownloads(source, baseDir string, naming map[string]string) *modelDownloads {
	d := &modelDownloads{source: source, active: make(map[downloadContentKey]*modelDownloadAttempt), unsafe: make(map[downloadContentKey]modelDownloadMarker), outcomes: make(map[downloadContentKey]modeldownload.Entry), writers: make(map[string]bool), readers: make(map[string]int), store: newModelDownloadStore(baseDir)}
	markers, err := d.store.load()
	if err != nil {
		d.recoveryRequired = true
		return d
	}
	d.unsafe = markers
	for key, marker := range markers {
		if naming[key.engine] != marker.Naming {
			d.recoveryRequired = true
		}
		d.outcomes[key] = modeldownload.Entry{ID: marker.ID, Engine: marker.Engine, Model: marker.Model, Phase: "unknown", BlocksReadiness: true, Reason: "interrupted"}
	}
	return d
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
	slots := len(d.unsafe)
	for k := range d.active {
		if _, ok := d.unsafe[k]; !ok {
			slots++
		}
	}
	if _, retained := d.unsafe[key]; !retained && slots >= modeldownload.MaxUnsafeEntries {
		return nil, errDownloadCapacity
	}
	if err := d.advanceLocked(); err != nil {
		return nil, err
	}
	d.removeRecentLocked(key)
	a := &modelDownloadAttempt{key: key, naming: naming, entry: modeldownload.Entry{ID: newOpID(), Engine: engine, Model: model, Phase: "queued", BlocksReadiness: true}}
	d.active[key] = a
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
	if err := d.advanceLocked(); err != nil {
		return err
	}
	a.entry.Phase, a.entry.Percent = phase, cloneDownloadPercent(percent)
	d.outcomes[a.key] = a.entry
	return nil
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
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.active[a.key] != a {
		return errDownloadBusy
	}
	if err := d.advanceLocked(); err != nil {
		return err
	}
	_, blocked := d.unsafe[a.key]
	a.entry.Phase, a.entry.Reason, a.entry.Persistence = phase, reason, persistence
	a.entry.BlocksReadiness, a.entry.Percent = blocked, nil
	delete(d.active, a.key)
	d.outcomes[a.key] = a.entry
	if !blocked {
		d.recent = append(d.recent, a.key)
		if len(d.recent) > modeldownload.MaxRecentEntries {
			delete(d.outcomes, d.recent[0])
			d.recent = d.recent[1:]
		}
	}
	return nil
}

func (d *modelDownloads) snapshot() modeldownload.Snapshot {
	d.mu.Lock()
	defer d.mu.Unlock()
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
	next := make(map[downloadContentKey]modelDownloadMarker, len(d.unsafe)+1)
	for key, marker := range d.unsafe {
		next[key] = marker
	}
	if remove {
		delete(next, a.key)
	} else {
		next[a.key] = modelDownloadMarker{ID: a.entry.ID, Engine: a.key.engine, Model: a.entry.Model, ContentKey: a.key.model, Naming: a.naming}
	}
	d.mu.Unlock()
	landed, uncertain, err := d.store.save(next)
	if landed {
		d.mu.Lock()
		d.unsafe = next
		d.mu.Unlock()
	}
	return uncertain, err
}

func (d *modelDownloads) blockedLocked(key downloadContentKey) bool {
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
	if d.blockedLocked(key) || d.writers[key.engine] && !ownedWriter {
		d.mu.Unlock()
		return nil, errDownloadBusy
	}
	d.readers[key.engine]++
	d.mu.Unlock()
	var once sync.Once
	return func() { once.Do(func() { d.mu.Lock(); d.readers[key.engine]--; d.mu.Unlock() }) }, nil
}

func (d *modelDownloads) acquireWrite(engine string) (func(), error) {
	d.mu.Lock()
	if d.recoveryRequired || d.writers[engine] || d.readers[engine] != 0 {
		d.mu.Unlock()
		return nil, errDownloadBusy
	}
	d.writers[engine] = true
	d.mu.Unlock()
	var once sync.Once
	return func() { once.Do(func() { d.mu.Lock(); delete(d.writers, engine); d.mu.Unlock() }) }, nil
}
