// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// Copy coordination stays at the destination's existing executor. A remote
// controller holds only a bounded tuple, request context and latest progress.
import (
	"bufio"
	"bytes"
	"context"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"nvpair-shared/engines"
)

const (
	controlCopyPath             = "/v1/models/copy"
	controlCopyCapabilitiesPath = "/v1/models/copy-capabilities"
	maxCopyControlBody          = 16 << 10
	maxActiveCopies             = 8
)

// A dispatched mutation without a trustworthy terminal acknowledgement cannot
// be classified as a failed copy: publication may already have completed.
type copyOutcomeUnknownError struct{ error }

func (e *copyOutcomeUnknownError) Unwrap() error { return e.error }

func copyUnknown(cause error) error {
	return &copyOutcomeUnknownError{fmt.Errorf("copy acknowledgement unavailable; operation outcome is unknown: %w", cause)}
}

func copyErrorStage(err error) string {
	var unknown *copyOutcomeUnknownError
	if errors.As(err, &unknown) {
		return "unknown"
	}
	if errors.Is(err, context.Canceled) {
		return "cancelled"
	}
	return "error"
}

func copyTerminalStage(stage string) bool {
	switch stage {
	case "done", "error", "cancelled", "unknown":
		return true
	}
	return false
}

type copyRequest struct {
	OpID              string `json:"opId"`
	SourceNodeID      string `json:"sourceNodeId"`
	DestinationNodeID string `json:"destinationNodeId,omitempty"`
	Engine            string `json:"engine"`
	Model             string `json:"model"`
	ExpectedExportID  string `json:"expectedExportId,omitempty"`
}

type activeCopy struct {
	copyRequest
	Progress *copyProgress `json:"progress,omitempty"`
	cancel   context.CancelFunc
	// Admission is visible while the durable owner publishes its marker.
	pendingPublication bool
	role               string
	authorityHeld      uint64
}

type copyCapabilities struct {
	Version            int      `json:"version"`
	Engine             string   `json:"engine"`
	Available          bool     `json:"available"`
	Installed          bool     `json:"installed"`
	ModelStore         string   `json:"modelStore"`
	ExportLayouts      []string `json:"exportLayouts"`
	ImportLayouts      []string `json:"importLayouts"`
	Writable           bool     `json:"writable"`
	Reason             string   `json:"reason"`
	IdentityBoundRetry bool     `json:"identityBoundRetry"`
	ReceiptDurability  bool     `json:"receiptDurability"`
}

// These flat control objects have exact field names. Inspect tokens before
// encoding/json can fold case, replace invalid UTF-8 or discard duplicates.
func decodeCopyObject(data []byte, target any, allowed ...string) error {
	if len(data) > maxCopyControlBody || !utf8.Valid(data) {
		return fmt.Errorf("invalid copy parameters")
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	tok, err := dec.Token()
	if err != nil || tok != json.Delim('{') {
		return fmt.Errorf("copy parameters must be an object")
	}
	seen := map[string]bool{}
	for dec.More() {
		tok, err = dec.Token()
		key, ok := tok.(string)
		if err != nil || !ok {
			return fmt.Errorf("invalid copy field")
		}
		known := false
		for _, field := range allowed {
			if key == field {
				known = true
				break
			}
		}
		if !known || seen[key] {
			return fmt.Errorf("unknown or duplicate copy field")
		}
		seen[key] = true
		var raw json.RawMessage
		if err := dec.Decode(&raw); err != nil || bytes.Equal(raw, []byte("null")) {
			return fmt.Errorf("invalid copy field value")
		}
		if !copyStringUnicode(raw) {
			return fmt.Errorf("copy fields must be valid Unicode strings")
		}
	}
	if _, err := dec.Token(); err != nil {
		return fmt.Errorf("invalid copy object")
	}
	if dec.Decode(new(json.RawMessage)) != io.EOF {
		return fmt.Errorf("trailing copy parameters")
	}
	return json.Unmarshal(data, target)
}

// encoding/json replaces lone JSON UTF-16 surrogates with U+FFFD. Reject them
// before decoding, while permitting an intentional literal replacement rune.
func copyStringUnicode(raw []byte) bool {
	if len(raw) < 2 || raw[0] != '"' {
		return false
	}
	for i := 1; i < len(raw)-1; i++ {
		if raw[i] != '\\' {
			continue
		}
		i++
		if raw[i] != 'u' {
			continue
		}
		if i+4 >= len(raw) {
			return false
		}
		value, err := strconv.ParseUint(string(raw[i+1:i+5]), 16, 16)
		if err != nil {
			return false
		}
		i += 4
		if value >= 0xdc00 && value <= 0xdfff {
			return false
		}
		if value < 0xd800 || value > 0xdbff {
			continue
		}
		if i+6 >= len(raw) || raw[i+1] != '\\' || raw[i+2] != 'u' {
			return false
		}
		low, err := strconv.ParseUint(string(raw[i+3:i+7]), 16, 16)
		if err != nil || low < 0xdc00 || low > 0xdfff {
			return false
		}
		i += 6
	}
	return true
}

func lowerHex(s string, n int) bool {
	if len(s) != n || strings.ToLower(s) != s {
		return false
	}
	_, err := hex.DecodeString(s)
	return err == nil
}

func copySelector(s string) bool {
	if s == "" || len(s) > 128 {
		return false
	}
	for _, c := range s {
		if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '-' || c == '_' || c == '.') {
			return false
		}
	}
	return true
}

func decodeCopyRequest(data []byte, inbound bool) (copyRequest, error) {
	var p copyRequest
	allowed := []string{"opId", "sourceNodeId", "engine", "model", "expectedExportId"}
	if !inbound {
		allowed = append(allowed, "destinationNodeId")
	}
	if err := decodeCopyObject(data, &p, allowed...); err != nil {
		return p, err
	}
	if !lowerHex(p.OpID, 32) {
		return p, fmt.Errorf("opId must be 32 lowercase hex characters")
	}
	if !copySelector(p.SourceNodeID) || (!inbound && !copySelector(p.DestinationNodeID)) {
		return p, fmt.Errorf("sourceNodeId and destinationNodeId are required device IDs")
	}
	if !inbound && p.SourceNodeID == p.DestinationNodeID {
		return p, fmt.Errorf("source and destination must be different devices")
	}
	if !copySelector(p.Engine) {
		return p, fmt.Errorf("engine is required and must be an engine manager ID")
	}
	if err := validateModelRequest(data, true); err != nil {
		return p, err
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return p, err
	}
	if _, present := fields["expectedExportId"]; present && !lowerHex(p.ExpectedExportID, 64) {
		return p, fmt.Errorf("expectedExportId must be lowercase SHA256 hex")
	}
	return p, nil
}

func copyConflict(message string) error {
	return &codedError{code: codePolicyConflict, message: message}
}

func (m *Manager) reserveCopy(parent context.Context, p copyRequest) (context.Context, func(), error) {
	m.copyMu.Lock()
	defer m.copyMu.Unlock()
	if m.copiesClosed {
		return nil, nil, fmt.Errorf("engine manager is shutting down")
	}
	if _, ok := m.copies[p.OpID]; ok {
		return nil, nil, copyConflict("copy opId is already active")
	}
	if len(m.copies) >= maxActiveCopies {
		return nil, nil, copyConflict("eight model copies are already active")
	}
	// Admission, identity binding, role binding, publication and retirement.
	// Preserve shutdown even if every remaining revision is reserved.
	if math.MaxUint64-m.copyRevision <= m.copyHeld || math.MaxUint64-m.copyRevision-m.copyHeld-1 < 5 {
		return nil, nil, errOwnerAdmissionCapacity
	}
	ctx, cancel := context.WithCancel(parent)
	entry := &activeCopy{copyRequest: p, cancel: cancel, pendingPublication: true, authorityHeld: 4}
	if m.copies == nil {
		m.copies = map[string]*activeCopy{}
	}
	m.copies[p.OpID] = entry
	m.copyRevision++
	m.copyHeld += entry.authorityHeld
	return ctx, func() {
		cancel()
		m.copyMu.Lock()
		if m.copies[p.OpID] == entry {
			delete(m.copies, p.OpID)
			m.retireCopyAuthorityLocked(entry)
		}
		m.copyMu.Unlock()
	}, nil
}

func (m *Manager) cancelCopy(opID string) bool {
	m.copyMu.Lock()
	entry := m.copies[opID]
	m.copyMu.Unlock()
	if entry == nil {
		return false
	}
	entry.cancel()
	return true
}

func (m *Manager) stopCopies() {
	m.copyMu.Lock()
	if !m.copiesClosed {
		m.copiesClosed = true
		m.copyRevision++
	}
	entries := make([]*activeCopy, 0, len(m.copies))
	for id, entry := range m.copies {
		entries = append(entries, entry)
		delete(m.copies, id)
		m.retireCopyAuthorityLocked(entry)
	}
	m.copyMu.Unlock()
	for _, entry := range entries {
		entry.cancel()
	}
}

func (m *Manager) retireCopyAuthorityLocked(entry *activeCopy) {
	m.copyHeld -= entry.authorityHeld
	entry.authorityHeld = 0
	m.copyRevision++
}

// These bindings happen once in the existing execution path. Their steps were
// reserved at admission, independently of progress publication.
func (m *Manager) bindCopyAuthorityLocked(entry *activeCopy, p copyRequest, role string) error {
	if entry.copyRequest != p || entry.role != role {
		reserved := uint64(1)
		if entry.pendingPublication {
			reserved++
		}
		if entry.authorityHeld <= reserved {
			remaining := math.MaxUint64 - m.copyRevision
			if remaining <= m.copyHeld || remaining-m.copyHeld <= 1 {
				return errOwnerAdmissionCapacity
			}
			entry.authorityHeld++
			m.copyHeld++
		}
		entry.copyRequest, entry.role = p, role
		entry.authorityHeld--
		m.copyHeld--
		m.copyRevision++
	}
	return nil
}

// Bind one live claimant to the descriptor-owned content identity before reads
// or effects. Retained provenance is reused only by its current incarnation.
func (m *Manager) bindCopyEvidence(p copyRequest, self string) (modelCopyMarker, *activeCopy, bool, error) {
	var marker modelCopyMarker
	d := m.copyJournal
	if d == nil {
		return marker, nil, false, errDownloadRecovery
	}
	d.mu.Lock()
	source := d.source
	d.mu.Unlock()
	// Naming is owned by the registry, not the caller or remote capability.
	// Controllers retain this configuration even when they do no artifact I/O.
	if m.copyRegistry == nil {
		return marker, nil, false, errDownloadRecovery
	}
	manifest, ok := m.copyRegistry.Get(p.Engine)
	if !ok {
		return marker, nil, false, errDownloadRecovery
	}
	naming := manifest.ModelNamingOrDefault()
	key, err := downloadKey(p.Engine, p.Model, naming)
	if err != nil {
		return marker, nil, false, err
	}
	role := "controller"
	if p.DestinationNodeID == self {
		role = "destination"
	}
	marker = modelCopyMarker{OpID: p.OpID, Engine: p.Engine, Model: p.Model, ContentKey: key.model, Naming: naming, ExpectedExportID: p.ExpectedExportID, Role: role, OwnerNodeID: self, SourceNodeID: p.SourceNodeID, DestinationNodeID: p.DestinationNodeID, ManagerIncarnation: source, Phase: "unsafe"}
	if marker.validate() != nil {
		return marker, nil, false, errDownloadPersistence
	}
	m.copyMu.Lock()
	entry := m.copies[p.OpID]
	live := !m.copiesClosed && entry != nil && entry.copyRequest == p
	if live {
		// A retained marker cannot cover concurrent effects for this tuple.
		// Bind the role under the same lock that excludes another claimant.
		for _, other := range m.copies {
			if other != entry && other.role == role && other.Engine == p.Engine && other.SourceNodeID == p.SourceNodeID && other.DestinationNodeID == p.DestinationNodeID && engines.NormalizeModelName(other.Model, naming) == key.model {
				err = errDownloadBusy
				break
			}
		}
	}
	if live && err == nil {
		err = m.bindCopyAuthorityLocked(entry, p, role)
	}
	m.copyMu.Unlock()
	if err != nil {
		return marker, nil, false, err
	}
	if !live {
		return marker, nil, false, errDownloadBusy
	}
	marker, retained, err := d.retainedCopyMarker(marker)
	return marker, entry, retained, err
}

// The callback returns only after local finalization or the identity-checked
// terminal transport. Errors and cancellation leave the original marker intact.
func (m *Manager) copyWithEvidence(ctx context.Context, p copyRequest, self string, effect func() (copyModelResult, error)) (copyModelResult, error) {
	var res copyModelResult
	marker, entry, retained, err := m.bindCopyEvidence(p, self)
	if err != nil {
		return res, err
	}
	if !retained {
		if err := m.copyJournal.changeCopyMarker(marker, false); err != nil {
			return res, err
		}
	}
	m.copyMu.Lock()
	live := !m.copiesClosed && m.copies[p.OpID] == entry
	if live && entry.pendingPublication {
		entry.pendingPublication = false
		entry.authorityHeld--
		m.copyHeld--
		m.copyRevision++
	}
	m.copyMu.Unlock()
	if err := ctx.Err(); err != nil {
		return res, err
	}
	if !live {
		return res, context.Canceled
	}
	res, err = effect()
	if err != nil {
		return res, err
	}
	// A fresh marker belongs to this synchronous effect. A reused alias-only
	// marker may describe another export's surviving partials: this result
	// cannot establish that earlier export's identity.
	return m.completeCopyEvidence(ctx, p, marker, entry, !retained || marker.ExpectedExportID != "", res)
}

func (m *Manager) completeCopyEvidence(ctx context.Context, p copyRequest, marker modelCopyMarker, entry *activeCopy, retire bool, res copyModelResult) (copyModelResult, error) {
	if !validCopyResult(res, p.Engine, p.Model) || p.ExpectedExportID != "" && res.ExportID != p.ExpectedExportID || marker.ExpectedExportID != "" && res.ExportID != marker.ExpectedExportID {
		return res, copyUnknown(errors.New("copy completion identity does not match reservation"))
	}
	// A positive result after shutdown/cancellation is not this live owner's
	// permission to retire evidence. A successor cannot complete this marker.
	m.copyMu.Lock()
	live := !m.copiesClosed && m.copies[p.OpID] == entry
	m.copyMu.Unlock()
	if err := ctx.Err(); err != nil {
		return res, err
	}
	if !live {
		return res, copyUnknown(errors.New("copy owner retired before completion"))
	}
	if retire {
		if err := m.copyJournal.changeCopyMarker(marker, true); err != nil {
			return res, copyUnknown(err)
		}
	} else if _, _, err := m.copyJournal.retainedCopyMarker(marker); err != nil {
		return res, err
	}
	return res, nil
}

func (m *Manager) activeCopies() []activeCopy {
	m.copyMu.Lock()
	defer m.copyMu.Unlock()
	out := make([]activeCopy, 0, len(m.copies))
	for _, entry := range m.copies {
		copy := activeCopy{copyRequest: entry.copyRequest, pendingPublication: entry.pendingPublication, role: entry.role}
		copy.Progress = cloneCopyProgress(entry.Progress)
		out = append(out, copy)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].OpID < out[j].OpID })
	return out
}

func cloneCopyProgress(progress *copyProgress) *copyProgress {
	if progress == nil {
		return nil
	}
	copy := *progress
	if progress.DownloadState != nil {
		state := progress.DownloadState.Clone()
		copy.DownloadState = &state
	}
	return &copy
}

// Scanner includes its registry-owned self record in discovery:nodes. Match
// that record's cluster principal to the live mesh; host and cert UUIDs are
// separate domains even on installations that currently use the same value.
func (m *Manager) copySelfID() (string, error) {
	if m.mesh == nil {
		return "", fmt.Errorf("model copy unavailable: this node is not clustered")
	}
	m.mesh.Refresh()
	if !m.mesh.Clustered() {
		return "", fmt.Errorf("model copy unavailable: this node is not clustered")
	}
	uuid := m.mesh.NodeUUID()
	if m.peers == nil {
		return "", fmt.Errorf("model copy unavailable: self directory identity is unknown")
	}
	m.peers.mu.RLock()
	defer m.peers.mu.RUnlock()
	self := ""
	for host, peer := range m.peers.peers {
		if peer.clusterUUID != uuid {
			continue
		}
		if !copySelector(host) || self != "" {
			return "", fmt.Errorf("model copy unavailable: ambiguous self directory identity")
		}
		self = host
	}
	if self == "" {
		return "", fmt.Errorf("model copy unavailable: self directory identity is unknown")
	}
	return self, nil
}

func (e *Executor) copyCapabilities(engine string) copyCapabilities {
	c := copyCapabilities{Version: 1, Engine: engine, ExportLayouts: []string{}, ImportLayouts: []string{}, IdentityBoundRetry: true, ReceiptDurability: true}
	_, store, err := e.modelStoreFor(engine)
	if err != nil {
		c.Reason = "Model copy is unavailable for this engine on this device."
		return c
	}
	c.Available, c.ModelStore = true, store.adapter
	c.ExportLayouts = []string{store.adapter}
	if store.adapter == "llamacpp" {
		c.ExportLayouts = append(c.ExportLayouts, llamaArtifactsLayout)
	}
	c.ImportLayouts = append([]string{}, c.ExportLayouts...)
	c.Installed, err = e.Detect(engine)
	if err != nil || !c.Installed {
		c.Reason = "Install this engine on the destination before copying."
		return c
	}
	info, err := os.Stat(nearestExistingDir(store.root))
	if err != nil || !info.IsDir() || info.Mode().Perm()&0o222 == 0 {
		c.Reason = "The destination model store is not writable."
		return c
	}
	c.Writable = true
	if err := e.copyAdmission(); err != nil {
		c.Writable = false
		c.Reason = err.Error()
	}
	return c
}

// Copy changes content, not engine activation. Saved Off and engine locks are
// irrelevant; the existing pause/recovery mutation gates still apply.
func (e *Executor) copyAdmission() error {
	if e.switchRecoveryPending.Load() {
		return fmt.Errorf("engine manager switch recovery is pending")
	}
	blocked, known := e.nodeProjection.managementBlocked()
	if !known {
		return errNoProjection
	}
	if blocked {
		return &localVetoError{code: codeLocalVeto, reason: nodePausedVeto}
	}
	return nil
}

func (m *Manager) copyClient(ctx context.Context, node string) (*remoteClient, error) {
	peer, ok := m.peers.lookup(node)
	if !ok {
		return nil, fmt.Errorf("node %s is not a discovered ec peer", node)
	}
	return m.remoteClient(ctx, peer)
}

func copyLayoutsIntersect(source, destination copyCapabilities) bool {
	for _, a := range source.ExportLayouts {
		for _, b := range destination.ImportLayouts {
			if a == b {
				return true
			}
		}
	}
	return false
}

func requireCopyDestination(c copyCapabilities) error {
	if c.Version != 1 || !c.IdentityBoundRetry || !c.ReceiptDurability {
		return fmt.Errorf("destination does not support this model copy protocol")
	}
	if !c.Available || !c.Installed || !c.Writable {
		if c.Reason == remoteManagementDeniedMessage {
			return &codedError{code: codeRemoteManagementDenied, message: c.Reason}
		}
		if c.Reason == nodePausedVeto {
			return &codedError{code: codeLocalVeto, message: c.Reason}
		}
		return fmt.Errorf("destination is not ready for model copy: %s", c.Reason)
	}
	return nil
}

// executeCopy reserves before directory lookup or network I/O. Inbound requests
// only ever execute locally; they cannot recursively dispatch to another peer.
func (m *Manager) executeCopy(parent context.Context, p copyRequest, inbound bool, authorize func() error, onProgress func(copyProgress)) (res copyModelResult, err error) {
	ctx, release, err := m.reserveCopy(parent, p)
	if err != nil {
		return res, err
	}
	defer release()
	self, err := m.copySelfID()
	if err != nil {
		return res, err
	}
	if inbound {
		p.DestinationNodeID = self
	}
	if p.SourceNodeID == p.DestinationNodeID {
		return res, fmt.Errorf("source and destination must be different devices")
	}
	m.copyMu.Lock()
	if entry := m.copies[p.OpID]; entry != nil {
		err = m.bindCopyAuthorityLocked(entry, p, entry.role)
	}
	m.copyMu.Unlock()
	if err != nil {
		return res, err
	}
	if authorize != nil {
		if err := authorize(); err != nil {
			return res, err
		}
	}
	if err := ctx.Err(); err != nil {
		return res, err
	}
	var latest copyProgress
	emit := func(cp copyProgress) {
		cp.OpID, cp.Node, cp.Engine, cp.Op = p.OpID, p.SourceNodeID, p.Engine, "copy"
		cp.SourceNodeID, cp.DestinationNodeID, cp.Model = p.SourceNodeID, p.DestinationNodeID, p.Model
		if cp.ExportID == "" {
			cp.ExportID = latest.ExportID
		}
		latest = cp
		m.copyMu.Lock()
		if entry := m.copies[p.OpID]; entry != nil {
			entry.Progress = cloneCopyProgress(&cp)
		}
		m.copyMu.Unlock()
		// Every Manager publishes its own accepted work, including incoming
		// destination copies. The optional observer carries the same operation
		// over HTTP; GUI subscription lifetime belongs to the broker.
		if err := m.codec.Notify("engine:remote-progress", cp); err != nil {
			m.cancelCopy(p.OpID)
		}
		if onProgress != nil {
			onProgress(cp)
		}
	}
	defer func() {
		if err != nil && errors.Is(ctx.Err(), context.Canceled) {
			err = ctx.Err()
		}
		var unknown *copyOutcomeUnknownError
		uncertain := errors.As(err, &unknown)
		var pathError *os.PathError
		var linkError *os.LinkError
		if errors.As(err, &pathError) || errors.As(err, &linkError) {
			err = &codedError{code: rpcCode(err), message: "Model copy could not access its files on the selected device."}
			if uncertain {
				err = copyUnknown(err)
			}
		}
		cp := latest
		cp.File = ""
		if err != nil {
			cp.Stage, cp.Percent, cp.Message = copyErrorStage(err), -1, err.Error()
		} else {
			cp.Stage, cp.Percent, cp.Message, cp.BytesDone, cp.BytesTotal, cp.ExportID = "done", 100, "", res.BytesTotal, res.BytesTotal, res.ExportID
		}
		emit(cp)
	}()
	progress := func(cp copyProgress) {
		if !copyTerminalStage(cp.Stage) {
			emit(cp)
		}
	}
	marker, entry, retained, err := m.bindCopyEvidence(p, self)
	if err != nil {
		return res, err
	}
	effect, verified, err := m.copyPrepare(ctx, p, self, progress)
	if err != nil {
		return res, err
	}
	if effect == nil {
		return m.completeCopyEvidence(ctx, p, marker, entry, retained && marker.ExpectedExportID != "", verified)
	}
	return m.copyWithEvidence(ctx, p, self, effect)
}

// Preparation performs only reads. Mutation dispatch and source listing/import
// remain in the returned effect, after confirmed journal publication.
func (m *Manager) prepareCopy(ctx context.Context, p copyRequest, self string, progress func(copyProgress)) (effect func() (copyModelResult, error), res copyModelResult, err error) {
	if p.DestinationNodeID != self {
		client, err := m.copyClient(ctx, p.DestinationNodeID)
		if err != nil {
			return nil, res, err
		}
		capability, err := client.copyCapabilities(ctx, p.Engine)
		if err != nil {
			return nil, res, err
		}
		if err := requireCopyDestination(capability); err != nil {
			return nil, res, err
		}
		// No source checks here: the authorized destination must first be able
		// to prove a completed identity-bound retry even with source offline.
		return func() (copyModelResult, error) { return client.copyModel(ctx, p, progress) }, res, nil
	}
	if err := m.exec.copyAdmission(); err != nil {
		return nil, res, err
	}
	if err := requireCopyDestination(m.exec.copyCapabilities(p.Engine)); err != nil {
		return nil, res, err
	}
	if p.ExpectedExportID != "" {
		verified, complete, err := m.exec.completedCopy(ctx, p.Engine, p.Model, p.ExpectedExportID, progress)
		if err != nil || complete {
			return nil, verified, err
		}
	}
	client, err := m.copyClient(ctx, p.SourceNodeID)
	if err != nil {
		return nil, res, err
	}
	source, err := client.copyCapabilities(ctx, p.Engine)
	if err != nil {
		return nil, res, err
	}
	destination := m.exec.copyCapabilities(p.Engine)
	if p.ExpectedExportID != "" && !source.IdentityBoundRetry {
		return nil, res, fmt.Errorf("source does not support identity-bound model copy retry")
	}
	if source.Version != 1 || !source.Available || !source.Installed || !copyLayoutsIntersect(source, destination) {
		return nil, res, fmt.Errorf("source and destination do not support a common copy layout")
	}
	return func() (copyModelResult, error) {
		return m.exec.copyModelFrom(ctx, client, p.Engine, p.Model, p.ExpectedExportID, source.ExportLayouts, destination.ImportLayouts, progress)
	}, res, nil
}

func (m *Manager) runCopy(ctx context.Context, msg *Message) {
	p, err := decodeCopyRequest(msg.Params, false)
	if err != nil {
		m.codec.RespondError(msg.ID, -32602, err.Error())
		return
	}
	res, err := m.executeCopy(ctx, p, false, nil, nil)
	m.respondOrErr(msg, map[string]any{"opId": p.OpID, "result": res}, err)
}

// This callback has no remote-destination dispatch mode. The control server
// receives only this destination-bound entry point from main.
func (m *Manager) copyFromPeer(ctx context.Context, p copyRequest, authorize func() error, emit func(copyProgress)) (copyModelResult, error) {
	return m.executeCopy(ctx, p, true, authorize, emit)
}

func (m *Manager) copyState(msg *Message, cancel bool) {
	var p struct {
		OpID string `json:"opId"`
	}
	allowed := []string{}
	if cancel {
		allowed = append(allowed, "opId")
	}
	err := decodeCopyObject(msg.Params, &p, allowed...)
	if err == nil && cancel && !lowerHex(p.OpID, 32) {
		err = fmt.Errorf("invalid opId")
	}
	if err != nil {
		m.codec.RespondError(msg.ID, -32602, err.Error())
		return
	}
	if cancel {
		m.codec.Respond(msg.ID, map[string]bool{"cancelled": m.cancelCopy(p.OpID)})
		return
	}
	m.codec.Respond(msg.ID, map[string]any{"operations": m.activeCopies()})
}

func (m *Manager) runCopyCapabilities(ctx context.Context, msg *Message) {
	var p struct {
		NodeID string `json:"nodeId"`
		Engine string `json:"engine"`
	}
	err := decodeCopyObject(msg.Params, &p, "nodeId", "engine")
	if err != nil || !copySelector(p.NodeID) || !copySelector(p.Engine) {
		m.codec.RespondError(msg.ID, -32602, "nodeId and engine are required")
		return
	}
	self, err := m.copySelfID()
	if err != nil {
		m.respondOrErr(msg, nil, err)
		return
	}
	if p.NodeID == self {
		m.codec.Respond(msg.ID, m.exec.copyCapabilities(p.Engine))
		return
	}
	client, err := m.copyClient(ctx, p.NodeID)
	if err != nil {
		m.respondOrErr(msg, nil, err)
		return
	}
	capability, err := client.copyCapabilities(ctx, p.Engine)
	m.respondOrErr(msg, capability, err)
}

func (s *controlServer) copyAuthorization(caller string) error {
	if !s.mesh.RemoteManagement(caller).Allowed {
		return &codedError{code: codeRemoteManagementDenied, message: remoteManagementDeniedMessage}
	}
	return s.exec.copyAdmission()
}

func (s *controlServer) handleCopyCapabilities(w http.ResponseWriter, r *http.Request, caller string) {
	q, err := url.ParseQuery(r.URL.RawQuery)
	if err != nil || len(q) != 1 || len(q["engine"]) != 1 || !copySelector(q.Get("engine")) {
		http.Error(w, "engine is required", http.StatusBadRequest)
		return
	}
	c := s.exec.copyCapabilities(q.Get("engine"))
	if err := s.copyAuthorization(caller); err != nil {
		c.Writable, c.Reason = false, err.Error()
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(c)
}

func (s *controlServer) handleCopy(w http.ResponseWriter, r *http.Request, caller string) {
	controller := http.NewResponseController(w)
	_ = controller.SetReadDeadline(time.Now().Add(15 * time.Second))
	data, err := readRemoteResponse(r.Body, maxCopyControlBody)
	_ = controller.SetReadDeadline(time.Time{})
	if err != nil {
		http.Error(w, "copy body exceeds bounds", http.StatusBadRequest)
		return
	}
	p, err := decodeCopyRequest(data, true)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if s.copyFromPeer == nil {
		http.Error(w, "model copy is unavailable", http.StatusServiceUnavailable)
		return
	}
	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()
	progress := make(chan copyProgress, 1)
	type outcome struct {
		result copyModelResult
		err    error
	}
	result := make(chan outcome, 1)
	go func() {
		res, err := s.copyFromPeer(ctx, p, func() error { return s.copyAuthorization(caller) }, func(cp copyProgress) {
			// Terminal delivery belongs exclusively to the response frame.
			if copyTerminalStage(cp.Stage) {
				return
			}
			select {
			case progress <- cp:
			default:
				select {
				case <-progress:
				default:
				}
				select {
				case progress <- cp:
				default:
				}
			}
		})
		result <- outcome{res, err}
	}()
	w.Header().Set("Content-Type", "application/x-ndjson")
	write := func(frame streamFrame) bool {
		_ = controller.SetWriteDeadline(time.Now().Add(15 * time.Second))
		if err := json.NewEncoder(w).Encode(frame); err != nil {
			cancel()
			return false
		}
		if err := controller.Flush(); err != nil {
			cancel()
			return false
		}
		return true
	}
	for {
		select {
		case cp := <-progress:
			if !write(copyProgressFrame(cp)) {
				<-result
				return
			}
		case out := <-result:
			frame := streamFrame{Type: "result", OpID: p.OpID, Engine: p.Engine, Op: "copy"}
			if out.err != nil {
				frame.Type, frame.Message, frame.Code = "error", out.err.Error(), wireCodes[rpcCode(out.err)]
				frame.Stage = copyErrorStage(out.err)
			} else {
				frame.Result, err = json.Marshal(out.result)
				if err != nil {
					frame.Type, frame.Message = "error", "invalid copy result"
				}
			}
			write(frame)
			return
		case <-ctx.Done():
			cancel()
			<-result
			return
		}
	}
}

func (c *remoteClient) copyCapabilities(ctx context.Context, engine string) (copyCapabilities, error) {
	var capability copyCapabilities
	ctx, cancel := context.WithTimeout(ctx, remoteResponseHeaderTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.base+controlCopyCapabilitiesPath+"?engine="+url.QueryEscape(engine), nil)
	if err != nil {
		return capability, err
	}
	resp, err := c.controlClient(false).Do(req)
	if err != nil {
		return capability, err
	}
	defer resp.Body.Close()
	data, err := readRemoteResponse(resp.Body, maxCopyControlBody)
	if err != nil {
		return capability, err
	}
	if resp.StatusCode == http.StatusNotFound || resp.StatusCode == http.StatusMethodNotAllowed {
		return capability, fmt.Errorf("device does not support model copy")
	}
	if resp.StatusCode != http.StatusOK {
		return capability, remoteStatusError("copy capabilities", resp.StatusCode, data)
	}
	if err := json.Unmarshal(data, &capability); err != nil || capability.Version != 1 || capability.Engine != engine {
		return capability, fmt.Errorf("unrecognized model copy capability")
	}
	return capability, nil
}

func (c *remoteClient) copyModel(ctx context.Context, p copyRequest, emit func(copyProgress)) (copyModelResult, error) {
	var res copyModelResult
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	destination := p.DestinationNodeID
	p.DestinationNodeID = "" // Recipient always owns the destination.
	body, err := json.Marshal(p)
	if err != nil {
		return res, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.base+controlCopyPath, bytes.NewReader(body))
	if err != nil {
		return res, err
	}
	req.Header.Set("Content-Type", "application/json")
	// Suppress net/http's stale-connection replay. The shared direct/reverse
	// adapter independently forbids alternate-route replay after dispatch.
	req.Close, req.GetBody = true, nil
	resp, err := c.controlClient(false).Do(req)
	if err != nil {
		return res, copyUnknown(err)
	}
	defer resp.Body.Close()
	stall := time.AfterFunc(copyStallTimeout, cancel)
	defer stall.Stop()
	if resp.StatusCode != http.StatusOK {
		data, err := readRemoteResponse(resp.Body, maxCopyControlBody)
		if err != nil {
			return res, err
		}
		return res, remoteStatusError("remote copy", resp.StatusCode, data)
	}
	reader := bufio.NewReaderSize(resp.Body, maxCopyProgressFrame)
	var previous streamFrame
	for {
		// Copy operation results contain counters/identity only, never an
		// artifact manifest. Every frame has the smaller progress-frame cap.
		data, err := reader.ReadSlice('\n')
		if err != nil {
			return res, copyUnknown(err)
		}
		if len(data) > maxCopyProgressFrame {
			return res, copyUnknown(fmt.Errorf("copy frame exceeds bounds"))
		}
		var frame streamFrame
		if err := json.Unmarshal(data, &frame); err != nil || !utf8.Valid(data) || !copyResponseIdentity(data, "opId", p.OpID) || !copyResponseIdentity(data, "engine", p.Engine) || !copyResponseIdentity(data, "op", "copy") || !copyResponseIdentity(data, "type", frame.Type) {
			return res, copyUnknown(fmt.Errorf("invalid copy stream"))
		}
		switch frame.Type {
		case "progress":
			if frame.SourceNodeID != p.SourceNodeID || frame.DestinationNodeID != destination || frame.Model != p.Model || frame.Node != p.SourceNodeID {
				return res, copyUnknown(fmt.Errorf("invalid copy progress identity"))
			}
			if copyTerminalStage(frame.Stage) {
				return res, copyUnknown(fmt.Errorf("unexpected terminal copy progress"))
			}
			if frame.BytesDone < 0 || frame.BytesTotal < 0 || frame.BytesDone > frame.BytesTotal || frame.Percent < 0 || frame.Percent > 100 || frame.ExportID != "" && !lowerHex(frame.ExportID, 64) {
				return res, copyUnknown(fmt.Errorf("invalid copy progress counters"))
			}
			if frame.Stage != previous.Stage || frame.File != previous.File || frame.Percent > previous.Percent || frame.BytesDone > previous.BytesDone {
				stall.Reset(copyStallTimeout)
			}
			previous = frame
			if emit != nil {
				emit(copyProgress{remoteProgress: remoteProgress{OpID: frame.OpID, Node: frame.Node, Engine: frame.Engine, Op: frame.Op, Stage: frame.Stage, Percent: frame.Percent, Message: frame.Message}, SourceNodeID: frame.SourceNodeID, DestinationNodeID: frame.DestinationNodeID, Model: frame.Model, ExportID: frame.ExportID, File: frame.File, BytesDone: frame.BytesDone, BytesTotal: frame.BytesTotal})
			}
		case "error":
			if frame.Stage == "unknown" {
				return res, &copyOutcomeUnknownError{errors.New(frame.Message)}
			}
			if frame.Stage == "cancelled" {
				return res, context.Canceled
			}
			if code, ok := codeOfWire(frame.Code); ok {
				return res, &codedError{code: code, message: frame.Message}
			}
			return res, fmt.Errorf("%s", frame.Message)
		case "result":
			payload, unambiguous := downloadIdentityField(data, "result")
			res, err = decodeCopyResult(payload)
			if err != nil || !unambiguous || !validCopyResult(res, p.Engine, p.Model) {
				return res, copyUnknown(fmt.Errorf("invalid terminal copy result"))
			}
			if p.ExpectedExportID != "" && res.ExportID != p.ExpectedExportID {
				return res, copyUnknown(fmt.Errorf("copy result does not match expected export identity"))
			}
			return res, nil
		default:
			return res, copyUnknown(fmt.Errorf("unrecognized copy stream frame"))
		}
	}
}

func copyResponseIdentity(data []byte, field, expected string) bool {
	raw, ok := downloadIdentityField(data, field)
	var value string
	return ok && copyStringUnicode(raw) && json.Unmarshal(raw, &value) == nil && value == expected
}

// Validate field presence and uniqueness before decoding optional zero values.
// In particular, a later empty exportId must not erase an earlier identity.
func decodeCopyResult(data []byte) (copyModelResult, error) {
	var res copyModelResult
	invalid := errors.New("invalid terminal copy result")
	decoder := json.NewDecoder(bytes.NewReader(data))
	if token, err := decoder.Token(); err != nil || token != json.Delim('{') || !utf8.Valid(data) {
		return res, invalid
	}
	seen := make(map[string]bool)
	for decoder.More() {
		token, err := decoder.Token()
		name, ok := token.(string)
		if err != nil || !ok || seen[name] {
			return res, invalid
		}
		var raw json.RawMessage
		if decoder.Decode(&raw) != nil || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
			return res, invalid
		}
		switch name {
		case "engine", "model", "layout", "exportId", "durability":
			if !copyStringUnicode(raw) {
				return res, invalid
			}
		case "files", "filesSkipped", "bytesTotal", "bytesCopied", "requiresConfiguration":
		default:
			return res, invalid
		}
		seen[name] = true
	}
	if _, err := decoder.Token(); err != nil || decoder.Decode(new(json.RawMessage)) != io.EOF {
		return res, invalid
	}
	for _, required := range []string{"engine", "model", "files", "filesSkipped", "bytesTotal", "bytesCopied"} {
		if !seen[required] {
			return res, invalid
		}
	}
	if json.Unmarshal(data, &res) != nil || seen["exportId"] && !lowerHex(res.ExportID, 64) {
		return res, invalid
	}
	return res, nil
}

func validCopyResult(res copyModelResult, engine, model string) bool {
	if res.Engine != engine || res.Model != model || res.Files <= 0 || res.FilesSkipped < 0 || res.FilesSkipped > res.Files || res.BytesTotal < 0 || res.BytesCopied < 0 {
		return false
	}
	if res.Layout == "" {
		return res.ExportID == "" && res.Durability == "" && !res.RequiresConfiguration
	}
	return res.Layout == llamaArtifactsLayout && lowerHex(res.ExportID, 64) && res.RequiresConfiguration && (res.Durability == copyDurabilityConfirmed || res.Durability == copyDurabilityUncertain)
}

func copyProgressFrame(cp copyProgress) streamFrame {
	return streamFrame{Type: "progress", OpID: cp.OpID, Engine: cp.Engine, Op: cp.Op, Stage: cp.Stage, Percent: cp.Percent, Message: cp.Message, Node: cp.Node, SourceNodeID: cp.SourceNodeID, DestinationNodeID: cp.DestinationNodeID, Model: cp.Model, ExportID: cp.ExportID, File: cp.File, BytesDone: cp.BytesDone, BytesTotal: cp.BytesTotal}
}
