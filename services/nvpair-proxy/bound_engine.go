// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/nodepolicy"
)

type boundEngineDeadlineKey struct{}

// This value is captured with the pinned certificate at authenticated ingress,
// before any request body or storage wait. HTTP fields never populate it.
type boundEnginePeerKey struct{}
type boundEnginePeer struct{ id, incarnation string }

func (f *facade) boundIngressPeerCurrent(ctx context.Context) bool {
	peer, ok := ctx.Value(boundEnginePeerKey{}).(boundEnginePeer)
	if !ok {
		return true
	} // Local requests and internal terminal callers.
	incarnation, current, diagnostic := f.host.mesh.PeerServiceIncarnationWithDiagnostic(peer.id)
	if current && incarnation == peer.incarnation {
		return true
	}
	if current {
		diagnostic.Predicate = "incarnation_mismatch"
	}
	diagnostic.ExpectedIncarnation, diagnostic.ObservedIncarnation = peer.incarnation, incarnation
	logBoundEngineManagementRefusal("proxy.peer_incarnation", peer.id, diagnostic)
	return false
}

func logBoundEngineManagementRefusal(site, peer string, diagnostic clustertrust.PeerServiceDiagnostic) {
	slog.Warn("bound engine management refused",
		"site", site, "reason", diagnostic.Predicate,
		"node_id", diagnostic.NodeID, "peer_id", peer, "cluster_id", diagnostic.ClusterID,
		"epoch", diagnostic.Epoch, "counter", diagnostic.Counter, "io_error_code", diagnostic.ErrorCode,
		"expected_incarnation", diagnostic.ExpectedIncarnation, "observed_incarnation", diagnostic.ObservedIncarnation,
		"changed_component_authority_compared", diagnostic.ChangedComponentCompared,
		"changed_component_authority_same", diagnostic.ChangedComponentSame,
		"permission", "not_checked")
}

var errBoundEnginePrepareTimeout = errors.New("bound engine preparation deadline expired")

// Only cold pinned hops use this header-phase timer. A context deadline would
// also kill an accepted stream. Retire the timer on headers, then retain the
// same cancellation owner until the transport body closes normally.
type boundEnginePreparationTransport struct{ http.RoundTripper }

func (t boundEnginePreparationTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	remaining := boundEngineRemaining(r.Context())
	if remaining < time.Millisecond {
		return nil, errBoundEnginePrepareTimeout
	}
	ctx, cancel := context.WithCancelCause(r.Context())
	expired := make(chan struct{})
	timer := time.AfterFunc(remaining, func() { cancel(errBoundEnginePrepareTimeout); close(expired) })
	resp, err := t.RoundTripper.RoundTrip(r.WithContext(ctx))
	if !timer.Stop() {
		<-expired
	}
	if errors.Is(context.Cause(ctx), errBoundEnginePrepareTimeout) || boundEngineRemaining(r.Context()) < time.Millisecond || errors.Is(err, clustertrust.ErrPeerSlotOccupied) {
		if resp != nil {
			_ = resp.Body.Close()
		}
		cancel(errBoundEnginePrepareTimeout)
		return nil, errBoundEnginePrepareTimeout
	}
	if err != nil {
		cancel(err)
		return resp, err
	}
	resp.Body = &boundEnginePreparationBody{ReadCloser: resp.Body, cancel: func() { cancel(nil) }}
	return resp, nil
}

type boundEnginePreparationBody struct {
	io.ReadCloser
	cancel context.CancelFunc
}

func (b *boundEnginePreparationBody) Close() error { defer b.cancel(); return b.ReadCloser.Close() }

func withBoundEngineDeadline(r *http.Request, start time.Time) *http.Request {
	deadline := start.Add(nodepolicy.EnginePrepareBudget(r.Header.Values(nodepolicy.EnginePrepareBudgetHeader)))
	if d, ok := r.Context().Deadline(); ok && d.Before(deadline) {
		deadline = d
	}
	if d, ok := r.Context().Value(routingDeadlineContextKey{}).(time.Time); ok && d.Before(deadline) {
		deadline = d
	}
	return r.WithContext(context.WithValue(r.Context(), boundEngineDeadlineKey{}, deadline))
}

func boundEngineRemaining(ctx context.Context) time.Duration {
	d, _ := ctx.Value(boundEngineDeadlineKey{}).(time.Time)
	remaining := min(time.Until(d), nodepolicy.MaxBoundEngineBudgetMs*time.Millisecond)
	if d, ok := ctx.Deadline(); ok {
		remaining = min(remaining, time.Until(d))
	}
	return max(remaining, 0)
}

func boundEngineRefusal(reason string) nodepolicy.BoundEnginePrepared {
	return nodepolicy.BoundEnginePrepared{Reason: reason, Code: -32000}
}

func isBoundEngineRefusal(reason string) bool {
	switch reason {
	case "engine-switch-timeout", "engine-switch-busy", "engine-switch-cooldown", "engine-locked", "management-denied", "model-unavailable":
		return true
	}
	return false
}

func writeBoundEngineRefusal(w http.ResponseWriter, result nodepolicy.BoundEnginePrepared) {
	reason := result.Reason
	if reason == "stale-revision" || reason == "unavailable" || reason == "" {
		reason = "model-unavailable"
	}
	w.Header().Set(nodepolicy.AdmissionHeader, reason)
	if result.SwitchRequestID != "" {
		w.Header().Set(nodepolicy.SwitchRequestIDHeader, result.SwitchRequestID)
	}
	switch reason {
	case "engine-switch-timeout", "engine-switch-busy", "engine-switch-cooldown", "draining", "no-fit", "busy":
		w.Header().Set("Retry-After", "1")
	case "paused":
		w.Header().Set("Retry-After", "30")
	}
	// Broker messages never enter the inference response. Keep the existing
	// structured inference shape and bounded operational text.
	writeInferenceErrorFields(w, http.StatusServiceUnavailable,
		inferenceErrorField{"code", []string{"admission"}},
		inferenceErrorField{"error", []string{"Could not prepare the engine for this model within the request budget."}},
		inferenceErrorField{"reason", []string{reason}},
	)
}

func (f *facade) boundModel(pol nodepolicy.Policy, model string) bool {
	return slices.ContainsFunc(pol.EngineBindings, func(ref nodepolicy.Role) bool {
		return ref.Engine == f.profile.Name && f.profile.normalizeModel(ref.Model) == f.profile.normalizeModel(model)
	})
}

func (f *facade) needsBoundPreparation(model string) bool {
	if !f.boundModel(f.host.admission.currentPolicy(), model) {
		return false
	}
	_, models := f.host.selfIdentity(f.profile.Name)
	return !f.currentLocalBackend().Healthy || !slices.ContainsFunc(models, func(m string) bool {
		return f.profile.normalizeModel(m) == f.profile.normalizeModel(model)
	})
}

// This evidence is request-local. A receipt authorizes no dispatch: the real
// backend, inventory and policy are read after preparation, then admission
// checks their freshness again before granting and before sending any bytes.
type boundEngineEvidence struct {
	backend         localBackend
	policy          uint64
	identity        nodepolicy.EngineIdentity
	peer            string
	incarnation     string
	switchRequestID string
}

func (f *facade) boundEvidenceCurrentLocked(e *boundEngineEvidence) bool {
	return e == nil || e.policy == f.host.admission.policyGeneration &&
		e.backend == f.currentLocalBackend() && e.backend.Healthy &&
		reflect.DeepEqual(e.identity, f.host.admission.strict.identity[f.profile.Name])
}

func (f *facade) boundPeerCurrent(e *boundEngineEvidence) bool {
	if e == nil || e.peer == "" {
		return true
	}
	incarnation, ok := f.host.mesh.PeerServiceIncarnation(e.peer)
	return ok && incarnation == e.incarnation
}

// prepareLocalBound never holds a ticket, placement lock or IPC reader. Peer
// identity comes exclusively from the pinned-ingress principal supplied here.
func (f *facade) prepareLocalBound(ctx context.Context, model, peer string) (*boundEngineEvidence, nodepolicy.BoundEnginePrepared) {
	p := f.host
	pol := p.admission.currentPolicy()
	if !f.localBindingConsistent(pol, model) {
		return nil, boundEngineRefusal("model-unavailable")
	}
	if !f.boundModel(pol, model) {
		return nil, boundEngineRefusal("model-unavailable")
	}
	if availability := p.admission.currentAvailability(); availability != nodepolicy.Available {
		reason := "draining"
		if availability == nodepolicy.Paused {
			reason = "paused"
		}
		return nil, boundEngineRefusal(reason)
	}
	budget := int(boundEngineRemaining(ctx).Milliseconds())
	if budget < 1 {
		return nil, boundEngineRefusal("engine-switch-timeout")
	}
	in := nodepolicy.PrepareBoundEngineRequest{PolicyRevision: pol.Revision, Engine: f.profile.Name, Model: model, BudgetMs: budget}
	if peer != "" {
		authority, ok := ctx.Value(boundEnginePeerKey{}).(boundEnginePeer)
		if !ok || authority.id != peer {
			reason := "peer_context_missing"
			if ok {
				reason = "peer_context_mismatch"
			}
			logBoundEngineManagementRefusal("proxy.peer_context", peer, clustertrust.PeerServiceDiagnostic{Predicate: reason})
			return nil, boundEngineRefusal("management-denied")
		}
		if !f.boundIngressPeerCurrent(ctx) {
			return nil, boundEngineRefusal("management-denied")
		}
		in.PeerID, in.PeerIncarnation = authority.id, authority.incarnation
	}
	result, err := p.prepareBoundEngine(ctx, in)
	if err != nil {
		reason := "engine-switch-busy"
		if errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled) {
			reason = "engine-switch-timeout"
		}
		return nil, boundEngineRefusal(reason)
	}
	if !result.Success {
		return nil, result
	}
	refuse := func() (*boundEngineEvidence, nodepolicy.BoundEnginePrepared) {
		result.Success, result.Reason = false, "model-unavailable"
		return nil, result
	}
	p.admission.mu.Lock()
	e := &boundEngineEvidence{backend: f.currentLocalBackend(), policy: p.admission.policyGeneration,
		identity: p.admission.strict.identity[f.profile.Name], peer: peer, incarnation: in.PeerIncarnation, switchRequestID: result.SwitchRequestID}
	current := p.admission.policy.Revision == pol.Revision && f.boundModel(p.admission.policy, model)
	p.admission.mu.Unlock()
	if !current || !e.backend.Healthy || !f.boundPeerCurrent(e) {
		return refuse()
	}
	prepareCtx, cancel := context.WithTimeout(ctx, boundEngineRemaining(ctx))
	defer cancel()
	if !f.boundInventoryContains(prepareCtx, e.backend, model) {
		if prepareCtx.Err() != nil {
			result.Success, result.Reason = false, "engine-switch-timeout"
			return nil, result
		}
		return refuse()
	}
	p.admission.mu.Lock()
	current = f.boundEvidenceCurrentLocked(e)
	p.admission.mu.Unlock()
	if !current || !f.boundPeerCurrent(e) {
		return refuse()
	}
	return e, result
}

// Query canonical inventory at the actual loopback backend after a switch;
// offers and delayed discovery snapshots cannot establish model presence.
// The existing inference buffer account owns this bounded temporary read.
func (f *facade) boundInventoryContains(ctx context.Context, backend localBackend, model string) bool {
	target, ok := backend.target()
	if !ok {
		return false
	}
	var route route
	for _, r := range f.profile.Routes {
		if r.Role.isModelList() {
			route = r
			break
		}
	}
	if route.Path == "" {
		return false
	}
	target.Path = route.upstreamPath()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, target.String(), nil)
	if err != nil {
		return false
	}
	withLocalCredential(req.Header, backend.Credential)
	client := *modelListClient
	client.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	resp, err := client.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return false
	}
	w := newBodyWorkspace(ctx)
	defer w.close()
	if err := w.acquire(maxModelListBytes + 1); err != nil {
		return false
	}
	body := make([]byte, maxModelListBytes+1)
	n, err := io.ReadFull(resp.Body, body)
	body = body[:n]
	if err != nil && err != io.EOF && err != io.ErrUnexpectedEOF || len(body) > maxModelListBytes || !ownedJSONValid(body) {
		return false
	}
	field := "models"
	if route.Role == roleModelListOpenAIGET {
		field = "data"
	}
	found := false
	visitArray(requestJSONField(body, field), func(raw []byte) {
		valid := isJSONObject(raw)
		visitObject(raw, func(key []byte, start, end int) {
			if (jsonKeyEqual(key, "id", true) || jsonKeyEqual(key, "model", true) || jsonKeyEqual(key, "name", true)) && nodepolicy.ValidateModelIdentifierJSON(raw[start:end]) != nil {
				valid = false
			}
		})
		if !valid {
			return
		}
		name := requestStructString(raw, "id", nil)
		if field == "models" {
			name = requestStructString(raw, "model", nil)
			if len(name) == 0 {
				name = requestStructString(raw, "name", nil)
			}
		}
		// Decode the bounded native identifier with the same owner as bodies.
		if len(name) > 0 && nodepolicy.ValidateModelIdentifierJSON(name) == nil {
			var decoded string
			if json.Unmarshal(name, &decoded) == nil && nodepolicy.ValidateModelIdentifier(decoded) == nil && f.profile.normalizeModel(decoded) == f.profile.normalizeModel(model) {
				found = true
			}
		}
	})
	return found
}

func (f *facade) beforeBoundDispatch(ticket *admissionTicket, e *boundEngineEvidence) bool {
	if e == nil {
		return true
	}
	if !f.boundPeerCurrent(e) {
		return false
	}
	a := f.host.admission
	a.mu.Lock()
	defer a.mu.Unlock()
	return ticket != nil && a.tickets[ticket.id] == ticket && f.boundEvidenceCurrentLocked(e) &&
		a.availability == nodepolicy.Available && !a.drain[f.profile.Name]
}

func setBoundEngineBudget(h http.Header, ctx context.Context) {
	h.Set(nodepolicy.EnginePrepareBudgetHeader, strconv.FormatInt(boundEngineRemaining(ctx).Milliseconds(), 10))
}

// Only explicit bindings widen a concrete OpenAI request's engine domain.
// Ordinary names stay on their entry engine, and native protocols stay local
// to a compatible facade. Each facade retains its own discovery/transport.
func (f *facade) resolveConcreteCandidates(model, path string) []candidate {
	out := f.resolveCandidates(model)
	if model == "" || !openAICompatibleInference[path] {
		return out
	}
	facades := f.host.enabledFacades()
	slices.SortFunc(facades, func(a, b *facade) int { return strings.Compare(a.profile.Name, b.profile.Name) })
	for _, g := range facades {
		if g == f || !isInferenceRequest(g.profile, http.MethodPost, path) {
			continue
		}
		for _, c := range g.resolveCandidates(model) {
			bound := false
			if c.self {
				bound = g.boundModel(f.host.admission.currentPolicy(), model)
			} else if n, ok := discoveredNode(g, c.id); ok {
				bound = g.nodeOffersBinding(n, model)
			}
			if bound {
				c.model = model
				out = append(out, c)
			}
		}
	}
	selected := f.SelectedID()
	slices.SortStableFunc(out, func(a, b candidate) int {
		if selected != "" && (a.id == selected) != (b.id == selected) {
			if a.id == selected {
				return -1
			}
			return 1
		}
		if a.preparation != b.preparation {
			if !a.preparation {
				return -1
			}
			return 1
		}
		return 0
	})
	return out
}

type boundEngineWaiters struct {
	mu      sync.Mutex
	pending map[string]chan nodepolicy.BoundEnginePrepared
}

// Callers hold their request buffer, but no placement, resident or execution
// ticket. Routing must refresh all dispatch evidence after success.
func (p *Proxy) prepareBoundEngine(ctx context.Context, in nodepolicy.PrepareBoundEngineRequest) (nodepolicy.BoundEnginePrepared, error) {
	in.RequestID, in.ProxyRunID = rand.Text(), p.runID
	if err := in.Validate(); err != nil {
		return nodepolicy.BoundEnginePrepared{}, err
	}
	ctx, cancel := context.WithTimeout(ctx, time.Duration(in.BudgetMs)*time.Millisecond)
	defer cancel()
	deadline, _ := ctx.Deadline()
	in.BudgetMs = min(in.BudgetMs, int(time.Until(deadline).Milliseconds()))
	if in.BudgetMs < 1 {
		return nodepolicy.BoundEnginePrepared{}, context.DeadlineExceeded
	}
	w := &p.boundEngines
	ch := make(chan nodepolicy.BoundEnginePrepared, 1)
	w.mu.Lock()
	if len(w.pending) >= bufferBudgetMaxWaiters {
		w.mu.Unlock()
		return nodepolicy.BoundEnginePrepared{}, errors.New("engine preparation queue is full")
	}
	if w.pending == nil {
		w.pending = make(map[string]chan nodepolicy.BoundEnginePrepared)
	}
	w.pending[in.RequestID] = ch
	w.mu.Unlock()
	defer func() { w.mu.Lock(); delete(w.pending, in.RequestID); w.mu.Unlock() }()
	if err := ctx.Err(); err != nil {
		return nodepolicy.BoundEnginePrepared{}, err
	}
	if err := p.codec.NotifyContext(ctx, nodepolicy.MethodPrepareBoundEngine, in); err != nil {
		if ctx.Err() != nil {
			w.mu.Lock()
			delete(w.pending, in.RequestID)
			w.mu.Unlock()
			p.cancelBoundEngineWaiter(in.RequestID)
		}
		return nodepolicy.BoundEnginePrepared{}, err
	}
	select {
	case reply := <-ch:
		return reply, nil
	case <-ctx.Done():
		w.mu.Lock()
		delete(w.pending, in.RequestID)
		w.mu.Unlock()
		p.cancelBoundEngineWaiter(in.RequestID)
		return nodepolicy.BoundEnginePrepared{}, ctx.Err()
	}
}

func (p *Proxy) cancelBoundEngineWaiter(id string) {
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	_ = p.codec.NotifyContext(ctx, nodepolicy.MethodCancelBoundEngine, nodepolicy.CancelBoundEngineRequest{RequestID: id, ProxyRunID: p.runID})
}

func (p *Proxy) receiveBoundEnginePrepared(raw json.RawMessage) {
	var reply nodepolicy.BoundEnginePrepared
	if json.Unmarshal(raw, &reply) != nil || reply.ProxyRunID != p.runID {
		return
	}
	w := &p.boundEngines
	w.mu.Lock()
	defer w.mu.Unlock()
	if ch := w.pending[reply.RequestID]; ch != nil {
		delete(w.pending, reply.RequestID)
		ch <- reply
	}
}
