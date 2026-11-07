// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"reflect"
	"slices"
	"strings"
	"time"

	"nvpair-shared/durablefile"
	"nvpair-shared/engines"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/modelconfig"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
)

type settingsWorkerError struct {
	code    int
	message string
}

func (e *settingsWorkerError) Error() string { return e.message }

func engineSettingsErrorCode(err error) int {
	return policyErrorCode(err)
}

// validateModelSettingsEntry runs before journal initialization or worker IO.
// The settings entry points reserve maintenance after validation, before either
// the ordinary or model-change branch can enter an owner or worker.
func validateModelSettingsEntry(p settings.Request) error {
	if err := p.Validate(); err != nil {
		return &invalidParamsError{err: err}
	}
	if p.ModelChange != nil && (p.Settings.ServerPort < 1 || p.Settings.ServerPort > 65535 || p.Settings.ProxyPort < 1 || p.Settings.ProxyPort > 65535 || p.Settings.ServerPort == p.Settings.ProxyPort) {
		return &invalidParamsError{err: fmt.Errorf("modelChange requires a complete target settings baseline")}
	}
	return nil
}

// This is a subrecord of the settings journal, including for load-only work.
// Every model change reserves the policy, including derived load metadata.
// Failed and uncertain operations keep their withdrawal until a fresh local
// supersession is accepted.
type modelSettingsOperation struct {
	Caller              string                  `json:"caller,omitempty"`
	Change              modelconfig.ModelChange `json:"change"`
	Review              settings.Preview        `json:"review"`
	ReviewedModelChange modelconfig.ModelChange `json:"reviewedModelChange,omitzero"`
	ConversionProfiles  []string                `json:"conversionProfiles,omitempty"`
	NativeBefore        settings.Config         `json:"nativeBefore"`
	Before              string                  `json:"before"`
	After               string                  `json:"after"`
	Implications        string                  `json:"implications"`
	Stage               string                  `json:"stage"`
	Attempted           bool                    `json:"attempted"`
	PolicyRevision      string                  `json:"policyRevision,omitempty"`
}

func validateStoredModelSettings(record *engineSettingsRecord) error {
	op := record.Model
	if op == nil {
		return nil
	}
	if err := op.Change.Validate(); err != nil {
		return err
	}
	if !slices.Contains([]string{"accepted", "acceptance-uncertain", "drained", "native-applying", "native-persisted", "policy-persisted", "activation-pending", "releasing", "completed", "completion-uncertain"}, op.Stage) {
		return fmt.Errorf("invalid model settings stage")
	}
	if _, err := op.NativeBefore.Canonical(); err != nil {
		return err
	}
	if _, err := record.Snapshot.Settings.Canonical(); err != nil {
		return err
	}
	if !op.Review.Settings.Equal(record.Snapshot.Settings) || record.Snapshot.RequestID == "" {
		return fmt.Errorf("model settings journal lost its accepted review")
	}
	if op.Change.Remove && (op.Change.PreviewDigest == "" || op.Change.PreviewDigest != op.Review.PreviewDigest) {
		return fmt.Errorf("model settings journal lost its accepted public removal review")
	}
	if _, err := op.reviewedRemoval(); err != nil {
		return err
	}
	if err := validateConversionProfiles(op.Review, op.ConversionProfiles); err != nil {
		return err
	}
	// Ordinary native previews return only the selected identity. Conversion
	// reviews additionally retain sibling identities, so losing their scope
	// cannot silently turn a fresh repair into an ordinary selected-only edit.
	requireScope := len(op.ConversionProfiles) != 0
	for _, load := range op.Review.Settings.ModelLoads {
		if _, converted := modelConversionInput(load); converted {
			if _, accepted := op.Review.Identities[load.ProfileKey]; accepted && (len(op.NativeBefore.ModelLoads) == 0 || load.ProfileKey != modelChangeKey(op.Change)) {
				requireScope = true
			}
		}
	}
	if requireScope {
		for _, load := range op.Review.Settings.ModelLoads {
			_, converted := modelConversionInput(load)
			if converted && !slices.Contains(op.ConversionProfiles, load.ProfileKey) {
				return fmt.Errorf("accepted conversion publication scope is incomplete")
			}
		}
	}
	for _, fingerprint := range []string{op.Before, op.After, op.Implications} {
		if value, err := hex.DecodeString(fingerprint); err != nil || len(value) != sha256.Size {
			return fmt.Errorf("invalid model settings policy fingerprint")
		}
	}
	return nil
}

func modelSettingsConflict(message string) error {
	return &settingsWorkerError{code: codePolicyConflict, message: message}
}

func (op modelSettingsOperation) reviewedRemoval() (*modelconfig.ModelChange, error) {
	var reviewed *modelconfig.ModelChange
	if op.ReviewedModelChange != (modelconfig.ModelChange{}) {
		reviewed = &op.ReviewedModelChange
	}
	return reviewedModelRemoval(op.Change, reviewed, op.Review.PreviewDigest, op.native())
}

// Recovery consumes accepted native changes and proof, independent of the
// currently available descriptor catalog. Descriptor-only removals leave the
// native settings unchanged and carry no native omission proof.
func (op modelSettingsOperation) native() bool {
	return op.Change.Load != nil || op.Change.Remove && (!op.NativeBefore.Equal(op.Review.Settings) || op.ReviewedModelChange.Remove)
}

// The public digest binds the broker review. Native omission authority is the
// original worker selection and digest, retained separately through recovery.
func reviewedModelRemoval(change modelconfig.ModelChange, reviewed *modelconfig.ModelChange, publicDigest string, native bool) (*modelconfig.ModelChange, error) {
	if !change.Remove || !native {
		if reviewed != nil {
			return nil, modelSettingsConflict("unexpected reviewed removal evidence")
		}
		return nil, nil
	}
	if reviewed == nil {
		return nil, modelSettingsConflict("original native removal review is missing; explicitly Recalculate")
	}
	if err := settings.ValidateReviewedModelChange(reviewed); err != nil {
		return nil, modelSettingsConflict(err.Error())
	}
	if modelChangeKey(*reviewed) != modelChangeKey(change) || reviewed.ModelID != change.ModelID || reviewed.SupersedesRequestID != "" || change.SupersedesRequestID == "" && reviewed.ExpectedPolicyRevision != change.ExpectedPolicyRevision || reviewed.PreviewDigest == publicDigest || reviewed.PreviewDigest == change.PreviewDigest {
		return nil, modelSettingsConflict("original native removal review does not match the accepted selection")
	}
	copy := reviewed.Clone()
	return &copy, nil
}

func modelSettingsHash(value any) string {
	// Canonical object ordering includes opaque nested request-option objects;
	// decoding with UseNumber preserves every numeric digit.
	raw := settingsJSON(value)
	var canonical any
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	if decoder.Decode(&canonical) == nil {
		raw = settingsJSON(canonical)
	}
	sum := sha256.Sum256(raw)
	return hex.EncodeToString(sum[:])
}

func modelProfileDisplayName(ref nodepolicy.ModelRef) string {
	if nodepolicy.ValidateProfileDisplayName(ref.Model) == nil {
		return ref.Model
	}
	return "model-" + modelSettingsHash(ref)[:16]
}

func modelPolicyFingerprint(policy nodepolicy.Policy) string {
	var fields map[string]json.RawMessage
	if json.Unmarshal(settingsJSON(policy), &fields) != nil {
		return ""
	}
	dropFields(fields, "revision", "availability", "pausedBy", "pauseMode")
	return modelSettingsHash(fields)
}

// Implications identify the served profile rather than its artifact selector.
// Availability is checked separately.
func modelPolicyImplications(policy nodepolicy.Policy, engine, model string) string {
	var profiles []nodepolicy.Profile
	for _, profile := range policy.Profiles {
		if profile.Engine == engine && profile.Key == model {
			profiles = append(profiles, profile)
		}
	}
	ref := nodepolicy.ModelRef{Engine: engine, Model: model}
	return modelSettingsHash(struct {
		Profiles                       []nodepolicy.Profile
		Resident, Served, Unrestricted bool
	}{profiles, slices.Contains(policy.ResidentModels, ref), slices.Contains(policy.ServedModels, ref), policy.ServedModels == nil})
}

func (b *Broker) modelSettingsPendingLocked(engine string) bool {
	record := b.engineSettings[engine]
	return record != nil && record.Model != nil && record.Model.Stage != "completed"
}

// Pending snapshots bypass live readback. Their running observation belongs to
// the old incarnation when loading a journal or replacing engine-manager, even
// though the accepted intent and reconciliation state remain authoritative.
// The caller holds engineConfigMu; same-incarnation acceptance uncertainty
// deliberately does not invalidate the running observation.
func (b *Broker) invalidatePendingModelSettingsRunningLocked() {
	for engine, record := range b.engineSettings {
		if b.modelSettingsPendingLocked(engine) {
			record.Snapshot.Running = false
		}
	}
}

func modelSettingsNeedsReconciliation(record *engineSettingsRecord) bool {
	return record.Model != nil && record.Model.Stage != "completed" && (record.Model.Stage == "activation-pending" || record.Model.Stage == "acceptance-uncertain" || record.Receipts[record.Snapshot.RequestID].Phase == "failed")
}

// An unreadable journal cannot establish whether unrecognizable bytes once
// contained model work. Keep ordinary restoration behavior, but never discard
// a model reservation already recognizable in the file or known in memory.
func (b *Broker) knownModelSettingsFenceLocked() bool {
	for _, record := range b.engineSettings {
		if record != nil && record.Model != nil && (record.Model.Stage != "completed" || b.engineSettingsError != nil && validateStoredModelSettings(record) != nil) {
			return true
		}
	}
	return false
}

func markModelAcceptanceUncertain(record *engineSettingsRecord) {
	record.Model.Stage, record.Snapshot.CoupledStage = "acceptance-uncertain", "acceptance-uncertain"
	record.Snapshot.Phase, record.Snapshot.Editable = "applying", false
	record.Snapshot.Error, record.Snapshot.UnresolvedError = "Interrupted before changes; preview again on this device.", "Interrupted before changes; preview again on this device."
	record.Snapshot.Reason = record.Snapshot.UnresolvedError
	receipt := record.Receipts[record.Snapshot.RequestID]
	receipt.Phase = "failed"
	record.Receipts[record.Snapshot.RequestID] = receipt
}

// Must run before taking policyRuntime.mu. Availability writers deliberately
// do not use this fence; they retain their existing independent owner.
func (b *Broker) modelPolicyReservationLocked() error {
	if err := b.loadEngineSettingsLocked(); err != nil {
		if b.knownModelSettingsFenceLocked() {
			return err
		}
		return nil
	}
	for _, record := range b.engineSettings {
		if record.Model != nil && record.Model.Stage != "completed" {
			return &policyConflictError{expected: "pending model settings " + record.Snapshot.RequestID, current: record.Model.PolicyRevision}
		}
	}
	return nil
}

// The journal is the reservation authority. Rebuild volatile admission and
// identity fences before replaying proxy state, advertisements or restoring engines.
func (b *Broker) reconstructModelSettingsLocked() {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	defer r.mu.Unlock()
	for engine, record := range b.engineSettings {
		if record == nil {
			continue
		}
		if b.modelSettingsPendingLocked(engine) {
			record.Snapshot.Phase = "applying"
			if record.Model.native() {
				r.drains[engine] = true
				delete(r.settingsIdentity, engine)
			}
		}
		if record.Model != nil {
			record.Snapshot.CoupledStage = record.Model.Stage
		}
	}
}

func modelChangeKey(change modelconfig.ModelChange) string {
	if change.ProfileKey != "" {
		return change.ProfileKey
	}
	return change.ModelID
}

func modelChangeNative(change modelconfig.ModelChange, loads []modelconfig.ModelLoad) bool {
	return change.Load != nil || change.Remove && slices.ContainsFunc(loads, func(load modelconfig.ModelLoad) bool {
		return load.ProfileKey == modelChangeKey(change)
	})
}

// Refuse undeclared membership and retargeting before native preview or writes.
// Artifact identity is owned by the manager's typed load, not Profile.Model.
func validateModelPolicyChange(policy nodepolicy.Policy, engine string, change modelconfig.ModelChange, loads []modelconfig.ModelLoad) error {
	key := modelChangeKey(change)
	if change.Policy != nil {
		for name, selected := range change.Policy.Roles {
			if selected && !slices.ContainsFunc(policy.Roles, func(role nodepolicy.Role) bool { return role.Name == name }) {
				return &invalidParamsError{err: fmt.Errorf("Role %s is not declared on this node; create it in node settings.", name)}
			}
		}
	}
	typed := change.Load != nil
	for _, load := range loads {
		if load.ProfileKey == key {
			typed = true
			if load.ModelID != change.ModelID {
				return &invalidParamsError{err: fmt.Errorf("A profile keeps its model; create a new profile.")}
			}
		}
	}
	if profile, ok := policy.ProfileByKey(key); ok {
		model := change.ModelID
		if typed {
			model = key
		}
		// An untyped descriptor names the artifact. Its first typed load uses
		// the profile key as served id while retaining that exact artifact.
		if profile.Engine != engine || profile.Model != model && (!typed || profile.Model != change.ModelID) {
			return &invalidParamsError{err: fmt.Errorf("A profile keeps its model; create a new profile.")}
		}
	}
	return nil
}

// Merge selected leaves without resolving metadata. The same patch constructs
// the peer inference-lock preflight and the accepted policy candidate.
func patchModelPolicy(stored nodepolicy.Policy, engine string, change modelconfig.ModelChange, loads []modelconfig.ModelLoad, typedRemoval bool) (nodepolicy.Policy, error) {
	var policy nodepolicy.Policy
	if err := json.Unmarshal(settingsJSON(stored), &policy); err != nil {
		return policy, err
	}
	key := modelChangeKey(change)
	ref := nodepolicy.ModelRef{Engine: engine, Model: key}
	patch := change.Policy
	if change.Remove {
		profile, exists := policy.ProfileByKey(key)
		policy.Profiles = slices.DeleteFunc(policy.Profiles, func(profile nodepolicy.Profile) bool { return profile.Key == key })
		for i := range policy.Roles {
			policy.Roles[i].Profiles = slices.DeleteFunc(policy.Roles[i].Profiles, func(member string) bool { return member == key })
		}
		if typedRemoval {
			policy.ServedModels = slices.DeleteFunc(policy.ServedModels, func(member nodepolicy.ModelRef) bool { return member == ref })
			policy.ResidentModels = slices.DeleteFunc(policy.ResidentModels, func(member nodepolicy.ModelRef) bool { return member == ref })
		} else if exists {
			resident := nodepolicy.ModelRef{Engine: engine, Model: profile.Model}
			policy.ResidentModels = slices.DeleteFunc(policy.ResidentModels, func(member nodepolicy.ModelRef) bool { return member == resident })
		}
	} else {
		if err := validateModelPolicyChange(policy, engine, change, loads); err != nil {
			return policy, err
		}
		i := slices.IndexFunc(policy.Profiles, func(p nodepolicy.Profile) bool { return p.Key == key })
		if i < 0 {
			displayName := modelProfileDisplayName(ref)
			model := change.ModelID
			if change.Load != nil || slices.ContainsFunc(loads, func(load modelconfig.ModelLoad) bool { return load.ProfileKey == key }) {
				model = key
			}
			policy.Profiles = append(policy.Profiles, nodepolicy.Profile{Key: key, DisplayName: displayName, Engine: engine, Model: model})
			i = len(policy.Profiles) - 1
		}
		profile := &policy.Profiles[i]
		if change.Load != nil || slices.ContainsFunc(loads, func(load modelconfig.ModelLoad) bool { return load.ProfileKey == key }) {
			profile.Model = key
		}
		if patch != nil {
			if patch.DisplayName != nil {
				profile.DisplayName = *patch.DisplayName
			}
			if patch.Icon != nil {
				profile.Icon = *patch.Icon
			}
			options := map[string]json.RawMessage{}
			if len(profile.RequestOptions) != 0 {
				if err := json.Unmarshal(profile.RequestOptions, &options); err != nil {
					return policy, &invalidParamsError{err: err}
				}
				if options == nil {
					options = map[string]json.RawMessage{}
				}
			}
			for key, value := range patch.Generation {
				if value == nil {
					delete(options, key)
				} else {
					options[key] = json.RawMessage(value.String())
				}
			}
			if len(options) == 0 {
				profile.RequestOptions = nil
			} else {
				profile.RequestOptions = settingsJSON(options)
			}
			if patch.RequestOptionsMode != nil {
				profile.RequestOptionsMode = nodepolicy.RequestOptionsMode(*patch.RequestOptionsMode)
			}
			if len(patch.Roles) > 0 {
				for i := range policy.Roles {
					role := &policy.Roles[i]
					selected, patched := patch.Roles[role.Name]
					if !patched {
						continue
					}
					if selected && !slices.Contains(role.Profiles, key) {
						role.Profiles = append(role.Profiles, key)
					} else if !selected {
						role.Profiles = slices.DeleteFunc(role.Profiles, func(member string) bool { return member == key })
					}
				}
			}
			member := func(list []nodepolicy.ModelRef, selected bool) []nodepolicy.ModelRef {
				out := slices.Clone(list)
				if selected && !slices.Contains(out, ref) {
					out = append(out, ref)
				}
				if !selected {
					out = slices.DeleteFunc(out, func(r nodepolicy.ModelRef) bool { return r == ref })
				}
				if out == nil {
					out = []nodepolicy.ModelRef{}
				}
				return out
			}
			if patch.Served != nil {
				if policy.ServedModels == nil && !*patch.Served {
					return policy, &invalidParamsError{err: fmt.Errorf("cannot exclude one model from unrestricted serving; select an explicit served list first")}
				}
				// Selecting an already-unrestricted model preserves unrestricted scope.
				if policy.ServedModels != nil {
					policy.ServedModels = member(policy.ServedModels, *patch.Served)
				}
			}
			if patch.Resident != nil {
				policy.ResidentModels = member(policy.ResidentModels, *patch.Resident)
			}
		}
	}
	return policy, nil
}

func modelConversionInput(load modelconfig.ModelLoad) (string, bool) {
	if load.Resolved == nil {
		return "", false
	}
	if load.Resolved.ResolverRevision == "external-conversion/llama" {
		return load.Resolved.ReviewedInputID, true
	}
	if strings.HasPrefix(load.Resolved.ResolverRevision, "auto/") {
		source, _, converted := strings.Cut(load.Resolved.ReviewedInputID, ":")
		// Auto records use "-" when no external preset was converted.
		return source, converted && source != "-"
	}
	return "", false
}

func validateConversionProfiles(review settings.Preview, keys []string) error {
	seen := map[string]bool{}
	for _, key := range keys {
		if seen[key] || !slices.ContainsFunc(review.Settings.ModelLoads, func(load modelconfig.ModelLoad) bool { return load.ProfileKey == key }) {
			return modelSettingsConflict("accepted conversion profile is missing or duplicated")
		}
		seen[key] = true
		if _, ok := review.Identities[key]; !ok {
			return fmt.Errorf("native preview did not return converted profile %q's model identity", key)
		}
	}
	return nil
}

// Conversion publication belongs to the accepted operation, not to provenance
// left in an ordinary typed load. Fresh local repair carries that authority and
// the unchanged siblings' accepted facts into its own review and journal.
func modelConversionReview(previous *modelSettingsOperation, change modelconfig.ModelChange, review settings.Preview, before settings.Config) (settings.Preview, []string, error) {
	var keys []string
	if previous != nil && previous.Stage != "completed" && len(previous.ConversionProfiles) != 0 {
		review = review.Clone()
		if review.Identities == nil {
			review.Identities = map[string]engines.ModelIdentity{}
		}
		for _, key := range previous.ConversionProfiles {
			if key == modelChangeKey(change) && change.Remove {
				continue
			}
			keys = append(keys, key)
			if key == modelChangeKey(change) {
				continue
			}
			old := slices.IndexFunc(previous.Review.Settings.ModelLoads, func(load modelconfig.ModelLoad) bool { return load.ProfileKey == key })
			next := slices.IndexFunc(review.Settings.ModelLoads, func(load modelconfig.ModelLoad) bool { return load.ProfileKey == key })
			if old < 0 || next < 0 || previous.Review.Settings.ModelLoads[old].Resolved == nil || review.Settings.ModelLoads[next].Resolved == nil {
				return review, nil, modelSettingsConflict("unfinished conversion lost a resolved sibling")
			}
			accepted := previous.Review.Settings.ModelLoads[old]
			resolved := *accepted.Resolved
			// Whole-engine digests change with the selected edit. The fresh
			// worker review owns those fences; sibling allocation and artifacts
			// must still equal the accepted facts before carrying its identity.
			resolved.SourceInputDigest = review.Settings.ModelLoads[next].Resolved.SourceInputDigest
			resolved.PresetDigest = review.Settings.ModelLoads[next].Resolved.PresetDigest
			accepted.Resolved = &resolved
			if !reflect.DeepEqual(accepted, review.Settings.ModelLoads[next]) {
				return review, nil, modelSettingsConflict("unfinished conversion sibling changed; preview its saved configuration explicitly")
			}
			identity, ok := previous.Review.Identities[key]
			if !ok {
				return review, nil, modelSettingsConflict("unfinished conversion lost an accepted sibling identity")
			}
			review.Identities[key] = identity
		}
	} else if len(before.ModelLoads) == 0 {
		for _, load := range review.Settings.ModelLoads {
			source, converted := modelConversionInput(load)
			if !converted {
				continue
			}
			if digest, err := hex.DecodeString(source); err != nil || len(digest) != sha256.Size {
				return review, nil, fmt.Errorf("converted profile %q has no accepted external source", load.ProfileKey)
			}
			keys = append(keys, load.ProfileKey)
		}
	}
	return review, keys, validateConversionProfiles(review, keys)
}

func (b *Broker) mergeModelPolicyFromNative(stored nodepolicy.Policy, engine string, change modelconfig.ModelChange, caller string, review settings.Preview, typedRemoval bool, conversion []string) (nodepolicy.Policy, error) {
	if err := validateConversionProfiles(review, conversion); err != nil {
		return stored, err
	}
	policy := stored
	if len(conversion) != 0 {
		policy.Profiles = slices.Clone(stored.Profiles)
		for _, key := range conversion {
			if profile, exists := policy.ProfileByKey(key); exists {
				if profile.Engine != engine || profile.Model != key {
					return stored, &invalidParamsError{err: fmt.Errorf("A profile keeps its model; create a new profile.")}
				}
				continue
			}
			ref := nodepolicy.ModelRef{Engine: engine, Model: key}
			policy.Profiles = append(policy.Profiles, nodepolicy.Profile{Key: key, DisplayName: modelProfileDisplayName(ref), Engine: engine, Model: key})
		}
	}
	return b.mergeModelPolicy(policy, engine, change, caller, review, typedRemoval)
}

// Review and replay use this accepted preview, never current telemetry
// or a new native resolution, so the after-fingerprint is reproducible.
// Opaque request-option numbers retain precision; qualification is preserved.
func (b *Broker) mergeModelPolicy(stored nodepolicy.Policy, engine string, change modelconfig.ModelChange, caller string, review settings.Preview, typedRemoval bool) (nodepolicy.Policy, error) {
	policy, err := patchModelPolicy(stored, engine, change, review.Settings.ModelLoads, typedRemoval)
	if err != nil {
		return policy, err
	}
	policy, err = deriveProfileContext(policy, map[string][]modelconfig.ModelLoad{engine: review.Settings.ModelLoads})
	if err != nil {
		return policy, err
	}
	for i := range policy.Profiles {
		profile := &policy.Profiles[i]
		if profile.Engine != engine {
			continue
		}
		if identity, ok := review.Identities[profile.Key]; ok {
			identity.Files = slices.Clone(identity.Files)
			profile.ModelIdentity = &identity
		} else if !change.Remove && change.Load != nil && profile.Key == modelChangeKey(change) {
			return policy, fmt.Errorf("native preview did not return the profile's model identity")
		}
	}
	// Use the policy owner's normalization as well as validation so the
	// recorded after-fingerprint survives its normal disk parse on restart.
	canonical, err := nodepolicy.Parse(settingsJSON(policy))
	if err != nil {
		return policy, &invalidParamsError{err: err}
	}
	policy = canonical
	if caller != "" {
		merged, err := nodepolicy.MergePeerSet(stored, withoutRevision(settingsJSON(policy.PeerProjection())))
		if err != nil {
			return policy, err
		}
		policy = merged
	}
	return policy, nil
}

func modelSettingsRequest(p settings.Request, caller string) settings.Request {
	p = p.Clone()
	p.NodeID = ""
	p.PreserveCORS = caller != ""
	if p.ModelChange != nil && p.ModelChange.Load != nil {
		slices.Sort(p.ModelChange.Load.Clear)
	}
	return p
}

// Calls are pinned to one worker and never run under the journal/policy mutex.
func modelSettingsCall(ctx context.Context, worker *rpcWorker, method string, params, target any) error {
	if worker == nil {
		return fmt.Errorf("engine manager unavailable")
	}
	raw, rpcErr, err := worker.CallNoTimeout(ctx, method, settingsJSON(params))
	if err != nil {
		return err
	}
	if rpcErr != nil {
		return &settingsWorkerError{code: rpcErr.Code, message: rpcErr.Message}
	}
	if target != nil {
		return json.Unmarshal(raw, target)
	}
	return nil
}

func (b *Broker) modelWorkerCurrent(worker *rpcWorker, source *engineManagerSource) bool {
	if worker == nil || source == nil || b.getEngineMgr() != worker || !b.engineMgrPairCurrent(worker, source) {
		return false
	}
	select {
	case <-worker.Done():
		return false
	default:
		return true
	}
}

// Preview reads native facts outside owner locks, then validates the captured
// journal/policy/source again. Apply validates the retained finite settings;
// it never repeats the portable patch or substitutes a new resolution.
type modelSettingsReviewSource struct {
	Launch              settings.LaunchState
	Worker              *rpcWorker
	Source              *engineManagerSource
	Memory              *modelconfig.TargetMemory
	Policy              string
	NativeDigest        string
	ReviewedModelChange *modelconfig.ModelChange
	ReviewedAt          time.Time
}

func (source modelSettingsReviewSource) native(change modelconfig.ModelChange) bool {
	return modelChangeNative(change, source.Launch.ModelLoads) || source.NativeDigest != "" || source.ReviewedModelChange != nil
}

func (source modelSettingsReviewSource) reviewedRemoval(change modelconfig.ModelChange, publicDigest string, native bool) (*modelconfig.ModelChange, error) {
	if native {
		if value, err := hex.DecodeString(source.NativeDigest); err != nil || len(value) != sha256.Size || source.NativeDigest == publicDigest {
			return nil, modelSettingsConflict("original native preview digest is missing or invalid; explicitly Recalculate")
		}
	}
	reviewed, err := reviewedModelRemoval(change, source.ReviewedModelChange, publicDigest, native)
	if err == nil && reviewed != nil && reviewed.PreviewDigest != source.NativeDigest {
		return nil, modelSettingsConflict("retained native removal evidence changed; explicitly Recalculate")
	}
	return reviewed, err
}

func (b *Broker) previewModelSettings(ctx context.Context, input settings.Request, caller string) (settings.Preview, error) {
	var source modelSettingsReviewSource
	preview, err := b.reviewModelSettings(ctx, input, caller, &source)
	if err == nil && len(preview.Errors) == 0 && preview.Conflict == nil {
		b.engineConfigMu.Lock()
		err = b.retainModelReviewLocked(modelSettingsRequest(input, caller), caller, preview, source)
		b.engineConfigMu.Unlock()
	}
	return preview, err
}

func (b *Broker) reviewModelSettings(ctx context.Context, input settings.Request, caller string, capture *modelSettingsReviewSource) (settings.Preview, error) {
	ctx, cancel := context.WithTimeout(ctx, settings.CallBudget)
	defer cancel()
	p := modelSettingsRequest(input, caller)
	// No journal initialization or worker I/O before the target caller check.
	b.engineConfigMu.Lock()
	err := b.authorizeSettingsCallerLocked(caller)
	b.engineConfigMu.Unlock()
	if err != nil {
		return settings.Preview{}, err
	}
	worker := b.getEngineMgr()
	b.workersMu.Lock()
	source := b.engineMgrSource
	b.workersMu.Unlock()
	var launch settings.LaunchState
	if err := modelSettingsCall(ctx, worker, "engine:get-launch", map[string]string{"engine": p.Engine}, &launch); err != nil {
		return settings.Preview{}, err
	}
	b.engineConfigMu.Lock()
	base, policy, err := b.modelSettingsBaselineLocked(p, caller, launch)
	b.engineConfigMu.Unlock()
	if err != nil {
		return settings.Preview{}, err
	}
	targetLoads := base.Snapshot.Settings.ModelLoads
	if base.Model != nil && base.Model.Change.Remove && p.ModelChange.Remove && p.ModelChange.SupersedesRequestID == base.Snapshot.RequestID {
		// The accepted native removal may already have landed. Its original
		// saved load still owns the selected artifact identity during cleanup.
		targetLoads = base.Model.NativeBefore.ModelLoads
	}
	if err := validateModelPolicyChange(policy, p.Engine, *p.ModelChange, targetLoads); err != nil {
		return settings.Preview{}, err
	}
	if caller != "" {
		candidate, err := patchModelPolicy(policy, p.Engine, *p.ModelChange, base.Snapshot.Settings.ModelLoads, p.ModelChange.Remove && modelChangeNative(*p.ModelChange, targetLoads))
		if err != nil {
			return settings.Preview{}, err
		}
		if _, err := nodepolicy.MergePeerSet(policy, withoutRevision(settingsJSON(candidate.PeerProjection()))); err != nil {
			return settings.Preview{}, err
		}
	}
	preview := settings.Preview{Settings: base.Snapshot.Settings.Clone(), Revision: base.Snapshot.Revision, PolicyRevision: policy.Revision}
	var nativeRemoval *modelconfig.ModelChange
	nativeDigest := ""
	if modelChangeNative(*p.ModelChange, targetLoads) {
		request := p.Clone()
		change := request.ModelChange
		change.Policy, change.PreviewDigest = nil, ""
		// During local supersession, the native store may differ from desired.
		request.Settings = settings.ConfigFromLaunch(launch, p.Settings.ProxyPort)
		// A failed removal may already have persisted native cleanup. Only an
		// exact same-profile journal supersession may review that landed state
		// without asking the manager to remove an absent load a second time.
		removed := base.Model != nil && base.Model.Change.Remove && p.ModelChange.Remove &&
			p.ModelChange.SupersedesRequestID == base.Snapshot.RequestID &&
			modelChangeKey(base.Model.Change) == modelChangeKey(*p.ModelChange) &&
			base.Model.Change.ModelID == p.ModelChange.ModelID &&
			request.Settings.Equal(base.Snapshot.Settings) &&
			!slices.ContainsFunc(request.Settings.ModelLoads, func(load modelconfig.ModelLoad) bool { return load.ProfileKey == modelChangeKey(*p.ModelChange) })
		if removed {
			request.ModelChange = nil
			nativeRemoval, err = base.Model.reviewedRemoval()
			if err != nil {
				return settings.Preview{}, err
			}
		}
		envelope := modelWorkerRequest(b, request, caller, true)
		envelope.ReviewedModelChange = nativeRemoval
		started, clockErr := noderec.MemoryBootClock()
		if err := modelSettingsCall(ctx, worker, "engine:preview-launch", envelope, &preview); err != nil {
			return preview, err
		}
		elapsed := modelWorkerElapsed(started, clockErr)
		if envelope.Memory != nil && !modelReviewMemoryCurrent(envelope.Memory, b.modelTargetMemory(), preview.Settings.ModelLoads, modelChangeKey(*p.ModelChange)) {
			if preview.Auto != nil && preview.Auto.Resolved != nil {
				return settings.Preview{}, modelSettingsConflict("target evidence changed during preview; explicitly Recalculate")
			}
			elapsed = -1
		}
		modelAgeAdvice(preview.Estimate, elapsed)
		if preview.Auto != nil {
			modelAgeAdvice(&preview.Auto.Estimate, elapsed)
		}
		if len(preview.Errors) > 0 || preview.Conflict != nil {
			return preview, nil
		}
		if removed {
			if !preview.Settings.Equal(request.Settings) {
				return settings.Preview{}, modelSettingsConflict("removed profile native state changed; preview again")
			}
			// Frozen validation does not mint a new removal proof.
			preview.PreviewDigest = nativeRemoval.PreviewDigest
		}
		if len(preview.PreviewDigest) != sha256.Size*2 {
			return settings.Preview{}, fmt.Errorf("native preview did not bind its source identity")
		}
		nativeDigest = preview.PreviewDigest
		if p.ModelChange.Remove && nativeRemoval == nil {
			nativeRemoval = &modelconfig.ModelChange{ProfileKey: p.ModelChange.ProfileKey, ModelID: p.ModelChange.ModelID, ExpectedPolicyRevision: p.ModelChange.ExpectedPolicyRevision, Remove: true, PreviewDigest: nativeDigest}
			if err := settings.ValidateReviewedModelChange(nativeRemoval); err != nil {
				return settings.Preview{}, err
			}
		}
		if _, err := preview.Settings.Canonical(); err != nil {
			return settings.Preview{}, err
		}
		if preview.Settings.ServerPort != p.Settings.ServerPort || preview.Settings.ProxyPort != p.Settings.ProxyPort {
			return settings.Preview{}, fmt.Errorf("model preview changed an unrelated port")
		}
		if len(preview.Settings.ModelLoads) == 0 && !p.ModelChange.Remove {
			return settings.Preview{}, fmt.Errorf("model preview did not resolve native load settings")
		}
		selected := slices.IndexFunc(preview.Settings.ModelLoads, func(load modelconfig.ModelLoad) bool { return load.ProfileKey == modelChangeKey(*p.ModelChange) })
		if p.ModelChange.Remove {
			if selected >= 0 {
				return settings.Preview{}, fmt.Errorf("model preview retained the removed profile load")
			}
		} else if selected < 0 || preview.Settings.ModelLoads[selected].ModelID != p.ModelChange.ModelID {
			return settings.Preview{}, fmt.Errorf("model preview did not resolve the selected profile load")
		}
		for _, load := range preview.Settings.ModelLoads {
			if load.Resolved == nil {
				return settings.Preview{}, fmt.Errorf("model preview returned an unresolved load")
			}
		}
	}
	preview, conversion, err := modelConversionReview(base.Model, *p.ModelChange, preview, settings.ConfigFromLaunch(launch, preview.Settings.ProxyPort))
	if err != nil {
		return settings.Preview{}, err
	}
	nextPolicy, err := b.mergeModelPolicyFromNative(policy, p.Engine, *p.ModelChange, caller, preview, nativeRemoval != nil, conversion)
	if err != nil {
		return settings.Preview{}, err
	}
	preview.Revision, preview.PolicyRevision = base.Snapshot.Revision, policy.Revision
	preview.Rebind = preview.Settings.ProxyPort != base.Snapshot.Settings.ProxyPort
	preview.Restart = launch.Running && !preview.Settings.Equal(settings.ConfigFromLaunch(launch, p.Settings.ProxyPort))
	// Supersession reviews the unresolved running intent as well as current
	// observation: a replacement manager may have persisted settings but stopped.
	if p.ModelChange.SupersedesRequestID != "" && modelChangeNative(*p.ModelChange, targetLoads) && base.Resume {
		preview.Restart = true
	}
	preview.Impact = b.modelSettingsImpact(p.Engine, launch, preview.Settings, preview.Restart)
	if modelPolicyFingerprint(policy) != modelPolicyFingerprint(nextPolicy) {
		preview.Diff = append(preview.Diff, modelconfig.FieldDiff{Field: "policy", Before: settingsJSON(policy.PeerProjection()), After: settingsJSON(nextPolicy.PeerProjection())})
	}
	// Re-read source state after worker inspection, before taking owner locks.
	var after settings.LaunchState
	if err := modelSettingsCall(ctx, worker, "engine:get-launch", map[string]string{"engine": p.Engine}, &after); err != nil {
		return settings.Preview{}, err
	}
	if !reflect.DeepEqual(launch, after) || !b.modelWorkerCurrent(worker, source) {
		return settings.Preview{}, modelSettingsConflict("native source changed during preview; preview again")
	}
	b.engineConfigMu.Lock()
	checked, latest, err := b.modelSettingsBaselineLocked(p, caller, after)
	if err == nil && (checked.Snapshot.Revision != base.Snapshot.Revision || latest.Revision != policy.Revision) {
		err = modelSettingsConflict("model settings changed during preview")
	}
	b.engineConfigMu.Unlock()
	if err != nil {
		return settings.Preview{}, err
	}
	request := p.Clone()
	request.RequestID, request.ModelChange.PreviewDigest = "", ""
	preview.PreviewDigest = ""
	// Estimates are advisory and may change or disappear between preview and
	// apply. They must not change the authority bound by the review digest.
	review := preview.Clone()
	review.Estimate = nil
	preview.PreviewDigest = modelSettingsHash(struct {
		Target                 string
		Caller                 string
		Request                settings.Request
		Policy, Native, Source string
		Preview                settings.Preview
	}{b.nodeID, caller, request, modelPolicyFingerprint(policy), nativeDigest, source.settingsEpoch, review})
	if err := modelconfig.CheckPayloadSize(preview); err != nil {
		preview.Estimate = nil
		if err := modelconfig.CheckPayloadSize(preview); err != nil {
			return settings.Preview{}, &invalidParamsError{err: err}
		}
	}
	if capture != nil {
		*capture = modelSettingsReviewSource{Launch: launch, Worker: worker, Source: source, Policy: modelPolicyFingerprint(policy), NativeDigest: nativeDigest, ReviewedModelChange: nativeRemoval}
	}
	return preview, nil
}

// Caller holds engineConfigMu; this only inspects local authority, never a worker.
func (b *Broker) modelSettingsBaselineLocked(p settings.Request, caller string, launch settings.LaunchState) (*engineSettingsRecord, nodepolicy.Policy, error) {
	fail := func(err error) (*engineSettingsRecord, nodepolicy.Policy, error) {
		return nil, nodepolicy.Policy{}, err
	}
	if err := b.authorizeSettingsCallerLocked(caller); err != nil {
		return fail(err)
	}
	if err := b.loadEngineSettingsLocked(); err != nil {
		return fail(err)
	}
	if launch.Engine != p.Engine || !launch.Editable {
		return fail(modelSettingsConflict("engine is not editable"))
	}
	proxy := b.settingsProxy(p.Engine)
	if proxy == nil {
		return fail(fmt.Errorf("proxy unavailable"))
	}
	ready, port := proxy.Status(p.Engine)
	if !ready {
		return fail(fmt.Errorf("proxy unavailable"))
	}
	native := settings.ConfigFromLaunch(launch, port)
	record := b.engineSettings[p.Engine].clone()
	if record == nil {
		record = &engineSettingsRecord{Snapshot: settings.Snapshot{NodeID: b.nodeID, Engine: p.Engine, Epoch: b.engineSettingsEpoch, Revision: 1, AppliedRevision: 1, Phase: "idle", Settings: native.Clone()}, Receipts: map[string]settingsReceipt{}}
	} else if !b.modelSettingsPendingLocked(p.Engine) && record.Snapshot.Phase != "applying" && record.Snapshot.Phase != "failed" && !record.Snapshot.Settings.Equal(native) {
		record.Snapshot.Settings = native.Clone()
		record.Snapshot.Revision++
		record.Snapshot.AppliedRevision = record.Snapshot.Revision
	}
	if record.Snapshot.Phase == "applying" && !modelSettingsNeedsReconciliation(record) {
		return fail(modelSettingsConflict("settings operation is still applying"))
	}
	if b.modelSettingsPendingLocked(p.Engine) {
		if caller != "" || p.ModelChange.SupersedesRequestID != record.Snapshot.RequestID || p.RequestID == record.Snapshot.RequestID {
			return fail(modelSettingsConflict("local fresh preview with supersedesRequestId required for " + record.Snapshot.RequestID))
		}
		if record.Model.native() && !modelChangeNative(*p.ModelChange, record.Model.NativeBefore.ModelLoads) {
			return fail(modelSettingsConflict("native reconciliation requires a fresh load preview"))
		}
		if modelChangeKey(record.Model.Change) == modelChangeKey(*p.ModelChange) && record.Model.Change.ModelID != p.ModelChange.ModelID {
			return fail(&invalidParamsError{err: fmt.Errorf("A profile keeps its model; create a new profile.")})
		}
	} else if p.ModelChange.SupersedesRequestID != "" {
		return fail(modelSettingsConflict("superseded operation is no longer pending"))
	}
	for engine, pending := range b.engineSettings {
		if engine != p.Engine && b.modelSettingsPendingLocked(engine) {
			return fail(modelSettingsConflict("policy is reserved by " + pending.Snapshot.RequestID))
		}
	}
	if p.ExpectedRevision != record.Snapshot.Revision {
		return fail(modelSettingsConflict("settings revision changed; reload and preview again"))
	}
	if err := p.ValidateModelChange(record.Snapshot.Settings, false); err != nil {
		return fail(&invalidParamsError{err: err})
	}
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	defer r.mu.Unlock()
	if err := checkExpectedRevision(r.current, p.ModelChange.ExpectedPolicyRevision); err != nil {
		return fail(err)
	}
	policy := r.current
	record.Snapshot.Running, record.Snapshot.Adopted = launch.Running, launch.Adopted
	record.Snapshot.WorkingSetLimitStatus = launch.WorkingSetLimitStatus
	record.Snapshot.EffectiveServerPort, record.Snapshot.EffectiveProxyPort = launch.EffectivePort, port
	record.Snapshot.Editable, record.Snapshot.Format = true, launch.Format
	record.Snapshot.EffectiveSettings = &native
	return record, policy, nil
}

func (b *Broker) modelSettingsImpact(engine string, launch settings.LaunchState, desired settings.Config, restart bool) *modelconfig.Impact {
	models := map[string]bool{}
	for _, load := range launch.ModelLoads {
		models[load.ProfileKey] = true
	}
	for _, load := range desired.ModelLoads {
		models[load.ProfileKey] = true
	}
	for _, config := range []settings.Config{settings.ConfigFromLaunch(launch, 0), desired} {
		if config.Serving != nil {
			for _, model := range config.Serving.Models {
				models[model.ModelID] = true
			}
		}
	}
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	_, known := r.loaded[engine]
	known = known && r.models.loadedSource == r.engineManagerSource && r.models.loadedProxy == b.policyProxy()
	if known {
		for _, model := range r.loaded[engine] {
			models[model] = true
		}
	}
	paused := r.managementBlockedLocked()
	r.mu.Unlock()
	scope := "none"
	if restart {
		scope = "whole-router"
	}
	interrupt, off := restart, !launch.Running && !restart || paused
	impact := &modelconfig.Impact{RestartScope: scope, MayInterruptActive: &interrupt, StaysOff: &off, ActiveWorkPolicy: "none"}
	if restart {
		seconds := int32(60)
		impact.DrainTimeoutSeconds, impact.ActiveWorkPolicy = &seconds, "drain-then-restart"
	}
	for model := range models {
		impact.AffectedModels = append(impact.AffectedModels, model)
	}
	slices.Sort(impact.AffectedModels)
	if !known {
		impact.Unknowns = []string{"Loaded router models have no current readback."}
	}
	return impact
}

func (b *Broker) applyModelSettings(ctx context.Context, input settings.Request, caller string) (settingsReceipt, error) {
	p := modelSettingsRequest(input, caller)
	if p.RequestID == "" || p.ModelChange.PreviewDigest == "" {
		return settingsReceipt{}, &invalidParamsError{err: fmt.Errorf("requestId and reviewed previewDigest are required")}
	}
	ctx, cancel := context.WithTimeout(ctx, settings.CallBudget)
	defer cancel()
	b.settingsApplyMu.Lock()
	defer b.settingsApplyMu.Unlock()
	hash := modelSettingsHash(struct {
		Caller  string
		Request settings.Request
	}{caller, p})
	// Receipt lookup precedes CAS, but never the fresh pin/grant/Pause checks.
	b.engineConfigMu.Lock()
	err := b.authorizeSettingsCallerLocked(caller)
	if err == nil {
		err = b.authorizeRemoteManagement(caller, "settings", p.RequestID)
	}
	if err == nil {
		err = b.loadEngineSettingsLocked()
	}
	if err == nil {
		if record := b.engineSettings[p.Engine]; record != nil {
			if receipt, exists := record.Receipts[p.RequestID]; exists {
				b.engineConfigMu.Unlock()
				if receipt.Hash != hash {
					return settingsReceipt{}, &invalidParamsError{err: fmt.Errorf("request identifier was already used for different content or caller")}
				}
				return receipt, nil
			}
		}
	}
	b.engineConfigMu.Unlock()
	if err != nil {
		return settingsReceipt{}, err
	}
	var source modelSettingsReviewSource
	preview, source, err := b.retainedModelReview(ctx, p, caller)
	if err != nil {
		return settingsReceipt{}, err
	}
	if len(preview.Errors) != 0 || preview.Conflict != nil {
		return settingsReceipt{}, &invalidParamsError{err: fmt.Errorf("resolve model preview errors before applying")}
	}
	if preview.PreviewDigest != p.ModelChange.PreviewDigest {
		return settingsReceipt{}, modelSettingsConflict("review changed; obtain a fresh model settings preview")
	}
	worker := source.Worker
	var launch settings.LaunchState
	if err := modelSettingsCall(ctx, worker, "engine:get-launch", map[string]string{"engine": p.Engine}, &launch); err != nil {
		return settingsReceipt{}, err
	}
	if !reflect.DeepEqual(launch, source.Launch) {
		return settingsReceipt{}, modelSettingsConflict("native source changed before acceptance; preview again")
	}
	b.engineConfigMu.Lock()
	defer b.engineConfigMu.Unlock()
	record, policy, err := b.modelSettingsBaselineLocked(p, caller, launch)
	if err != nil {
		return settingsReceipt{}, err
	}
	if modelPolicyFingerprint(policy) != source.Policy {
		return settingsReceipt{}, modelSettingsConflict("policy contents changed since review; explicitly Recalculate")
	}
	if !b.modelWorkerCurrent(worker, source.Source) || ctx.Err() != nil {
		return settingsReceipt{}, modelSettingsConflict("engine source changed before acceptance")
	}
	preview, conversion, err := modelConversionReview(record.Model, *p.ModelChange, preview, settings.ConfigFromLaunch(source.Launch, preview.Settings.ProxyPort))
	if err != nil {
		return settingsReceipt{}, err
	}
	next, err := b.mergeModelPolicyFromNative(policy, p.Engine, *p.ModelChange, caller, preview, source.ReviewedModelChange != nil, conversion)
	if err != nil {
		return settingsReceipt{}, err
	}
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	// Last admission decision shares the policy/availability ordering with the
	// durable journal write, with no worker I/O inside this section.
	if err := b.authorizeSettingsCallerLocked(caller); err != nil {
		r.mu.Unlock()
		return settingsReceipt{}, err
	}
	if err := b.authorizeRemoteManagementLocked(r, caller, "settings", p.RequestID); err != nil {
		r.mu.Unlock()
		return settingsReceipt{}, err
	}
	if err := checkExpectedRevision(r.current, p.ModelChange.ExpectedPolicyRevision); err != nil {
		r.mu.Unlock()
		return settingsReceipt{}, err
	}
	if time.Since(source.ReviewedAt) >= modelReviewTTL {
		r.mu.Unlock()
		return settingsReceipt{}, modelSettingsConflict("review expired before acceptance; explicitly Recalculate")
	}
	if !b.modelReviewRetainedLocked(p, caller, source.ReviewedAt, source) {
		r.mu.Unlock()
		return settingsReceipt{}, modelSettingsConflict("review superseded or evicted during validation; explicitly Recalculate")
	}
	if source.Memory != nil && !modelReviewMemoryCurrent(source.Memory, b.modelTargetMemory(), preview.Settings.ModelLoads, modelChangeKey(*p.ModelChange)) {
		r.mu.Unlock()
		return settingsReceipt{}, modelSettingsConflict("target memory source or age changed before acceptance; explicitly Recalculate")
	}
	previous := b.engineSettings[p.Engine]
	if previous != nil && previous.Model != nil && previous.Model.Stage != "completed" {
		// Retire the previous receipt only in the same write that accepts the
		// replacement; native readback is carried as the new operation's baseline.
		receipt := record.Receipts[previous.Snapshot.RequestID]
		receipt.Phase = "superseded"
		record.Receipts[previous.Snapshot.RequestID] = receipt
	}
	record.Previous = record.Snapshot.Settings.Clone()
	record.Resume = preview.Restart && !r.managementBlockedLocked()
	record.Explicit = true
	var reviewed modelconfig.ModelChange
	if source.ReviewedModelChange != nil {
		reviewed = source.ReviewedModelChange.Clone()
	}
	record.Model = &modelSettingsOperation{Caller: caller, Change: p.ModelChange.Clone(), Review: preview.Clone(), ReviewedModelChange: reviewed, ConversionProfiles: conversion, NativeBefore: settings.ConfigFromLaunch(launch, preview.Settings.ProxyPort), Before: modelPolicyFingerprint(policy), After: modelPolicyFingerprint(next), Implications: modelPolicyImplications(policy, p.Engine, modelChangeKey(*p.ModelChange)), Stage: "accepted"}
	record.Snapshot.Settings = preview.Settings.Clone()
	record.Snapshot.Revision++
	if record.Snapshot.Revision > settings.MaxSafeCounter {
		r.mu.Unlock()
		return settingsReceipt{}, fmt.Errorf("settings revision is exhausted")
	}
	record.Snapshot.RequestID, record.Snapshot.Phase = p.RequestID, "applying"
	record.Snapshot.Editable, record.Snapshot.Reason = false, "Model settings are being applied."
	record.Snapshot.CoupledStage, record.Snapshot.PolicyRevision = "accepted", policy.Revision
	record.Snapshot.Error, record.Snapshot.UnresolvedError = "", ""
	record.Receipts[p.RequestID] = settingsReceipt{Hash: hash, Revision: record.Snapshot.Revision, Phase: "applying"}
	b.engineSettings[p.Engine] = record
	if err := b.saveEngineSettingsLocked(); err != nil {
		if errors.Is(err, durablefile.ErrReplaced) {
			acceptedHash := modelSettingsHash(b.engineSettings)
			// Update only the exact replacement that just landed. If this
			// second write also fails, loading "accepted" remains non-replayable.
			markModelAcceptanceUncertain(record)
			path, pathErr := b.engineSettingsPath()
			var data []byte
			if pathErr == nil {
				data, pathErr = os.ReadFile(path)
			}
			var landed map[string]*engineSettingsRecord
			if pathErr == nil {
				pathErr = json.Unmarshal(data, &landed)
			}
			if pathErr == nil && modelSettingsHash(landed) != acceptedHash {
				pathErr = fmt.Errorf("acceptance readback differs from the landed journal")
			}
			if pathErr != nil {
				b.engineSettingsError = fmt.Errorf("acceptance reconciliation cannot replace changed or unreadable journal: %w", pathErr)
			} else if saveErr := b.saveEngineSettingsLocked(); saveErr != nil {
				record.Snapshot.UnresolvedError += "; uncertainty record durability could not be confirmed"
			}
			if record.Model.native() {
				r.drains[p.Engine] = true
				delete(r.settingsIdentity, p.Engine)
			}
		} else if previous == nil {
			delete(b.engineSettings, p.Engine)
		} else {
			b.engineSettings[p.Engine] = previous
		}
		r.mu.Unlock()
		return settingsReceipt{}, fmt.Errorf("model settings acceptance could not be confirmed; no native or policy effects were made: %w", err)
	}
	b.releaseAcceptedModelReviewLocked(p, caller)
	if record.Model.native() {
		r.drains[p.Engine] = true
		delete(r.settingsIdentity, p.Engine)
	}
	r.mu.Unlock()
	b.signalModelConfig()
	b.publishSettingsLocked()
	operationCtx, operationCancel := context.WithTimeout(context.Background(), settings.OperationBudget)
	defer operationCancel()
	_ = b.runModelSettingsOperationLocked(operationCtx, p.Engine, record)
	return record.Receipts[p.RequestID], nil
}

// Caller holds settingsApplyMu and engineConfigMu. Save before every effect;
// all native/proxy calls release engineConfigMu. Failed writes never release
// the journal-derived fence, including a terminal-result write failure.
func (b *Broker) saveModelStageLocked(record *engineSettingsRecord, stage string) error {
	previous := record.Model.Stage
	record.Model.Stage, record.Snapshot.CoupledStage = stage, stage
	if err := b.saveEngineSettingsLocked(); err != nil {
		if !errors.Is(err, durablefile.ErrReplaced) {
			record.Model.Stage, record.Snapshot.CoupledStage = previous, previous
		}
		return err
	}
	return nil
}

func (b *Broker) failModelSettingsLocked(record *engineSettingsRecord, err error) error {
	if record.Model.Stage == "completed" {
		record.Model.Stage, record.Snapshot.CoupledStage = "completion-uncertain", "completion-uncertain"
	}
	// The operation itself remains nonterminal. Existing mutation owners
	// already fence applying settings rows; the failed attempt is the
	// receipt, not permission to discard that reservation.
	record.Snapshot.Phase = "applying"
	record.Snapshot.Error, record.Snapshot.UnresolvedError = err.Error(), err.Error()
	record.Snapshot.Editable = false
	record.Snapshot.Reason = "Local fresh preview and supersession are required."
	receipt := record.Receipts[record.Snapshot.RequestID]
	receipt.Phase = "failed"
	record.Receipts[record.Snapshot.RequestID] = receipt
	if saveErr := b.saveEngineSettingsLocked(); saveErr != nil {
		record.Snapshot.UnresolvedError += "; reconciliation result could not be saved"
	}
	b.publishSettingsLocked()
	return err
}

func (b *Broker) checkModelPolicyLocked(record *engineSettingsRecord) error {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	defer r.mu.Unlock()
	op := record.Model
	fingerprint := modelPolicyFingerprint(r.current)
	if fingerprint != op.Before && fingerprint != op.After {
		return modelSettingsConflict("policy content conflicts with the accepted operation; local fresh preview required")
	}
	return nil
}

// The private same-operation writer is reachable only from a journal record.
// It preserves current availability/provenance and refuses any third content.
func (b *Broker) commitModelPolicyLocked(record *engineSettingsRecord) error {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	defer r.mu.Unlock()
	op := record.Model
	fingerprint := modelPolicyFingerprint(r.current)
	if fingerprint == op.After {
		op.PolicyRevision = r.current.Revision
		record.Snapshot.PolicyRevision = op.PolicyRevision
		return nil
	}
	if fingerprint != op.Before {
		return modelSettingsConflict("a third policy state cannot be overwritten")
	}
	policy, err := b.mergeModelPolicyFromNative(r.current, record.Snapshot.Engine, op.Change, op.Caller, op.Review, op.ReviewedModelChange.Remove, op.ConversionProfiles)
	if err != nil {
		return err
	}
	if modelPolicyFingerprint(policy) != op.After {
		return modelSettingsConflict("accepted policy patch no longer produces the reviewed content")
	}
	policy, err = r.publishLocked(policy)
	if err != nil && !errors.Is(err, durablefile.ErrReplaced) {
		return &policySaveError{err: err}
	}
	r.current, r.recoveryRequired, r.activationErr = policy, false, ""
	op.PolicyRevision = policy.Revision
	record.Snapshot.PolicyRevision = op.PolicyRevision
	if err != nil {
		return &policySaveError{err: err, revision: policy.Revision}
	}
	return nil
}

func observeModelSettingsLaunch(record *engineSettingsRecord, launch settings.LaunchState) {
	if launch.Engine != record.Snapshot.Engine {
		return
	}
	actual := settings.ConfigFromLaunch(launch, record.Snapshot.EffectiveProxyPort)
	record.Snapshot.EffectiveSettings = &actual
	record.Snapshot.Serving = launch.ServingReadback.Clone()
	record.Snapshot.Running, record.Snapshot.Adopted = launch.Running, launch.Adopted
	record.Snapshot.WorkingSetLimitStatus = launch.WorkingSetLimitStatus
	record.Snapshot.EffectiveServerPort = launch.EffectivePort
}

func (b *Broker) runModelSettingsOperationLocked(ctx context.Context, engine string, record *engineSettingsRecord) error {
	fail := func(err error) error { return b.failModelSettingsLocked(record, err) }
	reviewed, err := record.Model.reviewedRemoval()
	if err != nil {
		return fail(err)
	}
	if err := b.checkModelPolicyLocked(record); err != nil {
		return fail(err)
	}
	op := record.Model
	worker := b.getEngineMgr()
	b.workersMu.Lock()
	source := b.engineMgrSource
	b.workersMu.Unlock()
	if worker == nil {
		return fail(fmt.Errorf("engine manager unavailable; operation remains withdrawn"))
	}
	if op.native() {
		// A running restart has a disclosed 60-second wait after an acknowledged
		// fence. A lost drain ACK never authorizes a native effect.
		b.engineConfigMu.Unlock()
		err := b.drainModelSettings(ctx, engine, record.Resume)
		b.engineConfigMu.Lock()
		if err != nil {
			return fail(err)
		}
		if err := b.checkModelPolicyLocked(record); err != nil {
			return fail(err)
		}
		if err := b.saveModelStageLocked(record, "drained"); err != nil {
			return fail(err)
		}
		var launch settings.LaunchState
		var validated settings.Preview
		b.engineConfigMu.Unlock()
		err = modelSettingsCall(ctx, worker, "engine:get-launch", map[string]string{"engine": engine}, &launch)
		if err == nil {
			// Full frozen settings revalidation checks bindings/build without
			// running the portable patch or selecting a fresh Auto result.
			err = modelSettingsCall(ctx, worker, "engine:preview-launch", settings.WorkerRequest[settings.Request]{Request: settings.Request{Engine: engine, Settings: record.Snapshot.Settings.Clone()}, ReviewedModelChange: reviewed}, &validated)
		}
		if err == nil {
			var rechecked settings.LaunchState
			err = modelSettingsCall(ctx, worker, "engine:get-launch", map[string]string{"engine": engine}, &rechecked)
			if err == nil && !settings.ConfigFromLaunch(launch, record.Snapshot.Settings.ProxyPort).Equal(settings.ConfigFromLaunch(rechecked, record.Snapshot.Settings.ProxyPort)) {
				err = modelSettingsConflict("native overrides changed during frozen validation")
			}
			if err == nil {
				launch = rechecked
			}
		}
		b.engineConfigMu.Lock()
		if err != nil {
			return fail(err)
		}
		if !b.modelWorkerCurrent(worker, source) || launch.Engine != engine {
			return fail(modelSettingsConflict("native source changed during reconciliation"))
		}
		observeModelSettingsLaunch(record, launch)
		if len(validated.Errors) != 0 || validated.Conflict != nil || !validated.Settings.Equal(record.Snapshot.Settings) {
			return fail(modelSettingsConflict("native bindings or build changed; local fresh preview required"))
		}
		actual := settings.ConfigFromLaunch(launch, record.Snapshot.Settings.ProxyPort)
		paused, _ := b.settingsPause()
		resumeNeeded := record.Resume && !launch.Running && !paused
		if !actual.Equal(record.Snapshot.Settings) || resumeNeeded {
			if !actual.Equal(record.Snapshot.Settings) && !actual.Equal(op.NativeBefore) {
				return fail(modelSettingsConflict("native overrides changed to a third state; no replacement was attempted"))
			}
			// A replay may converge an exact unchanged before-state. An exact
			// desired readback skips persistence, but saved resume must still
			// converge through ConfigureLaunch's current activation guards.
			op.Attempted = true
			if err := b.saveModelStageLocked(record, "native-applying"); err != nil {
				if !errors.Is(err, durablefile.ErrReplaced) {
					op.Attempted = false
				}
				return fail(err)
			}
			id := record.Snapshot.RequestID
			b.settingsRelayMu.Lock()
			if b.activeSettings == nil {
				b.activeSettings = map[string]activeSettingsOperation{}
			}
			b.activeSettings[id] = activeSettingsOperation{Engine: engine, Port: record.Snapshot.Settings.ProxyPort}
			b.settingsRelayMu.Unlock()
			paused, _ := b.settingsPause()
			configure := settings.Configure{Engine: engine, Settings: record.Snapshot.Settings.Clone(), OperationID: id, Resume: record.Resume && !paused, ReviewedModelChange: reviewed}
			b.engineConfigMu.Unlock()
			configureErr := modelSettingsCall(ctx, worker, "engine:configure-launch", configure, nil)
			// A response is not persistence evidence, and loss is not no effect.
			launch = settings.LaunchState{}
			err = modelSettingsCall(ctx, worker, "engine:get-launch", map[string]string{"engine": engine}, &launch)
			b.settingsRelayMu.Lock()
			delete(b.activeSettings, id)
			b.settingsRelayMu.Unlock()
			b.engineConfigMu.Lock()
			if err != nil {
				record.Snapshot.EffectiveSettings = nil
				return fail(fmt.Errorf("native outcome uncertain: %w", err))
			}
			if !b.modelWorkerCurrent(worker, source) {
				record.Snapshot.EffectiveSettings = nil
				return fail(modelSettingsConflict("native outcome came from a replaced engine manager"))
			}
			observeModelSettingsLaunch(record, launch)
			if !settings.ConfigFromLaunch(launch, record.Snapshot.Settings.ProxyPort).Equal(record.Snapshot.Settings) {
				if configureErr != nil {
					return fail(configureErr)
				}
				return fail(modelSettingsConflict("native response did not persist the exact reviewed settings"))
			}
		}
		if !b.modelWorkerCurrent(worker, source) {
			return fail(modelSettingsConflict("engine manager changed before native persistence could be confirmed"))
		}
		// Persist desired/applied/observed separately, even if runtime startup
		// failed after exact persistence. Readiness still needs live evidence.
		if err := b.saveModelStageLocked(record, "native-persisted"); err != nil {
			return fail(err)
		}
	}
	if !b.modelWorkerCurrent(worker, source) {
		record.Snapshot.EffectiveSettings = nil
		return fail(modelSettingsConflict("engine manager changed before policy publication"))
	}
	if err := b.commitModelPolicyLocked(record); err != nil {
		return fail(err)
	}
	record.Snapshot.PolicyRevision = op.PolicyRevision
	if err := b.saveModelStageLocked(record, "policy-persisted"); err != nil {
		return fail(err)
	}
	// The policy owner publishes/activates while the admission drain stays set.
	b.engineConfigMu.Unlock()
	b.publishNodeProjection()
	b.activatePolicyNow()
	b.notifyPolicyChanged()
	var final settings.LaunchState
	err = modelSettingsCall(ctx, worker, "engine:get-launch", map[string]string{"engine": engine}, &final)
	b.engineConfigMu.Lock()
	if err != nil {
		record.Snapshot.EffectiveSettings = nil
		return fail(err)
	}
	if b.modelWorkerCurrent(worker, source) {
		observeModelSettingsLaunch(record, final)
	}
	if !b.modelWorkerCurrent(worker, source) || final.Engine != engine || !settings.ConfigFromLaunch(final, record.Snapshot.Settings.ProxyPort).Equal(record.Snapshot.Settings) {
		return fail(modelSettingsConflict("current-source native readback no longer matches the accepted settings"))
	}
	if err := b.checkModelPolicyLocked(record); err != nil {
		return fail(err)
	}
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	paused := r.managementBlockedLocked()
	activation := r.activationLocked(b.policyProxy(), r.current.Revision)
	r.mu.Unlock()
	if activation.Activation != activationActive {
		record.Snapshot.Error, record.Snapshot.UnresolvedError = "Policy activation is pending; engine remains withdrawn.", "Policy activation is pending; engine remains withdrawn."
		if err := b.saveModelStageLocked(record, "activation-pending"); err != nil {
			return fail(err)
		}
		b.publishSettingsLocked()
		return fmt.Errorf("policy activation is pending")
	}
	if paused && final.Running {
		return fail(modelSettingsConflict("Pause is pending native shutdown; engine remains withdrawn"))
	}
	if record.Resume && !paused && !final.Running {
		return fail(fmt.Errorf("settings are persisted but the requested running engine is not ready"))
	}
	if final.ServingReadback != nil && (final.ServingReadback.Status == "mismatch" || final.ServingReadback.Status == "unknown") {
		return fail(modelSettingsConflict("native serving readback is unresolved"))
	}
	if !b.modelWorkerCurrent(worker, source) {
		return fail(modelSettingsConflict("native source changed before completion"))
	}
	prior := record.clone()
	// The final receipt is durable before drain release. "releasing" retains
	// the reservation through an interrupted or unacknowledged release.
	record.Model.Stage, record.Snapshot.CoupledStage = "releasing", "releasing"
	record.Snapshot.Phase, record.Snapshot.Error, record.Snapshot.UnresolvedError = "applying", "", ""
	record.Snapshot.AppliedRevision = record.Snapshot.Revision
	record.Snapshot.Running, record.Snapshot.Serving = final.Running, final.ServingReadback.Clone()
	record.Snapshot.Editable, record.Snapshot.Reason = final.Editable, final.Reason
	receipt := record.Receipts[record.Snapshot.RequestID]
	receipt.Phase = "succeeded"
	record.Receipts[record.Snapshot.RequestID] = receipt
	if err := b.saveEngineSettingsLocked(); err != nil {
		*record = *prior
		return fail(err)
	}
	if op.native() {
		b.engineConfigMu.Unlock()
		_, err = b.pushEngineDrain(ctx, engine, false)
		b.engineConfigMu.Lock()
		if err != nil {
			return fail(err)
		}
	}
	if !b.modelWorkerCurrent(worker, source) {
		return fail(modelSettingsConflict("native source changed during release"))
	}
	resume := record.Resume
	record.Resume = false
	record.Snapshot.Phase = "succeeded"
	if err := b.saveModelStageLocked(record, "completed"); err != nil {
		record.Resume = resume
		if op.native() {
			b.engineConfigMu.Unlock()
			_, _ = b.pushEngineDrain(ctx, engine, true)
			b.engineConfigMu.Lock()
		}
		return fail(err)
	}
	r.mu.Lock()
	if op.native() {
		delete(r.drains, engine)
	}
	r.mu.Unlock()
	b.publishSettingsLocked()
	b.signalModelConfig()
	// Periodic advertising will probe current runtime; a receipt never turns a
	// backend healthy or manufactures strict model identity.
	return nil
}

// inspectModelSettings is a read, never a settings snapshot initialization or
// apply. Authorization is fresh at the existing target caller boundary.
func (b *Broker) inspectModelSettings(ctx context.Context, p modelconfig.InspectRequest, caller string) (modelconfig.InspectResult, error) {
	if err := p.Validate(); err != nil {
		return modelconfig.InspectResult{}, &invalidParamsError{err: err}
	}
	ctx, cancel := context.WithTimeout(ctx, settings.CallBudget)
	defer cancel()
	lease, leaseErr := b.reserveBuildApplySettings()
	if lease != nil {
		defer lease.Release()
	}
	b.engineConfigMu.Lock()
	err := b.authorizeSettingsCallerLocked(caller)
	b.engineConfigMu.Unlock()
	if err != nil {
		return modelconfig.InspectResult{}, err
	}
	p.NodeID = ""
	var raw json.RawMessage
	envelope := modelWorkerRequest(b, p, caller, leaseErr == nil)
	started, clockErr := noderec.MemoryBootClock()
	if err := b.settingsWorkerCall(ctx, "engine:inspect-model", envelope, &raw); err != nil {
		return modelconfig.InspectResult{}, err
	}
	result, err := modelconfig.DecodeInspectResult(raw)
	if err != nil {
		return modelconfig.InspectResult{}, err
	}
	elapsed := modelWorkerElapsed(started, clockErr)
	if envelope.Memory != nil && !modelReviewMemoryCurrent(envelope.Memory, b.modelTargetMemory(), nil, "") {
		elapsed = -1
	}
	modelAgeAdvice(&result.Estimate, elapsed)
	if result.Engine != p.Engine || result.ModelID != p.ModelID {
		return modelconfig.InspectResult{}, fmt.Errorf("model inspection: worker reply identity mismatch")
	}
	result.NodeID = b.nodeID
	if err := result.Validate(); err != nil {
		return modelconfig.InspectResult{}, err
	}
	return result, nil
}

func (b *Broker) recordModelMemory(ctx context.Context, relay settings.Relay, caller string) (modelconfig.InspectResult, error) {
	var engine, model string
	var request any
	if relay.Method == "set-stated-memory" {
		p := relay.SetStatedMemory.Clone()
		if err := p.Validate(); err != nil {
			return modelconfig.InspectResult{}, &invalidParamsError{err: err}
		}
		p.NodeID = ""
		engine, model, request = p.Engine, p.ModelID, p
	} else {
		p := relay.ClearStatedMemory.Clone()
		if err := p.Validate(); err != nil {
			return modelconfig.InspectResult{}, &invalidParamsError{err: err}
		}
		p.NodeID = ""
		engine, model, request = p.Engine, p.ModelID, p
	}
	ctx, cancel := context.WithTimeout(ctx, settings.CallBudget)
	defer cancel()
	b.settingsApplyMu.Lock()
	defer b.settingsApplyMu.Unlock()
	b.engineConfigMu.Lock()
	defer b.engineConfigMu.Unlock()
	if err := b.authorizeSettingsCallerLocked(caller); err != nil {
		return modelconfig.InspectResult{}, err
	}
	if err := b.authorizeRemoteManagement(caller, "settings", ""); err != nil {
		return modelconfig.InspectResult{}, err
	}
	worker := b.getEngineMgr()
	// The worker rechecks its current Pause projection
	// for this private remote marker, without status detection or native helpers.
	envelope := settings.WorkerRequest[any]{Request: request, Memory: b.modelTargetMemory(), NativeInitAllowed: false, RemoteMemoryRecord: caller != ""}
	var raw json.RawMessage
	if err := modelSettingsCall(ctx, worker, "engine:"+relay.Method, envelope, &raw); err != nil {
		return modelconfig.InspectResult{}, err
	}
	result, err := modelconfig.DecodeInspectResult(raw)
	if err != nil {
		return result, err
	}
	if result.Engine != engine || result.ModelID != model {
		return modelconfig.InspectResult{}, fmt.Errorf("memory profile reply identity mismatch")
	}
	result.NodeID = b.nodeID
	return result, result.Validate()
}

func (b *Broker) handleStatedMemory(msg *Message) {
	relay := settings.Relay{}
	var node string
	var err error
	if msg.Method == "engine:set-stated-memory" {
		p, e := modelconfig.DecodeSetStatedMemoryRequest(msg.Params)
		err = e
		relay.Method, relay.SetStatedMemory, node = "set-stated-memory", &p, p.NodeID
	} else {
		p, e := modelconfig.DecodeClearStatedMemoryRequest(msg.Params)
		err = e
		relay.Method, relay.ClearStatedMemory, node = "clear-stated-memory", &p, p.NodeID
	}
	if err != nil {
		_ = b.codec.RespondError(msg.ID, -32602, err.Error())
		return
	}
	if node != "" && node != b.nodeID {
		worker := b.getEngineMgr()
		if worker == nil {
			_ = b.codec.RespondError(msg.ID, -32000, "engine manager unavailable")
			return
		}
		err = worker.RelayRequest("engine:remote-"+relay.Method, msg.Params, func(raw json.RawMessage, rpcErr *RPCError, err error) {
			if err != nil {
				_ = b.codec.RespondError(msg.ID, -32000, err.Error())
			} else if rpcErr != nil {
				_ = b.codec.RespondError(msg.ID, rpcErr.Code, rpcErr.Message)
			} else {
				result, e := modelconfig.DecodeInspectResult(raw)
				if e != nil {
					_ = b.codec.RespondError(msg.ID, -32000, e.Error())
				} else {
					_ = b.codec.Respond(msg.ID, result)
				}
			}
		})
		if err != nil {
			_ = b.codec.RespondError(msg.ID, -32000, err.Error())
		}
		return
	}
	result, err := b.recordModelMemory(context.Background(), relay, "")
	if err != nil {
		_ = b.codec.RespondError(msg.ID, engineSettingsErrorCode(err), err.Error())
	} else {
		_ = b.codec.Respond(msg.ID, result)
	}
}

func (b *Broker) handleModelInspection(msg *Message) {
	p, err := modelconfig.DecodeInspectRequest(msg.Params)
	if err != nil {
		_ = b.codec.RespondError(msg.ID, -32602, err.Error())
		return
	}
	if p.NodeID != "" && p.NodeID != b.nodeID {
		worker := b.getEngineMgr()
		if worker == nil {
			_ = b.codec.RespondError(msg.ID, -32000, "engine manager unavailable")
			return
		}
		started, clockErr := noderec.MemoryBootClock()
		err = worker.RelayRequest("engine:remote-inspect-model", msg.Params, func(result json.RawMessage, rpcErr *RPCError, err error) {
			if err != nil {
				_ = b.codec.RespondError(msg.ID, -32000, err.Error())
			} else if rpcErr != nil {
				_ = b.codec.RespondError(msg.ID, rpcErr.Code, rpcErr.Message)
			} else {
				inspection, decodeErr := modelconfig.DecodeInspectResult(result)
				if decodeErr != nil {
					_ = b.codec.RespondError(msg.ID, -32000, decodeErr.Error())
					return
				}
				modelAgeAdvice(&inspection.Estimate, modelWorkerElapsed(started, clockErr))
				_ = b.codec.Respond(msg.ID, inspection)
			}
		})
		if err != nil {
			_ = b.codec.RespondError(msg.ID, -32000, err.Error())
		}
		return
	}
	result, err := b.inspectModelSettings(context.Background(), p, "")
	if err != nil {
		_ = b.codec.RespondError(msg.ID, engineSettingsErrorCode(err), err.Error())
	} else {
		_ = b.codec.Respond(msg.ID, result)
	}
}
