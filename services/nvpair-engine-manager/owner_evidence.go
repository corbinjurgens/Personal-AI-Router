// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"time"

	settings "nvpair-shared/enginesettings"
)

// desiredOwnerEvidence is a detached view of one saved document. Revisions
// describe readable semantics, not write durability, mutation history or an
// absence of intervening ABA edits. They also serve as protected semantic
// digests across restarts; a consumer must bind its fence to the manager's
// separate incarnation. No runtime stop or completion fact follows from State.
type desiredOwnerEvidence struct {
	State          desiredStateFile
	IntentRevision string
	LockRevision   string
}

func (s *desiredStateStore) ownerEvidence() (desiredOwnerEvidence, error) {
	return s.captureOwnerEvidence(s.readDocument)
}

func (s *desiredStateStore) captureOwnerEvidence(read func() ([]byte, error)) (desiredOwnerEvidence, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.path == "" {
		return desiredOwnerEvidence{}, errors.New("no user data directory is available")
	}
	data, err := read()
	if err != nil {
		return desiredOwnerEvidence{}, err
	}
	// Ordinary intent accessors retain their existing decoding behavior. Evidence
	// must not turn malformed null booleans into explicit Off or unlocked values.
	if err := validateDesiredOwnerDocument(data); err != nil {
		return desiredOwnerEvidence{}, err
	}
	state, err := decodeDesiredState(data)
	if err != nil {
		return desiredOwnerEvidence{}, err
	}
	for name, origin := range state.OffBy {
		enabled, present := state.Engines[name]
		if !present || enabled || (origin != intentLocal && origin != intentPeer) {
			return desiredOwnerEvidence{}, fmt.Errorf("invalid off provenance for %s", name)
		}
	}
	for name, enabled := range state.Engines {
		if !enabled {
			if _, present := state.OffBy[name]; !present {
				state.OffBy[name] = intentLocal
			}
		}
	}
	for name, locked := range state.Locked {
		enabled, present := state.Engines[name]
		if locked && (!present || enabled || state.OffBy[name] != intentLocal) {
			return desiredOwnerEvidence{}, fmt.Errorf("locked engine %s lacks local Off intent", name)
		}
	}
	if cp := state.PendingSwitch; cp != nil {
		for _, observed := range []settings.SwitchEngineState{cp.PreSource, cp.PreTarget, cp.Source, cp.Target} {
			if err := observed.Validate(); err != nil {
				return desiredOwnerEvidence{}, fmt.Errorf("invalid pending switch state: %w", err)
			}
		}
		for _, prior := range []struct {
			intent string
			origin string
		}{{cp.PreSource.Intent, cp.SourceProvenance}, {cp.PreTarget.Intent, cp.TargetProvenance}} {
			if prior.origin != "" && (prior.intent != settings.SwitchIntentOff || (prior.origin != string(intentLocal) && prior.origin != string(intentPeer))) {
				return desiredOwnerEvidence{}, errors.New("invalid pending switch Off provenance")
			}
		}
		if cp.Error != nil {
			if err := cp.Error.Validate(); err != nil {
				return desiredOwnerEvidence{}, fmt.Errorf("invalid pending switch error: %w", err)
			}
		}
	} else if state.PendingSwitchPorts != nil {
		return desiredOwnerEvidence{}, errors.New("pending switch endpoints lack a checkpoint")
	}
	// encoding/json sorts map keys. Explicit false lock entries and intent key
	// presence remain part of their respective baselines. Each decode owns all
	// maps and nested checkpoint pointers, so no store or other read aliases them.
	intent, err := switchDigest(struct {
		Engines map[string]bool         `json:"engines"`
		OffBy   map[string]intentOrigin `json:"offBy"`
	}{state.Engines, state.OffBy})
	if err != nil {
		return desiredOwnerEvidence{}, err
	}
	locks, err := switchDigest(state.Locked)
	if err != nil {
		return desiredOwnerEvidence{}, err
	}
	return desiredOwnerEvidence{State: state, IntentRevision: intent, LockRevision: locks}, nil
}

func validateDesiredOwnerDocument(data []byte) error {
	if err := validateOwnerUniqueMembers(json.NewDecoder(bytes.NewReader(data)), 0); err != nil {
		return err
	}
	if err := validateOwnerScalarPresence(data, reflect.TypeOf(desiredStateFile{})); err != nil {
		return err
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return fmt.Errorf("decode desired engine evidence: %w", err)
	}
	if fields == nil {
		return errors.New("desired engine evidence must be an object")
	}
	for name, raw := range fields {
		switch name {
		case "engines", "offBy", "locked", "processStops", "allocatedPorts":
			var entries map[string]json.RawMessage
			if err := json.Unmarshal(raw, &entries); err != nil || entries == nil {
				return fmt.Errorf("desired engine evidence %s must be an object", name)
			}
			if name == "engines" || name == "locked" {
				for engine, value := range entries {
					value = bytes.TrimSpace(value)
					if !bytes.Equal(value, []byte("true")) && !bytes.Equal(value, []byte("false")) {
						return fmt.Errorf("desired engine evidence %s.%s must be boolean", name, engine)
					}
				}
			}
		case "pendingSwitch", "pendingSwitchPorts":
			// The document decoder validates these optional checkpoint records.
		default:
			return fmt.Errorf("unrecognized desired engine evidence field %q", name)
		}
	}
	return nil
}

// JSON null must not silently become a known zero-valued scalar. Pointer fields
// retain their existing unknown/absent semantics; ordinary accessors are unchanged.
func validateOwnerScalarPresence(raw json.RawMessage, typ reflect.Type) error {
	if bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		if typ.Kind() == reflect.Pointer {
			return nil
		}
		return errors.New("null nonnullable field in desired engine evidence")
	}
	if typ.Kind() == reflect.Pointer {
		return validateOwnerScalarPresence(raw, typ.Elem())
	}
	if typ == reflect.TypeOf(time.Time{}) {
		return nil // The document decoder validates the timestamp string.
	}
	switch typ.Kind() {
	case reflect.Struct:
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(raw, &fields); err != nil {
			return err
		}
		for i := 0; i < typ.NumField(); i++ {
			field := typ.Field(i)
			if field.Anonymous {
				if err := validateOwnerScalarPresence(raw, field.Type); err != nil {
					return err
				}
				continue
			}
			name := strings.Split(field.Tag.Get("json"), ",")[0]
			if name == "" {
				name = field.Name
			}
			for key := range fields {
				if key != name && strings.EqualFold(key, name) {
					return errors.New("noncanonical field name in desired engine evidence")
				}
			}
			if value, ok := fields[name]; ok {
				if err := validateOwnerScalarPresence(value, field.Type); err != nil {
					return err
				}
			}
		}
	case reflect.Map:
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(raw, &fields); err != nil {
			return err
		}
		for _, value := range fields {
			if err := validateOwnerScalarPresence(value, typ.Elem()); err != nil {
				return err
			}
		}
	}
	return nil
}

// A map decode loses duplicate members, while decoding into the saved-state
// structs may merge them. Reject that ambiguity before either projection.
func validateOwnerUniqueMembers(dec *json.Decoder, depth int) error {
	if depth > 64 {
		return errors.New("desired engine evidence nesting exceeds its document shape")
	}
	token, err := dec.Token()
	if err != nil {
		return err
	}
	delim, composite := token.(json.Delim)
	if !composite {
		return nil
	}
	seen := map[string]bool{}
	for dec.More() {
		if delim == '{' {
			key, err := dec.Token()
			if err != nil {
				return err
			}
			name, ok := key.(string)
			if !ok || seen[name] {
				return errors.New("duplicate member in desired engine evidence")
			}
			seen[name] = true
		}
		if err := validateOwnerUniqueMembers(dec, depth+1); err != nil {
			return err
		}
	}
	_, err = dec.Token()
	return err
}
