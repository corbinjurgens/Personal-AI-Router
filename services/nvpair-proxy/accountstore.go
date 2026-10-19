// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// The account store is one JSON document under appdir, replaced whole through
// durablefile on every change:
//
//	{"version":1,"checksum":"<sha256 hex of the state bytes>","state":{...}}
//
// The checksum covers exactly the state bytes in the file. Loading refuses an
// unknown version, a checksum mismatch, unknown fields, any invalid record and
// a document larger than maxAccountStoreBytes; a store that fails to load is
// never rewritten. Records are secret-free: credential references are names,
// never values.

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"regexp"
	"strings"
	"time"
	"unicode"

	"nvpair-shared/nodepolicy"
)

const (
	accountStoreFileName = "proxy-accounts.json"
	accountLockFileName  = "proxy-accounts.lock"
	accountStoreVersion  = 1

	maxLedgerBindings   = 1024
	maxLedgerAccounts   = 1024
	maxLedgerHolds      = 4096
	maxResolvedHolds    = 256
	maxLedgerQuarantine = 1024
	maxLedgerTokens     = 1 << 24
	maxLedgerTextBytes  = 2048
)

type holdState string

const (
	// holdReserved is a hold this process reserved; its outcome is unknown.
	holdReserved holdState = "reserved"
	// holdUncertain is a hold recovered from an earlier process, or one whose
	// reservation landed without confirmed durability. It is never assumed
	// unsent and stays outstanding until an owner releases it.
	holdUncertain holdState = "uncertain"
	// holdReconciled is resolved with an actual amount.
	holdReconciled holdState = "reconciled"
)

type cooldownReason string

const (
	cooldownRateLimited cooldownReason = "rate-limited"
	cooldownExhausted   cooldownReason = "exhausted"
	cooldownProbeLost   cooldownReason = "probe-lost"
)

type accountOutcome string

const (
	outcomeOK                accountOutcome = "ok"
	outcomeRateLimited       accountOutcome = "rate-limited"
	outcomeExhausted         accountOutcome = "exhausted"
	outcomeCredentialFailure accountOutcome = "credential-failure"
	outcomeTransient         accountOutcome = "transient"
)

const (
	freezeUsageAboveBound     = "usage-above-bound"
	quarantineUsageAboveBound = "usage-above-bound"
	quarantineQualification   = "qualification-failure"
)

// ledgerState is the whole durable account state.
type ledgerState struct {
	// LastObservedMs is the latest wall-clock time any write observed. A clock
	// behind it refuses new reservations.
	LastObservedMs int64             `json:"lastObservedMs"`
	Bindings       []ledgerBinding   `json:"bindings"`
	Accounts       []ledgerAccount   `json:"accounts"`
	Holds          []ledgerHold      `json:"holds"`
	Quarantine     []quarantineEntry `json:"quarantine"`
}

// ledgerBinding binds a credential reference to the first account ID that
// used it, for good: retirement and re-addition keep it.
type ledgerBinding struct {
	Credential nodepolicy.CredentialRef `json:"credential"`
	Account    string                   `json:"account"`
}

type ledgerAccount struct {
	ID string `json:"id"`
	// LastConfiguredCredential survives retirement so delayed outcomes keep
	// the configured reference's failure. Absent in stores written before it.
	LastConfiguredCredential nodepolicy.CredentialRef `json:"lastConfiguredCredential,omitempty"`
	// Spent is the reconciled amount per UTC period key ("d:2006-01-02",
	// "m:2006-01"); only the current day and month are kept.
	Spent map[string]nodepolicy.NanoUSD `json:"spent"`
	// Generation counts restrictive transitions; a probe's success clears
	// suppression only when none happened since its reservation.
	Generation        int64              `json:"generation"`
	Cooldown          *accountCooldown   `json:"cooldown,omitempty"`
	CredentialFailure *credentialFailure `json:"credentialFailure,omitempty"`
	Frozen            *accountFreeze     `json:"frozen,omitempty"`
	// Probe is the attempt holding the single half-open probe, or empty.
	Probe string `json:"probe,omitempty"`
}

type accountCooldown struct {
	UntilMs int64          `json:"untilMs"`
	Reason  cooldownReason `json:"reason"`
}

type credentialFailure struct {
	Credential nodepolicy.CredentialRef `json:"credential"`
}

type accountFreeze struct {
	Attempt string `json:"attempt"`
	Reason  string `json:"reason"`
}

type tokenUsage struct {
	InputTokens  int64 `json:"inputTokens"`
	OutputTokens int64 `json:"outputTokens"`
}

// ledgerHold is one reservation with the configuration it was made under.
type ledgerHold struct {
	Attempt          string                      `json:"attempt"`
	Account          string                      `json:"account"`
	Deployment       string                      `json:"deployment"`
	Endpoint         string                      `json:"endpoint"`
	Model            string                      `json:"model"`
	Qualification    nodepolicy.QualificationRef `json:"qualification"`
	Credential       nodepolicy.CredentialRef    `json:"credential"`
	PolicyRevision   string                      `json:"policyRevision"`
	Currency         string                      `json:"currency"`
	Unit             string                      `json:"unit"`
	InputPerMillion  nodepolicy.NanoUSD          `json:"inputPerMillion"`
	OutputPerMillion nodepolicy.NanoUSD          `json:"outputPerMillion"`
	InputTokens      int64                       `json:"inputTokens"`
	OutputTokens     int64                       `json:"outputTokens"`
	Reserved         nodepolicy.NanoUSD          `json:"reserved"`
	ReservedAtMs     int64                       `json:"reservedAtMs"`
	// Generation is the account generation the reservation observed.
	Generation int64     `json:"generation"`
	Probe      bool      `json:"probe,omitempty"`
	State      holdState `json:"state"`
	// Outcome is the account outcome recorded for this attempt, once.
	Outcome accountOutcome     `json:"outcome,omitempty"`
	Usage   *tokenUsage        `json:"usage,omitempty"`
	Actual  nodepolicy.NanoUSD `json:"actual,omitempty"`
	// QualificationChecked records completed identity/bound classification,
	// either without a violation or with a durably recorded quarantine.
	QualificationChecked bool `json:"qualificationChecked,omitempty"`
}

// quarantineKey identifies a deployment by what it is, not by its name.
type quarantineKey struct {
	Endpoint      string                      `json:"endpoint"`
	Model         string                      `json:"model"`
	Qualification nodepolicy.QualificationRef `json:"qualification"`
}

type quarantineEntry struct {
	quarantineKey
	Reason  string `json:"reason"`
	SinceMs int64  `json:"sinceMs"`
}

func (h ledgerHold) key() quarantineKey {
	return quarantineKey{Endpoint: h.Endpoint, Model: h.Model, Qualification: h.Qualification}
}

func (h ledgerHold) outstanding() bool {
	return h.State == holdReserved || h.State == holdUncertain
}

func (h ledgerHold) evictable() bool {
	return h.State == holdReconciled && h.Outcome != "" && h.QualificationChecked
}

// maxAccountStoreBytes bounds the store both when it is written and when it is
// read, so a store this process writes is always one the next can load. A
// variable only so a test can reach it without writing megabytes.
var maxAccountStoreBytes = 8 << 20

var (
	errAccountStoreLocked  = errors.New("the account store is locked by another process")
	errAccountStoreInvalid = errors.New("the account store is invalid")
	errAccountStoreTooBig  = errors.New("the account store would exceed its size limit")
	periodKeyPattern       = regexp.MustCompile(`^(d:[0-9]{4}-[0-9]{2}-[0-9]{2}|m:[0-9]{4}-[0-9]{2})$`)
	ledgerAttemptPattern   = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$`)
	ledgerIDPattern        = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$`)
)

// clone copies s deeply enough that mutating the copy's slices and spend maps
// leaves s unchanged. Pointer fields are replaced, never mutated in place.
func (s ledgerState) clone() ledgerState {
	out := ledgerState{
		LastObservedMs: s.LastObservedMs,
		Bindings:       append([]ledgerBinding{}, s.Bindings...),
		Accounts:       make([]ledgerAccount, len(s.Accounts)),
		Holds:          append([]ledgerHold{}, s.Holds...),
		Quarantine:     append([]quarantineEntry{}, s.Quarantine...),
	}
	for i, a := range s.Accounts {
		spent := make(map[string]nodepolicy.NanoUSD, len(a.Spent))
		for k, v := range a.Spent {
			spent[k] = v
		}
		a.Spent = spent
		out.Accounts[i] = a
	}
	return out
}

// encodeLedger renders the store document. The state bytes are written once
// and the checksum is taken over exactly those bytes.
func encodeLedger(s ledgerState) ([]byte, error) {
	state, err := json.Marshal(s)
	if err != nil {
		return nil, err
	}
	sum := sha256.Sum256(state)
	var buf bytes.Buffer
	fmt.Fprintf(&buf, `{"version":%d,"checksum":"%s","state":`, accountStoreVersion, hex.EncodeToString(sum[:]))
	buf.Write(state)
	buf.WriteString("}\n")
	if buf.Len() > maxAccountStoreBytes {
		return nil, errAccountStoreTooBig
	}
	return buf.Bytes(), nil
}

// readAccountStore reads the store. A missing file is an empty state.
func readAccountStore(path string) (ledgerState, error) {
	f, err := os.Open(path)
	if errors.Is(err, os.ErrNotExist) {
		return emptyLedgerState(), nil
	}
	if err != nil {
		return ledgerState{}, err
	}
	defer f.Close()
	data, err := io.ReadAll(io.LimitReader(f, int64(maxAccountStoreBytes)+1))
	if err != nil {
		return ledgerState{}, err
	}
	if len(data) > maxAccountStoreBytes {
		return ledgerState{}, fmt.Errorf("%w: larger than %d bytes", errAccountStoreInvalid, maxAccountStoreBytes)
	}
	return decodeLedger(data)
}

func emptyLedgerState() ledgerState {
	return ledgerState{Bindings: []ledgerBinding{}, Accounts: []ledgerAccount{}, Holds: []ledgerHold{}, Quarantine: []quarantineEntry{}}
}

// decodeStrictJSON decodes one JSON value with no unknown fields and nothing
// after it.
func decodeStrictJSON(data []byte, v any) error {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(v); err != nil {
		return err
	}
	if _, err := dec.Token(); err != io.EOF {
		return errors.New("trailing data")
	}
	return nil
}

func decodeLedger(data []byte) (ledgerState, error) {
	invalid := func(format string, args ...any) (ledgerState, error) {
		return ledgerState{}, fmt.Errorf("%w: %s", errAccountStoreInvalid, fmt.Sprintf(format, args...))
	}
	var doc struct {
		Version  *int            `json:"version"`
		Checksum string          `json:"checksum"`
		State    json.RawMessage `json:"state"`
	}
	if err := decodeStrictJSON(data, &doc); err != nil {
		return invalid("document: %v", err)
	}
	if doc.Version == nil || *doc.Version != accountStoreVersion {
		return invalid("unsupported version")
	}
	sum := sha256.Sum256(doc.State)
	if doc.Checksum != hex.EncodeToString(sum[:]) {
		return invalid("checksum mismatch")
	}
	var s ledgerState
	if err := decodeStrictJSON(doc.State, &s); err != nil {
		return invalid("state: %v", err)
	}
	if err := s.validate(); err != nil {
		return invalid("%v", err)
	}
	return s, nil
}

func validLedgerText(s string) bool {
	return s != "" && len(s) <= maxLedgerTextBytes && strings.IndexFunc(s, unicode.IsControl) < 0
}

func validMoney(m nodepolicy.NanoUSD) bool {
	_, err := m.Int64()
	return err == nil
}

// validate checks every record and cross reference of a loaded state.
func (s ledgerState) validate() error {
	if s.Bindings == nil || s.Accounts == nil || s.Holds == nil || s.Quarantine == nil {
		return errors.New("bindings, accounts, holds and quarantine are required")
	}
	if s.LastObservedMs < 0 {
		return errors.New("lastObservedMs must not be negative")
	}
	if len(s.Bindings) > maxLedgerBindings || len(s.Accounts) > maxLedgerAccounts ||
		len(s.Holds) > maxLedgerHolds || len(s.Quarantine) > maxLedgerQuarantine {
		return errors.New("record count above its limit")
	}
	accounts := make(map[string]bool, len(s.Accounts))
	for i, a := range s.Accounts {
		if !ledgerIDPattern.MatchString(a.ID) || accounts[a.ID] {
			return fmt.Errorf("accounts[%d]: invalid or duplicate id", i)
		}
		accounts[a.ID] = true
		if a.LastConfiguredCredential != "" {
			if _, err := nodepolicy.ParseCredentialRef(string(a.LastConfiguredCredential)); err != nil {
				return fmt.Errorf("accounts[%d]: invalid last configured credential", i)
			}
		}
		if a.Spent == nil || a.Generation < 0 {
			return fmt.Errorf("accounts[%d]: spent and a nonnegative generation are required", i)
		}
		for k, v := range a.Spent {
			if !periodKeyPattern.MatchString(k) || !validMoney(v) {
				return fmt.Errorf("accounts[%d]: invalid spend record", i)
			}
			layout := "2006-01"
			if strings.HasPrefix(k, "d:") {
				layout = "2006-01-02"
			}
			if _, err := time.Parse(layout, k[2:]); err != nil {
				return fmt.Errorf("accounts[%d]: invalid spend period", i)
			}
		}
		if c := a.Cooldown; c != nil {
			switch c.Reason {
			case cooldownRateLimited, cooldownExhausted, cooldownProbeLost:
			default:
				return fmt.Errorf("accounts[%d]: invalid cooldown reason", i)
			}
			if c.UntilMs < 0 {
				return fmt.Errorf("accounts[%d]: invalid cooldown deadline", i)
			}
		}
		if f := a.CredentialFailure; f != nil {
			if _, err := nodepolicy.ParseCredentialRef(string(f.Credential)); err != nil {
				return fmt.Errorf("accounts[%d]: invalid credential failure", i)
			}
		}
		if f := a.Frozen; f != nil && (!ledgerAttemptPattern.MatchString(f.Attempt) || f.Reason != freezeUsageAboveBound) {
			return fmt.Errorf("accounts[%d]: invalid freeze", i)
		}
		if a.Probe != "" && !ledgerAttemptPattern.MatchString(a.Probe) {
			return fmt.Errorf("accounts[%d]: invalid probe", i)
		}
	}
	refs := make(map[nodepolicy.CredentialRef]bool, len(s.Bindings))
	for i, b := range s.Bindings {
		if _, err := nodepolicy.ParseCredentialRef(string(b.Credential)); err != nil || refs[b.Credential] || !accounts[b.Account] {
			return fmt.Errorf("bindings[%d]: invalid, duplicate or unknown account", i)
		}
		refs[b.Credential] = true
	}
	attempts := make(map[string]bool, len(s.Holds))
	resolved := 0
	for i, h := range s.Holds {
		if err := h.validate(); err != nil {
			return fmt.Errorf("holds[%d]: %w", i, err)
		}
		if attempts[h.Attempt] || !accounts[h.Account] {
			return fmt.Errorf("holds[%d]: duplicate attempt or unknown account", i)
		}
		attempts[h.Attempt] = true
		if h.evictable() {
			resolved++
		}
	}
	if resolved > maxResolvedHolds {
		return errors.New("resolved holds above their limit")
	}
	keys := make(map[quarantineKey]bool, len(s.Quarantine))
	for i, q := range s.Quarantine {
		if !validLedgerText(q.Endpoint) || !validLedgerText(q.Model) || q.Qualification.Validate() != nil ||
			(q.Reason != quarantineUsageAboveBound && q.Reason != quarantineQualification) || q.SinceMs < 0 || keys[q.quarantineKey] {
			return fmt.Errorf("quarantine[%d]: invalid or duplicate", i)
		}
		keys[q.quarantineKey] = true
	}
	return nil
}

func (h ledgerHold) validate() error {
	if !ledgerAttemptPattern.MatchString(h.Attempt) || !ledgerIDPattern.MatchString(h.Deployment) {
		return errors.New("invalid attempt or deployment")
	}
	if !validLedgerText(h.Endpoint) || !validLedgerText(h.Model) || h.Qualification.Validate() != nil {
		return errors.New("invalid deployment identity")
	}
	if _, err := nodepolicy.ParseCredentialRef(string(h.Credential)); err != nil {
		return errors.New("invalid credential reference")
	}
	if _, err := nodepolicy.CompareRevisions(h.PolicyRevision, "0"); err != nil || h.PolicyRevision == "" {
		return errors.New("invalid policy revision")
	}
	if h.Currency != nodepolicy.CurrencyUSD || h.Unit != nodepolicy.PriceUnit {
		return errors.New("unsupported currency or unit")
	}
	if h.InputTokens < 0 || h.InputTokens > maxLedgerTokens || h.OutputTokens < 1 || h.OutputTokens > maxLedgerTokens ||
		h.ReservedAtMs < 0 || h.Generation < 0 {
		return errors.New("invalid bounds")
	}
	reserved, err := reservationAmount(h.InputTokens, h.OutputTokens, h.InputPerMillion, h.OutputPerMillion)
	if err != nil || nodepolicy.NanoUSD(fmt.Sprint(reserved)) != h.Reserved {
		return errors.New("reserved amount does not match its bounds and prices")
	}
	switch h.Outcome {
	case "", outcomeOK, outcomeRateLimited, outcomeExhausted, outcomeCredentialFailure, outcomeTransient:
	default:
		return errors.New("invalid outcome")
	}
	switch h.State {
	case holdReserved, holdUncertain:
		if h.Usage != nil || h.Actual != "" {
			return errors.New("an outstanding hold has no actual amount")
		}
	case holdReconciled:
		if !validMoney(h.Actual) {
			return errors.New("a reconciled hold needs an actual amount")
		}
		if u := h.Usage; u != nil && (u.InputTokens < 0 || u.OutputTokens < 0) {
			return errors.New("invalid usage")
		}
	default:
		return errors.New("invalid state")
	}
	return nil
}
