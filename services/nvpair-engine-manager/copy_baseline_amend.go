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
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"nvpair-shared/durablefile"
	"nvpair-shared/modeldownload"
	"nvpair-shared/servicectl"
)

type copyBaselineOptions struct {
	Mode             bool
	Expected         string
	Disposition      string
	Attest           bool
	AllowOld         bool
	PreviousReviewed bool
}

func (o copyBaselineOptions) validate() error {
	if !o.Mode || !o.Attest || !lowerHex(o.Expected, 64) || o.Disposition != "complete-empty-history" {
		return errors.New("amendment requires explicit mode, lowercase SHA-256, complete-empty-history and true quiescence attestation")
	}
	return nil
}

func copyBaselineOwner(image string) bool {
	n := strings.ToLower(normalizeProcessName(image))
	return strings.HasPrefix(n, "nvpair-") || n == "nvpair" || n == "pair" || n == "nvidia pair" || strings.HasPrefix(n, "pair helper") || strings.HasPrefix(n, "nvidia pair helper") || strings.HasPrefix(n, "nvpair helper")
}

func checkCopyBaselineQuiescence(ctx context.Context, override string, self int, enumerate func(context.Context) ([]processRecord, error), endpoint func() (string, error), dial func(context.Context, string) (io.ReadWriteCloser, error)) error {
	if override != "" {
		return errors.New("service endpoint override must be reviewed and cleared")
	}
	ctx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	if err := ctx.Err(); err != nil {
		return err
	}
	records, err := enumerate(ctx)
	if err != nil {
		return err
	}
	found := false
	seen := make(map[int]bool)
	for _, record := range records {
		// Windows' system idle process legitimately has PID zero.
		if record.PID < 0 || normalizeProcessName(record.Image) == "" || seen[record.PID] {
			return errors.New("incomplete process enumeration")
		}
		seen[record.PID] = true
		if record.PID == self {
			found = true
			continue
		}
		if copyBaselineOwner(record.Image) {
			return errors.New("PAIR owner detected; hold offline interval")
		}
	}
	if !found {
		return errors.New("amendment PID missing from enumeration")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	path, err := endpoint()
	if err != nil || path == "" {
		return errors.New("service endpoint resolution failed")
	}
	conn, err := dial(ctx, path)
	if conn != nil {
		closeErr := conn.Close()
		return fmt.Errorf("service connection detected (close error: %v)", closeErr)
	}
	if ctx.Err() != nil {
		return ctx.Err()
	}
	if !errors.Is(err, servicectl.ErrNotRunning) {
		return fmt.Errorf("service absence unconfirmed: %v", err)
	}
	return nil
}

func hostCopyBaselineQuiescence() error {
	return checkCopyBaselineQuiescence(context.Background(), os.Getenv(servicectl.EndpointEnv), os.Getpid(), listProcessRecords, servicectl.Endpoint, servicectl.DialEndpoint)
}

func journalDigest(data []byte) string { sum := sha256.Sum256(data); return hex.EncodeToString(sum[:]) }

func readAmendJournal(path string) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	data, err := io.ReadAll(io.LimitReader(f, modeldownload.MaxJournalBytes+1))
	closeErr := f.Close()
	if err != nil {
		return nil, err
	}
	if closeErr != nil {
		return nil, closeErr
	}
	if len(data) > modeldownload.MaxJournalBytes {
		return nil, errDownloadPersistence
	}
	return data, nil
}

// Only the exact empty predecessor shape is qualified here. Ordinary loading
// remains strict; this in-memory addition is never the persisted serializer.
func emptyOldJournal(data []byte) error {
	if len(data) > modeldownload.MaxJournalBytes || downloadJournalObject(data, "version", "entries", "copies", "copiesComplete") != nil {
		return errDownloadPersistence
	}
	var object map[string]json.RawMessage
	if json.Unmarshal(data, &object) != nil {
		return errDownloadPersistence
	}
	object["copyRecoveryRequired"] = json.RawMessage("false")
	candidate, err := json.Marshal(object)
	if err != nil {
		return errDownloadPersistence
	}
	// Validate original UTF-8 too: encoding/json otherwise repairs it.
	if !json.Valid(data) || !bytes.Equal(bytes.ToValidUTF8(data, nil), data) {
		return errDownloadPersistence
	}
	d, c, _, _, err := decodeModelDownloadJournal(candidate)
	if err != nil || len(d) != 0 || len(c) != 0 {
		return errDownloadPersistence
	}
	return nil
}

type copyBaselineFS struct {
	lstat   func(string) (os.FileInfo, error)
	read    func(string) ([]byte, error)
	reserve func(string) (io.Closer, error)
}

func defaultCopyBaselineFS() copyBaselineFS {
	return copyBaselineFS{os.Lstat, readAmendJournal, func(path string) (io.Closer, error) {
		return os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	}}
}

// inspectAmendPath checks all intermediates, including ancestors above existing
// leaves. Definite absence is allowed only for previous-profile inspection.
func inspectAmendPath(path string, directory, missingOK bool, fs copyBaselineFS) (bool, error) {
	if !filepath.IsAbs(path) || filepath.Clean(path) != path {
		return false, errors.New("invalid profile path")
	}
	present := true
	foundExisting := false
	for p := path; ; p = filepath.Dir(p) {
		info, err := fs.lstat(p)
		if errors.Is(err, os.ErrNotExist) && missingOK {
			if foundExisting || filepath.Dir(p) == p {
				return false, errors.New("profile ancestor disappeared")
			}
			present = false
		} else if err != nil {
			return false, errors.New("profile inspection failed")
		} else if info.Mode()&os.ModeSymlink != 0 || (p != path || directory) && !info.IsDir() || p == path && !directory && !info.Mode().IsRegular() {
			return false, errors.New("unsafe profile path type")
		} else {
			foundExisting = true
		}
		if filepath.Dir(p) == p {
			break
		}
	}
	return present, nil
}

func inspectPreviousCopyBaseline(previous string, reviewed bool, fs copyBaselineFS) (string, error) {
	present, err := inspectAmendPath(previous, true, true, fs)
	if err != nil {
		return "unavailable", err
	}
	if !present {
		return "absent-app", nil
	}
	if !reviewed {
		return "review-required", errors.New("existing previous app requires true previous-profile-reviewed")
	}
	install := filepath.Join(previous, "engine-bin")
	present, err = inspectAmendPath(install, true, true, fs)
	if err != nil {
		return "unavailable", err
	}
	if !present {
		return "absent-journal", nil
	}
	path := filepath.Join(install, modelDownloadFile)
	present, err = inspectAmendPath(path, false, true, fs)
	if err != nil {
		return "unavailable", err
	}
	if !present {
		return "absent-journal", nil
	}
	data, err := fs.read(path)
	if err != nil {
		return "unavailable", err
	}
	d, c, _, _, err := decodeModelDownloadJournal(data)
	if err == nil && len(d) == 0 && len(c) == 0 {
		return "empty-current", nil
	}
	if emptyOldJournal(data) == nil {
		return "empty-old", nil
	}
	return "retained-or-invalid", errors.New("previous journal retained or invalid")
}

type copyBaselineResult struct {
	Before, After, Backup, Input, Previous, Load, Publication string
	Downloads, Copies                                         int
	Complete, Recovery, PreviousReviewed                      bool
}

func unavailableCopyBaselineResult(o copyBaselineOptions) copyBaselineResult {
	return copyBaselineResult{Before: "unavailable", After: "unavailable", Backup: "none", Input: "unknown", Previous: "unavailable", Load: "unavailable", Publication: "not-attempted", PreviousReviewed: o.PreviousReviewed}
}

func (r copyBaselineResult) summary() string {
	return fmt.Sprintf("before-sha256=%s after-sha256=%s backup=%s input=%s previous=%s previous-reviewed=%t publication=%s load=%s downloads=%d copies=%d copiesComplete=%t copyRecoveryRequired=%t; live source/incarnation and full owner/build-apply evidence: not sampled", r.Before, r.After, r.Backup, r.Input, r.Previous, r.PreviousReviewed, r.Publication, r.Load, r.Downloads, r.Copies, r.Complete, r.Recovery)
}

func amendCopyBaseline(current, previous string, options copyBaselineOptions, probe func() error, backup, journal durablefile.Replacer) (copyBaselineResult, error) {
	return amendCopyBaselineWithFS(current, previous, options, probe, backup, journal, defaultCopyBaselineFS())
}

// These repeated snapshots enforce reviewed identity under the attested operator
// exclusion interval. They are not an atomic CAS against ordinary startup.
// Failure retains reservations/backups/landed bytes; never restore or retry.
func amendCopyBaselineWithFS(current, previous string, o copyBaselineOptions, probe func() error, backup, journal durablefile.Replacer, fs copyBaselineFS) (r copyBaselineResult, err error) {
	r = unavailableCopyBaselineResult(o)
	if err = o.validate(); err != nil {
		return
	}
	if err = offlineEnvironmentGuard(); err != nil {
		return
	}
	if !filepath.IsAbs(current) || filepath.Clean(current) != current || current == previous || current == filepath.Dir(current) {
		return r, errors.New("invalid profile pair")
	}
	path := filepath.Join(current, "engine-bin", modelDownloadFile)
	inspect := func() error {
		if _, e := inspectAmendPath(path, false, false, fs); e != nil {
			return e
		}
		var e error
		r.Previous, e = inspectPreviousCopyBaseline(previous, o.PreviousReviewed, fs)
		return e
	}
	// Safely capture actual after-state on every readable refusal and attempted
	// publication. Unavailable is explicit rather than a fabricated digest/load.
	defer func() {
		if _, e := inspectAmendPath(path, false, false, fs); e != nil {
			if err == nil {
				err = e
				r.Publication = "readback-failed"
			}
			return
		}
		data, e := fs.read(path)
		if e != nil {
			if err == nil {
				err = e
				r.Publication = "readback-failed"
			}
			return
		}
		r.After = journalDigest(data)
		d, c, complete, recovery, e := decodeModelDownloadJournal(data)
		if e != nil {
			r.Load = "invalid"
			if err == nil {
				err = e
				r.Publication = "readback-failed"
			}
			return
		}
		r.Load, r.Downloads, r.Copies, r.Complete, r.Recovery = "strict-current", len(d), len(c), complete, recovery
		if err == nil && (len(d) != 0 || len(c) != 0 || !complete || recovery) {
			err = errors.New("publication readback mismatch")
			r.Publication = "readback-failed"
		}
	}()
	if err = inspect(); err != nil {
		return
	}
	if err = probe(); err != nil {
		return
	}
	data, e := fs.read(path)
	if e != nil {
		return r, e
	}
	r.Before = journalDigest(data)
	if r.Before != o.Expected {
		return r, errors.New("reviewed journal hash mismatch")
	}
	d, c, complete, recovery, e := decodeModelDownloadJournal(data)
	if e == nil {
		r.Input = "current"
		if o.AllowOld || len(d) != 0 || len(c) != 0 || complete || recovery {
			return r, errors.New("current journal ineligible")
		}
	} else {
		r.Input = "old-or-invalid"
		if !o.AllowOld || emptyOldJournal(data) != nil {
			return r, errors.New("old journal ineligible")
		}
		r.Input = "empty-old"
	}
	recheck := func() error {
		if e := inspect(); e != nil {
			return e
		}
		if e := probe(); e != nil {
			return e
		}
		now, e := fs.read(path)
		if e != nil {
			return e
		}
		if !bytes.Equal(now, data) || journalDigest(now) != o.Expected {
			return errors.New("journal changed during review interval")
		}
		return nil
	}
	if err = recheck(); err != nil {
		return
	}
	r.Backup = modelDownloadFile + ".before-copy-baseline-" + r.Before + ".bak"
	backupPath := filepath.Join(filepath.Dir(path), r.Backup)
	reservation, e := fs.reserve(backupPath)
	if e != nil {
		return r, e
	}
	if err = reservation.Close(); err != nil {
		return
	}
	if err = backup.WriteFile(backupPath, data, 0600); err != nil {
		return
	}
	if _, err = inspectAmendPath(backupPath, false, false, fs); err != nil {
		return
	}
	saved, e := fs.read(backupPath)
	if e != nil || !bytes.Equal(saved, data) {
		return r, errors.New("exact backup readback failed")
	}
	if err = recheck(); err != nil {
		return
	}
	store := newModelDownloadStore(filepath.Dir(path))
	store.replace = journal
	r.Publication = "attempted"
	landed, uncertain, e := store.save(nil, nil, true, false)
	if e != nil || !landed || uncertain {
		r.Publication = fmt.Sprintf("failed-landed-%t-uncertain-%t", landed, uncertain)
		return r, errors.New("publication unconfirmed; hold offline interval")
	}
	r.Publication = "confirmed"
	// Readback is in the defer so failure summaries describe actual landed bytes.
	return
}
