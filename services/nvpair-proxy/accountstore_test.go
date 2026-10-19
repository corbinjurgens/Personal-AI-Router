// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"nvpair-shared/nodepolicy"
)

// storeDocument renders a store document around state with a correct
// checksum, so a test can change the state and still reach validation.
func storeDocument(version int, state []byte) []byte {
	sum := sha256.Sum256(state)
	return []byte(fmt.Sprintf(`{"version":%d,"checksum":"%s","state":%s}`, version, hex.EncodeToString(sum[:]), state))
}

// validStore returns the bytes of a store holding one outstanding 500 hold
// and one reconciled hold.
func validStore(t *testing.T, dir string) []byte {
	t.Helper()
	l := openTestLedger(t, dir, newLedgerClock(), testRouting("1000000", "1000000", "1000000"))
	mustReserve(t, l, "chat-a", 100, 200)
	h := mustReserve(t, l, "chat-a", 1, 1)
	if err := l.reconcile(h.Attempt, nil); err != nil {
		t.Fatal(err)
	}
	l.close()
	data, err := os.ReadFile(filepath.Join(dir, accountStoreFileName))
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func storeState(t *testing.T, data []byte) []byte {
	t.Helper()
	var doc struct {
		State json.RawMessage `json:"state"`
	}
	if err := json.Unmarshal(data, &doc); err != nil {
		t.Fatal(err)
	}
	return doc.State
}

func TestAccountStoreRoundTripsWithItsChecksum(t *testing.T) {
	data := validStore(t, t.TempDir())
	if !strings.HasPrefix(string(data), `{"version":1,"checksum":"`) {
		t.Fatalf("store does not start with its version and checksum: %.60s", data)
	}
	got, err := decodeLedger(data)
	if err != nil {
		t.Fatalf("decodeLedger: %v", err)
	}
	again, err := encodeLedger(got)
	if err != nil || string(again) != string(data) {
		t.Fatalf("re-encoding changed the store (err %v)", err)
	}
	for _, secretish := range []string{"sk-", "Bearer"} {
		if strings.Contains(string(data), secretish) {
			t.Errorf("store contains %q", secretish)
		}
	}
}

// A store that fails to load refuses every reservation and is never
// rewritten, so the owner can inspect it.
func TestDamagedAccountStoreRefusesSpendAndIsKept(t *testing.T) {
	good := validStore(t, t.TempDir())
	state := storeState(t, good)
	tamper := func(old, repl string) []byte {
		if !strings.Contains(string(state), old) {
			t.Fatalf("state has no %q", old)
		}
		return storeDocument(1, []byte(strings.Replace(string(state), old, repl, 1)))
	}
	cases := map[string][]byte{
		"empty":                  {},
		"truncated":              good[:len(good)/2],
		"truncated tail":         good[:len(good)-2],
		"bit flip under the sum": []byte(strings.Replace(string(good), `"reserved":"500"`, `"reserved":"501"`, 1)),
		"checksum only":          []byte(strings.Replace(string(good), `"policyRevision":"7"`, `"policyRevision":"8"`, 1)),
		"unsupported version":    storeDocument(2, state),
		"missing version":        []byte(strings.Replace(string(good), `"version":1,`, ``, 1)),
		"unknown state field":    tamper(`"lastObservedMs"`, `"extra":1,"lastObservedMs"`),
		"amount off its bounds":  tamper(`"reserved":"500"`, `"reserved":"499"`),
		"non-canonical money":    tamper(`"reserved":"500"`, `"reserved":"0500"`),
		"money as a number":      tamper(`"reserved":"500"`, `"reserved":500`),
		"non-USD hold":           tamper(`"currency":"USD"`, `"currency":"EUR"`),
		"unknown hold state":     tamper(`"state":"reserved"`, `"state":"released"`),
		"invalid calendar day":   tamper(`d:2026-10-19`, `d:2026-02-30`),
		"invalid calendar month": tamper(`m:2026-10`, `m:2026-13`),
		"credential value shape": tamper(`"credential":"NVPAIR_PROVIDER_SECRET_PAID"`, `"credential":"sk-live"`),
		"trailing data":          append(append([]byte{}, good...), []byte("{}")...),
		"oversize":               []byte(strings.Repeat(" ", maxAccountStoreBytes+1)),
	}
	for name, data := range cases {
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, accountStoreFileName)
			if err := os.WriteFile(path, data, 0o600); err != nil {
				t.Fatal(err)
			}
			l := newAccountLedger()
			l.setDir(dir, nil)
			t.Cleanup(l.close)
			err := l.applyRouting(testRouting("1000000", "1000000", "1000000"), "1")
			if !errors.Is(err, errLedgerUnavailable) || !errors.Is(err, errAccountStoreInvalid) {
				t.Fatalf("applyRouting err = %v, want an invalid store", err)
			}
			_, err = l.reserve("chat-a", 1, 1)
			wantErr(t, "reserve on a damaged store", err, errLedgerUnavailable)
			after := readLedgerBytes(t, dir)
			if string(after) != string(data) {
				t.Fatal("the damaged store was rewritten")
			}
			if l.lock != nil {
				t.Fatal("the lock is still held after a failed open")
			}
		})
	}
}

func TestLargeNanoUSDHoldSurvivesTheStore(t *testing.T) {
	dir := t.TempDir()
	clock := newLedgerClock()
	routing := testRouting("9223372036854775807", "9223372036854775807", "9223372036854775807")
	routing.Deployments[0].ContextTokens = 2_000_000
	routing.Deployments[0].Prices.InputPerMillion = "9007199254740993"
	routing.Deployments[0].Prices.OutputPerMillion = "0"
	l := openTestLedger(t, dir, clock, routing)
	h := mustReserve(t, l, "chat-a", 1_000_000, 1)
	if h.Reserved != "9007199254740993" {
		t.Fatalf("reserved = %s, want exact money above the JavaScript safe integer range", h.Reserved)
	}
	l = reopen(t, l, dir, clock, routing)
	if got := l.state.hold(h.Attempt); got == nil || got.Reserved != h.Reserved || got.State != holdUncertain {
		t.Fatalf("recovered hold = %+v, want the exact uncertain hold", got)
	}
}

// A write that would produce a store too large to load is refused and the
// previous store stays loadable.
func TestAccountStoreSizeLimitKeepsThePreviousStore(t *testing.T) {
	dir := t.TempDir()
	clock := newLedgerClock()
	routing := testRouting("1000000", "1000000", "1000000")
	l := openTestLedger(t, dir, clock, routing)
	mustReserve(t, l, "chat-a", 1, 1)
	before, err := os.ReadFile(filepath.Join(dir, accountStoreFileName))
	if err != nil {
		t.Fatal(err)
	}
	limit := maxAccountStoreBytes
	maxAccountStoreBytes = len(before) + 16
	t.Cleanup(func() { maxAccountStoreBytes = limit })

	_, err = l.reserve("chat-a", 1, 1)
	wantErr(t, "reserve beyond the size limit", err, errAccountStoreTooBig)
	after := readLedgerBytes(t, dir)
	if string(after) != string(before) {
		t.Fatal("the store changed")
	}
	l = reopen(t, l, dir, clock, routing)
	if len(l.state.Holds) != 1 {
		t.Fatalf("reopened store holds %d, want the previous single hold", len(l.state.Holds))
	}
}

// Goroutines racing on first use open the store once, through the ledger's
// mutex: a second open in this process would be refused by the lock.
func TestConcurrentFirstUseOpensOnce(t *testing.T) {
	dir := t.TempDir()
	holder := newAccountLedger()
	holder.setDir(dir, nil)
	t.Cleanup(holder.close)
	routing := testRouting("1000000", "1000000", "1000000")
	if err := holder.applyRouting(routing, "1"); err != nil {
		t.Fatal(err)
	}
	l := newAccountLedger()
	l.setDir(dir, nil)
	t.Cleanup(l.close)
	err := l.applyRouting(routing, "1")
	wantErr(t, "open while another ledger holds the lock", err, errAccountStoreLocked)
	holder.close()

	var wg sync.WaitGroup
	for range 20 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := l.reserve("chat-a", 1, 1); err != nil {
				t.Errorf("reserve: %v", err)
			}
		}()
	}
	wg.Wait()
}

const (
	ledgerHelperMode = "NVPAIR_TEST_ACCOUNT_LEDGER_MODE"
	ledgerHelperDir  = "NVPAIR_TEST_ACCOUNT_LEDGER_DIR"
)

// TestAccountLedgerHelperProcess is the second process of the lock and crash
// tests. It does nothing unless a parent test starts it. It binds no listener.
func TestAccountLedgerHelperProcess(t *testing.T) {
	mode := os.Getenv(ledgerHelperMode)
	if mode == "" {
		return
	}
	l := newAccountLedger()
	l.setDir(os.Getenv(ledgerHelperDir), nil)
	err := l.applyRouting(testRouting("1000000", "1000", "1000000"), "1")
	switch {
	case errors.Is(err, errAccountStoreLocked):
		fmt.Println("result: locked")
		return
	case err != nil:
		fmt.Println("result: error", err)
		return
	}
	h, err := l.reserve("chat-a", 100, 200)
	if err != nil {
		fmt.Println("result: error", err)
		return
	}
	fmt.Println("result: reserved", h.Attempt)
	if mode == "hold" {
		// Keep the lock until the parent kills this process.
		io.Copy(io.Discard, os.Stdin)
	}
	// The process exits without closing the ledger, as a crash would.
}

// startLedgerHelper runs the helper in mode against dir and returns its first
// "result:" line.
func startLedgerHelper(t *testing.T, dir, mode string) (*exec.Cmd, string) {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=^TestAccountLedgerHelperProcess$", "-test.count=1")
	cmd.Env = append(os.Environ(), ledgerHelperMode+"="+mode, ledgerHelperDir+"="+dir)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		stdin.Close()
		cmd.Process.Kill()
		cmd.Wait()
	})
	lines := make(chan string, 1)
	go func() {
		sc := bufio.NewScanner(stdout)
		for sc.Scan() {
			if line, ok := strings.CutPrefix(sc.Text(), "result: "); ok {
				lines <- line
				break
			}
		}
		io.Copy(io.Discard, stdout)
		close(lines)
	}()
	select {
	case line := <-lines:
		return cmd, line
	case <-time.After(30 * time.Second):
		t.Fatal("the helper process reported nothing")
	}
	return nil, ""
}

func openRealClockLedger(t *testing.T, dir string, routing nodepolicy.Routing) (*accountLedger, error) {
	t.Helper()
	l := newAccountLedger()
	l.setDir(dir, nil)
	t.Cleanup(l.close)
	return l, l.applyRouting(routing, "2")
}

// Another process cannot open a store this one holds; a process killed while
// holding the store releases it, and its durable hold survives as uncertain.
func TestSecondProcessIsRefusedAndACrashKeepsTheHold(t *testing.T) {
	dir := t.TempDir()
	routing := testRouting("1000000", "1000", "1000000")
	parent, err := openRealClockLedger(t, dir, routing)
	if err != nil {
		t.Fatal(err)
	}
	if _, line := startLedgerHelper(t, dir, "probe"); line != "locked" {
		t.Fatalf("helper while the parent holds the store: %q, want locked", line)
	}
	parent.close()

	helper, line := startLedgerHelper(t, dir, "hold")
	attempt, ok := strings.CutPrefix(line, "reserved ")
	if !ok {
		t.Fatalf("helper: %q, want a reservation", line)
	}
	l, err := openRealClockLedger(t, dir, routing)
	wantErr(t, "open while the helper holds the store", err, errAccountStoreLocked)

	if err := helper.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	helper.Wait()
	if err := l.applyRouting(routing, "3"); err != nil {
		t.Fatalf("open after the helper was killed: %v", err)
	}
	h := l.state.hold(attempt)
	if h == nil || h.State != holdUncertain || h.Reserved != "500" {
		t.Fatalf("killed helper's hold = %+v, want an uncertain 500", h)
	}
	mustReserve(t, l, "chat-a", 100, 200)
	_, err = l.reserve("chat-a", 0, 1)
	wantErr(t, "reserve beyond the ceiling including the crashed hold", err, errAccountCeiling)
}

// A new process started after the account's credential reference was removed
// (the account retired) still holds the liability, and re-adding the account
// keeps counting it.
func TestReferenceSetChangeAcrossProcessesKeepsLiabilities(t *testing.T) {
	dir := t.TempDir()
	if _, line := startLedgerHelper(t, dir, "exit"); !strings.HasPrefix(line, "reserved ") {
		t.Fatalf("helper: %q, want a reservation", line)
	}
	retired := nodepolicy.Routing{
		Accounts: []nodepolicy.Account{{ID: "spare", Credential: otherCredential, Billing: nodepolicy.BillingPaid,
			PerRequestCeiling: "1", DailyCeiling: "1", MonthlyCeiling: "1"}},
	}
	var l *accountLedger
	var err error
	deadline := time.Now().Add(30 * time.Second)
	for {
		// The helper may still be exiting.
		if l, err = openRealClockLedger(t, dir, retired); !errors.Is(err, errAccountStoreLocked) || time.Now().After(deadline) {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if err != nil {
		t.Fatal(err)
	}
	if got, _ := l.state.outstanding("paid"); got != 500 {
		t.Fatalf("retired account outstanding = %d, want 500", got)
	}
	if err := l.applyRouting(testRouting("1000000", "1000", "1000000"), "3"); err != nil {
		t.Fatal(err)
	}
	mustReserve(t, l, "chat-b", 100, 200)
	_, err = l.reserve("chat-b", 0, 1)
	wantErr(t, "reserve beyond the ceiling including the earlier process's hold", err, errAccountCeiling)
}
