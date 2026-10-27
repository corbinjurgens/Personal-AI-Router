// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build darwin

package secretstore

import (
	"context"
	"errors"
	"sync"
	"testing"
	"unsafe"
)

type fakeDarwin struct {
	keychains                                              map[uintptr]bool
	arrays                                                 map[uintptr]uintptr
	defaultStatus                                          int32
	a                                                      *darwinAPI
	initial, enabled                                       uint8
	live                                                   map[uintptr]bool
	dicts                                                  map[uintptr]map[uintptr]uintptr
	texts                                                  map[uintptr]string
	data                                                   map[uintptr][]byte
	next                                                   uintptr
	allocations, failAllocation, releases, items, settings int
	readStatus, disableStatus, restoreStatus               int32
	updateStatus, addStatus, copyStatus, deleteStatus      int32
	duplicateUpdateStatus                                  int32
	updates, adds, reads, deletes                          int
	badType, nilData, missingResult                        bool
	dataSize                                               int
}

func newFakeDarwin(t *testing.T, enabled uint8) *fakeDarwin {
	t.Helper()
	f := &fakeDarwin{initial: enabled, enabled: enabled, live: make(map[uintptr]bool), dicts: make(map[uintptr]map[uintptr]uintptr), texts: make(map[uintptr]string), data: make(map[uintptr][]byte), next: 100, dataSize: 4}
	a := &darwinAPI{}
	f.a = a
	f.keychains = make(map[uintptr]bool)
	f.arrays = make(map[uintptr]uintptr)
	next := uintptr(1)
	var err error
	a.constants, err = darwinConstants(func(_ bool, _ string) (uintptr, error) { p := next; next++; return p, nil }, func(p uintptr) uintptr { return p + 1000 })
	wantError(t, err, nil)
	a.copyDefault = func(out *uintptr) int32 {
		if f.enabled != 0 {
			t.Fatal("default keychain accessed with interaction enabled")
		}
		*out = f.allocate()
		if *out != 0 {
			f.keychains[*out] = true
		}
		return f.defaultStatus
	}
	a.arrayCreate = func(_ uintptr, keychain *uintptr, n int, callbacks uintptr) uintptr {
		if n != 1 || !f.keychains[*keychain] || callbacks != a.constants["kCFTypeArrayCallBacks"] {
			t.Fatal("invalid default keychain search list")
		}
		ref := f.allocate()
		if ref != 0 {
			f.arrays[ref] = *keychain
		}
		return ref
	}
	a.interactionGet = func(p *uint8) int32 { *p = f.enabled; return f.readStatus }
	a.interactionSet = func(value uint8) int32 {
		f.settings++
		if f.settings%2 == 1 {
			f.enabled = value
			return f.disableStatus
		}
		f.enabled = value
		return f.restoreStatus
	}
	a.stringCreate = func(_ uintptr, p *byte, n int, _ uint32, _ uint8) uintptr {
		ref := f.allocate()
		if ref != 0 {
			f.texts[ref] = string(unsafe.Slice(p, n))
		}
		return ref
	}
	a.dataCreate = func(_ uintptr, p *byte, n int) uintptr {
		ref := f.allocate()
		if ref != 0 {
			f.data[ref] = append([]byte(nil), unsafe.Slice(p, n)...)
		}
		return ref
	}
	a.dictionaryCreate = func(_ uintptr, _ int, keys, values uintptr) uintptr {
		if keys != a.constants["kCFTypeDictionaryKeyCallBacks"] || values != a.constants["kCFTypeDictionaryValueCallBacks"] {
			t.Fatal("wrong callback addresses")
		}
		ref := f.allocate()
		if ref != 0 {
			f.dicts[ref] = make(map[uintptr]uintptr)
		}
		return ref
	}
	a.dictionarySet = func(dict, key, value uintptr) {
		if !f.live[dict] || value == 0 {
			t.Fatal("invalid CF dictionary write")
		}
		f.dicts[dict][key] = value
	}
	a.release = func(p uintptr) {
		if !f.live[p] {
			t.Fatal("unowned or double CF release")
		}
		delete(f.live, p)
		f.releases++
	}
	checkQuery := func(query uintptr) {
		f.items++
		if f.enabled != 0 {
			t.Fatal("item call while interaction enabled")
		}
		q := f.dicts[query]
		if keychain := q[a.constants["kSecUseKeychain"]]; keychain != 0 {
			if !f.keychains[keychain] || q[a.constants["kSecMatchSearchList"]] != 0 {
				t.Fatal("creation targeted another keychain")
			}
		} else if !f.keychains[f.arrays[q[a.constants["kSecMatchSearchList"]]]] {
			t.Fatal("query searched beyond the default keychain")
		}
		for key, want := range map[string]uintptr{"kSecClass": a.constants["kSecClassGenericPassword"], "kSecAttrSynchronizable": a.constants["kCFBooleanFalse"], "kSecUseAuthenticationUI": a.constants["kSecUseAuthenticationUIFail"]} {
			if q[a.constants[key]] != want {
				t.Fatalf("wrong query attribute %s", key)
			}
		}
		if f.texts[q[a.constants["kSecAttrService"]]] != servicePrefix+"purpose" || f.texts[q[a.constants["kSecAttrAccount"]]] != "id" {
			t.Fatal("wrong exact-entry query")
		}
	}
	a.update = func(query, attrs uintptr) int32 {
		checkQuery(query)
		f.updates++
		if len(f.dicts[attrs]) != 1 || f.dicts[attrs][a.constants["kSecValueData"]] == 0 {
			t.Fatal("update must contain only value data")
		}
		if f.dicts[query][a.constants["kSecValueData"]] != 0 {
			t.Fatal("replacement data leaked into update query")
		}
		if f.updates > 1 {
			return f.duplicateUpdateStatus
		}
		return f.updateStatus
	}
	a.add = func(query uintptr, out *uintptr) int32 {
		checkQuery(query)
		f.adds++
		if out != nil || f.dicts[query][a.constants["kSecValueData"]] == 0 {
			t.Fatal("wrong add attributes")
		}
		return f.addStatus
	}
	a.copyMatching = func(query uintptr, out *uintptr) int32 {
		checkQuery(query)
		f.reads++
		q := f.dicts[query]
		if q[a.constants["kSecReturnData"]] != a.constants["kCFBooleanTrue"] || q[a.constants["kSecMatchLimit"]] != a.constants["kSecMatchLimitOne"] {
			t.Fatal("wrong get shape")
		}
		if !f.missingResult {
			*out = f.allocate()
			f.data[*out] = []byte("fake")
		}
		return f.copyStatus
	}
	a.remove = func(query uintptr) int32 { checkQuery(query); f.deletes++; return f.deleteStatus }
	a.typeID = func(uintptr) uintptr {
		if f.badType {
			return 2
		}
		return 1
	}
	a.dataTypeID = func() uintptr { return 1 }
	a.dataLength = func(uintptr) int { return f.dataSize }
	a.dataBytes = func(p uintptr) *byte {
		if f.nilData {
			return nil
		}
		return unsafe.SliceData(f.data[p])
	}
	t.Cleanup(func() {
		if len(f.live) != 0 {
			t.Error("owned CF objects leaked")
		}
		if f.enabled != f.initial {
			t.Error("interaction state not restored")
		}
	})
	return f
}

func (f *fakeDarwin) allocate() uintptr {
	f.allocations++
	if f.allocations == f.failAllocation {
		return 0
	}
	f.next++
	f.live[f.next] = true
	return f.next
}

func TestDarwinDataSymbolIndirectionAndCallbacks(t *testing.T) {
	reads := make(map[uintptr]bool)
	addresses := make(map[string]uintptr)
	next := uintptr(10)
	constants, err := darwinConstants(func(security bool, name string) (uintptr, error) {
		if security != (name[:4] == "kSec") {
			t.Fatal("wrong framework for symbol")
		}
		next++
		addresses[name] = next
		return next, nil
	}, func(address uintptr) uintptr { reads[address] = true; return address + 100 })
	wantError(t, err, nil)
	for name, address := range addresses {
		callback := name == "kCFTypeDictionaryKeyCallBacks" || name == "kCFTypeDictionaryValueCallBacks" || name == "kCFTypeArrayCallBacks"
		want := address + 100
		if callback {
			want = address
		}
		if constants[name] != want || reads[address] == callback {
			t.Fatalf("incorrect data indirection for %s", name)
		}
	}
}

func TestDarwinSymbolFailuresAreSanitized(t *testing.T) {
	for _, mode := range []string{"lookup failure", "null address", "null data pointer"} {
		t.Run(mode, func(t *testing.T) {
			_, err := darwinConstants(func(bool, string) (uintptr, error) {
				if mode == "lookup failure" {
					return 0, errors.New("native diagnostic")
				}
				if mode == "null address" {
					return 0, nil
				}
				return 1, nil
			}, func(uintptr) uintptr { return 0 })
			wantError(t, err, ErrUnavailable)
		})
	}
}

func TestDarwinOperationsRestoreBothInitialInteractionStates(t *testing.T) {
	for _, test := range []struct {
		name    string
		enabled uint8
	}{{"initially enabled", 1}, {"initially disabled", 0}} {
		t.Run(test.name, func(t *testing.T) {
			f := newFakeDarwin(t, test.enabled)
			s := &Store{backend: f.a}
			wantError(t, s.Set(t.Context(), "purpose", "id", []byte("fake")), nil)
			value, err := s.Get(t.Context(), "purpose", "id")
			wantValue(t, value, []byte("fake"), err)
			wantError(t, s.Delete(t.Context(), "purpose", "id"), nil)
			if f.settings != 6 || f.items != 3 {
				t.Fatal("interaction guard did not surround each operation")
			}
		})
	}
}

func TestDarwinUpdateAddAndDuplicateRace(t *testing.T) {
	test := func(name string, update, add, retry int32, want error, wantUpdates, wantAdds int) {
		t.Run(name, func(t *testing.T) {
			f := newFakeDarwin(t, 1)
			f.updateStatus = update
			f.addStatus = add
			f.duplicateUpdateStatus = retry
			wantError(t, f.a.set(t.Context(), servicePrefix+"purpose", "id", []byte("fake")), want)
			if f.updates != wantUpdates || f.adds != wantAdds || f.deletes != 0 {
				t.Fatal("wrong replace sequence")
			}
		})
	}
	test("existing item", 0, 0, 0, nil, 1, 0)
	test("new item", errSecItemNotFound, 0, 0, nil, 1, 1)
	test("duplicate add races existing writer", errSecItemNotFound, errSecDuplicateItem, 0, nil, 2, 1)
	test("update denied", -25308, 0, 0, ErrUnavailable, 1, 0)
	test("add denied", errSecItemNotFound, -25308, 0, ErrUnavailable, 1, 1)
	test("duplicate retry denied", errSecItemNotFound, errSecDuplicateItem, -25308, ErrUnavailable, 2, 1)
	test("add missing is not absent", errSecItemNotFound, errSecItemNotFound, 0, ErrUnavailable, 1, 1)
}

func TestDarwinMissingAndUnavailableStatuses(t *testing.T) {
	for _, test := range []struct {
		name   string
		status int32
		want   error
	}{
		{"explicit missing", errSecItemNotFound, ErrNotFound}, {"locked", -25308, ErrUnavailable}, {"denied", -25293, ErrUnavailable}, {"unexpected", -1, ErrUnavailable},
	} {
		t.Run(test.name, func(t *testing.T) {
			f := newFakeDarwin(t, 1)
			f.copyStatus = test.status
			f.deleteStatus = test.status
			_, err := f.a.get(t.Context(), servicePrefix+"purpose", "id")
			wantError(t, err, test.want)
			wantError(t, f.a.delete(t.Context(), servicePrefix+"purpose", "id"), test.want)
		})
	}
}

func TestDarwinAllocationFailuresReleaseOwnership(t *testing.T) {
	for failure := 1; failure <= 10; failure++ {
		t.Run(string(rune('a'+failure-1)), func(t *testing.T) {
			f := newFakeDarwin(t, 1)
			f.failAllocation = failure
			f.updateStatus = errSecItemNotFound
			wantError(t, f.a.set(t.Context(), servicePrefix+"purpose", "id", []byte("fake")), ErrUnavailable)
		})
	}
}

func TestDarwinMalformedReadReleasesResult(t *testing.T) {
	for _, scenario := range []string{"missing result", "wrong CF type", "empty data", "oversized data", "null bytes"} {
		t.Run(scenario, func(t *testing.T) {
			f := newFakeDarwin(t, 1)
			switch scenario {
			case "missing result":
				f.missingResult = true
			case "wrong CF type":
				f.badType = true
			case "empty data":
				f.dataSize = 0
			case "oversized data":
				f.dataSize = 513
			case "null bytes":
				f.nilData = true
			}
			value, err := f.a.get(t.Context(), servicePrefix+"purpose", "id")
			wantError(t, err, ErrUnavailable)
			if value != nil {
				t.Fatal("malformed read returned data")
			}
		})
	}
}

func TestDarwinDefaultKeychainFailureDoesNotSearchOtherKeychains(t *testing.T) {
	for _, operation := range []string{"set", "get", "delete"} {
		t.Run(operation, func(t *testing.T) {
			f := newFakeDarwin(t, 1)
			f.defaultStatus = errSecItemNotFound
			var err error
			switch operation {
			case "set":
				err = f.a.set(t.Context(), servicePrefix+"purpose", "id", []byte("fake"))
			case "get":
				_, err = f.a.get(t.Context(), servicePrefix+"purpose", "id")
			case "delete":
				err = f.a.delete(t.Context(), servicePrefix+"purpose", "id")
			}
			wantError(t, err, ErrUnavailable)
			if f.items != 0 {
				t.Fatal("searched items after default keychain failure")
			}
		})
	}
}

func TestDarwinInteractionFailuresPreventItemAccessOrOverrideOutcome(t *testing.T) {
	for _, operation := range []string{"set", "get", "delete"} {
		for _, failure := range []string{"read setting", "disable", "restore"} {
			t.Run(operation+"/"+failure, func(t *testing.T) {
				f := newFakeDarwin(t, 1)
				switch failure {
				case "read setting":
					f.readStatus = -1
				case "disable":
					f.disableStatus = -1
				case "restore":
					f.restoreStatus = -1
				}
				var err error
				switch operation {
				case "set":
					err = f.a.set(t.Context(), servicePrefix+"purpose", "id", []byte("fake"))
				case "get":
					_, err = f.a.get(t.Context(), servicePrefix+"purpose", "id")
				case "delete":
					err = f.a.delete(t.Context(), servicePrefix+"purpose", "id")
				}
				wantError(t, err, ErrUnavailable)
				if failure != "restore" && f.items != 0 {
					t.Fatal("item called despite interaction guard failure")
				}
				if failure == "read setting" && f.settings != 0 {
					t.Fatal("unknown interaction state overwritten")
				}
				if failure != "read setting" && f.settings != 2 {
					t.Fatal("previous interaction setting not restored exactly once")
				}
			})
		}
	}
}

func TestDarwinConcurrentStoresShareInteractionGuard(t *testing.T) {
	f := newFakeDarwin(t, 1)
	var wg sync.WaitGroup
	for range 30 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			s := &Store{backend: f.a}
			wantError(t, s.Set(t.Context(), "purpose", "id", []byte("fake")), nil)
		}()
	}
	wg.Wait()
	if f.items != 30 || f.settings != 60 {
		t.Fatal("cross-Store interaction guard lost operations")
	}
}

func TestDarwinSynchronousMutationReportsCompletedOutcome(t *testing.T) {
	f := newFakeDarwin(t, 1)
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	update := f.a.update
	f.a.update = func(query, attrs uintptr) int32 { status := update(query, attrs); cancel(); return status }
	s := &Store{backend: f.a}
	wantError(t, s.Set(ctx, "purpose", "id", []byte("fake")), nil)
}
