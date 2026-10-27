// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build darwin

package secretstore

import (
	"context"
	"runtime"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

const (
	securityFramework             = "/System/Library/Frameworks/Security.framework/Security"
	coreFoundationFramework       = "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
	errSecItemNotFound      int32 = -25300
	errSecDuplicateItem     int32 = -25299
)

// The legacy file-based keychain ignores the query UI flag. Its interaction
// switch is process-wide, so every Store shares this lock through restoration.
var darwinInteractionMu sync.Mutex
var darwinLoadMu sync.Mutex
var loadedDarwin *darwinAPI

type darwinAPI struct {
	copyDefault      func(*uintptr) int32
	arrayCreate      func(uintptr, *uintptr, int, uintptr) uintptr
	copyMatching     func(uintptr, *uintptr) int32
	add              func(uintptr, *uintptr) int32
	update           func(uintptr, uintptr) int32
	remove           func(uintptr) int32
	interactionGet   func(*uint8) int32
	interactionSet   func(uint8) int32
	stringCreate     func(uintptr, *byte, int, uint32, uint8) uintptr
	dataCreate       func(uintptr, *byte, int) uintptr
	dataLength       func(uintptr) int
	dataBytes        func(uintptr) *byte
	typeID           func(uintptr) uintptr
	dataTypeID       func() uintptr
	dictionaryCreate func(uintptr, int, uintptr, uintptr) uintptr
	dictionarySet    func(uintptr, uintptr, uintptr)
	release          func(uintptr)
	constants        map[string]uintptr
}

func newNative() (backend, error) {
	darwinLoadMu.Lock()
	defer darwinLoadMu.Unlock()
	if loadedDarwin != nil {
		return loadedDarwin, nil
	}
	a, err := loadDarwin()
	if err != nil {
		return nil, ErrUnavailable
	}
	loadedDarwin = a
	return a, nil
}

func loadDarwin() (*darwinAPI, error) {
	sec, err := purego.Dlopen(securityFramework, purego.RTLD_NOW|purego.RTLD_LOCAL)
	if err != nil {
		return nil, ErrUnavailable
	}
	cf, err := purego.Dlopen(coreFoundationFramework, purego.RTLD_NOW|purego.RTLD_LOCAL)
	if err != nil {
		_ = purego.Dlclose(sec)
		return nil, ErrUnavailable
	}
	ok := false
	defer func() {
		if !ok {
			_ = purego.Dlclose(cf)
			_ = purego.Dlclose(sec)
		}
	}()
	a := &darwinAPI{}
	bindings := []struct {
		library uintptr
		name    string
		target  any
	}{
		{sec, "SecKeychainCopyDefault", &a.copyDefault}, {cf, "CFArrayCreate", &a.arrayCreate},
		{sec, "SecItemCopyMatching", &a.copyMatching}, {sec, "SecItemAdd", &a.add},
		{sec, "SecItemUpdate", &a.update}, {sec, "SecItemDelete", &a.remove},
		{sec, "SecKeychainGetUserInteractionAllowed", &a.interactionGet},
		{sec, "SecKeychainSetUserInteractionAllowed", &a.interactionSet},
		{cf, "CFStringCreateWithBytes", &a.stringCreate}, {cf, "CFDataCreate", &a.dataCreate},
		{cf, "CFDataGetLength", &a.dataLength}, {cf, "CFDataGetBytePtr", &a.dataBytes},
		{cf, "CFGetTypeID", &a.typeID}, {cf, "CFDataGetTypeID", &a.dataTypeID},
		{cf, "CFDictionaryCreateMutable", &a.dictionaryCreate},
		{cf, "CFDictionarySetValue", &a.dictionarySet}, {cf, "CFRelease", &a.release},
	}
	for _, binding := range bindings {
		p, err := purego.Dlsym(binding.library, binding.name)
		if err != nil || p == 0 {
			return nil, ErrUnavailable
		}
		purego.RegisterFunc(binding.target, p)
	}
	// Dlsym returns the address of an exported data variable. Read a CFTypeRef
	// through that address once. Callback symbols are structs, so use their
	// addresses directly. memcpy avoids uintptr-to-Go-pointer conversions.
	p, err := purego.Dlsym(cf, "memcpy")
	if err != nil || p == 0 {
		return nil, ErrUnavailable
	}
	var copyPointer func(*uintptr, uintptr, uintptr) uintptr
	purego.RegisterFunc(&copyPointer, p)
	a.constants, err = darwinConstants(func(security bool, name string) (uintptr, error) {
		lib := cf
		if security {
			lib = sec
		}
		return purego.Dlsym(lib, name)
	}, func(address uintptr) uintptr {
		var value uintptr
		copyPointer(&value, address, unsafe.Sizeof(value))
		return value
	})
	if err != nil {
		return nil, err
	}
	// Successful bindings retain their framework handles for process lifetime.
	ok = true
	return a, nil
}

func darwinConstants(lookup func(bool, string) (uintptr, error), read func(uintptr) uintptr) (map[string]uintptr, error) {
	constants := make(map[string]uintptr)
	for _, name := range []string{
		"kSecClass", "kSecClassGenericPassword", "kSecAttrService", "kSecAttrAccount",
		"kSecAttrSynchronizable", "kSecUseAuthenticationUI", "kSecUseAuthenticationUIFail",
		"kSecReturnData", "kSecMatchLimit", "kSecMatchLimitOne", "kSecValueData",
		"kSecMatchSearchList", "kSecUseKeychain", "kCFTypeArrayCallBacks",
		"kCFBooleanTrue", "kCFBooleanFalse", "kCFTypeDictionaryKeyCallBacks", "kCFTypeDictionaryValueCallBacks",
	} {
		p, err := lookup(name[:4] == "kSec", name)
		if err != nil || p == 0 {
			return nil, ErrUnavailable
		}
		if name != "kCFTypeDictionaryKeyCallBacks" && name != "kCFTypeDictionaryValueCallBacks" && name != "kCFTypeArrayCallBacks" {
			p = read(p)
		}
		if p == 0 {
			return nil, ErrUnavailable
		}
		constants[name] = p
	}
	return constants, nil
}

func (a *darwinAPI) withoutInteraction(ctx context.Context, op func() error) (err error) {
	darwinInteractionMu.Lock()
	defer darwinInteractionMu.Unlock()
	if err := ctx.Err(); err != nil {
		return err
	}
	var previous uint8
	if a.interactionGet(&previous) != 0 {
		return ErrUnavailable
	}
	defer func() {
		if a.interactionSet(previous) != 0 {
			err = ErrUnavailable
		}
	}()
	if a.interactionSet(0) != 0 {
		return ErrUnavailable
	}
	return op()
}

type cfScope struct {
	api        *darwinAPI
	owned      []uintptr
	keychain   uintptr
	searchList uintptr
}

func (s *cfScope) own(p uintptr) uintptr {
	if p != 0 {
		s.owned = append(s.owned, p)
	}
	return p
}

func (s *cfScope) close() {
	for i := len(s.owned) - 1; i >= 0; i-- {
		s.api.release(s.owned[i])
	}
}

func (s *cfScope) dictionary() uintptr {
	a := s.api
	return s.own(a.dictionaryCreate(0, 0, a.constants["kCFTypeDictionaryKeyCallBacks"], a.constants["kCFTypeDictionaryValueCallBacks"]))
}

func (s *cfScope) text(value string) uintptr {
	b := []byte(value)
	p := s.own(s.api.stringCreate(0, unsafe.SliceData(b), len(b), 0x08000100, 0)) // kCFStringEncodingUTF8
	runtime.KeepAlive(b)
	return p
}

func (s *cfScope) data(value []byte) uintptr {
	p := s.own(s.api.dataCreate(0, unsafe.SliceData(value), len(value)))
	runtime.KeepAlive(value)
	return p
}

func (s *cfScope) put(dict uintptr, key string, value uintptr) {
	s.api.dictionarySet(dict, s.api.constants[key], value)
}

func (s *cfScope) query(service, account string) uintptr {
	return s.entry(service, account, false)
}

func (s *cfScope) entry(service, account string, creation bool) uintptr {
	if s.keychain == 0 {
		var keychain uintptr
		status := s.api.copyDefault(&keychain)
		s.own(keychain)
		if status != 0 || keychain == 0 {
			return 0
		}
		s.keychain = keychain
		s.searchList = s.own(s.api.arrayCreate(0, &keychain, 1, s.api.constants["kCFTypeArrayCallBacks"]))
		if s.searchList == 0 {
			return 0
		}
	}
	dict := s.dictionary()
	serviceRef, accountRef := s.text(service), s.text(account)
	if dict == 0 || serviceRef == 0 || accountRef == 0 {
		return 0
	}
	s.put(dict, "kSecClass", s.api.constants["kSecClassGenericPassword"])
	s.put(dict, "kSecAttrService", serviceRef)
	s.put(dict, "kSecAttrAccount", accountRef)
	s.put(dict, "kSecAttrSynchronizable", s.api.constants["kCFBooleanFalse"])
	s.put(dict, "kSecUseAuthenticationUI", s.api.constants["kSecUseAuthenticationUIFail"])
	if creation {
		s.put(dict, "kSecUseKeychain", s.keychain)
	} else {
		s.put(dict, "kSecMatchSearchList", s.searchList)
	}
	return dict
}

func (a *darwinAPI) set(ctx context.Context, service, account string, value []byte) error {
	return a.withoutInteraction(ctx, func() error {
		s := cfScope{api: a}
		defer s.close()
		query, attrs, data := s.query(service, account), s.dictionary(), s.data(value)
		if query == 0 || attrs == 0 || data == 0 {
			return ErrUnavailable
		}
		s.put(attrs, "kSecValueData", data)
		status := a.update(query, attrs)
		if status == errSecItemNotFound {
			creation := s.entry(service, account, true)
			if creation == 0 {
				return ErrUnavailable
			}
			s.put(creation, "kSecValueData", data)
			status = a.add(creation, nil)
			if status == errSecDuplicateItem {
				status = a.update(query, attrs)
			}
		}
		if status != 0 {
			return ErrUnavailable
		}
		return nil
	})
}

func (a *darwinAPI) get(ctx context.Context, service, account string) (value []byte, err error) {
	err = a.withoutInteraction(ctx, func() error {
		s := cfScope{api: a}
		defer s.close()
		query := s.query(service, account)
		if query == 0 {
			return ErrUnavailable
		}
		s.put(query, "kSecReturnData", a.constants["kCFBooleanTrue"])
		s.put(query, "kSecMatchLimit", a.constants["kSecMatchLimitOne"])
		var result uintptr
		status := a.copyMatching(query, &result)
		s.own(result)
		if status == errSecItemNotFound {
			return ErrNotFound
		}
		if status != 0 || result == 0 || a.typeID(result) != a.dataTypeID() {
			return ErrUnavailable
		}
		n := a.dataLength(result)
		if n <= 0 || n > maxSecretBytes {
			return ErrUnavailable
		}
		p := a.dataBytes(result)
		if p == nil {
			return ErrUnavailable
		}
		value = append([]byte(nil), unsafe.Slice(p, n)...)
		return nil
	})
	if err != nil {
		return nil, err
	}
	return value, nil
}

func (a *darwinAPI) delete(ctx context.Context, service, account string) error {
	return a.withoutInteraction(ctx, func() error {
		s := cfScope{api: a}
		defer s.close()
		query := s.query(service, account)
		if query == 0 {
			return ErrUnavailable
		}
		switch a.remove(query) {
		case 0:
			return nil
		case errSecItemNotFound:
			return ErrNotFound
		default:
			return ErrUnavailable
		}
	})
}
