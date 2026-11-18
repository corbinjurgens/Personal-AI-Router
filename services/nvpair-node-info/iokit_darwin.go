// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build darwin

package main

import (
	"bytes"
	"fmt"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
	"howett.net/plist"
)

type ioAcceleratorAPI struct {
	serviceMatching  func(string) uintptr
	matchingServices func(uint32, uintptr, *uint32) int32
	iteratorNext     func(uint32) uint32
	objectRelease    func(uint32) int32
	objectClass      func(uint32, *byte) int32
	entryName        func(uint32, *byte) int32
	entryID          func(uint32, *uint64) int32
	createProperties func(uint32, *uintptr, uintptr, uint32) int32
	propertyListData func(uintptr, uintptr, int, uintptr, *uintptr) uintptr
	dataLength       func(uintptr) int
	dataBytes        func(uintptr) *byte
	release          func(uintptr)
}

var loadIOAccelerator = sync.OnceValues(func() (*ioAcceleratorAPI, error) {
	io, err := purego.Dlopen("/System/Library/Frameworks/IOKit.framework/IOKit", purego.RTLD_NOW|purego.RTLD_LOCAL)
	if err != nil {
		return nil, fmt.Errorf("load IOKit: %w", err)
	}
	cf, err := purego.Dlopen("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation", purego.RTLD_NOW|purego.RTLD_LOCAL)
	if err != nil {
		return nil, fmt.Errorf("load CoreFoundation: %w", err)
	}
	// Framework handles stay loaded for process lifetime, like gopsutil's
	// Darwin bindings. Every read owns and releases its registry/CF objects.
	a := &ioAcceleratorAPI{}
	for _, binding := range []struct {
		library uintptr
		name    string
		target  any
	}{
		{io, "IOServiceMatching", &a.serviceMatching},
		{io, "IOServiceGetMatchingServices", &a.matchingServices},
		{io, "IOIteratorNext", &a.iteratorNext},
		{io, "IOObjectRelease", &a.objectRelease},
		{io, "IOObjectGetClass", &a.objectClass},
		{io, "IORegistryEntryGetName", &a.entryName},
		{io, "IORegistryEntryGetRegistryEntryID", &a.entryID},
		{io, "IORegistryEntryCreateCFProperties", &a.createProperties},
		{cf, "CFPropertyListCreateData", &a.propertyListData},
		{cf, "CFDataGetLength", &a.dataLength},
		{cf, "CFDataGetBytePtr", &a.dataBytes},
		{cf, "CFRelease", &a.release},
	} {
		symbol, err := purego.Dlsym(binding.library, binding.name)
		if err != nil {
			return nil, fmt.Errorf("bind %s: %w", binding.name, err)
		}
		purego.RegisterFunc(binding.target, symbol)
	}
	return a, nil
})

// readIOAcceleratorEntries returns one decoded entry per matched
// IOAccelerator service, with EntryID, EntryName and ObjectClass set from
// IOKit, as ioreg -a -r -d 1 -c IOAccelerator would report them.
func readIOAcceleratorEntries() ([]ioRegistryEntry, error) {
	a, err := loadIOAccelerator()
	if err != nil {
		return nil, err
	}
	matching := a.serviceMatching("IOAccelerator")
	if matching == 0 {
		return nil, fmt.Errorf("IOServiceMatching returned no dictionary")
	}
	var iterator uint32
	// IOServiceGetMatchingServices consumes matching on success and failure.
	status := a.matchingServices(0, matching, &iterator) // kIOMainPortDefault
	if iterator != 0 {
		defer a.objectRelease(iterator)
	}
	if status != 0 {
		return nil, fmt.Errorf("IOServiceGetMatchingServices: %d", status)
	}
	var entries []ioRegistryEntry
	for service := a.iteratorNext(iterator); service != 0; service = a.iteratorNext(iterator) {
		entry, err := a.readEntry(service)
		if err != nil {
			return nil, err
		}
		entries = append(entries, entry)
	}
	return entries, nil
}

func (a *ioAcceleratorAPI) readEntry(service uint32) (ioRegistryEntry, error) {
	defer a.objectRelease(service)
	var entry ioRegistryEntry
	var properties uintptr
	status := a.createProperties(service, &properties, 0, 0)
	if properties != 0 {
		defer a.release(properties)
	}
	if status != 0 || properties == 0 {
		return entry, fmt.Errorf("IORegistryEntryCreateCFProperties: %d", status)
	}
	var cfError uintptr
	data := a.propertyListData(0, properties, 200, 0, &cfError) // kCFPropertyListBinaryFormat_v1_0
	if cfError != 0 {
		defer a.release(cfError)
	}
	if data == 0 {
		return entry, fmt.Errorf("CFPropertyListCreateData failed")
	}
	defer a.release(data)
	length, ptr := a.dataLength(data), a.dataBytes(data)
	if length <= 0 || ptr == nil {
		return entry, fmt.Errorf("CFPropertyListCreateData returned empty data")
	}
	// Decode while the owning CFData is alive, through the recorded-plist
	// decoder, including its handling of the literal VRAM,totalMB key.
	if _, err := plist.Unmarshal(unsafe.Slice(ptr, length), &entry); err != nil {
		return entry, fmt.Errorf("decode IOAccelerator properties: %w", err)
	}
	if status := a.entryID(service, &entry.EntryID); status != 0 {
		return entry, fmt.Errorf("IORegistryEntryGetRegistryEntryID: %d", status)
	}
	var name, class [128]byte // io_name_t
	if status := a.entryName(service, &name[0]); status != 0 {
		return entry, fmt.Errorf("IORegistryEntryGetName: %d", status)
	}
	if status := a.objectClass(service, &class[0]); status != 0 {
		return entry, fmt.Errorf("IOObjectGetClass: %d", status)
	}
	entry.EntryName = string(bytes.TrimRight(name[:], "\x00"))
	entry.ObjectClass = string(bytes.TrimRight(class[:], "\x00"))
	return entry, nil
}
