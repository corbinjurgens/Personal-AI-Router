// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package buildinfo exposes only the executable's bounded public source identity.
package buildinfo

import (
	"bytes"
	filebuildinfo "debug/buildinfo"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"runtime/debug"
	"strings"
)

// MaxExecutableBytes bounds inspection of a staged executable.
const MaxExecutableBytes int64 = 512 << 20

const maxMetadataReadBytes int64 = 32 << 20
const maxMetadataRecords = 100000

// Metadata preserves independently unknown VCS fields. ID requires both fields;
// an artifact hash belongs to the package inventory, not to embedded VCS data.
type Metadata struct {
	ID       ID
	Revision *string
	Dirty    *bool
}

// InspectFile reads embedded Go VCS metadata from a regular executable.
// An empty ID means that revision or dirty state is unknown. The result is
// diagnostic metadata, not a hash or an authentication assertion.
func InspectFile(path string) (ID, error) {
	metadata, err := InspectMetadata(path)
	return metadata.ID, err
}

func InspectMetadata(path string) (Metadata, error) {
	info, err := os.Lstat(path)
	if err != nil {
		return Metadata{}, err
	}
	if !info.Mode().IsRegular() || info.Size() <= 0 || info.Size() > MaxExecutableBytes {
		return Metadata{}, fmt.Errorf("executable is not a bounded regular file")
	}
	file, err := os.Open(path)
	if err != nil {
		return Metadata{}, err
	}
	defer file.Close()
	return inspectOpened(path, file, info)
}

// InspectOpenedMetadata inspects a caller-owned, verified descriptor without
// reopening its pathname. The caller retains rooted path/inventory validation;
// this parser preserves the same regular-file, identity and read-budget bounds.
// It neither closes nor changes the descriptor's offset.
func InspectOpenedMetadata(file *os.File) (Metadata, error) {
	info, err := file.Stat()
	if err != nil {
		return Metadata{}, err
	}
	return inspectOpened("", file, info)
}

func inspectOpened(path string, file *os.File, info os.FileInfo) (Metadata, error) {
	opened, err := file.Stat()
	if err != nil {
		return Metadata{}, err
	}
	if !opened.Mode().IsRegular() || opened.Size() <= 0 || opened.Size() > MaxExecutableBytes || !os.SameFile(info, opened) ||
		opened.Size() != info.Size() || !opened.ModTime().Equal(info.ModTime()) {
		return Metadata{}, fmt.Errorf("executable changed during inspection")
	}
	// Root-approved ReadFile deviation: Read uses the same standard parser on
	// the held descriptor, avoiding a path reopen after identity validation.
	// Preflight caps header-driven allocations before entering that parser.
	reader := &metadataReader{reader: io.NewSectionReader(file, 0, opened.Size()), remaining: maxMetadataReadBytes}
	if err := checkObjectHeaders(reader); err != nil {
		return Metadata{}, err
	}
	build, readErr := filebuildinfo.Read(reader)
	after, statErr := file.Stat()
	pathAfter, pathErr := after, statErr
	if path != "" {
		pathAfter, pathErr = os.Lstat(path)
	}
	if statErr != nil || pathErr != nil || !os.SameFile(opened, after) || !os.SameFile(opened, pathAfter) ||
		!pathAfter.Mode().IsRegular() || opened.Size() != after.Size() || !opened.ModTime().Equal(after.ModTime()) ||
		opened.Size() != pathAfter.Size() || !opened.ModTime().Equal(pathAfter.ModTime()) || reader.exhausted {
		return Metadata{}, fmt.Errorf("executable changed or metadata read budget exhausted")
	}
	if readErr != nil {
		return Metadata{}, readErr
	}
	return metadataFromSettings(build.Settings), nil
}

type metadataReader struct {
	reader    *io.SectionReader
	remaining int64
	exhausted bool
}

func (r *metadataReader) ReadAt(p []byte, offset int64) (int, error) {
	if int64(len(p)) > r.remaining || offset < 0 || offset > r.reader.Size() || int64(len(p)) > r.reader.Size()-offset {
		r.exhausted = true
		return 0, io.ErrUnexpectedEOF
	}
	r.remaining -= int64(len(p))
	return r.reader.ReadAt(p, offset)
}

// The standard parsers use bounded chunk reads for string/symbol data, but some
// object headers directly allocate slices (notably Mach-O relocation tables).
// Check those counts first. Current 64-bit PAIR targets need no legacy formats.
func checkObjectHeaders(r *metadataReader) error {
	var header [64]byte
	if _, err := r.ReadAt(header[:], 0); err != nil {
		return err
	}
	bad := fmt.Errorf("unsupported or oversized executable metadata headers")
	le := binary.LittleEndian
	switch {
	case bytes.Equal(header[:2], []byte("MZ")):
		offset := int64(le.Uint32(header[60:]))
		var pe [24]byte
		if _, err := r.ReadAt(pe[:], offset); err != nil {
			return err
		}
		if !bytes.Equal(pe[:4], []byte{'P', 'E', 0, 0}) || le.Uint16(pe[6:]) > 4096 || le.Uint32(pe[16:]) > maxMetadataRecords {
			return bad
		}
	case bytes.Equal(header[:4], []byte{0x7f, 'E', 'L', 'F'}):
		if header[4] != 2 || header[5] != 1 {
			return bad
		}
		sections, programs := le.Uint16(header[60:]), le.Uint16(header[56:])
		if sections == 0 || sections > 4096 || programs > 4096 {
			return bad
		}
		entrySize, names := le.Uint16(header[58:]), le.Uint16(header[62:])
		if entrySize != 64 || names >= sections {
			return bad
		}
		table := le.Uint64(header[40:])
		if table > uint64(r.reader.Size()) || uint64(sections)*64 > uint64(r.reader.Size())-table {
			return bad
		}
		var nameHeader [64]byte
		if _, err := r.ReadAt(nameHeader[:], int64(table)+int64(names)*64); err != nil {
			return err
		}
		if le.Uint64(nameHeader[32:]) > 1<<20 || le.Uint64(nameHeader[8:])&0x800 != 0 {
			return bad
		}
	case le.Uint32(header[:4]) == 0xfeedfacf:
		commands, size := le.Uint32(header[16:]), le.Uint32(header[20:])
		if commands > 4096 || size > 1<<20 || int64(size) > r.reader.Size()-32 {
			return bad
		}
		offset := int64(32)
		end := offset + int64(size)
		for i := uint32(0); i < commands; i++ {
			var command [8]byte
			if _, err := r.ReadAt(command[:], offset); err != nil {
				return err
			}
			kind, length := le.Uint32(command[:4]), le.Uint32(command[4:])
			if length < 8 || int64(length) > end-offset {
				return bad
			}
			switch kind {
			case 0xb: // LC_DYSYMTAB, nindirectsyms is allocated directly.
				var data [80]byte
				if length < 80 {
					return bad
				}
				if _, err := r.ReadAt(data[:], offset); err != nil {
					return err
				}
				if le.Uint32(data[60:]) > maxMetadataRecords {
					return bad
				}
			case 0x19: // LC_SEGMENT_64 and each section's relocation count.
				var data [72]byte
				if length < 72 {
					return bad
				}
				if _, err := r.ReadAt(data[:], offset); err != nil {
					return err
				}
				n := le.Uint32(data[64:])
				if n > 4096 || uint64(n)*80+72 > uint64(length) {
					return bad
				}
				for j := uint32(0); j < n; j++ {
					var section [80]byte
					if _, err := r.ReadAt(section[:], offset+72+int64(j)*80); err != nil {
						return err
					}
					if le.Uint32(section[60:]) > maxMetadataRecords {
						return bad
					}
				}
			}
			offset += int64(length)
		}
		if offset != end {
			return bad
		}
	default:
		return bad
	}
	return nil
}

func metadataFromSettings(settings []debug.BuildSetting) Metadata {
	var metadata Metadata
	var vcs, revision, modified string
	for _, setting := range settings {
		switch setting.Key {
		case "vcs":
			vcs = setting.Value
		case "vcs.revision":
			revision = setting.Value
		case "vcs.modified":
			modified = setting.Value
		}
	}
	if vcs != "git" {
		return metadata
	}
	if (len(revision) == 40 || len(revision) == 64) && valid("git:"+revision) {
		metadata.Revision = &revision
	}
	if modified == "true" || modified == "false" {
		dirty := modified == "true"
		metadata.Dirty = &dirty
	}
	metadata.ID = fromSettings(settings)
	return metadata
}

// ID is an optional Git source revision, with +dirty for modified sources.
// It is diagnostic metadata, not an artifact hash or a trust assertion.
type ID string

// Current reads this executable's embedded Go VCS metadata. Unknown metadata
// remains empty rather than implying an unmodified source revision.
func Current() ID {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return ""
	}
	return fromSettings(info.Settings)
}

func fromSettings(settings []debug.BuildSetting) ID {
	var vcs, revision, modified string
	for _, setting := range settings {
		switch setting.Key {
		case "vcs":
			vcs = setting.Value
		case "vcs.revision":
			revision = setting.Value
		case "vcs.modified":
			modified = setting.Value
		}
	}
	if vcs != "git" || (modified != "true" && modified != "false") || (len(revision) != 40 && len(revision) != 64) {
		return ""
	}
	id := "git:" + revision
	if modified == "true" {
		id += "+dirty"
	}
	if !valid(id) {
		return ""
	}
	return ID(id)
}

func valid(value string) bool {
	if len(value) > 74 || !strings.HasPrefix(value, "git:") {
		return false
	}
	revision := strings.TrimSuffix(value[4:], "+dirty")
	if len(revision) != 40 && len(revision) != 64 {
		return false
	}
	for _, c := range revision {
		if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f') {
			return false
		}
	}
	return true
}

// UnmarshalJSON isolates malformed optional metadata from its enclosing node.
// The encoded bound permits JSON Unicode escapes for every valid ASCII byte.
func (id *ID) UnmarshalJSON(data []byte) error {
	*id = ""
	if len(data) > 74*6+2 {
		return nil
	}
	var value string
	if err := json.Unmarshal(data, &value); err != nil {
		return nil
	}
	if valid(value) {
		*id = ID(value)
	}
	return nil
}
