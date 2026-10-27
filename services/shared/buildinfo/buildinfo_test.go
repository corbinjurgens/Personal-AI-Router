// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package buildinfo

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"runtime/debug"
	"strings"
	"testing"

	"nvpair-shared/durablefile"
)

func TestOpenedMetadataDoesNotReadSubstitutedPath(t *testing.T) {
	path := filepath.Join(t.TempDir(), "candidate")
	const revision = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	data := metadataFixture(t, "build\tvcs=git\nbuild\tvcs.revision="+revision+"\nbuild\tvcs.modified=false\n")
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	file, err := durablefile.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	if err := durablefile.WriteFile(path, []byte("substituted non-Go bytes"), 0600); err != nil {
		t.Fatal(err)
	}
	got, err := InspectOpenedMetadata(file)
	if err != nil {
		t.Fatal(err)
	}
	if got.Revision == nil || *got.Revision != revision {
		t.Fatalf("metadata did not come from held descriptor: %+v", got)
	}
}

// A tiny inert ELF with a real inline Go build-info section, never compiled or
// executed. The standard parser still decodes the actual embedded metadata.
func metadataFixture(t *testing.T, settings string) []byte {
	t.Helper()
	framed := "0123456789abcdef" + "path\tfixture\n" + settings + "0123456789abcdef"
	blob := make([]byte, 32)
	copy(blob, []byte("\xff Go buildinf:"))
	blob[14], blob[15] = 8, 2
	blob = binary.AppendUvarint(blob, uint64(len("go1.25.0")))
	blob = append(blob, "go1.25.0"...)
	blob = binary.AppendUvarint(blob, uint64(len(framed)))
	blob = append(blob, framed...)
	names := []byte("\x00.go.buildinfo\x00.shstrtab\x00")
	data := make([]byte, 312)
	copy(data, []byte{0x7f, 'E', 'L', 'F', 2, 1, 1})
	le := binary.LittleEndian
	le.PutUint16(data[16:], 2)
	le.PutUint16(data[18:], 62)
	le.PutUint32(data[20:], 1)
	le.PutUint64(data[32:], 64)
	le.PutUint64(data[40:], 120)
	le.PutUint16(data[52:], 64)
	le.PutUint16(data[54:], 56)
	le.PutUint16(data[56:], 1)
	le.PutUint16(data[58:], 64)
	le.PutUint16(data[60:], 3)
	le.PutUint16(data[62:], 2)
	program := data[64:120]
	le.PutUint32(program, 1) // PT_LOAD makes the build-info address readable.
	le.PutUint32(program[4:], 6)
	le.PutUint64(program[8:], 312)
	le.PutUint64(program[16:], 4096)
	le.PutUint64(program[24:], 4096)
	le.PutUint64(program[32:], uint64(len(blob)))
	le.PutUint64(program[40:], uint64(len(blob)))
	le.PutUint64(program[48:], 16)
	section := data[184:248]
	le.PutUint32(section, 1)
	le.PutUint32(section[4:], 1)
	le.PutUint64(section[8:], 3)
	le.PutUint64(section[16:], 4096)
	le.PutUint64(section[24:], 312)
	le.PutUint64(section[32:], uint64(len(blob)))
	le.PutUint64(section[48:], 16)
	section = data[248:312]
	le.PutUint32(section, 15)
	le.PutUint32(section[4:], 3)
	le.PutUint64(section[24:], uint64(312+len(blob)))
	le.PutUint64(section[32:], uint64(len(names)))
	data = append(data, blob...)
	return append(data, names...)
}

func TestEmbeddedFileIdentityKeepsMissingVCSUnknown(t *testing.T) {
	path := filepath.Join(t.TempDir(), "inert-service")
	data := metadataFixture(t, "")
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	got, err := InspectMetadata(path)
	if err != nil {
		t.Fatalf("inspect inert fixture: %v", err)
	}
	if got.ID != "" || got.Revision != nil || got.Dirty != nil {
		t.Fatalf("missing VCS = %+v", got)
	}
	// The artifact remains measurable independently of the absent source stamp.
	hash := sha256.Sum256(data)
	if hash == ([sha256.Size]byte{}) {
		t.Fatal("missing artifact hash")
	}
	id, err := InspectFile(path)
	if err != nil || id != "" {
		t.Fatalf("optional ID = %q, err %v", id, err)
	}
}

func TestMetadataRevisionDoesNotAcceptBuildIDSuffix(t *testing.T) {
	got := metadataFromSettings([]debug.BuildSetting{
		{Key: "vcs", Value: "git"},
		{Key: "vcs.revision", Value: "0123456789abcdef0123456789abcdef01234567+dirty"},
		{Key: "vcs.modified", Value: "false"},
	})
	if got.ID != "" || got.Revision != nil {
		t.Fatalf("malformed revision = %+v", got)
	}
}

func TestEmbeddedFileIdentityPreservesActualSourceFields(t *testing.T) {
	const revision = "0123456789abcdef0123456789abcdef01234567"
	test := func(name, modified string, wantDirty *bool, wantID ID) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			settings := "build\tvcs=git\nbuild\tvcs.revision=" + revision + "\n"
			if modified != "" {
				settings += "build\tvcs.modified=" + modified + "\n"
			}
			path := filepath.Join(t.TempDir(), "inert-service")
			if err := os.WriteFile(path, metadataFixture(t, settings), 0600); err != nil {
				t.Fatal(err)
			}
			got, err := InspectMetadata(path)
			if err != nil {
				t.Fatal(err)
			}
			if got.ID != wantID || got.Revision == nil || *got.Revision != revision {
				t.Fatalf("source = %+v", got)
			}
			if wantDirty == nil {
				if got.Dirty != nil {
					t.Fatalf("invented dirty state %v", *got.Dirty)
				}
			} else if got.Dirty == nil || *got.Dirty != *wantDirty {
				t.Fatalf("dirty = %v, want %v", got.Dirty, *wantDirty)
			}
		})
	}
	clean, dirty := false, true
	test("clean", "false", &clean, ID("git:"+revision))
	test("dirty", "true", &dirty, ID("git:"+revision+"+dirty"))
	test("missing dirty does not mean clean", "", nil, "")
}

func TestMetadataReaderEnforcesReadBudgetAndDescriptorRange(t *testing.T) {
	data := bytes.NewReader(make([]byte, 128))
	reader := &metadataReader{reader: io.NewSectionReader(data, 0, data.Size()), remaining: 16}
	if _, err := reader.ReadAt(make([]byte, 16), 0); err != nil {
		t.Fatal(err)
	}
	if _, err := reader.ReadAt(make([]byte, 1), 0); err == nil || !reader.exhausted {
		t.Fatal("read budget not enforced")
	}
	reader = &metadataReader{reader: io.NewSectionReader(data, 0, data.Size()), remaining: 1024}
	if _, err := reader.ReadAt(make([]byte, 2), 127); err == nil || !reader.exhausted {
		t.Fatal("descriptor range not enforced")
	}
}

func TestMetadataInspectionRejectsNonRegularAndOversizedFiles(t *testing.T) {
	root := t.TempDir()
	if _, err := InspectMetadata(root); err == nil {
		t.Fatal("accepted a directory")
	}
	path := filepath.Join(root, "oversized")
	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := file.Truncate(MaxExecutableBytes + 1); err != nil {
		file.Close()
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := InspectMetadata(path); err == nil {
		t.Fatal("accepted an oversized regular file")
	}
}

func TestMetadataInspectionRejectsSubstitutedPath(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "candidate")
	data := metadataFixture(t, "")
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		t.Fatal(err)
	}
	// Supply a different path after opening, without relying on the host's
	// ability to rename an open executable. The held descriptor is unchanged.
	replacement := filepath.Join(root, "replacement")
	if err := os.WriteFile(replacement, data, 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := inspectOpened(replacement, file, info); err == nil {
		t.Fatal("accepted substituted candidate path")
	}
}

func TestMetadataInspectionRejectsMalformedAllocationHeaders(t *testing.T) {
	test := func(name string, data []byte) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "inert")
			if err := os.WriteFile(path, data, 0600); err != nil {
				t.Fatal(err)
			}
			if _, err := InspectMetadata(path); err == nil {
				t.Fatal("accepted malformed metadata")
			}
		})
	}
	elf := metadataFixture(t, "")
	binary.LittleEndian.PutUint16(elf[60:], 0xffff)
	test("ELF section count", elf)
	pe := make([]byte, 128)
	copy(pe, "MZ")
	binary.LittleEndian.PutUint32(pe[60:], 64)
	copy(pe[64:], []byte{'P', 'E', 0, 0})
	binary.LittleEndian.PutUint32(pe[80:], 0xffffffff)
	test("PE symbol count", pe)
	macho := make([]byte, 128)
	binary.LittleEndian.PutUint32(macho, 0xfeedfacf)
	binary.LittleEndian.PutUint32(macho[16:], 1)
	binary.LittleEndian.PutUint32(macho[20:], 80)
	binary.LittleEndian.PutUint32(macho[32:], 0xb)
	binary.LittleEndian.PutUint32(macho[36:], 80)
	binary.LittleEndian.PutUint32(macho[92:], 0xffffffff)
	test("Mach-O indirect symbol allocation", macho)
	test("truncated header", []byte("MZ"))
	test("not executable", []byte(strings.Repeat("inert", 32)))
}

func TestFromSettingsRequiresCompleteGitMetadata(t *testing.T) {
	test := func(name, vcs, revision, modified string, want ID) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			settings := []debug.BuildSetting{
				{Key: "vcs", Value: vcs},
				{Key: "vcs.revision", Value: revision},
				{Key: "vcs.modified", Value: modified},
				{Key: "path", Value: "private/path"},
			}
			if got := fromSettings(settings); got != want {
				t.Fatalf("identity = %q, want %q", got, want)
			}
		})
	}
	const sha1 = "0123456789abcdef0123456789abcdef01234567"
	const sha256 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	test("clean SHA-1", "git", sha1, "false", "git:0123456789abcdef0123456789abcdef01234567")
	test("dirty SHA-1", "git", sha1, "true", "git:0123456789abcdef0123456789abcdef01234567+dirty")
	test("clean SHA-256", "git", sha256, "false", "git:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")
	test("dirty SHA-256", "git", sha256, "true", "git:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef+dirty")
	test("missing VCS", "", sha1, "false", "")
	test("non Git", "hg", sha1, "false", "")
	test("missing revision", "git", "", "false", "")
	test("missing modification state", "git", sha1, "", "")
	test("invalid modification state", "git", sha1, "False", "")
	test("uppercase revision", "git", strings.ToUpper(sha1), "false", "")
	test("abbreviated revision", "git", "01234567", "false", "")
	test("non hex revision", "git", "g123456789abcdef0123456789abcdef01234567", "false", "")
	test("revision with dirty suffix", "git", sha1+"+dirty", "false", "")
	if got := fromSettings(nil); got != "" {
		t.Fatalf("missing settings = %q, want unknown", got)
	}
}

func TestOptionalIDDecoderPreservesNeighborFields(t *testing.T) {
	const clean = "git:0123456789abcdef0123456789abcdef01234567"
	test := func(name, raw string, want ID) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			value := struct {
				NodeID string `json:"nodeId"`
				Build  ID     `json:"serviceBuildId"`
			}{Build: ID(clean)}
			if err := json.Unmarshal([]byte(`{"nodeId":"neighbor","serviceBuildId":`+raw+`}`), &value); err != nil {
				t.Fatalf("decode optional field: %v", err)
			}
			if value.NodeID != "neighbor" || value.Build != want {
				t.Fatalf("decoded = %+v, want node neighbor and build %q", value, want)
			}
		})
	}
	test("clean", `"`+clean+`"`, ID(clean))
	test("dirty", `"`+clean+`+dirty"`, "git:0123456789abcdef0123456789abcdef01234567+dirty")
	test("escaped ASCII", `"\u0067it:0123456789abcdef0123456789abcdef01234567"`, ID(clean))
	test("null clears previous", `null`, "")
	test("number", `42`, "")
	test("boolean", `true`, "")
	test("object", `{"revision":"ignored"}`, "")
	test("array", `[]`, "")
	test("empty", `""`, "")
	test("uppercase", `"git:0123456789ABCDEF0123456789abcdef01234567"`, "")
	test("wrong prefix", `"svn:0123456789abcdef0123456789abcdef01234567"`, "")
	test("trailing whitespace", `"`+clean+` "`, "")
	test("escaped invalid character", `"git:0123456789abcdef0123456789abcdef0123456\u0067"`, "")
	test("oversize decoded", `"git:`+strings.Repeat("a", 65)+`+dirty"`, "")
	test("oversize encoded", `"`+strings.Repeat("a", 445)+`"`, "")
	const maxID = "git:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef+dirty"
	test("maximum length", `"`+maxID+`"`, ID(maxID))
	var escaped strings.Builder
	escaped.WriteByte('"')
	for _, c := range maxID {
		const hex = "0123456789abcdef"
		escaped.WriteString(`\u00`)
		escaped.WriteByte(hex[c>>4])
		escaped.WriteByte(hex[c&15])
	}
	escaped.WriteByte('"')
	test("maximum escaped length", escaped.String(), ID(maxID))
}

func TestOptionalIDMissingIsZeroAndOmitted(t *testing.T) {
	var value struct {
		Build ID `json:"serviceBuildId,omitempty"`
	}
	if err := json.Unmarshal([]byte(`{}`), &value); err != nil {
		t.Fatal(err)
	}
	if value.Build != "" {
		t.Fatalf("missing identity = %q", value.Build)
	}
	encoded, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	if string(encoded) != `{}` {
		t.Fatalf("unknown identity encoded as %s, want {}", encoded)
	}
}

func TestOptionalIDDoesNotHideInvalidOuterJSON(t *testing.T) {
	var value struct {
		Build ID `json:"serviceBuildId"`
	}
	if err := json.Unmarshal([]byte(`{"serviceBuildId":"bad",`), &value); err == nil {
		t.Fatal("malformed enclosing JSON was accepted")
	}
}
