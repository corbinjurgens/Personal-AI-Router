// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package buildapply

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// This ELF contains an inline Go build-info section. It is synthetic data,
// never compiled, loaded or executed, including when these tests run on Windows.
func packageELF(settings string) []byte {
	framed := "0123456789abcdefpath\tfixture\n" + settings + "0123456789abcdef"
	blob := make([]byte, 32)
	copy(blob, "\xff Go buildinf:")
	blob[14], blob[15] = 8, 2
	blob = binary.AppendUvarint(blob, 8)
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
	le.PutUint32(program, 1)
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
	return append(append(data, blob...), names...)
}

func packageWrite(t *testing.T, path string, data []byte) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0700); err != nil {
		t.Fatal(err)
	}
}

func packageFixture(t *testing.T, settings string) string {
	t.Helper()
	root, err := filepath.EvalSymlinks(packageFixtureTempDir(t))
	if err != nil {
		t.Fatal(err)
	}
	image := packageELF(settings)
	for _, name := range electronRuntimeFiles("linux") {
		packageWrite(t, filepath.Join(root, filepath.FromSlash(name)), []byte("inert runtime"))
	}
	packageWrite(t, filepath.Join(root, "nvpair"), image)
	packageWrite(t, filepath.Join(root, "resources", "app.asar"), packageASAR(t, `function guiFacts(){return {buildId:"git:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",pid:42}}`))
	type file struct {
		FileName string `json:"fileName"`
		Size     int64  `json:"size"`
		SHA256   string `json:"sha256"`
	}
	files := []file{}
	for _, name := range bundledPrograms {
		packageWrite(t, filepath.Join(root, "resources", "cli-bin", name), image)
		hash := sha256.Sum256(image)
		files = append(files, file{name, int64(len(image)), hex.EncodeToString(hash[:])})
	}
	data, err := json.Marshal(struct {
		Source, Platform, Arch string
		Files                  []file
	}{"services-build", "linux", "x64", files})
	if err != nil {
		t.Fatal(err)
	}
	packageWrite(t, filepath.Join(root, "resources", "cli-bin", "manifest.json"), data)
	return root
}

func packageASAR(t *testing.T, main string) []byte {
	t.Helper()
	archive := asarEntry{Files: map[string]asarEntry{}}
	files := []struct{ path, value string }{{"package.json", `{"main":"./out/main/index.js"}`}, {"out/main/index.js", main}, {"out/preload/index.js", "inert preload"}, {"out/ui/index.html", "inert HTML"}}
	var payload []byte
	for _, file := range files {
		node := archive.Files
		parts := strings.Split(file.path, "/")
		for _, part := range parts[:len(parts)-1] {
			next, ok := node[part]
			if !ok {
				next = asarEntry{Files: map[string]asarEntry{}}
				node[part] = next
			}
			node = next.Files
		}
		node[parts[len(parts)-1]] = asarEntry{Offset: strconv.Itoa(len(payload)), Size: int64(len(file.value))}
		payload = append(payload, file.value...)
	}
	header, err := json.Marshal(archive)
	if err != nil {
		t.Fatal(err)
	}
	headerSize := (8 + len(header) + 3) &^ 3
	data := make([]byte, 8+headerSize)
	le := binary.LittleEndian
	le.PutUint32(data, 4)
	le.PutUint32(data[4:], uint32(headerSize))
	le.PutUint32(data[8:], uint32(headerSize-4))
	le.PutUint32(data[12:], uint32(len(header)))
	copy(data[16:], header)
	return append(data, payload...)
}

func TestPackageMissingVCSKeepsActualArtifactHash(t *testing.T) {
	root := packageFixture(t, "")
	p, err := InspectPackage(context.Background(), root, "linux", "amd64")
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	if p.Service.SourceRevision.Known || p.Service.Dirty.Known || p.Service.BuildID.Known {
		t.Fatalf("invented VCS identity: %+v", p.Service)
	}
	if !p.GUI.BuildID.Known || *p.GUI.BuildID.Value != "git:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" {
		t.Fatalf("independent GUI stamp = %+v", p.GUI)
	}
	hash := sha256.Sum256(packageELF(""))
	if !p.Service.SHA256.Known || *p.Service.SHA256.Value != hex.EncodeToString(hash[:]) {
		t.Fatalf("artifact hash = %+v", p.Service.SHA256)
	}
	if p.SHA256 == "" || p.Bytes == 0 {
		t.Fatal("package bytes were not measured")
	}
}

func TestPackageMissingElectronRuntimeRefuses(t *testing.T) {
	for _, name := range electronRuntimeFiles("linux") {
		t.Run(name, func(t *testing.T) {
			root := packageFixture(t, "")
			if err := os.Remove(filepath.Join(root, filepath.FromSlash(name))); err != nil {
				t.Fatal(err)
			}
			p, err := InspectPackage(context.Background(), root, "linux", "amd64")
			if err == nil {
				p.Close()
				t.Fatal("incomplete Electron distribution accepted")
			}
			if !strings.Contains(err.Error(), "missing regular Electron runtime file") {
				t.Fatalf("refusal = %v", err)
			}
		})
	}
}

func TestPackagePresentVCSRetainsActualMetadata(t *testing.T) {
	const revision = "0123456789abcdef0123456789abcdef01234567"
	root := packageFixture(t, "build\tvcs=git\nbuild\tvcs.revision="+revision+"\nbuild\tvcs.modified=true\n")
	p, err := InspectPackage(context.Background(), root, "linux", "amd64")
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	if !p.Service.SourceRevision.Known || *p.Service.SourceRevision.Value != revision || !p.Service.Dirty.Known || !*p.Service.Dirty.Value || !p.Service.BuildID.Known || *p.Service.BuildID.Value != "git:"+revision+"+dirty" {
		t.Fatalf("actual metadata = %+v", p.Service)
	}
}

func TestPackageForgedManifestCannotHideActualBytes(t *testing.T) {
	root := packageFixture(t, "")
	changed := packageELF("")
	changed[15] = 1 // Same-size ELF padding change.
	packageWrite(t, filepath.Join(root, "resources", "cli-bin", "nvpair-service"), changed)
	p, err := InspectPackage(context.Background(), root, "linux", "amd64")
	if err == nil {
		p.Close()
		t.Fatal("accepted manifest with forged content hash")
	}
	if !strings.Contains(err.Error(), "manifest differs from measured package bytes") {
		t.Fatalf("did not compare actual content hash: %v", err)
	}
}

func TestPackageRepeatedInspectionIsReadOnly(t *testing.T) {
	root := packageFixture(t, "")
	first, err := InspectPackage(context.Background(), root, "linux", "amd64")
	if err != nil {
		t.Fatal(err)
	}
	defer first.Close()
	for i := 0; i < 2; i++ {
		p, err := InspectPackage(context.Background(), root, "linux", "amd64")
		if err != nil {
			t.Fatal(err)
		}
		if first.SHA256 != p.SHA256 || len(first.Entries) != len(p.Entries) {
			p.Close()
			t.Fatal("preview changed package inventory or bytes")
		}
		for j, entry := range first.Entries {
			next := p.Entries[j]
			if entry.Path != next.Path || entry.SHA256 != next.SHA256 || !sameIdentity(entry.info, next.info) {
				p.Close()
				t.Fatal("preview changed file identity or bytes")
			}
		}
		p.Close()
	}
	if err := first.Revalidate(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestPackageBoundsWithSmallFixtures(t *testing.T) {
	cases := []struct {
		name   string
		limits packageLimits
		want   string
	}{
		{"single file", packageLimits{1024, 3, 100, 4096}, "byte limit"},
		{"total bytes", packageLimits{3, 1024, 100, 4096}, "byte limit"},
		{"entries", packageLimits{1024, 1024, 1, 4096}, "entry limit"},
		{"UTF8 relative path", packageLimits{1024, 1024, 100, 3}, "relative package path"},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			root := packageFixtureTempDir(t)
			packageWrite(t, filepath.Join(root, "éé"), []byte("four"))
			if test.limits.path != 3 {
				packageWrite(t, filepath.Join(root, "other"), []byte("four"))
			}
			real, err := filepath.EvalSymlinks(root)
			if err != nil {
				t.Fatal(err)
			}
			p, err := inspectPackage(context.Background(), real, "linux", "amd64", test.limits)
			if err == nil {
				p.Close()
				t.Fatal("oversized package accepted")
			}
			if !strings.Contains(err.Error(), test.want) {
				t.Fatalf("refusal = %v", err)
			}
		})
	}
}

func TestPackageWrongPlatformAndArchitectureRefuse(t *testing.T) {
	for _, test := range []struct{ name, platform, arch string }{{"PE required", "windows", "amd64"}, {"ARM required", "linux", "arm64"}} {
		t.Run(test.name, func(t *testing.T) {
			root := packageFixture(t, "")
			if test.platform == "windows" {
				packageWrite(t, filepath.Join(root, "PAIR.exe"), packageELF(""))
			}
			if p, err := InspectPackage(context.Background(), root, test.platform, test.arch); err == nil {
				p.Close()
				t.Fatal("accepted wrong-platform image")
			}
		})
	}
}

func TestPackageRevalidationRejectsEqualByteSubstitution(t *testing.T) {
	root := packageFixture(t, "")
	p, err := InspectPackage(context.Background(), root, "linux", "amd64")
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	path := filepath.Join(root, "nvpair")
	replacement := filepath.Join(root, "new-image")
	packageWrite(t, replacement, packageELF(""))
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(replacement, path); err != nil {
		t.Fatal(err)
	}
	if err := p.Revalidate(context.Background()); err == nil {
		t.Fatal("equal-byte image substitution retained consent")
	}
	if f, err := p.OpenVerifiedFile(context.Background(), "nvpair"); err == nil {
		f.Close()
		t.Fatal("at-use open accepted an equal-byte substituted image")
	}
}

func TestPackageVerifiedOpenUsesTheMeasuredDescriptor(t *testing.T) {
	root := packageFixture(t, "")
	p, err := InspectPackage(context.Background(), root, "linux", "amd64")
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	f, err := p.OpenVerifiedFile(context.Background(), "nvpair")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var header [4]byte
	if _, err := f.Read(header[:]); err != nil {
		t.Fatal(err)
	}
	if string(header[:]) != "\x7fELF" {
		t.Fatal("verified descriptor was not positioned at the selected image")
	}
}

func TestPackageInventoryMatchesThePackagingOwner(t *testing.T) {
	read := func(name string) string {
		t.Helper()
		data, err := os.ReadFile(filepath.Join("..", "..", "..", "desktop", "src", "shared", "constants", name))
		if err != nil {
			t.Fatal(err)
		}
		return string(data)
	}
	source := read("modular-binaries.ts")
	want := []string{}
	for _, match := range regexp.MustCompile(`baseName:\s*'([^']+)'`).FindAllStringSubmatch(source, -1) {
		want = append(want, match[1])
	}
	service := regexp.MustCompile(`SERVICE_BINARY_BASE_NAME\s*=\s*'([^']+)'`).FindStringSubmatch(source)
	if len(service) != 2 {
		t.Fatal("missing service packaging owner")
	}
	want = append(want, service[1])
	dispatcher := regexp.MustCompile(`INFERENCE_DISPATCHER_BASE_NAME\s*=\s*'([^']+)'`).FindStringSubmatch(read("inference-dispatcher.ts"))
	if len(dispatcher) != 2 {
		t.Fatal("missing dispatcher packaging owner")
	}
	want = append(want, dispatcher[1])
	got := append([]string(nil), bundledPrograms...)
	sort.Strings(got)
	sort.Strings(want)
	if len(got) != len(want) {
		t.Fatalf("package programs = %v, packaging owner = %v", got, want)
	}
	for i, name := range got {
		if name != want[i] {
			t.Fatalf("package programs = %v, packaging owner = %v", got, want)
		}
	}
}

func TestPackageManifestClaimsDoNotSupplyMissingSourceOrSettings(t *testing.T) {
	root := packageFixture(t, "")
	path := filepath.Join(root, "resources", "cli-bin", "manifest.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		t.Fatal(err)
	}
	fields["sourceFingerprint"] = json.RawMessage(`"forged-clean-source"`)
	fields["serviceBuildId"] = json.RawMessage(`"git:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"`)
	fields["expectedSettings"] = json.RawMessage(`{"stopEngines":false,"unloadModels":false,"unchanged":true}`)
	data, err = json.Marshal(fields)
	if err != nil {
		t.Fatal(err)
	}
	packageWrite(t, path, data)
	p, err := InspectPackage(context.Background(), root, "linux", "amd64")
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	if p.Service.BuildID.Known || p.Service.SourceRevision.Known || p.Service.Dirty.Known || !p.Service.SHA256.Known {
		t.Fatalf("manifest supplied Go authority: %+v", p.Service)
	}
}

func TestPackageCaseCollisionRefuses(t *testing.T) {
	root := packageFixtureTempDir(t)
	packageWrite(t, filepath.Join(root, "file"), []byte("a"))
	packageWrite(t, filepath.Join(root, "FILE"), []byte("b"))
	entries, err := os.ReadDir(root)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 2 {
		t.Skip("filesystem folds case aliases")
	}
	real, err := filepath.EvalSymlinks(root)
	if err != nil {
		t.Fatal(err)
	}
	held, err := OpenRealRoot(real)
	if err != nil {
		t.Fatal(err)
	}
	defer held.Close()
	p := &Package{Root: held, Platform: "linux"}
	if err := p.inventory(context.Background(), operationLimits); err == nil || !strings.Contains(err.Error(), "case-colliding") {
		t.Fatalf("collision refusal = %v", err)
	}
}
