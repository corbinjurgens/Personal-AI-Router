// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package buildapply

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"unicode"

	"nvpair-shared/buildinfo"
)

// PackageEntry describes measured bytes. info is deliberately private: the
// inventory is never authorization supplied by a candidate manifest.
type PackageEntry struct {
	Path   string
	Size   int64
	Mode   os.FileMode
	SHA256 string
	Link   string
	info   os.FileInfo
}

type Package struct {
	Root                   *RealRoot
	Layout                 Layout
	Platform, Architecture string
	Entries                []PackageEntry
	Bytes                  int64
	SHA256                 string
	Service                Identity
	GUI                    Identity
}

type packageLimits struct {
	total, file   int64
	entries, path int
}

var operationLimits = packageLimits{MaxPackageBytes, MaxFileBytes, MaxEntries, MaxPathBytes}

// ErrPackageTooLarge distinguishes the fixed package bounds from malformed input.
var ErrPackageTooLarge = errors.New("staged build exceeds the bounded package limits")

// InspectPackage performs no writes and executes no images. The returned root
// must be closed by its owner, including when the preview slot is retired.
func InspectPackage(ctx context.Context, path, platform, arch string) (*Package, error) {
	return inspectPackage(ctx, path, platform, arch, operationLimits)
}

func inspectPackage(ctx context.Context, path, platform, arch string, limits packageLimits) (_ *Package, err error) {
	layout, ok := LayoutFor(platform)
	if !ok || (arch != "amd64" && arch != "arm64") {
		return nil, errors.New("unsupported package platform or architecture")
	}
	root, err := OpenRealRoot(path)
	if err != nil {
		return nil, err
	}
	defer func() {
		if err != nil {
			root.Close()
		}
	}()
	if layout.RootName != "" && filepath.Base(root.Path) != layout.RootName {
		return nil, errors.New("unexpected application bundle name")
	}
	if err := nativePathCheck(root.Path, root.info, true); err != nil {
		return nil, err
	}
	p := &Package{Root: root, Layout: layout, Platform: platform, Architecture: arch}
	if err = p.inventory(ctx, limits); err != nil {
		return nil, err
	}
	if err = p.validateLayout(); err != nil {
		return nil, err
	}
	if err = p.readIdentities(ctx); err != nil {
		return nil, err
	}
	if err = p.Root.Revalidate(); err != nil {
		return nil, err
	}
	return p, nil
}

func (p *Package) Close() error { return p.Root.Close() }

// OpenVerifiedFile opens through the retained root, checks the selected file's
// identity and bytes, and returns that same descriptor positioned at zero. A
// later copy owner must hash its copy too; a preview never freezes disk bytes.
func (p *Package) OpenVerifiedFile(ctx context.Context, path string) (*os.File, error) {
	if !validRelative(path) {
		return nil, errors.New("invalid package file path")
	}
	if err := p.Root.Revalidate(); err != nil {
		return nil, err
	}
	dir, err := p.Root.root.Open(".")
	if err != nil {
		return nil, err
	}
	dirInfo, statErr := dir.Stat()
	if statErr == nil {
		statErr = nativeOpenedCheck(dir, dirInfo)
	}
	dir.Close()
	if statErr != nil {
		return nil, statErr
	}
	entry := p.entry(path)
	if entry == nil || !entry.Mode.IsRegular() {
		return nil, errors.New("file was not selected in the package inventory")
	}
	f, err := p.Root.root.Open(filepath.FromSlash(path))
	if err != nil {
		return nil, err
	}
	refuse := func(err error) (*os.File, error) { f.Close(); return nil, err }
	hash, _, err := hashOpened(ctx, f, entry.info, MaxFileBytes)
	if err != nil {
		return refuse(err)
	}
	if hash != entry.SHA256 {
		return refuse(errors.New("selected file bytes changed"))
	}
	after, err := p.Root.root.Lstat(filepath.FromSlash(path))
	if err != nil || !sameIdentity(entry.info, after) {
		return refuse(errors.New("selected file path changed"))
	}
	if err := nativeOpenedCheck(f, after); err != nil {
		return refuse(err)
	}
	if err := p.Root.Revalidate(); err != nil {
		return refuse(err)
	}
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return refuse(err)
	}
	return f, nil
}

func (p *Package) inventory(ctx context.Context, limits packageLimits) error {
	seen := map[string]bool{}
	var walk func(string) error
	walk = func(dir string) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := nativeLocalPath(filepath.Join(p.Root.Path, filepath.FromSlash(dir))); err != nil {
			return err
		}
		file, err := p.Root.root.Open(dir)
		if err != nil {
			return err
		}
		defer file.Close()
		held, err := file.Stat()
		if err != nil {
			return err
		}
		if err := nativeOpenedCheck(file, held); err != nil {
			return err
		}
		for {
			children, readErr := file.ReadDir(128)
			for _, child := range children {
				if err := ctx.Err(); err != nil {
					return err
				}
				name := child.Name()
				if dir != "." {
					name = dir + "/" + name
				}
				if len(name) > limits.path {
					return fmt.Errorf("%w: relative package path byte limit", ErrPackageTooLarge)
				}
				if !validRelative(name) {
					return errors.New("invalid or oversized relative package path")
				}
				if len(p.Entries) >= limits.entries {
					return fmt.Errorf("%w: entry limit", ErrPackageTooLarge)
				}
				fold := foldPackagePath(name)
				if seen[fold] {
					return errors.New("case-colliding package paths")
				}
				seen[fold] = true
				if child.Name() == ".git" || child.Name() == ".hg" || child.Name() == ".svn" {
					return errors.New("source checkout content")
				}
				if err := nativeLocalPath(filepath.Join(p.Root.Path, filepath.FromSlash(name))); err != nil {
					return err
				}
				info, err := p.Root.root.Lstat(filepath.FromSlash(name))
				if err != nil {
					return err
				}
				if err := nativeInfoCheck(info, false); err != nil {
					return err
				}
				entry := PackageEntry{Path: name, Size: info.Size(), Mode: info.Mode(), info: info}
				switch {
				case info.Mode()&os.ModeSymlink != 0:
					if p.Platform != "darwin" || !strings.HasPrefix(name, "Contents/Frameworks/") || !strings.Contains(name, ".framework/") {
						return errors.New("package link outside a macOS framework")
					}
					link, err := p.Root.root.Readlink(filepath.FromSlash(name))
					if err != nil {
						return err
					}
					if filepath.IsAbs(link) {
						return errors.New("absolute framework link")
					}
					// Root.Stat resolves the complete chain under the held root.
					// Unlike EvalSymlinks followed by containment, it refuses an
					// escaping chain before reading an external target.
					resolvedInfo, err := p.Root.root.Stat(filepath.FromSlash(name))
					if err != nil {
						return err
					}
					if !resolvedInfo.IsDir() && !resolvedInfo.Mode().IsRegular() {
						return errors.New("framework link resolves to a special file")
					}
					entry.Link = link
				case info.IsDir():
					entry.Size = 0
				case info.Mode().IsRegular():
					if info.Size() < 0 || info.Size() > limits.file || info.Size() > limits.total-p.Bytes {
						return fmt.Errorf("%w: file or total byte limit", ErrPackageTooLarge)
					}
					opened, err := p.Root.root.Open(filepath.FromSlash(name))
					if err != nil {
						return err
					}
					hash, size, hashErr := hashOpened(ctx, opened, info, limits.file)
					closeErr := opened.Close()
					if hashErr != nil {
						return hashErr
					}
					if closeErr != nil {
						return closeErr
					}
					entry.SHA256 = hash
					entry.Size = size
					p.Bytes += size
				default:
					return errors.New("special file in package")
				}
				p.Entries = append(p.Entries, entry)
				if info.IsDir() {
					if err := walk(name); err != nil {
						return err
					}
				}
				after, err := p.Root.root.Lstat(filepath.FromSlash(name))
				if err != nil || !sameIdentity(info, after) {
					return errors.New("package changed during inventory")
				}
			}
			if readErr == io.EOF {
				break
			}
			if readErr != nil {
				return readErr
			}
		}
		return nil
	}
	if err := walk("."); err != nil {
		return err
	}
	sort.Slice(p.Entries, func(i, j int) bool { return p.Entries[i].Path < p.Entries[j].Path })
	hash := sha256.New()
	for _, entry := range p.Entries {
		// Length-delimited JSON records avoid path/hash/size concatenation ambiguity.
		encoded, err := json.Marshal(struct {
			Path       string
			Size       int64
			Mode       uint32
			Hash, Link string
		}{entry.Path, entry.Size, uint32(entry.Mode), entry.SHA256, entry.Link})
		if err != nil {
			return err
		}
		fmt.Fprintf(hash, "%d:", len(encoded))
		hash.Write(encoded)
	}
	p.SHA256 = hex.EncodeToString(hash.Sum(nil))
	return p.Root.Revalidate()
}

func foldPackagePath(path string) string {
	var result strings.Builder
	for _, r := range path {
		canonical := r
		for next := unicode.SimpleFold(r); next != r; next = unicode.SimpleFold(next) {
			if next < canonical {
				canonical = next
			}
		}
		result.WriteRune(canonical)
	}
	return result.String()
}

func hashOpened(ctx context.Context, f *os.File, before os.FileInfo, limit int64) (string, int64, error) {
	opened, err := f.Stat()
	if err != nil {
		return "", 0, err
	}
	if !sameIdentity(before, opened) || !opened.Mode().IsRegular() {
		return "", 0, errors.New("file was substituted")
	}
	if err := nativeOpenedCheck(f, opened); err != nil {
		return "", 0, err
	}
	hash := sha256.New()
	buffer := make([]byte, 128<<10)
	var total int64
	for {
		if err := ctx.Err(); err != nil {
			return "", 0, err
		}
		n, readErr := f.Read(buffer)
		total += int64(n)
		if total > limit || total > before.Size() {
			return "", 0, errors.New("file grew beyond inventory bounds")
		}
		hash.Write(buffer[:n])
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return "", 0, readErr
		}
	}
	after, err := f.Stat()
	if err != nil || total != before.Size() || !sameIdentity(before, after) {
		return "", 0, errors.New("file changed during hashing")
	}
	return hex.EncodeToString(hash.Sum(nil)), total, nil
}

// Revalidate repeats both the bounded inventory/hash and file identities. Equal
// bytes at a substituted inode are still a new selection needing new consent.
func (p *Package) Revalidate(ctx context.Context) error {
	if err := p.Root.Revalidate(); err != nil {
		return err
	}
	next := &Package{Root: p.Root, Platform: p.Platform, Architecture: p.Architecture, Layout: p.Layout}
	if err := next.inventory(ctx, operationLimits); err != nil {
		return err
	}
	if p.SHA256 != next.SHA256 || len(p.Entries) != len(next.Entries) {
		return errors.New("selected package content changed")
	}
	for i, e := range p.Entries {
		if e.Path != next.Entries[i].Path || !sameIdentity(e.info, next.Entries[i].info) {
			return errors.New("selected package identity changed")
		}
	}
	return nil
}

func (p *Package) entry(path string) *PackageEntry {
	i := sort.Search(len(p.Entries), func(i int) bool { return p.Entries[i].Path >= path })
	if i < len(p.Entries) && p.Entries[i].Path == path {
		return &p.Entries[i]
	}
	return nil
}

var bundledPrograms = []string{"nvpair-proxy", "nvpair-node-scanner", "nvpair-node-info", "nvpair-workload-manager", "nvpair-cluster-manager", "nvpair-errors", "nvpair-manual-nodes", "nvpair-node-settings", "nvpair-engine-manager", "nvpair-job-scheduler", "nvpair-ui-broker", "nvpair-tui", "nvpair-service", "inference-dispatcher"}

// These are runtime dependencies of the packaged Electron distribution, not
// optional source/build descriptors. Every selected byte is still inventoried.
func electronRuntimeFiles(platform string) []string {
	if platform == "darwin" {
		return []string{"Contents/Info.plist", "Contents/Frameworks/Electron Framework.framework/Versions/A/Electron Framework", "Contents/Frameworks/Electron Framework.framework/Versions/A/Resources/Info.plist", "Contents/Frameworks/Electron Framework.framework/Versions/A/Resources/icudtl.dat", "Contents/Frameworks/Electron Framework.framework/Versions/A/Resources/v8_context_snapshot.bin", "Contents/Frameworks/Electron Framework.framework/Versions/A/Resources/en.lproj/locale.pak"}
	}
	files := []string{"icudtl.dat", "resources.pak", "chrome_100_percent.pak", "chrome_200_percent.pak", "v8_context_snapshot.bin", "locales/en-US.pak"}
	if platform == "windows" {
		return append(files, "snapshot_blob.bin", "ffmpeg.dll", "libEGL.dll", "libGLESv2.dll", "d3dcompiler_47.dll", "dxcompiler.dll", "dxil.dll", "vk_swiftshader.dll", "vk_swiftshader_icd.json", "vulkan-1.dll")
	}
	return append(files, "libffmpeg.so", "libEGL.so", "libGLESv2.so", "libvk_swiftshader.so", "vk_swiftshader_icd.json", "libvulkan.so.1", "chrome_crashpad_handler", "chrome-sandbox")
}

func (p *Package) validateLayout() error {
	for _, name := range electronRuntimeFiles(p.Platform) {
		entry := p.entry(name)
		if entry == nil || !entry.Mode.IsRegular() || entry.Size == 0 {
			return fmt.Errorf("missing regular Electron runtime file: %s", name)
		}
	}
	if p.Platform == "windows" {
		for _, entry := range p.Entries {
			base := strings.ToLower(entry.Path)
			if !strings.Contains(base, "/") && strings.HasPrefix(base, "uninstall") && strings.HasSuffix(base, ".exe") {
				return errors.New("installer-owned applications are not unpacked build targets")
			}
		}
	}
	suffix := ""
	manifestPlatform := p.Platform
	manifestArch := p.Architecture
	if p.Platform == "windows" {
		suffix = ".exe"
		manifestPlatform = "win32"
	}
	if manifestArch == "amd64" {
		manifestArch = "x64"
	}
	required := []string{p.Layout.Executable}
	for _, name := range bundledPrograms {
		required = append(required, p.Layout.ServiceDir+"/"+name+suffix)
	}
	for _, name := range required {
		e := p.entry(name)
		if e == nil || !e.Mode.IsRegular() || e.Size == 0 {
			return fmt.Errorf("missing regular package executable: %s", name)
		}
		f, err := p.Root.root.Open(filepath.FromSlash(name))
		if err != nil {
			return err
		}
		err = checkImage(f, p.Platform, p.Architecture)
		closeErr := f.Close()
		if err != nil {
			return fmt.Errorf("%s: %w", name, err)
		}
		if closeErr != nil {
			return closeErr
		}
		// Windows does not expose POSIX execute bits, including for inert
		// cross-platform package fixtures. Native Unix validation requires them.
		if runtime.GOOS != "windows" && p.Platform != "windows" && e.Mode.Perm()&0100 == 0 {
			return errors.New("package executable permission is absent")
		}
	}
	resources := strings.TrimSuffix(p.Layout.ServiceDir, "/cli-bin")
	if e := p.entry(resources + "/app.asar"); e == nil || !e.Mode.IsRegular() || e.Size == 0 {
		return errors.New("missing packaged GUI application")
	}
	manifest := p.entry(p.Layout.ServiceDir + "/manifest.json")
	if manifest == nil || !manifest.Mode.IsRegular() || manifest.Size > MaxRequestBytes*64 {
		return errors.New("missing or oversized modular manifest")
	}
	f, err := p.Root.root.Open(filepath.FromSlash(manifest.Path))
	if err != nil {
		return err
	}
	defer f.Close()
	data, err := io.ReadAll(io.LimitReader(f, MaxRequestBytes*64+1))
	if err != nil || len(data) > MaxRequestBytes*64 {
		return errors.New("oversized modular manifest")
	}
	if err := uniqueJSON(json.NewDecoder(bytes.NewReader(data)), 0); err != nil {
		return errors.New("ambiguous modular manifest")
	}
	var m struct {
		Source, Platform, Arch string
		Files                  []struct {
			FileName string
			Size     int64
			SHA256   string
		}
	}
	if err := json.Unmarshal(data, &m); err != nil {
		return errors.New("invalid modular manifest")
	}
	if m.Source != "services-build" || m.Platform != manifestPlatform || m.Arch != manifestArch {
		return errors.New("manifest target disagrees with host or artifact")
	}
	want := map[string]bool{}
	for _, program := range bundledPrograms {
		want[program+suffix] = true
	}
	if len(m.Files) != len(want) {
		return errors.New("incomplete modular manifest")
	}
	for _, file := range m.Files {
		if !want[file.FileName] {
			return errors.New("unexpected or duplicate modular manifest entry")
		}
		delete(want, file.FileName)
		e := p.entry(p.Layout.ServiceDir + "/" + file.FileName)
		if e == nil || e.Size != file.Size || e.SHA256 != file.SHA256 {
			return errors.New("manifest differs from measured package bytes")
		}
	}
	for _, e := range p.Entries {
		if strings.HasPrefix(e.Path, p.Layout.ServiceDir+"/") && e.Path != manifest.Path {
			name := strings.TrimPrefix(e.Path, p.Layout.ServiceDir+"/")
			valid := false
			for _, program := range bundledPrograms {
				if name == program+suffix {
					valid = true
				}
			}
			if !valid {
				return errors.New("unexpected modular binary inventory")
			}
		}
	}
	return nil
}

// checkImage reads only fixed-size headers, never an image loader or parser
// whose allocation sizes are controlled by candidate headers.
func checkImage(f *os.File, platform, arch string) error {
	var header [64]byte
	if _, err := f.ReadAt(header[:], 0); err != nil {
		return err
	}
	le := binary.LittleEndian
	switch platform {
	case "windows":
		if string(header[:2]) != "MZ" {
			return errors.New("not a Windows PE image")
		}
		var pe [26]byte
		offset := int64(le.Uint32(header[60:]))
		if _, err := f.ReadAt(pe[:], offset); err != nil {
			return err
		}
		want := uint16(0x8664)
		if arch == "arm64" {
			want = 0xaa64
		}
		if string(pe[:4]) != "PE\x00\x00" || le.Uint16(pe[4:]) != want || le.Uint16(pe[24:]) != 0x20b {
			return errors.New("wrong PE architecture or format")
		}
	case "linux":
		want := uint16(62)
		if arch == "arm64" {
			want = 183
		}
		if string(header[:4]) != "\x7fELF" || header[4] != 2 || header[5] != 1 || le.Uint16(header[18:]) != want {
			return errors.New("wrong ELF architecture or format")
		}
	case "darwin":
		want := uint32(0x1000007)
		if arch == "arm64" {
			want = 0x100000c
		}
		if le.Uint32(header[:4]) != 0xfeedfacf || le.Uint32(header[4:]) != want {
			return errors.New("wrong Mach-O architecture or format")
		}
	default:
		return errors.New("unsupported image platform")
	}
	return nil
}

func (p *Package) readIdentities(ctx context.Context) error {
	service := p.entry(p.Layout.ServiceDir + "/" + p.Layout.ServiceExecutable)
	p.Service.SHA256 = Known(service.SHA256)
	p.GUI.SHA256 = Known(p.entry(p.Layout.Executable).SHA256)
	// The trusted bounded buildinfo reader revalidates its held descriptor.
	// Missing VCS is distinct from a malformed/non-Go service image.
	file, err := p.OpenVerifiedFile(ctx, service.Path)
	if err != nil {
		return err
	}
	metadata, err := buildinfo.InspectOpenedMetadata(file)
	closeErr := file.Close()
	if err != nil {
		return fmt.Errorf("cannot inspect bundled Go service: %w", err)
	}
	if closeErr != nil {
		return closeErr
	}
	if metadata.ID != "" {
		p.Service.BuildID = Known(string(metadata.ID))
	}
	if metadata.Revision != nil {
		p.Service.SourceRevision = Known(*metadata.Revision)
	}
	if metadata.Dirty != nil {
		p.Service.Dirty = Known(*metadata.Dirty)
	}
	if err := p.readGUIIdentity(); err != nil {
		return err
	}
	return nil
}

type asarEntry struct {
	Files    map[string]asarEntry `json:"files"`
	Offset   string               `json:"offset"`
	Size     int64                `json:"size"`
	Unpacked bool                 `json:"unpacked"`
	Link     string               `json:"link"`
}

// The stamp is the build-time constant in the native main GUIFacts object,
// not a manifest/sourceFingerprint claim. Missing or ambiguous stamps stay
// unknown. This is diagnostic identity for unsigned selected bytes.
var guiStamp = regexp.MustCompile(`(?:\bbuildId|"buildId"|'buildId')\s*:\s*["'](git:[a-f0-9]{40}(?:[a-f0-9]{24})?(?:\+dirty)?)["']`)

func (p *Package) readGUIIdentity() error {
	resources := strings.TrimSuffix(p.Layout.ServiceDir, "/cli-bin")
	entry := p.entry(resources + "/app.asar")
	f, err := p.Root.root.Open(filepath.FromSlash(entry.Path))
	if err != nil {
		return err
	}
	defer f.Close()
	held, err := f.Stat()
	if err != nil || !sameIdentity(entry.info, held) {
		return errors.New("GUI archive changed")
	}
	if err := nativeOpenedCheck(f, held); err != nil {
		return err
	}
	var prefix [16]byte
	if _, err := f.ReadAt(prefix[:], 0); err != nil {
		return errors.New("invalid GUI archive header")
	}
	le := binary.LittleEndian
	headerSize, jsonSize := int64(le.Uint32(prefix[4:8])), int64(le.Uint32(prefix[12:16]))
	if le.Uint32(prefix[:4]) != 4 || headerSize < 8 || headerSize > 16<<20 || jsonSize <= 0 || jsonSize > headerSize-8 || 8+headerSize > held.Size() || int64(le.Uint32(prefix[8:12])) != headerSize-4 {
		return errors.New("oversized or malformed GUI archive header")
	}
	data := make([]byte, int(jsonSize))
	if _, err := f.ReadAt(data, 16); err != nil {
		return err
	}
	var archive asarEntry
	if err := json.Unmarshal(data, &archive); err != nil {
		return errors.New("invalid GUI archive inventory")
	}
	get := func(path string, limit int64) ([]byte, error) {
		if !validRelative(path) {
			return nil, errors.New("invalid GUI archive path")
		}
		node := archive
		for _, part := range strings.Split(path, "/") {
			next, ok := node.Files[part]
			if !ok {
				return nil, errors.New("missing GUI archive entry")
			}
			node = next
		}
		if node.Files != nil || node.Unpacked || node.Link != "" || node.Size <= 0 || node.Size > limit {
			return nil, errors.New("unsupported GUI archive entry")
		}
		offset, err := strconv.ParseInt(node.Offset, 10, 64)
		if err != nil || offset < 0 || offset > held.Size()-8-headerSize || node.Size > held.Size()-8-headerSize-offset {
			return nil, errors.New("GUI archive entry escapes data region")
		}
		value := make([]byte, int(node.Size))
		_, err = f.ReadAt(value, 8+headerSize+offset)
		return value, err
	}
	packageJSON, err := get("package.json", MaxRequestBytes*4)
	if err != nil {
		return err
	}
	var descriptor struct {
		Main string `json:"main"`
	}
	if json.Unmarshal(packageJSON, &descriptor) != nil || (descriptor.Main != "./out/main/index.js" && descriptor.Main != "out/main/index.js") {
		return errors.New("unexpected packaged GUI entry point")
	}
	main, err := get("out/main/index.js", 32<<20)
	if err != nil {
		return err
	}
	if _, err := get("out/preload/index.js", 32<<20); err != nil {
		return err
	}
	if _, err := get("out/ui/index.html", 2<<20); err != nil {
		return err
	}
	stamp := ""
	matches := guiStamp.FindAllSubmatch(main, 17)
	if len(matches) <= 16 {
		for _, match := range matches {
			value := string(match[1])
			if stamp != "" && stamp != value {
				stamp = ""
				break
			}
			stamp = value
		}
	}
	if stamp != "" {
		var id buildinfo.ID
		encoded, _ := json.Marshal(stamp)
		if err := json.Unmarshal(encoded, &id); err == nil && id != "" {
			p.GUI.BuildID = Known(stamp)
			p.GUI.SourceRevision = Known(strings.TrimSuffix(strings.TrimPrefix(stamp, "git:"), "+dirty"))
			p.GUI.Dirty = Known(strings.HasSuffix(stamp, "+dirty"))
		}
	}
	after, err := f.Stat()
	pathAfter, pathErr := p.Root.root.Lstat(filepath.FromSlash(entry.Path))
	if err != nil || pathErr != nil || !sameIdentity(held, after) || !sameIdentity(held, pathAfter) {
		return errors.New("GUI archive changed during inspection")
	}
	return nil
}
