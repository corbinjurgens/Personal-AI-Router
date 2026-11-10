// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package buildapply

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"unicode/utf8"
)

// RealRoot holds the directory descriptor as well as its original name and
// identity. A resolved string alone is never authority for a later operation.
type RealRoot struct {
	Path string
	root *os.Root
	info os.FileInfo
}

func localRootSyntax(path string) error {
	if !filepath.IsAbs(path) || !utf8.ValidString(path) || len(path) > MaxPathBytes || strings.ContainsAny(path, "\x00\r\n") || strings.HasPrefix(path, `\\`) || strings.HasPrefix(path, "//") {
		return errors.New("an absolute local real path is required")
	}
	withoutVolume := path[len(filepath.VolumeName(path)):]
	if strings.Contains(withoutVolume, ":") {
		return errors.New("alternate-stream path")
	}
	for _, part := range strings.FieldsFunc(withoutVolume, func(c rune) bool { return c == '/' || c == '\\' }) {
		if !validRelative(part) {
			return errors.New("ambiguous or traversing path")
		}
	}
	return nil
}

// ValidateLocalPath rejects network/device/ambiguous spellings and nonlocal
// mounts before opening the selected path. It creates no filesystem probe.
func ValidateLocalPath(path string) error {
	if err := localRootSyntax(path); err != nil {
		return err
	}
	return nativeLocalPath(path)
}

// resolveRoot resolves the links above a root PAIR selects or owns, once, and
// returns the real spelling every later check, comparison and open uses. It
// resolves only the parent: the final component stays as named, so the strict
// walk in checkRealAncestors still refuses a root that is itself a link, and
// nothing below the root is ever resolved.
//
// A link in an ancestor the user or the OS owns (a linked HOME or ~/Library,
// macOS's /var -> /private/var, a linked parent of a selected bundle) adds no
// attack surface: whoever can replace it already controls the account. The
// time-of-check risk these checks exist for is a link swapped inside PAIR's
// tree between check and use, which the held os.Root and the identity
// revalidation cover. Missing trailing directories are kept as named so a root
// that does not exist yet still resolves (journal creation, absent stores).
//
// Go's EvalSymlinks does not evaluate Windows mount points (junctions), so a
// junction above a root still fails here and stays refused.
func resolveRoot(path string) (string, error) {
	if err := localRootSyntax(path); err != nil {
		return "", err
	}
	path = filepath.Clean(path)
	parent := filepath.Dir(path)
	if parent == path {
		return path, nil
	}
	rest := []string{filepath.Base(path)}
	for {
		resolved, err := filepath.EvalSymlinks(parent)
		if err == nil {
			real := filepath.Join(append([]string{resolved}, rest...)...)
			if err := localRootSyntax(real); err != nil {
				return "", err
			}
			return real, nil
		}
		next := filepath.Dir(parent)
		if !os.IsNotExist(err) || next == parent {
			return "", err
		}
		rest = append([]string{filepath.Base(parent)}, rest...)
		parent = next
	}
}

// checkRealAncestors refuses a link anywhere in path, including path itself.
// Callers pass a path resolveRoot produced, or one derived below such a root,
// so a link it finds is at or inside a root PAIR selects or owns.
func checkRealAncestors(path string) (os.FileInfo, error) {
	ancestors := []string{}
	for current := path; ; current = filepath.Dir(current) {
		ancestors = append(ancestors, current)
		if filepath.Dir(current) == current {
			break
		}
	}
	var selected os.FileInfo
	// Inspect parents first. Lstat of a child still follows its ancestors, so
	// checking leaf-to-root could read through a substituted network link.
	for i := len(ancestors) - 1; i >= 0; i-- {
		current := ancestors[i]
		if err := ValidateLocalPath(current); err != nil {
			return nil, err
		}
		info, err := os.Lstat(current)
		if err != nil {
			return nil, err
		}
		if !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
			return nil, fmt.Errorf("linked root or ancestor: %s", current)
		}
		if err := nativePathCheck(current, info, false); err != nil {
			return nil, err
		}
		selected = info
	}
	return selected, nil
}

// OpenRealRoot holds a directory whose every component is real. path must
// already be resolved (resolveRoot) or derived below a resolved root; entry
// points that take an outside spelling resolve it first.
func OpenRealRoot(path string) (*RealRoot, error) {
	if err := localRootSyntax(path); err != nil {
		return nil, err
	}
	path = filepath.Clean(path)
	info, err := checkRealAncestors(path)
	if err != nil {
		return nil, err
	}
	root, err := os.OpenRoot(path)
	if err != nil {
		return nil, err
	}
	r := &RealRoot{Path: path, root: root, info: info}
	if err := r.Revalidate(); err != nil {
		root.Close()
		return nil, err
	}
	return r, nil
}

func (r *RealRoot) Close() error { return r.root.Close() }

func sameIdentity(a, b os.FileInfo) bool {
	return os.SameFile(a, b) && a.Mode() == b.Mode() && a.Size() == b.Size() && a.ModTime().Equal(b.ModTime())
}

func (r *RealRoot) Revalidate() error {
	current, err := checkRealAncestors(r.Path)
	if err != nil {
		return err
	}
	held, err := r.root.Stat(".")
	if err != nil {
		return err
	}
	if !current.IsDir() || !os.SameFile(r.info, current) || !os.SameFile(r.info, held) {
		return errors.New("selected root was substituted")
	}
	return nil
}

func validRelative(path string) bool {
	if path == "" || !utf8.ValidString(path) || len(path) > MaxPathBytes || strings.ContainsAny(path, "\\:*?<>|\"\x00\r\n") || strings.HasPrefix(path, "/") {
		return false
	}
	for _, part := range strings.Split(path, "/") {
		if part == "" || part == "." || part == ".." || strings.TrimRight(part, " .") != part {
			return false
		}
		base := strings.ToUpper(strings.SplitN(part, ".", 2)[0])
		if base == "CON" || base == "PRN" || base == "AUX" || base == "NUL" || base == "CLOCK$" || base == "CONIN$" || base == "CONOUT$" {
			return false
		}
		runes := []rune(base)
		if len(runes) == 4 && (strings.HasPrefix(base, "COM") || strings.HasPrefix(base, "LPT")) && strings.ContainsRune("123456789¹²³", runes[3]) {
			return false
		}
	}
	return true
}

func containsPath(root, path string) bool {
	rel, err := filepath.Rel(root, path)
	return err == nil && (rel == "." || rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator)) && !filepath.IsAbs(rel))
}

// ValidateLocations rejects both containment directions: a bundle cannot own a
// profile/source/model tree, nor be nested in one. Exclusions are trusted owner
// paths, never supplied by a candidate descriptor. Every path is resolved
// first and only resolved paths are compared, so a selection cannot reach a
// protected tree through a link that a string comparison would miss.
func ValidateLocations(source, target string, excluded []string) error {
	if !filepath.IsAbs(source) || !filepath.IsAbs(target) {
		return errors.New("unresolved bundle root")
	}
	selected := make([]string, 0, 2)
	for _, path := range []string{source, target} {
		real, err := resolveRoot(path)
		if err != nil {
			return err
		}
		if _, err := checkRealAncestors(real); err != nil {
			return err
		}
		selected = append(selected, real)
	}
	if containsPath(selected[0], selected[1]) || containsPath(selected[1], selected[0]) {
		return errors.New("source and target overlap")
	}
	protectedRoots := make([]string, 0, len(excluded))
	for _, protected := range excluded {
		if protected == "" {
			continue
		}
		if err := localRootSyntax(protected); err != nil {
			return errors.New("protected root is unresolved")
		}
		real, err := resolveRoot(protected)
		if err != nil {
			return errors.New("protected root is linked or unavailable")
		}
		if err := nativeLocalPath(real); err != nil {
			return err
		}
		// Missing default stores are harmless. A protected root that is
		// itself a link, or is otherwise unavailable, cannot authorize a
		// selection: do not follow it to an unknown filesystem merely to
		// improve preview.
		if _, err := checkRealAncestors(real); err != nil && !os.IsNotExist(err) {
			return errors.New("protected root is linked or unavailable")
		}
		protectedRoots = append(protectedRoots, real)
	}
	for _, path := range selected {
		for _, protected := range protectedRoots {
			if containsPath(path, protected) || containsPath(protected, path) {
				return errors.New("bundle overlaps protected data or model root")
			}
		}
		for parent := path; ; parent = filepath.Dir(parent) {
			for _, marker := range []string{".git", ".hg", ".svn"} {
				if _, err := os.Lstat(filepath.Join(parent, marker)); err == nil {
					return errors.New("source checkout is not an installation")
				} else if !os.IsNotExist(err) {
					return err
				}
			}
			if filepath.Dir(parent) == parent {
				break
			}
		}
	}
	return nil
}

// TargetFromExecutables derives a target solely from the actual native GUI and
// current service paths. Callers independently inspect the GUI process lifetime.
func TargetFromExecutables(gui, service, platform string) (string, error) {
	layout, ok := LayoutFor(platform)
	if !ok || !filepath.IsAbs(gui) || !filepath.IsAbs(service) {
		return "", errors.New("unsupported installation")
	}
	for _, path := range []string{gui, service} {
		if err := localRootSyntax(path); err != nil {
			return "", err
		}
	}
	named := filepath.Dir(gui)
	if platform == "darwin" {
		named = filepath.Dir(filepath.Dir(named))
	}
	// Links above the installation are resolved; the bundle root and the
	// directories inside it that hold both executables must be real.
	root, err := resolveRoot(named)
	if err != nil {
		return "", err
	}
	wantGUI := filepath.Join(root, filepath.FromSlash(layout.Executable))
	wantService := filepath.Join(root, filepath.FromSlash(layout.ServiceDir), layout.ServiceExecutable)
	for _, path := range []string{root, filepath.Dir(wantGUI), filepath.Dir(wantService)} {
		if _, err := checkRealAncestors(path); err != nil {
			return "", err
		}
	}
	for _, pair := range [][2]string{{gui, wantGUI}, {service, wantService}} {
		a, err := os.Lstat(pair[0])
		if err != nil {
			return "", err
		}
		b, err := os.Lstat(pair[1])
		if err != nil {
			return "", err
		}
		if !a.Mode().IsRegular() || !b.Mode().IsRegular() || !os.SameFile(a, b) {
			return "", errors.New("GUI and service do not belong to one native installation")
		}
	}
	if layout.RootName != "" && filepath.Base(root) != layout.RootName {
		return "", errors.New("unexpected application bundle name")
	}
	return root, nil
}

// CheckTargetParent checks the existing directory without creating a probe or
// prepared sibling. Filesystem permissions/space can change; Task 3 must repeat
// this immediately before preparing its same-parent sibling.
func CheckTargetParent(target *RealRoot, bytes int64) error {
	if err := target.Revalidate(); err != nil {
		return err
	}
	parent, err := OpenRealRoot(filepath.Dir(target.Path))
	if err != nil {
		return err
	}
	defer parent.Close()
	if target.Path == parent.Path {
		return errors.New("volume root is not an installation")
	}
	if err := nativeTargetParent(target.Path, parent.Path, bytes); err != nil {
		return fmt.Errorf("target parent: %w", err)
	}
	return parent.Revalidate()
}

// ProcessIdentity is local-only evidence, never a wire observation or PID-only
// authority. Re-read it at use and compare the creation token and executable.
type ProcessIdentity struct{ Executable, Created string }

func InspectProcess(pid int) (ProcessIdentity, error) {
	if pid <= 0 || int64(pid) > 1<<31-1 {
		return ProcessIdentity{}, errors.New("invalid process identity")
	}
	return nativeProcessIdentity(pid)
}

// ReadLocalFile is a bounded, rooted read of an existing owner file. It never
// creates missing state and refuses links, nonlocal mounts and substitutions.
// The file and the directory holding it must be real; links above that
// directory are resolved (resolveRoot).
func ReadLocalFile(path string, limit int64) ([]byte, error) {
	if limit <= 0 || limit > MaxFileBytes {
		return nil, errors.New("invalid local read bound")
	}
	if err := localRootSyntax(path); err != nil {
		return nil, err
	}
	dir, err := resolveRoot(filepath.Dir(path))
	if err != nil {
		return nil, err
	}
	root, err := OpenRealRoot(dir)
	if err != nil {
		return nil, err
	}
	defer root.Close()
	name := filepath.Base(path)
	before, err := root.root.Lstat(name)
	if err != nil {
		return nil, err
	}
	if !before.Mode().IsRegular() || before.Size() > limit {
		return nil, errors.New("owner file is not a bounded regular file")
	}
	if err := nativeInfoCheck(before, false); err != nil {
		return nil, err
	}
	f, err := root.root.Open(name)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	held, err := f.Stat()
	if err != nil || !sameIdentity(before, held) {
		return nil, errors.New("owner file changed")
	}
	if err := nativeOpenedCheck(f, held); err != nil {
		return nil, err
	}
	data, err := io.ReadAll(io.LimitReader(f, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(data)) > limit {
		return nil, errors.New("owner file exceeds read bound")
	}
	after, err := f.Stat()
	named, nameErr := root.root.Lstat(name)
	if err != nil || nameErr != nil || !sameIdentity(before, after) || !sameIdentity(before, named) {
		return nil, errors.New("owner file changed during read")
	}
	if err := root.Revalidate(); err != nil {
		return nil, err
	}
	return data, nil
}
