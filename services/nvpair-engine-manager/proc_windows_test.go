// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build windows

package main

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"golang.org/x/sys/windows"

	"nvpair-shared/testbin"
)

// Only reference paths use aliases. Every listener, including a restored cold
// launch, executes an existing fixed-path image covered by the test manifest.
func windowsImageFixture(t *testing.T) (string, string) {
	t.Helper()
	dir := testbin.Dir(t, "nvpair-engine-manager", "stop-orphan", "fakeorphan")
	bin := filepath.Join(dir, testbin.Exe("fakeorphan"))
	copyFile(t, fakeEngineBin, bin)
	return dir, bin
}

func windowsImageJunction(t *testing.T, root, target string) string {
	t.Helper()
	link := filepath.Join(root, "install alias")
	rel, err := filepath.Rel(root, link)
	if err != nil || !filepath.IsLocal(rel) {
		t.Fatalf("junction must be contained in its private root: root=%q link=%q err=%v", root, link, err)
	}
	junction(t, link, target)
	t.Cleanup(func() {
		// Remove only the owned reparse entry, never recursively remove its
		// target (which can be a persistent, firewall-covered fixture path).
		info, err := os.Lstat(link)
		if err != nil || info.Mode()&(os.ModeSymlink|os.ModeIrregular) == 0 {
			t.Errorf("owned junction changed before cleanup: %q info=%v err=%v", link, info, err)
			return
		}
		if err := os.Remove(link); err != nil {
			t.Errorf("unlink owned junction %q: %v", link, err)
			return
		}
		if _, err := os.Lstat(link); !os.IsNotExist(err) {
			t.Errorf("junction remains after unlink: %q err=%v", link, err)
		}
		if _, err := os.Stat(target); err != nil {
			t.Errorf("junction cleanup changed target %q: %v", target, err)
		}
	})
	resolved, err := filepath.EvalSymlinks(link)
	if err != nil {
		t.Fatalf("resolve private junction: %v", err)
	}
	got, err := os.Stat(resolved)
	if err != nil {
		t.Fatal(err)
	}
	want, err := os.Stat(target)
	if err != nil || !os.SameFile(got, want) {
		t.Fatalf("junction target=%q, want %q: %v", resolved, target, err)
	}
	return link
}

func windowsShortImage(t *testing.T, bin string) string {
	t.Helper()
	path, err := windows.UTF16PtrFromString(bin)
	if err != nil {
		t.Fatal(err)
	}
	buf := make([]uint16, 32768)
	n, err := windows.GetShortPathName(path, &buf[0], uint32(len(buf)))
	if errors.Is(err, windows.ERROR_NOT_SUPPORTED) {
		t.Skipf("fixture volume does not support short paths: %v", err)
	}
	if err != nil || n == 0 || n >= uint32(len(buf)) {
		t.Fatalf("GetShortPathName(%q) length=%d: %v", bin, n, err)
	}
	short := windows.UTF16ToString(buf[:n])
	if strings.EqualFold(filepath.Clean(short), filepath.Clean(bin)) {
		t.Skipf("fixture has no distinct 8.3 alias on this volume: %q", bin)
	}
	t.Logf("8.3 reference=%q physical executable=%q", short, bin)
	return short
}

func windowsImageAliases(t *testing.T, test func(*testing.T, string, string, string)) {
	t.Helper()
	t.Run("junction", func(t *testing.T) {
		dir, bin := windowsImageFixture(t)
		aliasDir := windowsImageJunction(t, t.TempDir(), dir)
		test(t, dir, bin, filepath.Join(aliasDir, filepath.Base(bin)))
	})
	t.Run("8.3 short name when available", func(t *testing.T) {
		dir, bin := windowsImageFixture(t)
		test(t, dir, bin, windowsShortImage(t, bin))
	})
}

func windowsImageExecutor(t *testing.T, dir, bin, reference string, port int) *Executor {
	t.Helper()
	m := testEngineManifest(bin)
	p := m.Platforms[hostKey()]
	p.Detect = []string{reference}
	p.Runtime.Port = port
	p.Runtime.Health = nil
	p.Runtime.Stop = &StopSpec{Signal: "kill"}
	p.Uninstall = &Uninstall{Remove: []string{bin}}
	m.Platforms[hostKey()] = p
	ex := managedFixtureExecutor(t, m, dir)
	t.Cleanup(ex.StopAll)
	if installed, err := ex.Detect(m.Engine); err != nil || !installed {
		t.Fatalf("detect reference %q: installed=%v err=%v", reference, installed, err)
	}
	return ex
}

func TestWindowsEngineImageAliasesIdentifyPhysicalListener(t *testing.T) {
	windowsImageAliases(t, func(t *testing.T, dir, bin, reference string) {
		port := privateTestPorts(t, 1)[0]
		child := spawnFakeListener(t, bin, port)
		image := imagePathForPID(child.Process.Pid)
		if !isOurEngineImage(image, reference) || !isOurEngineImage(reference, image) {
			t.Fatalf("physical listener image=%q reference=%q must match in either direction", image, reference)
		}
		if !isManagedInstallPath(reference, dir) {
			t.Fatalf("alias %q must remain contained in managed install %q", reference, dir)
		}
		if !isManagedInstallPath(bin, filepath.Dir(reference)) {
			t.Fatalf("physical image %q must remain contained in aliased install %q", bin, filepath.Dir(reference))
		}
	})
}

func TestWindowsEngineImageSymlinkIdentity(t *testing.T) {
	dir, bin := windowsImageFixture(t)
	reference := filepath.Join(t.TempDir(), "image alias.exe")
	if err := os.Symlink(bin, reference); errors.Is(err, windows.ERROR_PRIVILEGE_NOT_HELD) {
		t.Skip("file symlink creation requires developer mode or symlink privilege")
	} else if err != nil {
		t.Fatal(err)
	}
	port := privateTestPorts(t, 1)[0]
	child := spawnFakeListener(t, bin, port)
	image := imagePathForPID(child.Process.Pid)
	if !isOurEngineImage(image, reference) || !isOurEngineImage(reference, image) || !isManagedInstallPath(reference, dir) {
		t.Fatalf("symlink must identify the physical managed executable: image=%q reference=%q install=%q", image, reference, dir)
	}
}

func TestWindowsEngineImageRefusesDistinctFiles(t *testing.T) {
	root := t.TempDir()
	bin := filepath.Join(root, "managed", "engine.exe")
	if err := os.MkdirAll(filepath.Dir(bin), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(bin, []byte("same contents"), 0o600); err != nil {
		t.Fatal(err)
	}
	test := func(name, contents string) {
		t.Run(name, func(t *testing.T) {
			image := filepath.Join(t.TempDir(), filepath.Base(bin))
			if err := os.WriteFile(image, []byte(contents), 0o600); err != nil {
				t.Fatal(err)
			}
			if isOurEngineImage(image, bin) || isOurEngineImage(bin, image) {
				t.Fatal("distinct files with the same filename must never match")
			}
		})
	}
	test("same filename and bytes", "same contents")
	test("same filename different bytes", "different contents")
}

func TestWindowsEngineImageFailsClosed(t *testing.T) {
	dir := t.TempDir()
	missing := filepath.Join(dir, "missing.exe")
	malformed := dir + "\x00"
	if isOurEngineImage(missing, missing) || isOurEngineImage(dir, dir) || isOurEngineImage(malformed, malformed) {
		t.Fatal("matching spellings of missing, directory or malformed images are not executable identity")
	}
}

func TestWindowsEngineImageUnavailableIdentityFailsClosed(t *testing.T) {
	dir := t.TempDir()
	bin := filepath.Join(dir, "locked.exe")
	if err := os.WriteFile(bin, []byte("image fixture"), 0o600); err != nil {
		t.Fatal(err)
	}
	path, err := windows.UTF16PtrFromString(bin)
	if err != nil {
		t.Fatal(err)
	}
	h, err := windows.CreateFile(path, windows.GENERIC_READ, 0, nil, windows.OPEN_EXISTING, windows.FILE_ATTRIBUTE_NORMAL, 0)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := windows.CloseHandle(h); err != nil {
			t.Errorf("close private exclusive content handle: %v", err)
		}
	})
	// A zero-access metadata open can succeed despite an exclusive content
	// handle. Establish actual unavailable identity before testing that refusal;
	// otherwise access-denied identity coverage remains explicitly untested.
	info, statErr := os.Stat(bin)
	if statErr == nil && os.SameFile(info, info) {
		t.Skip("exclusive content handle leaves metadata identity readable; access-denied identity remains untested")
	}
	if isOurEngineImage(bin, bin) {
		t.Fatal("unreadable file identity must fail closed even for identical path strings")
	}
}

func TestWindowsEngineImageManagedContainmentFailsClosed(t *testing.T) {
	dir := t.TempDir()
	bin := filepath.Join(dir, "engine.exe")
	if err := os.WriteFile(bin, []byte("image fixture"), 0o600); err != nil {
		t.Fatal(err)
	}
	cases := []struct{ name, image, install string }{
		{"missing executable", filepath.Join(dir, "missing.exe"), dir},
		{"missing install directory", bin, filepath.Join(dir, "missing")},
		{"directory is not an executable", dir, dir},
		{"file is not an install directory", bin, bin},
		{"malformed executable", bin + "\x00", dir},
		{"malformed install directory", bin, dir + "\x00"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if isManagedInstallPath(tc.image, tc.install) {
				t.Fatalf("invalid containment must fail closed: image=%q install=%q", tc.image, tc.install)
			}
		})
	}
}

func TestWindowsEngineImageAliasStop(t *testing.T) {
	windowsImageAliases(t, func(t *testing.T, dir, bin, reference string) {
		port := privateTestPorts(t, 1)[0]
		child := spawnFakeListener(t, bin, port)
		ex := windowsImageExecutor(t, dir, bin, reference, port)
		engine := filepath.Base(dir)
		if st, err := ex.Status(engine); err != nil || !st.Running {
			t.Fatalf("adopt physical listener: status=%+v err=%v", st, err)
		}
		if err := ex.Stop(engine); err != nil {
			t.Fatalf("stop aliased managed orphan: %v", err)
		}
		if portServing(port) || !orphanPIDExited(child.Process.Pid) || !fileExists(bin) {
			t.Fatal("Stop must terminate the physical orphan and preserve its executable")
		}
		if enabled, known, err := ex.desired.get(engine); err != nil || !known || enabled {
			t.Fatalf("desired state=(%v,%v,%v), want known Off", enabled, known, err)
		}
	})
}

func TestWindowsEngineImageAliasRestoreReclaim(t *testing.T) {
	windowsImageAliases(t, func(t *testing.T, dir, bin, reference string) {
		port := privateTestPorts(t, 1)[0]
		child := spawnFakeListener(t, bin, port)
		ex := windowsImageExecutor(t, dir, bin, reference, port)
		engine := filepath.Base(dir)
		if err := ex.desired.set(engine, true, intentLocal); err != nil {
			t.Fatal(err)
		}
		st, err := ex.state(engine)
		if err != nil {
			t.Fatal(err)
		}
		// Exercise restore's shared reclaim owner with the alias, then use a
		// canonical Detect reference for the cold launch. Detect also selects
		// the spawn path, so a full aliased restore would execute an uncovered
		// alias instead of the fixed physical test image.
		st.opMu.Lock()
		reclaimed, err := ex.reclaimOrphan(context.Background(), st, engine, port)
		st.opMu.Unlock()
		if err != nil || !reclaimed || !orphanPIDExited(child.Process.Pid) || portServing(port) {
			t.Fatalf("restore reclaim owner must stop aliased orphan: reclaimed=%v err=%v", reclaimed, err)
		}
		cold := windowsImageExecutor(t, dir, bin, bin, port)
		cold.desired = ex.desired
		if err := cold.RestoreEnabled(context.Background()); err != nil {
			t.Fatalf("cold restore through fixed physical image: %v", err)
		}
		st, err = cold.state(engine)
		if err != nil {
			t.Fatal(err)
		}
		st.mu.Lock()
		owned, adopted := st.proc != nil, st.adopted
		st.mu.Unlock()
		pid, image, found := pidOnPort(port)
		if !owned || adopted || !orphanPIDExited(child.Process.Pid) || !found || pid == child.Process.Pid || !isOurEngineImage(image, bin) {
			t.Fatalf("restore must replace orphan with owned cold launch: owned=%v adopted=%v listener=(%d,%q,%v) oldPID=%d", owned, adopted, pid, image, found, child.Process.Pid)
		}
	})
}

func TestWindowsEngineImageAliasUninstall(t *testing.T) {
	windowsImageAliases(t, func(t *testing.T, dir, bin, reference string) {
		port := privateTestPorts(t, 1)[0]
		child := spawnFakeListener(t, bin, port)
		ex := windowsImageExecutor(t, dir, bin, reference, port)
		if err := ex.Uninstall(context.Background(), filepath.Base(dir)); err != nil {
			t.Fatalf("uninstall aliased managed orphan: %v", err)
		}
		if portServing(port) || !orphanPIDExited(child.Process.Pid) || fileExists(bin) {
			t.Fatal("Uninstall must stop the physical orphan before removing its fixture executable")
		}
	})
}

func TestWindowsEngineImageAliasPreservesDistinctListener(t *testing.T) {
	testWindowsProtectedImage(t, false)
}

func TestWindowsEngineImageAliasPreservesExternalInstall(t *testing.T) {
	testWindowsProtectedImage(t, true)
}

func testWindowsProtectedImage(t *testing.T, externalInstall bool) {
	t.Helper()
	test := func(operation string) {
		t.Run(operation, func(t *testing.T) {
			managedDir := testbin.Dir(t, "nvpair-engine-manager", "uninstall-managed-same-name", "fakeu")
			externalDir := testbin.Dir(t, "nvpair-engine-manager", "uninstall-external-same-name", "fakeu")
			bin := filepath.Join(managedDir, testbin.Exe("fakeu"))
			externalBin := filepath.Join(externalDir, filepath.Base(bin))
			copyFile(t, fakeEngineBin, bin)
			copyFile(t, fakeEngineBin, externalBin)
			var reference string
			if externalInstall {
				// The detected image really is the listener, but the reference
				// escapes its nominal managed install through a junction.
				managedDir = filepath.Join(t.TempDir(), "managed")
				if err := os.Mkdir(managedDir, 0o700); err != nil {
					t.Fatal(err)
				}
				reference = filepath.Join(windowsImageJunction(t, managedDir, externalDir), filepath.Base(bin))
			} else {
				reference = filepath.Join(windowsImageJunction(t, t.TempDir(), managedDir), filepath.Base(bin))
			}
			port := privateTestPorts(t, 1)[0]
			child := spawnFakeListener(t, externalBin, port)
			ex := windowsImageExecutor(t, managedDir, bin, reference, port)
			engine := filepath.Base(managedDir)
			image := imagePathForPID(child.Process.Pid)
			if isOurEngineImage(image, reference) != externalInstall || isManagedInstallPath(reference, managedDir) == externalInstall {
				t.Fatalf("invalid protection fixture: image=%q reference=%q managed=%q", image, reference, managedDir)
			}
			if operation == "restore" {
				if err := ex.desired.set(engine, true, intentLocal); err != nil {
					t.Fatal(err)
				}
				if err := ex.RestoreEnabled(context.Background()); err != nil {
					t.Fatalf("restore must preserve external adoption: %v", err)
				}
			} else if operation == "stop" {
				if status, err := ex.Status(engine); err != nil || !status.Running {
					t.Fatalf("adopt external listener: status=%+v err=%v", status, err)
				}
				if err := ex.Stop(engine); err == nil || !strings.Contains(err.Error(), "external management") {
					t.Fatalf("Stop must refuse external listener: %v", err)
				}
			} else {
				if err := ex.Uninstall(context.Background(), engine); err == nil {
					t.Fatal("Uninstall must refuse external listener/install")
				}
			}
			pid, _, found := pidOnPort(port)
			if !found || pid != child.Process.Pid || !pidAlive(pid) || !fileExists(bin) || !fileExists(externalBin) || !fileExists(reference) {
				t.Fatal("refused reclaim must preserve the original listener and all fixture files/references")
			}
		})
	}
	test("stop")
	test("restore")
	test("uninstall")
}
