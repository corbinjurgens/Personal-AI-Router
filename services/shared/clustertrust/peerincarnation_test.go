// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package clustertrust

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"
)

type incarnationPin struct {
	NodeUUID       string `json:"nodeUuid"`
	CertPem        string `json:"certPem"`
	ClusterID      string `json:"clusterId"`
	AdmissionEpoch uint64 `json:"admissionEpoch"`
}

type incarnationFixture struct {
	mesh *Mesh
	dir  string
	pin  incarnationPin
}

func newIncarnationFixture(t *testing.T) incarnationFixture {
	t.Helper()
	dir := t.TempDir()
	selfPEM, selfKey, _ := genLeaf(t, "self")
	peerPEM, _, _ := genLeaf(t, "peer")
	writeIdentity(t, dir, selfPEM, selfKey)
	writeAdmission(t, dir, "cluster", 3)
	f := incarnationFixture{dir: dir, pin: incarnationPin{"peer", string(peerPEM), "cluster", 7}}
	f.writePin(t)
	f.mesh = Open(dir)
	return f
}

func writeIncarnationFile(t *testing.T, path string, data []byte) {
	t.Helper()
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatalf("write authority: %v", err)
	}
}

func (f incarnationFixture) writePin(t *testing.T) {
	t.Helper()
	if err := os.MkdirAll(filepath.Join(f.dir, "trusted"), 0o700); err != nil {
		t.Fatalf("mkdir trusted: %v", err)
	}
	data, err := json.Marshal(f.pin)
	if err != nil {
		t.Fatalf("encode pin: %v", err)
	}
	writeIncarnationFile(t, filepath.Join(f.dir, "trusted", "peer.json"), data)
}

func requireIncarnation(t *testing.T, m *Mesh) string {
	t.Helper()
	incarnation, ok := m.PeerServiceIncarnation("peer")
	if !ok || incarnation == "" {
		t.Fatalf("incarnation = %q, %v; want nonempty current authority", incarnation, ok)
	}
	return incarnation
}

func requireNoIncarnation(t *testing.T, m *Mesh, peer string) {
	t.Helper()
	if incarnation, ok := m.PeerServiceIncarnation(peer); ok || incarnation != "" {
		t.Fatalf("incarnation = %q, %v; want empty, false", incarnation, ok)
	}
}

func TestMeshPeerServiceIncarnationStableReadOnlySnapshot(t *testing.T) {
	f := newIncarnationFixture(t)
	paths := []string{"admission.json", "trusted/peer.json", "node.crt", "node.key"}
	before := make(map[string][]byte)
	for _, path := range paths {
		data, err := os.ReadFile(filepath.Join(f.dir, path))
		if err != nil {
			t.Fatalf("read before lookup: %v", err)
		}
		before[path] = data
	}
	first := requireIncarnation(t, f.mesh)
	f.mesh.Refresh()
	if got := requireIncarnation(t, f.mesh); got != first {
		t.Fatalf("no-op refresh changed incarnation: %q -> %q", first, got)
	}
	if got := requireIncarnation(t, Open(f.dir)); got != first {
		t.Fatalf("reopening changed incarnation: %q -> %q", first, got)
	}
	for _, path := range paths {
		data, err := os.ReadFile(filepath.Join(f.dir, path))
		if err != nil {
			t.Fatalf("read after lookup: %v", err)
		}
		if !bytes.Equal(data, before[path]) {
			t.Fatalf("lookup rewrote %s", path)
		}
	}
}

func TestMeshPeerServiceIncarnationSameCertificateReadmission(t *testing.T) {
	t.Run("peer admission", func(t *testing.T) {
		f := newIncarnationFixture(t)
		first := requireIncarnation(t, f.mesh)
		f.pin.AdmissionEpoch = 8
		f.writePin(t)
		if got := requireIncarnation(t, f.mesh); got == first {
			t.Fatal("same-cert peer readmission reused old incarnation")
		}
	})
	t.Run("local admission", func(t *testing.T) {
		f := newIncarnationFixture(t)
		first := requireIncarnation(t, f.mesh)
		writeAdmission(t, f.dir, "cluster", 4)
		if got := requireIncarnation(t, f.mesh); got == first {
			t.Fatal("same-cert local readmission reused old incarnation")
		}
	})
	t.Run("cluster identity", func(t *testing.T) {
		f := newIncarnationFixture(t)
		first := requireIncarnation(t, f.mesh)
		writeAdmission(t, f.dir, "other:cluster", 3)
		f.pin.ClusterID = "other:cluster"
		f.writePin(t)
		if got := requireIncarnation(t, f.mesh); got == first {
			t.Fatal("different cluster reused old incarnation")
		}
	})
}

func TestMeshPeerServiceIncarnationRejectsLocalAuthority(t *testing.T) {
	test := func(name, record string) {
		t.Run(name, func(t *testing.T) {
			f := newIncarnationFixture(t)
			writeIncarnationFile(t, filepath.Join(f.dir, admissionFileName), []byte(record))
			// Leave the Mesh cache clustered to prove fresh durable authority wins.
			requireNoIncarnation(t, f.mesh, "peer")
		})
	}
	test("malformed JSON", `{`)
	test("null record", `null`)
	test("cleared admission", `{"counter":3,"activated":3,"retired":true}`)
	test("missing cluster", `{"counter":3,"epoch":3}`)
	test("zero epoch", `{"counter":3,"clusterId":"cluster","epoch":0}`)
	test("negative epoch", `{"counter":3,"clusterId":"cluster","epoch":-1}`)
	test("noninteger epoch", `{"counter":3,"clusterId":"cluster","epoch":1.5}`)
	test("string epoch", `{"counter":3,"clusterId":"cluster","epoch":"3"}`)
	test("missing counter", `{"clusterId":"cluster","epoch":3}`)
	test("epoch exceeds counter", `{"counter":2,"clusterId":"cluster","epoch":3}`)
	test("epoch exceeds activated", `{"counter":4,"activated":2,"clusterId":"cluster","epoch":3}`)
	test("activated exceeds counter", `{"counter":3,"activated":4,"clusterId":"cluster","epoch":3}`)
	test("retired active pair", `{"counter":3,"retired":true,"clusterId":"cluster","epoch":3}`)
	t.Run("missing admission", func(t *testing.T) {
		f := newIncarnationFixture(t)
		if err := os.Remove(filepath.Join(f.dir, admissionFileName)); err != nil {
			t.Fatal(err)
		}
		requireNoIncarnation(t, f.mesh, "peer")
	})
	t.Run("unreadable admission", func(t *testing.T) {
		f := newIncarnationFixture(t)
		path := filepath.Join(f.dir, admissionFileName)
		if err := os.Remove(path); err != nil {
			t.Fatal(err)
		}
		if err := os.Mkdir(path, 0o700); err != nil {
			t.Fatal(err)
		}
		requireNoIncarnation(t, f.mesh, "peer")
	})
}

func TestMeshPeerServiceIncarnationRejectsPeerAuthority(t *testing.T) {
	test := func(name string, mutate func(*testing.T, *incarnationFixture)) {
		t.Run(name, func(t *testing.T) {
			f := newIncarnationFixture(t)
			mutate(t, &f)
			requireNoIncarnation(t, f.mesh, "peer")
		})
	}
	test("zero epoch", func(t *testing.T, f *incarnationFixture) {
		f.pin.AdmissionEpoch = 0
		f.writePin(t)
	})
	test("cross cluster", func(t *testing.T, f *incarnationFixture) {
		f.pin.ClusterID = "other-cluster"
		f.writePin(t)
	})
	test("missing cluster", func(t *testing.T, f *incarnationFixture) {
		f.pin.ClusterID = ""
		f.writePin(t)
	})
	test("inner UUID differs from filename", func(t *testing.T, f *incarnationFixture) {
		f.pin.NodeUUID = "stranger"
		f.writePin(t)
	})
	test("certificate principal differs from filename", func(t *testing.T, f *incarnationFixture) {
		certPEM, _, _ := genLeaf(t, "stranger")
		f.pin.CertPem = string(certPEM)
		f.writePin(t)
	})
	test("malformed certificate", func(t *testing.T, f *incarnationFixture) {
		f.pin.CertPem = "invalid"
		f.writePin(t)
	})
	test("certificate differs from current Mesh pin", func(t *testing.T, f *incarnationFixture) {
		certPEM, _, _ := genLeaf(t, "peer")
		f.pin.CertPem = string(certPEM)
		f.writePin(t)
	})
	test("pin removed before refresh", func(t *testing.T, f *incarnationFixture) {
		if err := os.Remove(filepath.Join(f.dir, "trusted", "peer.json")); err != nil {
			t.Fatal(err)
		}
	})
	test("malformed JSON", func(t *testing.T, f *incarnationFixture) {
		writeIncarnationFile(t, filepath.Join(f.dir, "trusted", "peer.json"), []byte(`{`))
	})
	test("malformed epoch", func(t *testing.T, f *incarnationFixture) {
		data, err := json.Marshal(map[string]string{
			"nodeUuid": "peer", "certPem": f.pin.CertPem, "clusterId": "cluster", "admissionEpoch": "7",
		})
		if err != nil {
			t.Fatal(err)
		}
		writeIncarnationFile(t, filepath.Join(f.dir, "trusted", "peer.json"), data)
	})
	test("pin unreadable", func(t *testing.T, f *incarnationFixture) {
		path := filepath.Join(f.dir, "trusted", "peer.json")
		if err := os.Remove(path); err != nil {
			t.Fatal(err)
		}
		if err := os.Mkdir(path, 0o700); err != nil {
			t.Fatal(err)
		}
	})
}

func TestMeshPeerServiceIncarnationRejectsInvalidMeshAndPaths(t *testing.T) {
	var nilMesh *Mesh
	requireNoIncarnation(t, nilMesh, "peer")
	requireNoIncarnation(t, Open(""), "peer")
	f := newIncarnationFixture(t)
	requireNoIncarnation(t, f.mesh, "self")
	requireNoIncarnation(t, f.mesh, "unknown")
	for _, peer := range []string{"", ".", "..", "../peer", "peer/child", `..\peer`, `C:peer`, "/peer", "peer\x00", " peer"} {
		t.Run(fmt.Sprintf("rejects %q before file reads", peer), func(t *testing.T) {
			reads := 0
			read := func(path string) ([]byte, error) {
				reads++
				return os.ReadFile(path)
			}
			if got, ok := f.mesh.peerServiceIncarnation(peer, read); ok || got != "" {
				t.Fatalf("incarnation = %q, %v; want empty, false", got, ok)
			}
			if reads != 0 {
				t.Fatalf("unsafe peer caused %d file reads", reads)
			}
		})
	}
	t.Run("no loaded identity", func(t *testing.T) {
		if err := os.Remove(filepath.Join(f.dir, "node.key")); err != nil {
			t.Fatal(err)
		}
		requireNoIncarnation(t, Open(f.dir), "peer")
	})
}

func TestMeshPeerServiceIncarnationRejectsAuthorityChangeDuringRead(t *testing.T) {
	test := func(name string, afterRead int, change func(*testing.T, *incarnationFixture)) {
		t.Run(name, func(t *testing.T) {
			f := newIncarnationFixture(t)
			reads := 0
			read := func(path string) ([]byte, error) {
				data, err := os.ReadFile(path)
				if err != nil {
					return nil, err
				}
				reads++
				if reads == afterRead {
					change(t, &f)
				}
				return data, nil
			}
			if got, ok := f.mesh.peerServiceIncarnation("peer", read); ok || got != "" {
				t.Fatalf("changing snapshot returned %q, %v", got, ok)
			}
			if reads < afterRead {
				t.Fatalf("fault was not injected: reads = %d, want at least %d", reads, afterRead)
			}
		})
	}
	test("local readmission after first admission read", 1, func(t *testing.T, f *incarnationFixture) {
		writeAdmission(t, f.dir, "cluster", 4)
	})
	test("same-cert peer readmission after first pin read", 2, func(t *testing.T, f *incarnationFixture) {
		f.pin.AdmissionEpoch = 8
		f.writePin(t)
	})
	test("pin removal after first pin read", 2, func(t *testing.T, f *incarnationFixture) {
		if err := os.Remove(filepath.Join(f.dir, "trusted", "peer.json")); err != nil {
			t.Fatal(err)
		}
	})
	test("local teardown after final pin read", 3, func(t *testing.T, f *incarnationFixture) {
		writeAdmission(t, f.dir, "", 0)
	})
	test("cluster replacement during snapshot", 2, func(t *testing.T, f *incarnationFixture) {
		writeAdmission(t, f.dir, "replacement", 4)
		f.pin.ClusterID = "replacement"
		f.writePin(t)
	})
	test("Mesh pin removed during final admission read", 4, func(t *testing.T, f *incarnationFixture) {
		if err := os.Remove(filepath.Join(f.dir, "trusted", "peer.json")); err != nil {
			t.Fatal(err)
		}
		f.mesh.Refresh()
	})
	test("Mesh identity replaced during final admission read", 4, func(t *testing.T, f *incarnationFixture) {
		certPEM, keyPEM, _ := genLeaf(t, "replacement-self")
		writeIdentity(t, f.dir, certPEM, keyPEM)
		f.mesh.Refresh()
	})
}

func TestMeshPeerServiceIncarnationRecoversAfterPartialAuthority(t *testing.T) {
	f := newIncarnationFixture(t)
	first := requireIncarnation(t, f.mesh)
	writeAdmission(t, f.dir, "", 0)
	f.mesh.Refresh()
	requireNoIncarnation(t, f.mesh, "peer")
	writeAdmission(t, f.dir, "new-cluster", 4)
	f.mesh.Refresh()
	requireNoIncarnation(t, f.mesh, "peer")
	f.pin.ClusterID, f.pin.AdmissionEpoch = "new-cluster", 8
	f.writePin(t)
	f.mesh.Refresh()
	if got := requireIncarnation(t, f.mesh); got == first {
		t.Fatal("complete readmission reused old authority")
	}
}

func TestMeshPeerServiceIncarnationConcurrentRefresh(t *testing.T) {
	f := newIncarnationFixture(t)
	want := requireIncarnation(t, f.mesh)
	var workers sync.WaitGroup
	workers.Add(2)
	go func() {
		defer workers.Done()
		for range 50 {
			f.mesh.Refresh()
		}
	}()
	go func() {
		defer workers.Done()
		for range 50 {
			if got, ok := f.mesh.PeerServiceIncarnation("peer"); !ok || got != want {
				t.Errorf("stable concurrent authority = %q, %v; want %q, true", got, ok, want)
				return
			}
		}
	}()
	workers.Wait()
}
