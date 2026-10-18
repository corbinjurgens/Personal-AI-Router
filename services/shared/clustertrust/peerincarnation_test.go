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
	f.writeMember(t)
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

func (f incarnationFixture) writeMember(t *testing.T) {
	t.Helper()
	data, err := json.Marshal([]struct {
		NodeUUID       string `json:"nodeUuid"`
		State          string `json:"state"`
		ClusterID      string `json:"clusterId"`
		AdmissionEpoch uint64 `json:"admissionEpoch"`
	}{{"peer", "member", f.pin.ClusterID, f.pin.AdmissionEpoch}})
	if err != nil {
		t.Fatalf("encode member: %v", err)
	}
	writeIncarnationFile(t, filepath.Join(f.dir, membersFileName), data)
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
	paths := []string{"admission.json", "members.json", "trusted/peer.json", "node.crt", "node.key"}
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
		requireNoIncarnation(t, f.mesh, "peer")
		f.writeMember(t)
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
		f.writeMember(t)
		if got := requireIncarnation(t, f.mesh); got == first {
			t.Fatal("different cluster reused old incarnation")
		}
	})
}

func TestMeshPeerServiceIncarnationIgnoresStableMemberMetadata(t *testing.T) {
	f := newIncarnationFixture(t)
	before := requireIncarnation(t, f.mesh)
	writeIncarnationFile(t, filepath.Join(f.dir, membersFileName), []byte(`[
		{"nodeUuid":"peer","state":"member","clusterId":"cluster","admissionEpoch":7,"name":"renamed","ipAddress":"192.0.2.40","port":25000,"lastSeen":100},
		{"nodeUuid":"other","state":"member","clusterId":"cluster","admissionEpoch":9}
	]`))
	if got := requireIncarnation(t, f.mesh); got != before {
		t.Fatalf("non-authority member metadata changed incarnation: %q -> %q", before, got)
	}
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

func TestMeshPeerServiceIncarnationChangesOnSameEpochCertificateReplacement(t *testing.T) {
	t.Run("peer certificate", func(t *testing.T) {
		f := newIncarnationFixture(t)
		before := requireIncarnation(t, f.mesh)
		peerPEM, _, _ := genLeaf(t, "peer")
		f.pin.CertPem = string(peerPEM)
		f.writePin(t)
		// A new durable pin cannot be combined with the Mesh's old loaded leaf.
		requireNoIncarnation(t, f.mesh, "peer")
		f.mesh.Refresh()
		if got := requireIncarnation(t, f.mesh); got == before {
			t.Fatal("same-epoch peer certificate replacement reused old authority")
		}
	})
	t.Run("local certificate", func(t *testing.T) {
		f := newIncarnationFixture(t)
		before := requireIncarnation(t, f.mesh)
		selfPEM, selfKey, _ := genLeaf(t, "self")
		writeIdentity(t, f.dir, selfPEM, selfKey)
		f.mesh.Refresh()
		if got := requireIncarnation(t, f.mesh); got == before {
			t.Fatal("same-epoch local certificate replacement reused old authority")
		}
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

func TestMeshPeerServiceIncarnationRejectsDurableMemberAuthority(t *testing.T) {
	test := func(name, record string) {
		t.Run(name, func(t *testing.T) {
			f := newIncarnationFixture(t)
			writeIncarnationFile(t, filepath.Join(f.dir, membersFileName), []byte(record))
			requireNoIncarnation(t, f.mesh, "peer")
			if !f.mesh.Clustered() {
				t.Fatal("service-route rejection changed compatibility membership")
			}
		})
	}
	test("malformed JSON", `[`)
	test("object instead of roster", `{}`)
	test("null roster", `null`)
	test("empty roster", `[]`)
	test("missing peer", `[{"nodeUuid":"other","state":"member","clusterId":"cluster","admissionEpoch":7}]`)
	test("pending inbound", `[{"nodeUuid":"peer","state":"pending-inbound","clusterId":"cluster","admissionEpoch":7}]`)
	test("pending outbound", `[{"nodeUuid":"peer","state":"pending-outbound","clusterId":"cluster","admissionEpoch":7}]`)
	test("missing state", `[{"nodeUuid":"peer","clusterId":"cluster","admissionEpoch":7}]`)
	test("wrong cluster", `[{"nodeUuid":"peer","state":"member","clusterId":"other","admissionEpoch":7}]`)
	test("zero epoch", `[{"nodeUuid":"peer","state":"member","clusterId":"cluster","admissionEpoch":0}]`)
	test("malformed epoch", `[{"nodeUuid":"peer","state":"member","clusterId":"cluster","admissionEpoch":"7"}]`)
	test("pin newer than member", `[{"nodeUuid":"peer","state":"member","clusterId":"cluster","admissionEpoch":6}]`)
	test("pin older than member", `[{"nodeUuid":"peer","state":"member","clusterId":"cluster","admissionEpoch":8}]`)
	test("duplicate current members", `[{"nodeUuid":"peer","state":"member","clusterId":"cluster","admissionEpoch":7},{"nodeUuid":"peer","state":"member","clusterId":"cluster","admissionEpoch":7}]`)
	test("duplicate pending member", `[{"nodeUuid":"peer","state":"member","clusterId":"cluster","admissionEpoch":7},{"nodeUuid":"peer","state":"pending-inbound","clusterId":"cluster","admissionEpoch":7}]`)
	t.Run("missing roster", func(t *testing.T) {
		f := newIncarnationFixture(t)
		if err := os.Remove(filepath.Join(f.dir, membersFileName)); err != nil {
			t.Fatal(err)
		}
		requireNoIncarnation(t, f.mesh, "peer")
	})
	t.Run("unreadable roster", func(t *testing.T) {
		f := newIncarnationFixture(t)
		path := filepath.Join(f.dir, membersFileName)
		if err := os.Remove(path); err != nil {
			t.Fatal(err)
		}
		if err := os.Mkdir(path, 0o700); err != nil {
			t.Fatal(err)
		}
		requireNoIncarnation(t, f.mesh, "peer")
	})
}

func TestMeshPeerServiceIncarnationSameCertificateRepairRequiresBothRecords(t *testing.T) {
	test := func(name string, memberFirst bool) {
		t.Run(name, func(t *testing.T) {
			f := newIncarnationFixture(t)
			before := requireIncarnation(t, f.mesh)
			f.pin.AdmissionEpoch = 8
			if memberFirst {
				f.writeMember(t)
			} else {
				f.writePin(t)
			}
			f.mesh.Refresh()
			requireNoIncarnation(t, f.mesh, "peer")
			if memberFirst {
				f.writePin(t)
			} else {
				f.writeMember(t)
			}
			f.mesh.Refresh()
			if got := requireIncarnation(t, f.mesh); got == before {
				t.Fatal("complete same-cert readmission reused old authority")
			}
		})
	}
	test("pin before member", false)
	test("member before pin", true)
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
	test("member replacement after first member read", 3, func(t *testing.T, f *incarnationFixture) {
		f.pin.AdmissionEpoch = 8
		f.writeMember(t)
	})
	test("member removal after first member read", 3, func(t *testing.T, f *incarnationFixture) {
		if err := os.Remove(filepath.Join(f.dir, membersFileName)); err != nil {
			t.Fatal(err)
		}
	})
	test("local teardown after final pin read", 5, func(t *testing.T, f *incarnationFixture) {
		writeAdmission(t, f.dir, "", 0)
	})
	test("cluster replacement during snapshot", 2, func(t *testing.T, f *incarnationFixture) {
		writeAdmission(t, f.dir, "replacement", 4)
		f.pin.ClusterID = "replacement"
		f.writePin(t)
	})
	test("Mesh pin removed during final admission read", 6, func(t *testing.T, f *incarnationFixture) {
		if err := os.Remove(filepath.Join(f.dir, "trusted", "peer.json")); err != nil {
			t.Fatal(err)
		}
		f.mesh.Refresh()
	})
	test("Mesh identity replaced during final admission read", 6, func(t *testing.T, f *incarnationFixture) {
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
	requireNoIncarnation(t, f.mesh, "peer")
	f.writeMember(t)
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

func TestMeshAdmissionEpochStableReadOnlySnapshot(t *testing.T) {
	f := newIncarnationFixture(t)
	paths := []string{"admission.json", "members.json", "trusted/peer.json", "node.crt", "node.key"}
	before := make(map[string][]byte)
	for _, path := range paths {
		data, err := os.ReadFile(filepath.Join(f.dir, path))
		if err != nil {
			t.Fatalf("read before lookup: %v", err)
		}
		before[path] = data
	}
	if epoch, ok := f.mesh.LocalAdmissionEpoch(); !ok || epoch != 3 {
		t.Fatalf("local epoch = %d, %v; want 3, true", epoch, ok)
	}
	if epoch, ok := f.mesh.PeerAdmissionEpoch("peer"); !ok || epoch != 7 {
		t.Fatalf("peer epoch = %d, %v; want 7, true", epoch, ok)
	}
	for _, path := range paths {
		data, err := os.ReadFile(filepath.Join(f.dir, path))
		if err != nil {
			t.Fatalf("read after lookup: %v", err)
		}
		if !bytes.Equal(data, before[path]) {
			t.Fatalf("epoch lookup rewrote %s", path)
		}
	}
}

func TestMeshAdmissionEpochLocalWithoutPeers(t *testing.T) {
	dir := t.TempDir()
	certPEM, keyPEM, _ := genLeaf(t, "self")
	writeIdentity(t, dir, certPEM, keyPEM)
	writeAdmission(t, dir, "cluster", 11)
	m := Open(dir)
	if epoch, ok := m.LocalAdmissionEpoch(); !ok || epoch != 11 {
		t.Fatalf("local epoch without peers = %d, %v; want 11, true", epoch, ok)
	}
	if epoch, ok := m.PeerAdmissionEpoch("peer"); ok || epoch != 0 {
		t.Fatalf("absent peer epoch = %d, %v; want 0, false", epoch, ok)
	}
}

func TestMeshAdmissionEpochSameCertificatePeerReadmission(t *testing.T) {
	f := newIncarnationFixture(t)
	f.pin.AdmissionEpoch = 8
	f.writePin(t)
	if epoch, ok := f.mesh.PeerAdmissionEpoch("peer"); ok || epoch != 0 {
		t.Fatalf("partial peer readmission epoch = %d, %v; want 0, false", epoch, ok)
	}
	f.writeMember(t)
	if epoch, ok := f.mesh.PeerAdmissionEpoch("peer"); !ok || epoch != 8 {
		t.Fatalf("peer readmission epoch = %d, %v; want 8, true", epoch, ok)
	}
}

func TestMeshAdmissionEpochSameCertificateLocalReadmission(t *testing.T) {
	f := newIncarnationFixture(t)
	writeAdmission(t, f.dir, "cluster", 4)
	if epoch, ok := f.mesh.LocalAdmissionEpoch(); !ok || epoch != 4 {
		t.Fatalf("local readmission epoch = %d, %v; want 4, true", epoch, ok)
	}
}

func TestMeshAdmissionEpochRejectsInvalidLocalAuthority(t *testing.T) {
	test := func(name, record string) {
		t.Run(name, func(t *testing.T) {
			f := newIncarnationFixture(t)
			writeIncarnationFile(t, filepath.Join(f.dir, admissionFileName), []byte(record))
			if epoch, ok := f.mesh.LocalAdmissionEpoch(); ok || epoch != 0 {
				t.Fatalf("invalid local epoch = %d, %v; want 0, false", epoch, ok)
			}
			if epoch, ok := f.mesh.PeerAdmissionEpoch("peer"); ok || epoch != 0 {
				t.Fatalf("peer epoch with invalid local authority = %d, %v; want 0, false", epoch, ok)
			}
		})
	}
	test("malformed JSON", `{`)
	test("missing cluster", `{"counter":3,"epoch":3}`)
	test("zero epoch", `{"counter":3,"clusterId":"cluster","epoch":0}`)
	test("missing counter", `{"clusterId":"cluster","epoch":3}`)
	test("epoch exceeds activated", `{"counter":4,"activated":2,"clusterId":"cluster","epoch":3}`)
	test("activated exceeds counter", `{"counter":3,"activated":4,"clusterId":"cluster","epoch":3}`)
	test("retired admission", `{"counter":3,"retired":true,"clusterId":"cluster","epoch":3}`)
	t.Run("missing admission", func(t *testing.T) {
		f := newIncarnationFixture(t)
		if err := os.Remove(filepath.Join(f.dir, admissionFileName)); err != nil {
			t.Fatal(err)
		}
		if epoch, ok := f.mesh.LocalAdmissionEpoch(); ok || epoch != 0 {
			t.Fatalf("missing local epoch = %d, %v; want 0, false", epoch, ok)
		}
	})
}

func TestMeshAdmissionEpochRejectsInvalidPeerAuthority(t *testing.T) {
	test := func(name string, change func(*testing.T, *incarnationFixture)) {
		t.Run(name, func(t *testing.T) {
			f := newIncarnationFixture(t)
			change(t, &f)
			if epoch, ok := f.mesh.PeerAdmissionEpoch("peer"); ok || epoch != 0 {
				t.Fatalf("invalid peer epoch = %d, %v; want 0, false", epoch, ok)
			}
		})
	}
	test("zero pin epoch", func(t *testing.T, f *incarnationFixture) {
		f.pin.AdmissionEpoch = 0
		f.writePin(t)
	})
	test("cross cluster pin", func(t *testing.T, f *incarnationFixture) {
		f.pin.ClusterID = "other"
		f.writePin(t)
	})
	test("pin and roster disagree", func(t *testing.T, f *incarnationFixture) {
		f.pin.AdmissionEpoch = 8
		f.writeMember(t)
	})
	test("missing roster", func(t *testing.T, f *incarnationFixture) {
		if err := os.Remove(filepath.Join(f.dir, membersFileName)); err != nil {
			t.Fatal(err)
		}
	})
	test("durable certificate differs from loaded pin", func(t *testing.T, f *incarnationFixture) {
		certPEM, _, _ := genLeaf(t, "peer")
		f.pin.CertPem = string(certPEM)
		f.writePin(t)
	})
}

func TestMeshAdmissionEpochRejectsInvalidMeshAndPaths(t *testing.T) {
	var nilMesh *Mesh
	for _, m := range []*Mesh{nilMesh, Open(""), Open(t.TempDir())} {
		if epoch, ok := m.LocalAdmissionEpoch(); ok || epoch != 0 {
			t.Fatalf("invalid Mesh local epoch = %d, %v; want 0, false", epoch, ok)
		}
		if epoch, ok := m.PeerAdmissionEpoch("peer"); ok || epoch != 0 {
			t.Fatalf("invalid Mesh peer epoch = %d, %v; want 0, false", epoch, ok)
		}
	}
	f := newIncarnationFixture(t)
	for _, peer := range []string{"self", "unknown", "", ".", "..", "../peer", `..\peer`, `C:peer`, "peer\x00", " peer"} {
		t.Run(fmt.Sprintf("rejects %q", peer), func(t *testing.T) {
			if epoch, ok := f.mesh.PeerAdmissionEpoch(peer); ok || epoch != 0 {
				t.Fatalf("invalid principal epoch = %d, %v; want 0, false", epoch, ok)
			}
		})
	}
}

func TestMeshAdmissionEpochRejectsLocalAuthorityChangeDuringRead(t *testing.T) {
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
			if epoch, ok := f.mesh.localAdmissionEpoch(read); ok || epoch != 0 {
				t.Fatalf("changing local epoch = %d, %v; want 0, false", epoch, ok)
			}
			if reads < afterRead {
				t.Fatalf("fault was not injected: reads = %d, want at least %d", reads, afterRead)
			}
		})
	}
	test("readmission after first read", 1, func(t *testing.T, f *incarnationFixture) {
		writeAdmission(t, f.dir, "cluster", 4)
	})
	test("teardown during final read", 2, func(t *testing.T, f *incarnationFixture) {
		writeAdmission(t, f.dir, "", 0)
		if err := os.Remove(filepath.Join(f.dir, "trusted", "peer.json")); err != nil {
			t.Fatal(err)
		}
		f.mesh.Refresh()
	})
	test("identity replacement during final read", 2, func(t *testing.T, f *incarnationFixture) {
		certPEM, keyPEM, _ := genLeaf(t, "replacement-self")
		writeIdentity(t, f.dir, certPEM, keyPEM)
		f.mesh.Refresh()
	})
}

func TestMeshAdmissionEpochRejectsPeerAuthorityChangeDuringRead(t *testing.T) {
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
			if epoch, ok := f.mesh.peerAdmissionEpoch("peer", read); ok || epoch != 0 {
				t.Fatalf("changing peer epoch = %d, %v; want 0, false", epoch, ok)
			}
			if reads < afterRead {
				t.Fatalf("fault was not injected: reads = %d, want at least %d", reads, afterRead)
			}
		})
	}
	test("local readmission after first read", 1, func(t *testing.T, f *incarnationFixture) {
		writeAdmission(t, f.dir, "cluster", 4)
	})
	test("same certificate peer readmission after pin read", 2, func(t *testing.T, f *incarnationFixture) {
		f.pin.AdmissionEpoch = 8
		f.writePin(t)
		f.writeMember(t)
	})
	test("roster change after member read", 3, func(t *testing.T, f *incarnationFixture) {
		f.pin.AdmissionEpoch = 8
		f.writeMember(t)
	})
	test("loaded pin removed during final read", 6, func(t *testing.T, f *incarnationFixture) {
		if err := os.Remove(filepath.Join(f.dir, "trusted", "peer.json")); err != nil {
			t.Fatal(err)
		}
		f.mesh.Refresh()
	})
}
