// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"net"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	settings "nvpair-shared/enginesettings"
	"nvpair-shared/modelconfig"
)

func removalRelayPreview(t *testing.T, f modelLoadFixture, envelope settings.WorkerRequest[settings.Request]) Message {
	t.Helper()
	before := f.st.plat.Runtime
	server, client := net.Pipe()
	t.Cleanup(func() { _ = server.Close(); _ = client.Close() })
	if err := client.SetDeadline(time.Now().Add(5 * time.Second)); err != nil {
		t.Fatal(err)
	}
	m := &Manager{exec: f.ex, codec: NewCodec(server)}
	raw, err := json.Marshal(envelope)
	if err != nil {
		t.Fatal(err)
	}
	id := json.RawMessage(`1`)
	if !m.handleSettingsMessage(context.Background(), &Message{JSONRPC: "2.0", ID: &id, Method: "engine:preview-launch", Params: raw}) {
		t.Fatal("preview-launch was not handled")
	}
	var reply Message
	if err := json.NewDecoder(client).Decode(&reply); err != nil {
		t.Fatal(err)
	}
	assertNoSettingsOverride(t, f.ex)
	if !reflect.DeepEqual(before, f.st.plat.Runtime) || f.st.running || f.st.proc != nil || len(m.settingsRelay.pending) != 0 {
		t.Fatal("preview changed runtime, activated an engine or leaked a relay")
	}
	return reply
}

func TestModelSettingsRemovalRelayAcceptsFrozenPreview(t *testing.T) {
	test := func(name string, last bool, wantLoads int) {
		t.Run(name, func(t *testing.T) {
			f, preview, change := reviewedRemovalConversionFixture(t, last)
			reply := removalRelayPreview(t, f, settings.WorkerRequest[settings.Request]{Request: settings.Request{Engine: "fake", Settings: preview.Settings}, ReviewedModelChange: change})
			if reply.Error != nil {
				t.Fatalf("reviewed frozen removal refused: %+v", reply.Error)
			}
			var got settings.Preview
			if err := modelconfig.DecodeStrict(reply.Result, &got); err != nil {
				t.Fatal(err)
			}
			if len(got.Errors) != 0 || got.Conflict != nil || !got.Settings.Equal(preview.Settings) || len(got.Settings.ModelLoads) != wantLoads {
				t.Fatalf("frozen removal changed or refused: %+v", got)
			}
		})
	}
	test("partial removal", false, 1)
	test("final removal", true, 0)
}

func TestModelSettingsRemovalRelayRefusesInvalidNativeProof(t *testing.T) {
	test := func(name string, last bool, edit func(modelLoadFixture, *settings.WorkerRequest[settings.Request])) {
		t.Run(name, func(t *testing.T) {
			f, preview, change := reviewedRemovalConversionFixture(t, last)
			envelope := settings.WorkerRequest[settings.Request]{Request: settings.Request{Engine: "fake", Settings: preview.Settings}, ReviewedModelChange: change}
			edit(f, &envelope)
			reply := removalRelayPreview(t, f, envelope)
			if reply.Error == nil {
				t.Fatalf("invalid frozen removal produced success: %s", reply.Result)
			}
		})
	}
	missing := func(_ modelLoadFixture, p *settings.WorkerRequest[settings.Request]) {
		p.ReviewedModelChange.PreviewDigest = ""
	}
	wrong := func(_ modelLoadFixture, p *settings.WorkerRequest[settings.Request]) {
		p.ReviewedModelChange.PreviewDigest = strings.Repeat("f", 64)
	}
	test("partial missing native digest", false, missing)
	test("final missing native digest", true, missing)
	test("partial wrong native digest", false, wrong)
	test("final wrong native digest", true, wrong)
	test("wrong selected profile", false, func(_ modelLoadFixture, p *settings.WorkerRequest[settings.Request]) {
		p.ReviewedModelChange.ProfileKey = "medium-model"
	})
	test("extra omission", false, func(_ modelLoadFixture, p *settings.WorkerRequest[settings.Request]) {
		p.Request.Settings.ModelLoads = nil
	})
	test("portable patch beside frozen settings", false, func(_ modelLoadFixture, p *settings.WorkerRequest[settings.Request]) {
		p.Request.ModelChange = p.ReviewedModelChange
	})
}

func TestModelSettingsRemovalRelayRefusesChangedNativeSource(t *testing.T) {
	t.Run("malformed original source", func(t *testing.T) {
		f, preview, change := reviewedRemovalConversionFixture(t, false)
		writeTestFile(t, f.source, []byte("not a preset"))
		reply := removalRelayPreview(t, f, settings.WorkerRequest[settings.Request]{Request: settings.Request{Engine: "fake", Settings: preview.Settings}, ReviewedModelChange: change})
		if reply.Error == nil {
			t.Fatal("malformed original source passed frozen preview")
		}
	})
	t.Run("final removal native build drift", func(t *testing.T) {
		f := nativeModelLoadConversionFixture(t)
		writeTestFile(t, f.source, []byte("[weak-model]\nmodel = weak.gguf\nctx-size = 4096\nparallel = 1\ngpu-layers = 0\n"))
		request := settingsRequest(t, f.ex)
		request.ModelChange = &modelconfig.ModelChange{ProfileKey: "weak-model", ModelID: "weak-model", Remove: true, ExpectedPolicyRevision: "0"}
		preview, err := f.ex.PreviewLaunch(request)
		if err != nil || len(preview.Errors) != 0 || len(preview.Settings.ModelLoads) != 0 || len(preview.PreviewDigest) != 64 {
			t.Fatalf("native removal setup: preview=%+v err=%v", preview, err)
		}
		change := request.ModelChange.Clone()
		change.PreviewDigest = preview.PreviewDigest
		writeTestFile(t, filepath.Join(f.root, "libggml-base.dylib"), []byte("changed native library content after review"))
		reply := removalRelayPreview(t, f, settings.WorkerRequest[settings.Request]{Request: settings.Request{Engine: "fake", Settings: preview.Settings}, ReviewedModelChange: &change})
		if reply.Error == nil {
			t.Fatal("changed native build passed frozen final-removal preview")
		}
	})
}

func TestModelSettingsRemovalRelayKeepsOrdinaryPreview(t *testing.T) {
	f := newModelLoadFixture(t)
	request := settingsRequest(t, f.ex)
	reply := removalRelayPreview(t, f, settings.WorkerRequest[settings.Request]{Request: request})
	if reply.Error != nil {
		t.Fatalf("ordinary preview refused: %+v", reply.Error)
	}
	var got settings.Preview
	if err := modelconfig.DecodeStrict(reply.Result, &got); err != nil {
		t.Fatal(err)
	}
	if len(got.Errors) != 0 || got.Conflict != nil || !got.Settings.Equal(request.Settings) {
		t.Fatalf("ordinary preview changed or refused: %+v", got)
	}
}
