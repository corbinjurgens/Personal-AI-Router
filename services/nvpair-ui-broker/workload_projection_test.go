// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"nvpair-ui-broker/workloadstore"
)

const privacyInfo = `{"id":"7","model":"model","engine":"ollama","runId":"run-1","state":"running","originatedFrom":"host","scheduledOn":"peer","createdAt":1,"startedAt":null,"completedAt":null,"error":null,"requesterId":null,"requestedModel":"weak","seq":2}`

func privacyEnvelope(info string) json.RawMessage {
	return json.RawMessage(`{"workloadInfo":` + info + `}`)
}

func privacyObject(t *testing.T, raw json.RawMessage) map[string]json.RawMessage {
	t.Helper()
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err != nil || obj == nil {
		t.Fatalf("expected JSON object: %v", err)
	}
	return obj
}

func assertPrivacyInfo(t *testing.T, raw json.RawMessage, want string) {
	t.Helper()
	got := privacyObject(t, raw)
	expected := privacyObject(t, json.RawMessage(want))
	// This literal is deliberately independent of the production projection.
	allowed := strings.Fields("id model engine runId state originatedFrom scheduledOn createdAt startedAt completedAt error requesterId requestedModel seq")
	for key := range got {
		found := false
		for _, name := range allowed {
			found = found || key == name
		}
		if !found {
			t.Fatal("unknown field escaped projection")
		}
	}
	if !reflect.DeepEqual(got, expected) {
		t.Fatal("projection changed legitimate fields")
	}
}

type privacyWriteCloser struct{ *syncBuffer }

func (privacyWriteCloser) Close() error { return nil }

func privacyBroker(t *testing.T) (*Broker, *syncBuffer, *syncBuffer, *syncBuffer) {
	t.Helper()
	b, client := newPolicyBroker(t)
	b.nodeID = "host"
	b.workloads = workloadstore.New()
	b.workloadsSubscribed = true
	scheduler, manager := &syncBuffer{}, &syncBuffer{}
	b.setScheduler(&rpcWorker{peer: NewPeer(NewCodec(scheduler))})
	b.setWorkloadMgr(&workloadManagerProcess{stdin: privacyWriteCloser{manager}})
	return b, client, scheduler, manager
}

func TestWorkloadPrivacyProjection(t *testing.T) {
	for _, method := range []string{"workload:submitted", "workload:started", "workload:completed", "workload:errored", "workloads:upsert"} {
		t.Run(method, func(t *testing.T) {
			input := json.RawMessage(`{"synthetic-secret-key":"synthetic-secret-value","workloadInfo":` + strings.TrimSuffix(privacyInfo, "}") + `,"synthetic-nested-secret":{"body":"synthetic-secret-value"}}}`)
			projected, category, count := projectWorkloadFrame(method, input)
			if category != "stripped" || count != 2 {
				t.Fatalf("diagnostic = %s/%d", category, count)
			}
			env := privacyObject(t, projected)
			if len(env) != 1 {
				t.Fatal("outer unknown field escaped")
			}
			assertPrivacyInfo(t, env["workloadInfo"], privacyInfo)
		})
	}
	t.Run("optional absence", func(t *testing.T) {
		info := `{"id":"7","model":"model","engine":"ollama","state":"queued","originatedFrom":"host"}`
		out, category, _ := projectWorkloadFrame("workload:started", privacyEnvelope(info))
		if category != "accepted" {
			t.Fatal(category)
		}
		assertPrivacyInfo(t, privacyObject(t, out)["workloadInfo"], info)
	})
	t.Run("nullable values present", func(t *testing.T) {
		info := strings.NewReplacer(`"startedAt":null`, `"startedAt":2`, `"completedAt":null`, `"completedAt":3`, `"error":null`, `"error":"legitimate error"`, `"requesterId":null`, `"requesterId":"requester"`).Replace(privacyInfo)
		out, category, _ := projectWorkloadFrame("workload:completed", privacyEnvelope(info))
		if category != "accepted" {
			t.Fatal(category)
		}
		assertPrivacyInfo(t, privacyObject(t, out)["workloadInfo"], info)
	})
	t.Run("removal", func(t *testing.T) {
		out, category, count := projectWorkloadFrame("workloads:remove", json.RawMessage(`{"workloadId":"7","originatedFrom":"host","synthetic-secret-key":"synthetic-secret-value"}`))
		if category != "stripped" || count != 1 || !reflect.DeepEqual(privacyObject(t, out), privacyObject(t, json.RawMessage(`{"workloadId":"7","originatedFrom":"host"}`))) {
			t.Fatal("removal projection lost identity or retained unknown fields")
		}
	})
	t.Run("well formed unknown numeric metadata", func(t *testing.T) {
		out, category, count := projectWorkloadFrame("workload:started", json.RawMessage(`{"workloadInfo":`+privacyInfo+`,"synthetic-secret-key":1e999}`))
		if category != "stripped" || count != 1 {
			t.Fatal("well formed unknown metadata rejected")
		}
		assertPrivacyInfo(t, privacyObject(t, out)["workloadInfo"], privacyInfo)
	})
}

func TestWorkloadPrivacyProjectionRejectsBeforeEffects(t *testing.T) {
	cases := map[string]json.RawMessage{
		"null envelope":          json.RawMessage(`null`),
		"array envelope":         json.RawMessage(`[]`),
		"missing info":           json.RawMessage(`{}`),
		"null info":              json.RawMessage(`{"workloadInfo":null}`),
		"array info":             json.RawMessage(`{"workloadInfo":[]}`),
		"trailing JSON":          append(privacyEnvelope(privacyInfo), []byte(` {}`)...),
		"duplicate outer":        json.RawMessage(`{"workloadInfo":{},"workloadInfo":` + privacyInfo + `}`),
		"duplicate known":        privacyEnvelope(strings.TrimSuffix(privacyInfo, "}") + `,"id":"8"}`),
		"duplicate unknown":      privacyEnvelope(strings.TrimSuffix(privacyInfo, "}") + `,"synthetic-secret-key":{"x":1,"x":2}}`),
		"duplicate inside array": privacyEnvelope(strings.TrimSuffix(privacyInfo, "}") + `,"synthetic-secret-key":[{"x":1,"x":2}]}`),
		"escaped duplicate":      privacyEnvelope(strings.TrimSuffix(privacyInfo, "}") + `,"\u0069d":"8"}`),
	}
	for _, key := range strings.Fields("id engine runId originatedFrom scheduledOn model state requestedModel createdAt seq startedAt completedAt error requesterId") {
		obj := privacyObject(t, json.RawMessage(privacyInfo))
		obj[key] = json.RawMessage(`{}`)
		raw, err := json.Marshal(obj)
		if err != nil {
			t.Fatal(err)
		}
		cases["wrong type "+key] = privacyEnvelope(string(raw))
		if key != "startedAt" && key != "completedAt" && key != "error" && key != "requesterId" {
			obj[key] = json.RawMessage(`null`)
			raw, err = json.Marshal(obj)
			if err != nil {
				t.Fatal(err)
			}
			cases["null "+key] = privacyEnvelope(string(raw))
		}
	}
	for _, key := range []string{"id", "engine", "runId"} {
		cases["empty "+key] = privacyEnvelope(strings.Replace(privacyInfo, `"`+key+`":"`+map[string]string{"id": "7", "engine": "ollama", "runId": "run-1"}[key]+`"`, `"`+key+`":""`, 1))
	}
	for _, key := range []string{"id", "model", "engine", "state", "originatedFrom"} {
		obj := privacyObject(t, json.RawMessage(privacyInfo))
		delete(obj, key)
		raw, err := json.Marshal(obj)
		if err != nil {
			t.Fatal(err)
		}
		cases["missing "+key] = privacyEnvelope(string(raw))
	}
	cases["invalid identity"] = privacyEnvelope(strings.Replace(privacyInfo, "run-1", "invalid/run", 1))
	for _, value := range []string{"-1", "1.5", "9223372036854775808"} {
		cases["invalid integer "+value] = privacyEnvelope(strings.Replace(privacyInfo, `"seq":2`, `"seq":`+value, 1))
	}
	for name, input := range cases {
		t.Run(name, func(t *testing.T) {
			logs := captureLogs(t)
			b, client, scheduler, manager := privacyBroker(t)
			b.routeProxyWorkload("workload:started", input)
			b.forwardWorkloadManagerNotification("workloads:upsert", input)
			if b.workloads.Len() != 0 || len(client.frames()) != 0 || len(scheduler.frames()) != 0 || len(manager.frames()) != 0 {
				t.Fatal("malformed frame caused effects")
			}
			if bufferText(logs) != "" {
				t.Fatal("malformed frame logged input")
			}
		})
	}
	for _, input := range []string{`null`, `[]`, `{"workloadId":null}`, `{"workloadId":7}`, `{"workloadId":""}`, `{"workloadId":"7","originatedFrom":null}`, `{"workloadId":"7","originatedFrom":"bad/origin"}`, `{"workloadId":"7","workloadId":"8"}`} {
		t.Run("malformed removal "+input, func(t *testing.T) {
			b, client, scheduler, manager := privacyBroker(t)
			b.routeProxyWorkload("workloads:remove", json.RawMessage(input))
			b.forwardWorkloadManagerNotification("workloads:remove", json.RawMessage(input))
			if len(client.frames())+len(scheduler.frames())+len(manager.frames()) != 0 {
				t.Fatal("malformed removal escaped")
			}
		})
	}
}

func TestWorkloadPrivacyProjectionLiveEffects(t *testing.T) {
	logs := captureLogs(t)
	b, client, scheduler, manager := privacyBroker(t)
	info := strings.TrimSuffix(privacyInfo, "}") + `,"synthetic-secret-key":"synthetic-secret-value"}`
	b.routeProxyWorkload("workload:started", privacyEnvelope(info))
	for _, output := range []*syncBuffer{client, scheduler, manager} {
		frames := output.frames()
		if len(frames) != 1 {
			t.Fatalf("frames = %d", len(frames))
		}
		assertPrivacyInfo(t, privacyObject(t, frames[0].Params)["workloadInfo"], privacyInfo)
	}
	snapshot := b.workloads.Snapshot()
	if len(snapshot) != 1 {
		t.Fatal("missing stored workload")
	}
	assertPrivacyInfo(t, snapshot[0], privacyInfo)
	if strings.Contains(bufferText(logs), "synthetic-secret") {
		t.Fatal("secret logged")
	}
}

func TestWorkloadPrivacyProjectionPeerEffects(t *testing.T) {
	logs := captureLogs(t)
	b, client, scheduler, manager := privacyBroker(t)
	info := strings.TrimSuffix(privacyInfo, "}") + `,"synthetic-secret-key":"synthetic-secret-value"}`
	b.forwardWorkloadManagerNotification("workloads:upsert", json.RawMessage(`{"synthetic-outer-secret":"synthetic-secret-value","workloadInfo":`+info+`}`))
	for _, output := range []*syncBuffer{client, scheduler} {
		frames := output.frames()
		if len(frames) != 1 {
			t.Fatal("missing peer update")
		}
		env := privacyObject(t, frames[0].Params)
		if len(env) != 1 {
			t.Fatal("outer field escaped")
		}
		assertPrivacyInfo(t, env["workloadInfo"], privacyInfo)
	}
	if len(manager.frames()) != 0 {
		t.Fatal("peer event rebroadcast")
	}
	assertPrivacyInfo(t, b.workloads.Snapshot()[0], privacyInfo)
	if strings.Contains(bufferText(logs), "synthetic-secret") || strings.Contains(bufferText(logs), "synthetic-outer") {
		t.Fatal("secret logged")
	}
}

func TestWorkloadPrivacyProjectionLocalStampAndRemoval(t *testing.T) {
	b, client, scheduler, manager := privacyBroker(t)
	local := strings.Replace(privacyInfo, `"originatedFrom":"host"`, `"originatedFrom":""`, 1)
	b.routeProxyWorkload("workload:started", privacyEnvelope(local))
	for _, output := range []*syncBuffer{client, scheduler, manager} {
		frames := output.frames()
		if len(frames) != 1 {
			t.Fatal("missing local update")
		}
		assertPrivacyInfo(t, privacyObject(t, frames[0].Params)["workloadInfo"], privacyInfo)
	}
	b.routeProxyWorkload("workloads:remove", json.RawMessage(`{"workloadId":"7","synthetic-secret-key":"synthetic-secret-value"}`))
	if b.workloads.Len() != 0 {
		t.Fatal("removal failed to update store")
	}
	for _, output := range []*syncBuffer{client, scheduler, manager} {
		frames := output.frames()
		if len(frames) != 2 {
			t.Fatal("missing removal")
		}
		if !reflect.DeepEqual(privacyObject(t, frames[1].Params), privacyObject(t, json.RawMessage(`{"workloadId":"7","originatedFrom":"host"}`))) {
			t.Fatal("removal lost identity or forwarded unknown metadata")
		}
	}
}

func TestWorkloadPrivacyProjectionRestartInitial(t *testing.T) {
	path := filepath.Join(t.TempDir(), "history.json")
	s := workloadstore.New().WithPersistence(path)
	info := strings.Replace(privacyInfo, `"state":"running"`, `"state":"completed"`, 1)
	info = strings.Replace(info, `"completedAt":null`, `"completedAt":`+jsonNumber(t, time.Now().UnixMilli()), 1)
	dirty := strings.TrimSuffix(info, "}") + `,"synthetic-secret-key":"synthetic-secret-value"}`
	in, ok := workloadstore.ParseIncoming(json.RawMessage(dirty))
	if !ok || !s.Apply(in) {
		t.Fatal("seed persisted record")
	}
	if err := s.Flush(); err != nil {
		t.Fatal(err)
	}
	b, client := newPolicyBroker(t)
	b.workloads = workloadstore.New().WithPersistence(path)
	if err := b.workloads.Load(); err != nil {
		t.Fatal(err)
	}
	if len(privacyObject(t, b.workloads.Snapshot()[0])) != 15 {
		t.Fatal("fixture did not retain unknown field across restart")
	}
	b.handleMessage(clientRequest(1, "workloads:get-initial", nil))
	frames := client.frames()
	if len(frames) != 1 {
		t.Fatal("missing initial response")
	}
	var result workloadsInitialResult
	if err := json.Unmarshal(frames[0].Result, &result); err != nil {
		t.Fatal(err)
	}
	if len(result.Workloads) != 1 {
		t.Fatal("missing projected historic record")
	}
	assertPrivacyInfo(t, result.Workloads[0], info)
}

func jsonNumber(t *testing.T, n int64) string {
	t.Helper()
	raw, err := json.Marshal(n)
	if err != nil {
		t.Fatal(err)
	}
	return string(raw)
}

func TestWorkloadPrivacyProjectionReplay(t *testing.T) {
	b, _, scheduler, manager := privacyBroker(t)
	dirty := strings.TrimSuffix(privacyInfo, "}") + `,"synthetic-secret-key":"synthetic-secret-value"}`
	in, ok := workloadstore.ParseIncoming(json.RawMessage(dirty))
	if !ok || !b.workloads.Apply(in) {
		t.Fatal("seed old record")
	}
	b.rehydrateWorkloadManager(b.getWorkloadMgr())
	if b.replayActiveWorkloadsToScheduler(b.getScheduler()) != 1 {
		t.Fatal("missing scheduler replay")
	}
	for _, output := range []*syncBuffer{scheduler, manager} {
		frames := output.frames()
		if len(frames) != 1 {
			t.Fatalf("replay frames = %d", len(frames))
		}
		assertPrivacyInfo(t, privacyObject(t, frames[0].Params)["workloadInfo"], privacyInfo)
	}
}
