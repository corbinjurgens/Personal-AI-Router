// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/json"
	"log/slog"
	"os"
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
			if strings.Count(bufferText(logs), "category=malformed count=0") != 2 {
				t.Fatal("malformed frame must report fixed diagnostics")
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

func TestWorkloadManualDestinationReachesConsumers(t *testing.T) {
	for name, destination := range map[string]string{
		"space": "manual node", "slash": "lab/gpu", "non-ASCII": "手動ノード",
		"accented Latin": "nœud café",
		"byte bound":     strings.Repeat("a", 256), "UTF-8 byte bound": strings.Repeat("界", 85) + "a",
	} {
		t.Run(name, func(t *testing.T) {
			raw, err := json.Marshal(destination)
			if err != nil {
				t.Fatal(err)
			}
			info := strings.Replace(privacyInfo, `"scheduledOn":"peer"`, `"scheduledOn":`+string(raw), 1)
			b, client, scheduler, manager := privacyBroker(t)
			b.routeProxyWorkload("workload:started", privacyEnvelope(info))
			snapshot := b.workloads.Snapshot()
			if len(snapshot) != 1 {
				t.Fatal("manual destination workload missing from store")
			}
			assertPrivacyInfo(t, snapshot[0], info)
			for _, output := range []*syncBuffer{client, scheduler, manager} {
				frames := output.frames()
				if len(frames) != 1 {
					t.Fatal("manual destination workload missing from consumer")
				}
				assertPrivacyInfo(t, privacyObject(t, frames[0].Params)["workloadInfo"], info)
			}
			assertPrivacyInfo(t, projectedWorkloadInfo(snapshot[0]), info)
			replay := b.activeLocalReplayFrames()
			if len(replay) != 1 {
				t.Fatal("manual destination workload missing from replay")
			}
			assertPrivacyInfo(t, privacyObject(t, replay[0].params)["workloadInfo"], info)
		})
	}
}

func TestWorkloadManualDestinationRejectsBeforeEffects(t *testing.T) {
	for name, value := range map[string]string{
		"empty": `""`, "newline": `"node\nname"`, "NUL": `"node\u0000name"`,
		"DEL": `"node\u007fname"`, "C1 control": `"node\u0085name"`,
		"left-to-right embedding":    `"node\u202aname"`,
		"right-to-left embedding":    `"node\u202bname"`,
		"pop directional formatting": `"node\u202cname"`,
		"left-to-right override":     `"node\u202dname"`,
		"right-to-left override":     `"node\u202ename"`,
		"left-to-right isolate":      `"node\u2066name"`,
		"right-to-left isolate":      `"node\u2067name"`,
		"first strong isolate":       `"node\u2068name"`,
		"pop directional isolate":    `"node\u2069name"`,
		"left-to-right mark":         `"node\u200ename"`,
		"right-to-left mark":         `"node\u200fname"`,
		"Arabic letter mark":         `"node\u061cname"`,
		"zero-width space":           `"node\u200bname"`,
		"zero-width non-joiner":      `"node\u200cname"`,
		"zero-width joiner":          `"node\u200dname"`,
		"word joiner":                `"node\u2060name"`,
		"soft hyphen":                `"node\u00adname"`,
		"byte order mark":            `"node\ufeffname"`,
		"over byte bound":            `"` + strings.Repeat("a", 257) + `"`,
		"UTF-8 over byte bound":      `"` + strings.Repeat("界", 86) + `"`,
		"invalid UTF-8":              string([]byte{'"', 'n', 0xff, '"'}),
	} {
		t.Run(name, func(t *testing.T) {
			input := privacyEnvelope(strings.Replace(privacyInfo, `"scheduledOn":"peer"`, `"scheduledOn":`+value, 1))
			projected, category, count := projectWorkloadFrame("workload:started", input)
			if projected != nil || category != "malformed" || count != 0 {
				t.Fatalf("projection = %s, %s/%d", projected, category, count)
			}
			b, client, scheduler, manager := privacyBroker(t)
			b.routeProxyWorkload("workload:started", input)
			if b.workloads.Len() != 0 || len(client.frames())+len(scheduler.frames())+len(manager.frames()) != 0 {
				t.Fatal("invalid destination caused effects")
			}
		})
	}
}

func TestWorkloadStrictIdentitiesRemainStrict(t *testing.T) {
	for _, key := range []string{"id", "engine", "runId", "originatedFrom"} {
		t.Run(key, func(t *testing.T) {
			for _, value := range []string{"manual node", "lab/gpu", "手動", strings.Repeat("a", 129)} {
				obj := privacyObject(t, json.RawMessage(privacyInfo))
				raw, err := json.Marshal(value)
				if err != nil {
					t.Fatal(err)
				}
				obj[key] = raw
				info, err := json.Marshal(obj)
				if err != nil {
					t.Fatal(err)
				}
				projected, category, _ := projectWorkloadFrame("workload:started", privacyEnvelope(string(info)))
				if projected != nil || category != "malformed" {
					t.Fatal("strict identity accepted a manual destination ID")
				}
			}
		})
	}
}

func TestWorkloadProjectionDiagnostics(t *testing.T) {
	for _, site := range []string{"proxy", "event", "replay", "info"} {
		t.Run(site, func(t *testing.T) {
			for _, category := range []string{"accepted", "stripped", "malformed"} {
				t.Run(category, func(t *testing.T) {
					info := privacyInfo
					wantCount := 0
					if category == "stripped" {
						info = strings.TrimSuffix(info, "}") + `,"private-key-sentinel":"private-value-sentinel"}`
						wantCount = 1
					} else if category == "malformed" {
						info = strings.Replace(info, `"scheduledOn":"peer"`, `"scheduledOn":"private/value\n"`, 1)
					}
					b, _, _, _ := privacyBroker(t)
					var logs bytes.Buffer
					previous := slog.Default()
					slog.SetDefault(slog.New(slog.NewJSONHandler(&logs, &slog.HandlerOptions{Level: slog.LevelDebug})))
					t.Cleanup(func() { slog.SetDefault(previous) })
					wantMessage := "projected workload " + site
					switch site {
					case "proxy":
						wantMessage = "projected proxy workload"
						b.routeProxyWorkload("workload:started", privacyEnvelope(info))
					case "event":
						b.emitWorkloadEvent("workloads:upsert", privacyEnvelope(info))
					case "replay":
						in, ok := workloadstore.ParseIncoming(json.RawMessage(info))
						if !ok || !b.workloads.Apply(in) {
							t.Fatal("seed replay record")
						}
						b.activeLocalReplayFrames()
					case "info":
						projectedWorkloadInfo(json.RawMessage(info))
					}
					found := 0
					decoder := json.NewDecoder(&logs)
					for decoder.More() {
						var record map[string]json.RawMessage
						if err := decoder.Decode(&record); err != nil {
							t.Fatal(err)
						}
						var message string
						if err := json.Unmarshal(record["msg"], &message); err != nil {
							t.Fatal(err)
						}
						if message != wantMessage {
							continue // The proxy also runs the event projection.
						}
						found++
						want := map[string]json.RawMessage{"level": json.RawMessage(`"DEBUG"`), "category": json.RawMessage(`"` + category + `"`), "count": json.RawMessage(jsonNumber(t, int64(wantCount)))}
						delete(record, "time")
						delete(record, "msg")
						if !reflect.DeepEqual(record, want) {
							t.Fatalf("diagnostic fields = %s, want category/count only", record)
						}
					}
					if found != 1 {
						t.Fatalf("diagnostic records = %d, want 1", found)
					}
				})
			}
		})
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

func TestWorkloadPrivacyProjectionPersistedRetirement(t *testing.T) {
	logs := captureLogs(t)
	b, client, scheduler, manager := privacyBroker(t)
	path := filepath.Join(t.TempDir(), "history.json")
	// All records survive startup, then cross the age cap together. Include a
	// valid retirement so silence cannot pass by disabling retirement entirely.
	expires := time.Now().Add(2 * time.Second)
	completed := expires.Add(-workloadstore.DefaultHistoryMaxAge).UnixMilli()
	terminal := strings.Replace(privacyInfo, `"state":"running"`, `"state":"completed"`, 1)
	terminal = strings.Replace(terminal, `"completedAt":null`, `"completedAt":`+jsonNumber(t, completed), 1)
	badID := strings.Replace(terminal, `"id":"7"`, `"id":"private/body-sentinel"`, 1)
	badOrigin := strings.Replace(terminal, `"originatedFrom":"host"`, `"originatedFrom":"private/origin-sentinel"`, 1)
	if err := os.WriteFile(path, []byte("["+terminal+","+badID+","+badOrigin+"]"), 0o600); err != nil {
		t.Fatal(err)
	}
	b.workloads = workloadstore.New().WithPersistence(path)
	if err := b.workloads.Load(); err != nil {
		t.Fatal(err)
	}
	if b.workloads.Len() != 3 {
		t.Fatal("loaded fixtures must survive startup retention")
	}
	for _, input := range []string{
		`{"workloadId":"private/body-sentinel","originatedFrom":"host"}`,
		`{"workloadId":"7","originatedFrom":"private/origin-sentinel"}`,
	} {
		projected, category, count := projectWorkloadFrame("workloads:remove", json.RawMessage(input))
		if projected != nil || category != "malformed" || count != 0 {
			t.Fatal("malformed fixture must be rejected by the removal projection")
		}
	}
	b.handleMessage(clientRequest(1, "workloads:get-initial", nil))
	frames := client.frames()
	if len(frames) != 1 {
		t.Fatal("missing initial response")
	}
	var initial workloadsInitialResult
	if err := json.Unmarshal(frames[0].Result, &initial); err != nil {
		t.Fatal(err)
	}
	if len(initial.Workloads) != 1 {
		t.Fatal("initial response must omit malformed persisted identities")
	}
	assertPrivacyInfo(t, initial.Workloads[0], terminal)
	b.retireWorkloadHistory()
	if b.workloads.Len() != 3 || len(client.frames()) != 1 || len(scheduler.frames()) != 0 {
		t.Fatal("records within retention must not be retired")
	}
	time.Sleep(time.Until(expires.Add(time.Millisecond)))
	b.retireWorkloadHistory()
	if b.workloads.Len() != 0 {
		t.Fatal("expired records must leave the store, including malformed identities")
	}
	frames = client.frames()
	if len(frames) != 2 {
		t.Fatalf("client frames = %d, want initial response and one valid retirement", len(frames))
	}
	want := privacyObject(t, json.RawMessage(`{"workloadId":"7","originatedFrom":"host"}`))
	assertRemoval := func(frame Message) {
		t.Helper()
		if frame.Method != "workloads:remove" || !reflect.DeepEqual(privacyObject(t, frame.Params), want) {
			t.Fatal("retirement must preserve only the valid removal identity")
		}
	}
	assertRemoval(frames[1])
	schedulerFrames := scheduler.frames()
	if len(schedulerFrames) != 1 {
		t.Fatalf("scheduler frames = %d, want one valid retirement", len(schedulerFrames))
	}
	assertRemoval(schedulerFrames[0])
	if len(manager.frames()) != 0 {
		t.Fatal("retirement must not be broadcast to peers")
	}
	if strings.Contains(bufferText(logs), "private/") {
		t.Fatal("retirement diagnostics leaked a malformed identity")
	}
	if strings.Count(bufferText(logs), `msg="dropped workload history removal" category=malformed count=1`) != 2 {
		t.Fatal("malformed retirements must report only fixed-category/count diagnostics")
	}
	b.retireWorkloadHistory()
	if len(client.frames()) != 2 || len(scheduler.frames()) != 1 || len(manager.frames()) != 0 {
		t.Fatal("a repeated retirement must be silent")
	}
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
