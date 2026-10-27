// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build linux

package secretstore

import (
	"context"
	"errors"
	"reflect"
	"testing"
	"time"

	"github.com/godbus/dbus/v5"
)

const fakeCollection dbus.ObjectPath = "/collection/default"
const fakeItem dbus.ObjectPath = "/collection/default/item"
const fakeSession dbus.ObjectPath = "/session/private"

type fakeLinuxBus struct {
	t                                       *testing.T
	collection                              dbus.ObjectPath
	items                                   []dbus.ObjectPath
	collectionLocked, itemLocked            bool
	createPrompt, deletePrompt              dbus.ObjectPath
	failMethod                              string
	nativeError                             error
	replies                                 map[string][]any
	before                                  func(context.Context, string)
	methods                                 []string
	closed, sessionClosed, creates, deletes int
	closeError                              error
	value                                   []byte
}

func newFakeLinux(t *testing.T) (*linuxAPI, *fakeLinuxBus) {
	t.Helper()
	b := &fakeLinuxBus{t: t, collection: fakeCollection, items: []dbus.ObjectPath{fakeItem}, createPrompt: "/", deletePrompt: "/", replies: make(map[string][]any), value: []byte("fake"), nativeError: errors.New("native diagnostic")}
	a := &linuxAPI{open: func(ctx context.Context) (linuxBus, error) {
		deadline, ok := ctx.Deadline()
		if !ok || time.Until(deadline) > secretCallLimit {
			t.Fatal("operation has no bounded context")
		}
		return b, nil
	}}
	return a, b
}

func (b *fakeLinuxBus) call(ctx context.Context, path dbus.ObjectPath, method string, args ...any) *dbus.Call {
	b.methods = append(b.methods, method)
	if b.before != nil {
		b.before(ctx, method)
	}
	if method == b.failMethod {
		return &dbus.Call{Err: b.nativeError}
	}
	if reply, ok := b.replies[method]; ok {
		return &dbus.Call{Body: reply}
	}
	reply := func(values ...any) *dbus.Call { return &dbus.Call{Body: values} }
	switch method {
	case secretService + ".ReadAlias":
		if path != secretRoot || !reflect.DeepEqual(args, []any{"default"}) {
			b.t.Fatal("wrong default collection lookup")
		}
		return reply(b.collection)
	case "org.freedesktop.DBus.Properties.Get":
		if len(args) != 2 || args[1] != "Locked" {
			b.t.Fatal("wrong property lookup")
		}
		if path == fakeCollection && args[0] == secretCollection {
			return reply(dbus.MakeVariant(b.collectionLocked))
		}
		if path == fakeItem && args[0] == secretItem {
			return reply(dbus.MakeVariant(b.itemLocked))
		}
		b.t.Fatal("wrong lock object")
	case secretCollection + ".SearchItems":
		want := map[string]string{"service": servicePrefix + "purpose", "account": "id"}
		if path != fakeCollection || len(args) != 1 || !reflect.DeepEqual(args[0], want) {
			b.t.Fatal("wrong exact item attributes")
		}
		return reply(b.items)
	case secretService + ".OpenSession":
		if path != secretRoot || len(args) != 2 || args[0] != "plain" || !reflect.DeepEqual(args[1], dbus.MakeVariant("")) {
			b.t.Fatal("wrong session negotiation")
		}
		return reply(dbus.MakeVariant(""), fakeSession)
	case secretCollection + ".CreateItem":
		if path != fakeCollection || len(args) != 3 || args[2] != true {
			b.t.Fatal("wrong create/replace operation")
		}
		props, ok := args[0].(map[string]dbus.Variant)
		if !ok || len(props) != 2 || props[secretItem+".Label"].Value() != "PAIR credential" || !reflect.DeepEqual(props[secretItem+".Attributes"].Value(), map[string]string{"service": servicePrefix + "purpose", "account": "id"}) {
			b.t.Fatal("wrong item metadata")
		}
		secret, ok := args[1].(linuxSecret)
		if !ok || secret.Session != fakeSession || len(secret.Parameters) != 0 || secret.ContentType != secretContentType || !validSecret(secret.Value) {
			b.t.Fatal("wrong secret tuple (values omitted)")
		}
		b.creates++
		if b.createPrompt == "/" {
			b.value = append([]byte(nil), secret.Value...)
		}
		return reply(fakeItem, b.createPrompt)
	case secretItem + ".GetSecret":
		if path != fakeItem || !reflect.DeepEqual(args, []any{fakeSession}) {
			b.t.Fatal("wrong get session")
		}
		return reply([]any{fakeSession, []byte{}, b.value, secretContentType})
	case secretItem + ".Delete":
		if path != fakeItem || len(args) != 0 {
			b.t.Fatal("wrong exact item delete")
		}
		b.deletes++
		if b.deletePrompt == "/" {
			b.value = nil
		}
		return reply(b.deletePrompt)
	case secretSession + ".Close":
		if path != fakeSession || len(args) != 0 {
			b.t.Fatal("wrong session cleanup")
		}
		b.sessionClosed++
		return reply()
	default:
		b.t.Fatalf("unexpected native method %s", method)
	}
	return &dbus.Call{Err: ErrUnavailable}
}

func (b *fakeLinuxBus) close() error { b.closed++; return b.closeError }

func linuxTestOperation(t *testing.T, a *linuxAPI, operation string) error {
	t.Helper()
	switch operation {
	case "set":
		return a.set(t.Context(), servicePrefix+"purpose", "id", []byte("replacement-fake"))
	case "get":
		_, err := a.get(t.Context(), servicePrefix+"purpose", "id")
		return err
	case "delete":
		return a.delete(t.Context(), servicePrefix+"purpose", "id")
	default:
		t.Fatal("unknown fake operation")
		return nil
	}
}

func TestLinuxUsesPrivateConnectionsAndClosesSessions(t *testing.T) {
	a, b := newFakeLinux(t)
	wantError(t, linuxTestOperation(t, a, "set"), nil)
	value, err := a.get(t.Context(), servicePrefix+"purpose", "id")
	wantValue(t, value, []byte("replacement-fake"), err)
	wantError(t, linuxTestOperation(t, a, "delete"), nil)
	if b.closed != 3 || b.sessionClosed != 2 || b.creates != 1 || b.deletes != 1 {
		t.Fatal("wrong connection/session lifecycle or mutation count")
	}
}

func TestLinuxRefusesLockedCollectionsAndItems(t *testing.T) {
	for _, operation := range []string{"set", "get", "delete"} {
		for _, locked := range []string{"collection", "item"} {
			t.Run(operation+"/"+locked, func(t *testing.T) {
				a, b := newFakeLinux(t)
				b.collectionLocked = locked == "collection"
				b.itemLocked = locked == "item"
				wantError(t, linuxTestOperation(t, a, operation), ErrUnavailable)
				if b.creates != 0 || b.deletes != 0 || b.sessionClosed != 0 || b.closed != 1 {
					t.Fatal("locked store was accessed")
				}
			})
		}
	}
}

func TestLinuxMissingDefaultAndDuplicateMatchesAreUnavailable(t *testing.T) {
	for _, scenario := range []string{"missing default", "invalid default", "duplicate matches", "invalid item"} {
		for _, operation := range []string{"set", "get", "delete"} {
			t.Run(scenario+"/"+operation, func(t *testing.T) {
				a, b := newFakeLinux(t)
				switch scenario {
				case "missing default":
					b.collection = "/"
				case "invalid default":
					b.collection = "invalid"
				case "duplicate matches":
					b.items = []dbus.ObjectPath{fakeItem, fakeItem}
				case "invalid item":
					b.items = []dbus.ObjectPath{"/"}
				}
				wantError(t, linuxTestOperation(t, a, operation), ErrUnavailable)
				if b.creates != 0 || b.deletes != 0 {
					t.Fatal("ambiguous or unavailable collection was mutated")
				}
			})
		}
	}
}

func TestLinuxOnlyUnlockedSuccessfulSearchCanReportAbsence(t *testing.T) {
	a, b := newFakeLinux(t)
	b.items = nil
	wantError(t, linuxTestOperation(t, a, "get"), ErrNotFound)
	wantError(t, linuxTestOperation(t, a, "delete"), ErrNotFound)
	wantError(t, linuxTestOperation(t, a, "set"), nil)
	if b.creates != 1 || b.deletes != 0 {
		t.Fatal("absent item sequence is wrong")
	}
}

func TestLinuxRefusesPromptsWithoutInvokingThem(t *testing.T) {
	for _, operation := range []string{"set", "delete"} {
		t.Run(operation, func(t *testing.T) {
			a, b := newFakeLinux(t)
			b.createPrompt = "/prompt/required"
			b.deletePrompt = "/prompt/required"
			wantError(t, linuxTestOperation(t, a, operation), ErrUnavailable)
			wantValue(t, b.value, []byte("fake"), nil)
			if b.creates+b.deletes != 1 {
				t.Fatal("prompting mutation was retried")
			}
		})
	}
}

func TestLinuxUnavailableBusAndDaemonAreSanitizedAndRetryable(t *testing.T) {
	a, b := newFakeLinux(t)
	open := a.open
	attempts := 0
	a.open = func(ctx context.Context) (linuxBus, error) {
		attempts++
		if attempts == 1 {
			return nil, errors.New("bus diagnostic")
		}
		return open(ctx)
	}
	wantError(t, linuxTestOperation(t, a, "get"), ErrUnavailable)
	b.failMethod = secretService + ".ReadAlias"
	b.nativeError = dbus.NewError("org.freedesktop.DBus.Error.ServiceUnknown", []any{"daemon diagnostic"})
	wantError(t, linuxTestOperation(t, a, "get"), ErrUnavailable)
	b.failMethod = ""
	wantError(t, linuxTestOperation(t, a, "get"), nil)
	if attempts != 3 || b.closed != 2 {
		t.Fatal("unavailable native startup latched or failed connection leaked")
	}
}

func TestLinuxFailuresCloseConnectionsAndSessions(t *testing.T) {
	for _, test := range []struct {
		method, operation string
		sessionClosed     int
	}{
		{secretService + ".ReadAlias", "get", 0}, {"org.freedesktop.DBus.Properties.Get", "get", 0},
		{secretCollection + ".SearchItems", "get", 0}, {secretService + ".OpenSession", "get", 0},
		{secretItem + ".GetSecret", "get", 1}, {secretCollection + ".CreateItem", "set", 1},
		{secretItem + ".Delete", "delete", 0}, {secretSession + ".Close", "set", 0},
	} {
		t.Run(test.method, func(t *testing.T) {
			a, b := newFakeLinux(t)
			b.failMethod = test.method
			wantError(t, linuxTestOperation(t, a, test.operation), ErrUnavailable)
			if b.closed != 1 || b.sessionClosed != test.sessionClosed {
				t.Fatal("failed operation leaked connection/session")
			}
			if b.creates > 1 || b.deletes > 1 {
				t.Fatal("ambiguous mutation retried")
			}
		})
	}
}

func TestLinuxMalformedRepliesAreUnavailable(t *testing.T) {
	for _, test := range []struct {
		name, method, operation string
		body                    []any
	}{
		{"alias type", secretService + ".ReadAlias", "get", []any{true}},
		{"string alias", secretService + ".ReadAlias", "get", []any{string(fakeCollection)}},
		{"lock type", "org.freedesktop.DBus.Properties.Get", "get", []any{dbus.MakeVariant("false")}},
		{"raw lock boolean", "org.freedesktop.DBus.Properties.Get", "get", []any{false}},
		{"search type", secretCollection + ".SearchItems", "get", []any{"none"}},
		{"empty string search get", secretCollection + ".SearchItems", "get", []any{[]string{}}},
		{"empty string search delete", secretCollection + ".SearchItems", "delete", []any{[]string{}}},
		{"string item path", secretCollection + ".SearchItems", "get", []any{[]string{string(fakeItem)}}},
		{"invalid session", secretService + ".OpenSession", "get", []any{dbus.MakeVariant(""), dbus.ObjectPath("/")}},
		{"string session path", secretService + ".OpenSession", "get", []any{dbus.MakeVariant(""), string(fakeSession)}},
		{"raw session output", secretService + ".OpenSession", "get", []any{"", fakeSession}},
		{"wrong negotiation", secretService + ".OpenSession", "get", []any{dbus.MakeVariant("unexpected"), fakeSession}},
		{"missing reply", secretItem + ".GetSecret", "get", nil},
		{"wrong secret type", secretItem + ".GetSecret", "get", []any{true}},
		{"invalid create item", secretCollection + ".CreateItem", "set", []any{dbus.ObjectPath("/"), dbus.ObjectPath("/")}},
		{"string create item", secretCollection + ".CreateItem", "set", []any{string(fakeItem), dbus.ObjectPath("/")}},
		{"string create prompt", secretCollection + ".CreateItem", "set", []any{fakeItem, "/"}},
		{"malformed delete prompt", secretItem + ".Delete", "delete", []any{true}},
		{"string delete prompt", secretItem + ".Delete", "delete", []any{"/"}},
		{"nonempty close reply", secretSession + ".Close", "get", []any{false}},
	} {
		t.Run(test.name, func(t *testing.T) {
			a, b := newFakeLinux(t)
			b.replies[test.method] = test.body
			wantError(t, linuxTestOperation(t, a, test.operation), ErrUnavailable)
			if b.closed != 1 {
				t.Fatal("malformed response leaked private connection")
			}
		})
	}
}

func TestLinuxValidatesSecretTuple(t *testing.T) {
	for _, scenario := range []string{"wrong session", "nonempty parameters", "wrong content type", "empty value", "invalid UTF-8", "string session", "wrong parameters type", "wrong value type", "wrong content type type", "extra field", "missing field", "Go struct instead of decoded tuple"} {
		t.Run(scenario, func(t *testing.T) {
			a, b := newFakeLinux(t)
			secret := []any{fakeSession, []byte{}, []byte("fake"), secretContentType}
			switch scenario {
			case "wrong session":
				secret[0] = dbus.ObjectPath("/session/other")
			case "nonempty parameters":
				secret[1] = []byte{1}
			case "wrong content type":
				secret[3] = "application/octet-stream"
			case "empty value":
				secret[2] = []byte(nil)
			case "invalid UTF-8":
				secret[2] = []byte{0xff}
			case "string session":
				secret[0] = string(fakeSession)
			case "wrong parameters type":
				secret[1] = []string{}
			case "wrong value type":
				secret[2] = "fake"
			case "wrong content type type":
				secret[3] = dbus.MakeVariant(secretContentType)
			case "extra field":
				secret = append(secret, false)
			case "missing field":
				secret = secret[:3]
			}
			b.replies[secretItem+".GetSecret"] = []any{secret}
			if scenario == "Go struct instead of decoded tuple" {
				b.replies[secretItem+".GetSecret"] = []any{linuxSecret{fakeSession, []byte{}, []byte("fake"), secretContentType}}
			}
			wantError(t, linuxTestOperation(t, a, "get"), ErrUnavailable)
		})
	}
}

func TestLinuxCallerCancellationAndDeadlineReturnContextSentinels(t *testing.T) {
	t.Run("in-flight cancellation", func(t *testing.T) {
		a, b := newFakeLinux(t)
		ctx, cancel := context.WithCancel(t.Context())
		defer cancel()
		b.before = func(_ context.Context, method string) {
			if method == secretItem+".GetSecret" {
				cancel()
			}
		}
		_, err := a.get(ctx, servicePrefix+"purpose", "id")
		wantError(t, err, context.Canceled)
	})
	t.Run("shorter caller deadline", func(t *testing.T) {
		a, b := newFakeLinux(t)
		ctx, cancel := context.WithTimeout(t.Context(), 10*time.Millisecond)
		defer cancel()
		b.before = func(callCtx context.Context, method string) {
			if method == secretItem+".GetSecret" {
				want, _ := ctx.Deadline()
				got, _ := callCtx.Deadline()
				if !got.Equal(want) {
					t.Fatal("caller deadline extended")
				}
				<-callCtx.Done()
			}
		}
		_, err := a.get(ctx, servicePrefix+"purpose", "id")
		wantError(t, err, context.DeadlineExceeded)
	})
	t.Run("native timeout without caller deadline", func(t *testing.T) {
		a, b := newFakeLinux(t)
		b.failMethod = secretItem + ".GetSecret"
		b.nativeError = context.DeadlineExceeded
		wantError(t, linuxTestOperation(t, a, "get"), ErrUnavailable)
	})
}

func TestLinuxConnectionCloseFailureDoesNotRetryWrite(t *testing.T) {
	a, b := newFakeLinux(t)
	b.closeError = errors.New("close diagnostic")
	wantError(t, linuxTestOperation(t, a, "set"), ErrUnavailable)
	if b.creates != 1 {
		t.Fatal("write retried on cleanup failure")
	}
}

func TestLinuxLocalSessionBusAddressPolicy(t *testing.T) {
	for _, test := range []struct{ name, address, want string }{
		{"unix path", "unix:path=/run/user/1000/bus", "/run/user/1000/bus"},
		{"escaped path", "unix:path=/tmp/test%20bus,guid=abc", "/tmp/test bus"},
		{"abstract socket", "unix:abstract=test-bus", "\x00test-bus"},
	} {
		t.Run(test.name, func(t *testing.T) {
			got, err := sessionBusSocket(test.address)
			wantError(t, err, nil)
			if got != test.want {
				t.Fatal("wrong local transport address")
			}
		})
	}
	for _, address := range []string{"tcp:host=localhost,port=1", "autolaunch:", "unix:path=relative", "unix:path=/tmp/a;unix:path=/tmp/b", "unix:path=/tmp/%00", "unix:path=/tmp/a,abstract=b", "unix:path=/tmp/%zz", "unix:guid=abc"} {
		_, err := sessionBusSocket(address)
		wantError(t, err, ErrUnavailable)
	}
}
