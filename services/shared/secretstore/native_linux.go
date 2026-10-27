// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build linux

package secretstore

import (
	"context"
	"net"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/godbus/dbus/v5"
)

const (
	secretDestination                 = "org.freedesktop.secrets"
	secretService                     = "org.freedesktop.Secret.Service"
	secretCollection                  = "org.freedesktop.Secret.Collection"
	secretItem                        = "org.freedesktop.Secret.Item"
	secretSession                     = "org.freedesktop.Secret.Session"
	secretContentType                 = "text/plain; charset=utf8"
	secretCallLimit                   = 3 * time.Second
	secretRoot        dbus.ObjectPath = "/org/freedesktop/secrets"
)

type linuxBus interface {
	call(context.Context, dbus.ObjectPath, string, ...any) *dbus.Call
	close() error
}

type linuxAPI struct {
	open func(context.Context) (linuxBus, error)
}
type privateBus struct {
	conn *dbus.Conn
	stop func() bool
}

func newNative() (backend, error) { return &linuxAPI{open: openPrivateBus}, nil }

// Only a local Unix session bus is supported. Never autolaunch a daemon or
// accept a TCP address: the plain secret session relies on local OS transport.
func sessionBusSocket(address string) (string, error) {
	if address == "" {
		return "/run/user/" + strconv.Itoa(os.Geteuid()) + "/bus", nil
	}
	if !strings.HasPrefix(address, "unix:") || strings.Contains(address, ";") {
		return "", ErrUnavailable
	}
	var socket string
	for _, field := range strings.Split(strings.TrimPrefix(address, "unix:"), ",") {
		key, escaped, ok := strings.Cut(field, "=")
		value, err := url.PathUnescape(escaped)
		if !ok || err != nil || strings.ContainsRune(value, 0) {
			return "", ErrUnavailable
		}
		switch key {
		case "path":
			if socket != "" || !filepath.IsAbs(value) {
				return "", ErrUnavailable
			}
			socket = value
		case "abstract":
			if socket != "" || value == "" {
				return "", ErrUnavailable
			}
			socket = "\x00" + value
		case "guid": // Bus identity metadata, not a transport endpoint.
		default:
			return "", ErrUnavailable
		}
	}
	if socket == "" {
		return "", ErrUnavailable
	}
	return socket, nil
}

func openPrivateBus(ctx context.Context) (linuxBus, error) {
	socket, err := sessionBusSocket(os.Getenv("DBUS_SESSION_BUS_ADDRESS"))
	if err != nil {
		return nil, ErrUnavailable
	}
	raw, err := (&net.Dialer{}).DialContext(ctx, "unix", socket)
	if err != nil {
		return nil, ErrUnavailable
	}
	// Bound dialing, authentication, Hello, all method calls and cleanup. Auth
	// has no context API in godbus, so the transport deadline and close cover it.
	deadline, ok := ctx.Deadline()
	if !ok {
		_ = raw.Close()
		return nil, ErrUnavailable
	}
	if err := raw.SetDeadline(deadline); err != nil {
		_ = raw.Close()
		return nil, ErrUnavailable
	}
	stop := context.AfterFunc(ctx, func() { _ = raw.Close() })
	conn, err := dbus.NewConn(raw, dbus.WithContext(ctx))
	if err != nil {
		stop()
		_ = raw.Close()
		return nil, ErrUnavailable
	}
	b := &privateBus{conn: conn, stop: stop}
	if err := conn.Auth([]dbus.Auth{dbus.AuthExternal(strconv.Itoa(os.Geteuid()))}); err != nil {
		_ = b.close()
		return nil, ErrUnavailable
	}
	if err := conn.Hello(); err != nil {
		_ = b.close()
		return nil, ErrUnavailable
	}
	return b, nil
}

func (b *privateBus) call(ctx context.Context, path dbus.ObjectPath, method string, args ...any) *dbus.Call {
	return b.conn.Object(secretDestination, path).CallWithContext(ctx, method, 0, args...)
}

func (b *privateBus) close() error { b.stop(); return b.conn.Close() }

type linuxSecret struct {
	Session     dbus.ObjectPath
	Parameters  []byte
	Value       []byte
	ContentType string
}

type linuxOperation struct {
	bus     linuxBus
	ctx     context.Context
	session dbus.ObjectPath
}

func (a *linuxAPI) run(parent context.Context, op func(*linuxOperation) error) (err error) {
	if err := parent.Err(); err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(parent, secretCallLimit)
	defer cancel()
	bus, err := a.open(ctx)
	if err != nil {
		if parent.Err() != nil {
			return parent.Err()
		}
		return ErrUnavailable
	}
	o := &linuxOperation{bus: bus, ctx: ctx}
	defer func() {
		if o.session != "" {
			if _, callErr := replyBody(o.bus.call(ctx, o.session, secretSession+".Close"), 0); callErr != nil {
				err = ErrUnavailable
			}
		}
		if bus.close() != nil {
			err = ErrUnavailable
		}
		if parent.Err() != nil {
			err = parent.Err()
		}
	}()
	err = op(o)
	return err
}

func validObject(path dbus.ObjectPath) bool { return path != "/" && path.IsValid() }

// godbus Store coerces convertible values, including strings to paths and raw
// values to variants. Inspect decoded wire types before accepting native facts.
func replyBody(call *dbus.Call, count int) ([]any, error) {
	if call == nil || call.Err != nil || len(call.Body) != count {
		return nil, ErrUnavailable
	}
	return call.Body, nil
}

func (o *linuxOperation) unlocked(path dbus.ObjectPath, iface string) error {
	body, err := replyBody(o.bus.call(o.ctx, path, "org.freedesktop.DBus.Properties.Get", iface, "Locked"), 1)
	if err != nil {
		return ErrUnavailable
	}
	locked, ok := body[0].(dbus.Variant)
	if !ok {
		return ErrUnavailable
	}
	v, ok := locked.Value().(bool)
	if !ok || v {
		return ErrUnavailable
	}
	return nil
}

func (o *linuxOperation) find(service, account string) (dbus.ObjectPath, dbus.ObjectPath, error) {
	body, err := replyBody(o.bus.call(o.ctx, secretRoot, secretService+".ReadAlias", "default"), 1)
	if err != nil {
		return "", "", ErrUnavailable
	}
	collection, ok := body[0].(dbus.ObjectPath)
	if !ok || !validObject(collection) {
		return "", "", ErrUnavailable
	}
	if err := o.unlocked(collection, secretCollection); err != nil {
		return "", "", err
	}
	attributes := map[string]string{"service": service, "account": account}
	body, err = replyBody(o.bus.call(o.ctx, collection, secretCollection+".SearchItems", attributes), 1)
	if err != nil {
		return "", "", ErrUnavailable
	}
	items, ok := body[0].([]dbus.ObjectPath)
	if !ok || len(items) > 1 {
		return "", "", ErrUnavailable
	}
	if len(items) == 0 {
		return collection, "", nil
	}
	if !validObject(items[0]) {
		return "", "", ErrUnavailable
	}
	if err := o.unlocked(items[0], secretItem); err != nil {
		return "", "", err
	}
	return collection, items[0], nil
}

func (o *linuxOperation) openSession() error {
	body, err := replyBody(o.bus.call(o.ctx, secretRoot, secretService+".OpenSession", "plain", dbus.MakeVariant("")), 2)
	if err != nil {
		return ErrUnavailable
	}
	session, ok := body[1].(dbus.ObjectPath)
	if !ok || !validObject(session) {
		return ErrUnavailable
	}
	o.session = session
	output, ok := body[0].(dbus.Variant)
	if !ok {
		return ErrUnavailable
	}
	if value, ok := output.Value().(string); !ok || value != "" {
		return ErrUnavailable
	}
	return nil
}

func (a *linuxAPI) set(ctx context.Context, service, account string, value []byte) error {
	return a.run(ctx, func(o *linuxOperation) error {
		collection, _, err := o.find(service, account)
		if err != nil {
			return err
		}
		if err := o.openSession(); err != nil {
			return err
		}
		properties := map[string]dbus.Variant{
			secretItem + ".Label":      dbus.MakeVariant("PAIR credential"),
			secretItem + ".Attributes": dbus.MakeVariant(map[string]string{"service": service, "account": account}),
		}
		secret := linuxSecret{o.session, []byte{}, value, secretContentType}
		body, err := replyBody(o.bus.call(o.ctx, collection, secretCollection+".CreateItem", properties, secret, true), 2)
		if err != nil {
			return ErrUnavailable
		}
		item, itemOK := body[0].(dbus.ObjectPath)
		prompt, promptOK := body[1].(dbus.ObjectPath)
		if !itemOK || !promptOK || !validObject(item) || prompt != "/" {
			return ErrUnavailable
		}
		return nil
	})
}

func (a *linuxAPI) get(ctx context.Context, service, account string) (value []byte, err error) {
	err = a.run(ctx, func(o *linuxOperation) error {
		_, item, err := o.find(service, account)
		if err != nil {
			return err
		}
		if item == "" {
			return ErrNotFound
		}
		if err := o.openSession(); err != nil {
			return err
		}
		body, err := replyBody(o.bus.call(o.ctx, item, secretItem+".GetSecret", o.session), 1)
		if err != nil {
			return ErrUnavailable
		}
		// D-Bus structs decode as []any, not a Go struct. Require the exact
		// (oayays) tuple instead of permitting Store's recursive conversions.
		fields, ok := body[0].([]any)
		if !ok || len(fields) != 4 {
			return ErrUnavailable
		}
		session, sessionOK := fields[0].(dbus.ObjectPath)
		parameters, parametersOK := fields[1].([]byte)
		secretValue, valueOK := fields[2].([]byte)
		contentType, contentOK := fields[3].(string)
		if !sessionOK || !parametersOK || !valueOK || !contentOK || session != o.session || len(parameters) != 0 || contentType != secretContentType || !validSecret(secretValue) {
			return ErrUnavailable
		}
		value = secretValue
		return nil
	})
	if err != nil {
		return nil, err
	}
	return value, nil
}

func (a *linuxAPI) delete(ctx context.Context, service, account string) error {
	return a.run(ctx, func(o *linuxOperation) error {
		_, item, err := o.find(service, account)
		if err != nil {
			return err
		}
		if item == "" {
			return ErrNotFound
		}
		body, err := replyBody(o.bus.call(o.ctx, item, secretItem+".Delete"), 1)
		if err != nil {
			return ErrUnavailable
		}
		prompt, ok := body[0].(dbus.ObjectPath)
		if !ok || prompt != "/" {
			return ErrUnavailable
		}
		return nil
	})
}
