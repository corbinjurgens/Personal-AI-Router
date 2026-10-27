// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build windows

package secretstore

import (
	"context"

	"github.com/danieljoos/wincred"
)

type windowsAPI struct {
	read   func(string) (*wincred.GenericCredential, error)
	write  func(*wincred.GenericCredential) error
	remove func(*wincred.GenericCredential) error
}

func newNative() (backend, error) {
	return &windowsAPI{
		read:   wincred.GetGenericCredential,
		write:  (*wincred.GenericCredential).Write,
		remove: (*wincred.GenericCredential).Delete,
	}, nil
}

func (w *windowsAPI) set(ctx context.Context, service, account string, value []byte) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	c := wincred.NewGenericCredential(service + "/" + account)
	c.UserName = account
	c.Persist = wincred.PersistLocalMachine
	c.CredentialBlob = value
	if w.write(c) != nil {
		return ErrUnavailable
	}
	return nil
}

func (w *windowsAPI) get(ctx context.Context, service, account string) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	c, err := w.read(service + "/" + account)
	if err == wincred.ErrElementNotFound {
		return nil, ErrNotFound
	}
	if err != nil || c == nil {
		return nil, ErrUnavailable
	}
	return c.CredentialBlob, nil
}

func (w *windowsAPI) delete(ctx context.Context, service, account string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	err := w.remove(wincred.NewGenericCredential(service + "/" + account))
	if err == wincred.ErrElementNotFound {
		return ErrNotFound
	}
	if err != nil {
		return ErrUnavailable
	}
	return nil
}
