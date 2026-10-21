// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"errors"
	"io"
	"sync"
)

type stdioTransport struct {
	io.ReadCloser
	io.WriteCloser
	closeOnce sync.Once
	closeErr  error
}

func newStdioTransport() (io.ReadWriteCloser, error) {
	input, output, err := prepareStdio()
	if err != nil {
		return nil, err
	}
	return &stdioTransport{ReadCloser: input, WriteCloser: output}, nil
}

func (s *stdioTransport) Close() error {
	s.closeOnce.Do(func() {
		// Interrupt an active notification before waking the protocol reader.
		outErr := s.WriteCloser.Close()
		inErr := s.ReadCloser.Close()
		s.closeErr = errors.Join(outErr, inErr)
	})
	return s.closeErr
}
