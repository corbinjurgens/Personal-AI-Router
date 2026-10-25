// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"io"
	"os"

	"nvpair-shared/pipeio"
)

type stdioTransport struct {
	io.Reader
	io.Writer
}

func newStdioTransport() (io.ReadWriteCloser, error) {
	input, err := pipeio.NewReader(os.Stdin)
	if err != nil {
		return nil, err
	}
	return &stdioTransport{
		Reader: input,
		Writer: os.Stdout,
	}, nil
}

func (s *stdioTransport) Close() error {
	return nil
}
