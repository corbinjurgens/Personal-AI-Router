// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !darwin && !windows && !linux

package secretstore

func newNative() (backend, error) { return nil, ErrUnavailable }
