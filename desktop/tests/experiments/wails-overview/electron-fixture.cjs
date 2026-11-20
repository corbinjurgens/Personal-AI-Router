// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const { tmpdir } = require('node:os')

exports.app = { getPath: () => tmpdir() }
exports.BrowserWindow = { getAllWindows: () => [] }
