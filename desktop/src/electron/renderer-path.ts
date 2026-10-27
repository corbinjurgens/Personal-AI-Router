// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { join } from 'path'

export function rendererPath(): string {
    return join(__dirname, '../ui/index.html')
}
