// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { safeHandle } from './safe-handle'
import { chooseModelFile } from '@/electron/models/choose-model-file'

export function registerModelsIpc(): void {
    safeHandle('models:choose-file', () => chooseModelFile())
}
