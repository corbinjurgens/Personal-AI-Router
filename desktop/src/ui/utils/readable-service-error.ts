// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** A service error message without its JSON-RPC code prefix ("-32000: "), for display. */
export function readableServiceError(message: string): string {
    return message.replace(/^-?\d+:\s*/, '')
}
