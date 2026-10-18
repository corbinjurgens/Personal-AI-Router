// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Reserved provider-secret namespace. Only the launch that starts
 * `nvpair-service` carries it; every other child the desktop starts loses it.
 * Matches `services/shared/secretenv`.
 *
 * This is launch filtering, not OS isolation: a child the framework starts on
 * its own still inherits the Electron process environment.
 */
const PROVIDER_SECRET_PREFIX = 'NVPAIR_PROVIDER_SECRET_'

/**
 * Copy of `env` without any reserved provider-secret variable. Membership is
 * case-insensitive on every platform, so a case variant is never forwarded.
 * Every other entry is kept.
 */
export function removeProviderSecrets(env: NodeJS.ProcessEnv): NodeJS.ProcessEnv {
    return Object.fromEntries(
        Object.entries(env).filter(([key]) => !key.toUpperCase().startsWith(PROVIDER_SECRET_PREFIX))
    )
}
