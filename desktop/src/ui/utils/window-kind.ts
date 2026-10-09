// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Whether this renderer is the tray popup. Main loads the popup with
 * `?window=tray` (see `createTrayWindow`) and Overview with no query. The
 * router in `App.tsx` picks the surface from it, and store initialization uses
 * it to set up only what the popup displays.
 */
export const isTrayWindow = new URLSearchParams(window.location.search).get('window') === 'tray'
