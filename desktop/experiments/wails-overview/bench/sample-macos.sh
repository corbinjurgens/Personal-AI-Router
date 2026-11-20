#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eu
export LC_ALL=C TZ=UTC
exec node "$(dirname "$0")/sample-macos.mjs" "$@"
