// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package engines

import "strings"

// NormalizeModelName applies descriptor-selected naming to a model identifier.
// Every naming rule is exact, so it only trims surrounding whitespace.
// Callers must exempt reserved routing aliases before applying model naming.
func NormalizeModelName(model, naming string) string {
	return strings.TrimSpace(model)
}
