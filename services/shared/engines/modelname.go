// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package engines

import "strings"

// NormalizeModelName applies descriptor-selected naming to a model identifier.
// Callers must exempt reserved routing aliases before applying model naming.
func NormalizeModelName(model, naming string) string {
	model = strings.TrimSpace(model)
	if naming != ModelNamingImpliedLatestTag || model == "" {
		return model
	}
	name := model[strings.LastIndex(model, "/")+1:]
	if name != "" && !strings.ContainsAny(name, ":@") {
		return model + ":latest"
	}
	return model
}
