// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package enginesettings

import "time"

// Settings calls wait for stop, rebind, and readiness. Configure allows ten
// minutes for the engine's readiness probe, which covers the bundled llama.cpp
// probe and leaves room for a slower descriptor engine, and adds time for stop
// and rebind. Each caller leaves headroom for the layer it waits on. Desktop's outer
// envelope is fourteen minutes (modular-runtime.ts).
const (
	ConfigureBudget = 10*time.Minute + 30*time.Second
	OperationBudget = ConfigureBudget + 30*time.Second
	CallBudget      = OperationBudget + time.Minute
	RelayBudget     = CallBudget + time.Minute
)
