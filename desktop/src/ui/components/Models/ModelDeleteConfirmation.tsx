// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ConfirmModal } from '@/ui/components/ConfirmModal'
import type { useModelDelete } from '@/ui/hooks/useModelDelete'

export function ModelDeleteConfirmation({
    deletion
}: {
    deletion: ReturnType<typeof useModelDelete>
}) {
    return (
        <ConfirmModal
            open={deletion.open}
            onOpenChange={open => {
                if (!open) deletion.dismiss()
            }}
            title="Delete model"
            message={deletion.loading ? 'Checking profile usage…' : deletion.message}
            confirmLabel="Delete model"
            confirmColor="danger"
            confirmDisabled={deletion.loading}
            onConfirm={deletion.confirm}
        />
    )
}
