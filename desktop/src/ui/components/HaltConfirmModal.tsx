// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ConfirmModal } from '@/ui/components/ConfirmModal'
import { HALT_CONFIRM_TITLE, HALT_CONSEQUENCE } from '@/shared/constants/availability-copy'

/** The one confirmation shown before a local Halt, from the header and from the node card. */
export function HaltConfirmModal({
    open,
    onOpenChange,
    confirmDisabled,
    onConfirm
}: {
    open: boolean
    onOpenChange: (open: boolean) => void
    confirmDisabled: boolean
    onConfirm: () => void
}) {
    return (
        <ConfirmModal
            open={open}
            onOpenChange={onOpenChange}
            title={HALT_CONFIRM_TITLE}
            message={HALT_CONSEQUENCE}
            confirmLabel="Halt"
            confirmColor="danger"
            confirmDisabled={confirmDisabled}
            onConfirm={onConfirm}
        />
    )
}
