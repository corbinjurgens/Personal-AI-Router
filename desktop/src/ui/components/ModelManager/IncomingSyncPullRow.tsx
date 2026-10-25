// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Flex, ProgressBar, Stack, Text } from '@nvidia/foundations-react-core'
import { formatDownloadStatus } from '@/ui/utils/formatters'
import type { IncomingSyncRow } from '@/ui/types/model-manager'

export function IncomingSyncPullRow({ row }: { row: IncomingSyncRow }) {
    return (
        <Stack gap="1" className="min-w-0 grow">
            <Flex wrap="wrap" gap="1">
                <Text kind="body/regular/sm" className="text-subtle-color">
                    {formatDownloadStatus(row.status, row.current)}
                    {row.current === true &&
                    row.status === 'in_progress' &&
                    row.percent !== undefined
                        ? ` · ${row.percent}%`
                        : ''}
                </Text>
            </Flex>
            {row.current === true &&
                row.status === 'in_progress' &&
                (row.percent === undefined ? (
                    <ProgressBar
                        size="small"
                        kind="indeterminate"
                        aria-label={`Downloading ${row.rawModel}`}
                    />
                ) : (
                    <ProgressBar
                        size="small"
                        value={row.percent}
                        aria-label={`Downloading ${row.rawModel}`}
                    />
                ))}
            {row.current !== true && (
                <Text kind="body/regular/sm" className="text-subtle-color">
                    Disconnected or download status unavailable
                </Text>
            )}
            {row.current === true && (row.status === 'queued' || row.status === 'in_progress') && (
                <Text kind="body/regular/sm" className="text-subtle-color">
                    Download in progress
                </Text>
            )}
            {row.persistence === 'uncertain' && (
                <Text kind="body/regular/sm" className="text-subtle-color">
                    Download record durability unconfirmed
                </Text>
            )}
        </Stack>
    )
}
