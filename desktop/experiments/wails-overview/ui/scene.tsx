// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Stack, Text } from '@nvidia/foundations-react-core'
import OverviewColumns from '@/ui/components/MainApp/OverviewColumns'
import { ContentBaseClass } from '@/ui/constants/app'
import { useJobsFilterState } from '@/ui/hooks/useJobsFilterState'
import type { OverviewStatus } from './overview-session'

export function OverviewScene({
    status,
    nodesFirst,
    generation
}: {
    status: OverviewStatus
    nodesFirst: boolean
    generation: number
}) {
    const jobsFilter = useJobsFilterState()

    return (
        <Stack className="overflow-hidden bg-black h-full w-full">
            <div className="absolute top-0 right-0 w-full h-full app-bg" />
            <Stack className="flex-1 min-h-0 min-w-0 grow px-4 relative">
                <Stack className={ContentBaseClass}>
                    {status === 'ready' ? (
                        <OverviewColumns
                            key={generation}
                            jobsFilter={jobsFilter}
                            nodesFirst={nodesFirst}
                        />
                    ) : (
                        <Stack className={ContentBaseClass} align="center" justify="center">
                            <Text role="status" kind="body/regular/sm">
                                {status === 'connecting'
                                    ? 'Loading Overview'
                                    : 'Overview data unavailable'}
                            </Text>
                        </Stack>
                    )}
                </Stack>
            </Stack>
        </Stack>
    )
}
