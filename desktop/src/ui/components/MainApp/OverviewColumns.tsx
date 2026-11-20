// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import 'overlayscrollbars/styles/overlayscrollbars.css'
import { Flex } from '@nvidia/foundations-react-core'
import { useCallback } from 'react'
import NodeList from '@/ui/components/NodeList/NodeList'
import WorkloadListView from '@/ui/components/Workloads/WorkloadListView'
import WorkloadNodeConnections from '@/ui/components/Workloads/WorkloadNodeConnections'
import { CONNECTIONS_WIDTH, ContentBaseClass } from '@/ui/constants/app'
import type { useJobsFilterState } from '@/ui/hooks/useJobsFilterState'
import type { JobsFilterType } from '@/ui/types/types'

const connectorOverlap = `-${CONNECTIONS_WIDTH / 2}px`

export default function OverviewColumns({
    jobsFilter: [filter, setFilter],
    nodesFirst = false
}: {
    jobsFilter: ReturnType<typeof useJobsFilterState>
    nodesFirst?: boolean
}) {
    const handleSetFilter = useCallback(
        (type: JobsFilterType, checked: boolean) => {
            setFilter(prev => ({ ...prev, [type]: checked }))
        },
        [setFilter]
    )

    return (
        <Flex
            className={ContentBaseClass}
            style={nodesFirst ? { flexDirection: 'row-reverse' } : undefined}
        >
            <WorkloadListView filter={filter} setFilter={handleSetFilter} />
            <WorkloadNodeConnections
                style={
                    nodesFirst
                        ? { marginRight: connectorOverlap }
                        : { marginLeft: connectorOverlap }
                }
            />
            <NodeList />
        </Flex>
    )
}
