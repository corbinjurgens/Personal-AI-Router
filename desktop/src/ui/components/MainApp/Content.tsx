// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Stack } from '@nvidia/foundations-react-core'
import { ContentBaseClass } from '@/ui/constants/app'
import ContentBar from './ContentBar'
import OverviewColumns from './OverviewColumns'
import Settings from '@/ui/components/Settings'
import { ModelsView } from '@/ui/components/Models/ModelsView'
import ProfilesView from '@/ui/components/Profiles/ProfilesView'
import DiscoverView from '@/ui/components/Discover/DiscoverView'
import { useJobsFilterState } from '@/ui/hooks/useJobsFilterState'
import { useOverviewUiStore } from '@/ui/stores/overview-ui.store'

export default function ClusterContent({
    setAddNodeModalOpen,
    setEndPointModalOpen
}: {
    setAddNodeModalOpen?: (open: boolean) => void
    setEndPointModalOpen: (open: boolean) => void
}) {
    const jobsFilter = useJobsFilterState()
    const activeTab = useOverviewUiStore(state => state.activeTab)

    return (
        <Stack className={ContentBaseClass}>
            <ContentBar
                setAddNodeModalOpen={setAddNodeModalOpen}
                setEndPointModalOpen={setEndPointModalOpen}
            />
            {activeTab === 'settings' ? (
                <Settings />
            ) : activeTab === 'models' ? (
                <Stack className="grow min-h-0 overflow-y-auto">
                    <ModelsView />
                </Stack>
            ) : activeTab === 'discover' ? (
                <Stack className="grow min-h-0 overflow-hidden">
                    <DiscoverView />
                </Stack>
            ) : activeTab === 'profiles' ? (
                <Stack className="grow min-h-0 overflow-y-auto">
                    <ProfilesView />
                </Stack>
            ) : (
                <OverviewColumns jobsFilter={jobsFilter} />
            )}
        </Stack>
    )
}
