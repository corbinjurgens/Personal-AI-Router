// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { create } from 'zustand'
import type { SettingsWindowTab } from '@/ui/types/settings-window'
import type { OverviewTab } from '@/shared/types/overview'

interface OverviewUiStore {
    /** Which top-level Overview tab is showing the content area. */
    activeTab: OverviewTab
    /** Active sub-tab when the Settings content is shown. */
    settingsSubTab: SettingsWindowTab
    settingsNodeId: string | null
    settingsProfileKey: string | null
    showProfileSummaries: boolean
    setShowProfileSummaries: (show: boolean) => void
    selectSettingsNode: (nodeId: string | null) => void
    setActiveTab: (tab: OverviewTab) => void
    /** Switch to the Settings tab, optionally selecting a sub-tab. */
    openSettings: (subTab?: SettingsWindowTab) => void
    setSettingsSubTab: (subTab: SettingsWindowTab) => void
    /** Open node settings for the requested node. */
    focusNodeSettings: (nodeId: string, profileKey?: string) => void
}

export const useOverviewUiStore = create<OverviewUiStore>(set => ({
    activeTab: 'overview',
    settingsSubTab: 'cluster',
    settingsNodeId: null,
    settingsProfileKey: null,
    showProfileSummaries: false,
    setShowProfileSummaries: show => set({ showProfileSummaries: show }),
    selectSettingsNode: nodeId =>
        set(state => ({
            settingsNodeId: nodeId,
            settingsProfileKey: nodeId === state.settingsNodeId ? state.settingsProfileKey : null
        })),

    setActiveTab: tab => set({ activeTab: tab }),

    openSettings: subTab =>
        set(state => ({
            activeTab: 'settings',
            settingsSubTab: subTab ?? state.settingsSubTab
        })),

    setSettingsSubTab: subTab => set({ settingsSubTab: subTab }),

    focusNodeSettings: (nodeId, profileKey) =>
        set({
            activeTab: 'settings',
            settingsSubTab: 'routing',
            settingsNodeId: nodeId,
            settingsProfileKey: profileKey ?? null
        })
}))
