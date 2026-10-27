// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ipcRenderer } from 'electron'
import { invokeAndUnwrap } from './unwrap'
import type { IpcChannelMap, IpcPushChannelMap } from '@/shared/types/ipc-channels'

export interface IBuildApplyApi {
    choosePreview(): Promise<IpcChannelMap['build-apply:preview']['response']>
    start(
        request: IpcChannelMap['build-apply:start']['request']
    ): Promise<IpcChannelMap['build-apply:start']['response']>
    status(
        request: IpcChannelMap['build-apply:status']['request']
    ): Promise<IpcChannelMap['build-apply:status']['response']>
    onChanged(callback: (operation: IpcPushChannelMap['build-apply:changed']) => void): () => void
}

export const buildApplyApi: IBuildApplyApi = {
    choosePreview: () =>
        invokeAndUnwrap<IpcChannelMap['build-apply:preview']['response']>(
            ipcRenderer.invoke('build-apply:preview')
        ),
    start: request =>
        invokeAndUnwrap<IpcChannelMap['build-apply:start']['response']>(
            ipcRenderer.invoke('build-apply:start', request)
        ),
    status: request =>
        invokeAndUnwrap<IpcChannelMap['build-apply:status']['response']>(
            ipcRenderer.invoke('build-apply:status', request)
        ),
    onChanged: callback => {
        const handler = (
            _event: Electron.IpcRendererEvent,
            operation: IpcPushChannelMap['build-apply:changed']
        ): void => callback(operation)
        ipcRenderer.on('build-apply:changed', handler)
        return () => ipcRenderer.removeListener('build-apply:changed', handler)
    }
}
