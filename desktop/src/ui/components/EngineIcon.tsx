// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { CSSProperties } from 'react'
import { type EngineType } from '@/shared/types/engines'
import { engineAdapterType, engineLabel } from '@/shared/utils/engines'
import { GenericEngine } from '@/ui/components/icons'
import { useEngineDescriptor } from '@/ui/stores/engine-status.store'
import ollamaIcon from '@/ui/assets/engine-icons/ollama.png?inline'
import lmStudioIcon from '@/ui/assets/engine-icons/lm-studio.png?inline'
import llamaCppIcon from '@/ui/assets/engine-icons/llama-cpp.svg?inline'

/**
 * An engine's icon. A variant of a known family (a llama.cpp fork) shows its
 * family's icon; an engine of an unknown family, or one this node has no
 * descriptor for, shows a generic engine glyph.
 */
export default function EngineIcon({ type, size = 32 }: { type: EngineType; size?: number }) {
    const descriptor = useEngineDescriptor(type)
    const family = engineAdapterType(type, descriptor)
    const dimension = `${size}px`
    const imgStyle: CSSProperties = { width: '100%', height: '100%', objectFit: 'contain' }
    const containerStyle: CSSProperties = {
        width: dimension,
        minWidth: dimension,
        maxWidth: dimension,
        height: dimension,
        minHeight: dimension,
        maxHeight: dimension,
        backgroundColor: '#fff',
        borderRadius: '25%',
        overflow: 'hidden'
    }

    if (family === 'ollama') {
        return (
            <div style={containerStyle}>
                <img src={ollamaIcon} alt="Ollama" style={imgStyle} />
            </div>
        )
    }

    if (family === 'lm-studio') {
        imgStyle.objectFit = 'cover'

        return (
            <div style={containerStyle}>
                <img src={lmStudioIcon} alt="LM Studio" style={imgStyle} />
            </div>
        )
    }

    if (family === 'llama-cpp') {
        return (
            <div style={containerStyle}>
                <img src={llamaCppIcon} alt="llama.cpp" style={imgStyle} />
            </div>
        )
    }

    return (
        <div
            style={{
                ...containerStyle,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#000'
            }}
            role="img"
            aria-label={engineLabel(type, descriptor)}
        >
            <GenericEngine style={{ width: '62%', height: '62%' }} />
        </div>
    )
}
