// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { JsonValue } from '@/shared/types/json'

// Source offsets are used only after JSON.parse has validated the document.
// Skipping strings and containers keeps opaque JSON out of JavaScript Number.
function valueEnd(text: string, start: number): number {
    let depth = 0
    for (let index = start; index < text.length; index++) {
        const character = text[index]
        if (character === '"') {
            for (index++; index < text.length; index++) {
                if (text[index] === '\\') index++
                else if (text[index] === '"') break
            }
            if (depth === 0) return index + 1
        } else if (character === '{' || character === '[') {
            depth++
        } else if (character === '}' || character === ']') {
            if (depth === 0) return index
            if (--depth === 0) return index + 1
        } else if (depth === 0 && /[\s,]/.test(character)) {
            return index
        }
    }
    return text.length
}

function whitespaceEnd(text: string, start: number): number {
    while (start < text.length && /\s/.test(text[start])) start++
    return start
}

/** Locate a field in validated JSON, following JSON.parse's last-key-wins rule. */
function fieldSpan(
    text: string,
    path: readonly string[]
): { start: number; end: number } | undefined {
    let start = whitespaceEnd(text, 0)
    let end = text.length
    for (const key of path) {
        if (text[start] !== '{') return undefined
        let found: { start: number; end: number } | undefined
        let index = whitespaceEnd(text, start + 1)
        while (index < end && text[index] !== '}') {
            const keyEnd = valueEnd(text, index)
            const name: string = JSON.parse(text.slice(index, keyEnd))
            index = whitespaceEnd(text, whitespaceEnd(text, keyEnd) + 1)
            const fieldEnd = valueEnd(text, index)
            if (name === key) found = { start: index, end: fieldEnd }
            index = whitespaceEnd(text, fieldEnd)
            if (text[index] === ',') index = whitespaceEnd(text, index + 1)
        }
        if (!found) return undefined
        start = found.start
        end = found.end
    }
    return { start, end }
}

export function jsonFieldSource(text: string, path: readonly string[]): string | undefined {
    const span = fieldSpan(text, path)
    return span ? text.slice(span.start, span.end) : undefined
}

/** Replace one owned field without re-encoding any other policy source. */
export function replaceJsonFieldSource(text: string, key: string, replacement: string): string {
    const object: JsonValue = JSON.parse(text)
    if (object === null || typeof object !== 'object' || Array.isArray(object))
        throw new Error('The policy must be a JSON object')
    JSON.parse(replacement)
    const span = fieldSpan(text, [key])
    if (span) return text.slice(0, span.start) + replacement + text.slice(span.end)
    const close = valueEnd(text, whitespaceEnd(text, 0)) - 1
    return (
        text.slice(0, close) +
        (Object.keys(object).length ? ',' : '') +
        JSON.stringify(key) +
        ':' +
        replacement +
        text.slice(close)
    )
}

/** Remove only insignificant whitespace from validated JSON for a line frame. */
export function compactJsonSource(text: string): string {
    let compact = ''
    for (let index = 0; index < text.length; index++) {
        const character = text[index]
        if (character === '"') {
            const end = valueEnd(text, index)
            compact += text.slice(index, end)
            index = end - 1
        } else if (!/[\t\n\r ]/.test(character)) {
            compact += character
        }
    }
    return compact
}

/** Format validated JSON for the editor without re-encoding any value token. */
export function prettyJsonSource(text: string): string {
    const compact = compactJsonSource(text)
    let pretty = ''
    let depth = 0
    for (let index = 0; index < compact.length; index++) {
        const character = compact[index]
        if (character === '"') {
            const end = valueEnd(compact, index)
            pretty += compact.slice(index, end)
            index = end - 1
        } else if (character === '{' || character === '[') {
            pretty += character
            depth++
            if (compact[index + 1] !== '}' && compact[index + 1] !== ']') {
                pretty += '\n' + '  '.repeat(depth)
            }
        } else if (character === '}' || character === ']') {
            depth--
            if (compact[index - 1] !== '{' && compact[index - 1] !== '[') {
                pretty += '\n' + '  '.repeat(depth)
            }
            pretty += character
        } else if (character === ',') {
            pretty += ',\n' + '  '.repeat(depth)
        } else if (character === ':') {
            pretty += ': '
        } else {
            pretty += character
        }
    }
    return pretty
}
