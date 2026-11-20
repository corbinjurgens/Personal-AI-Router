// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { builtinModules } from 'node:module'
import { fileURLToPath } from 'node:url'
import type { InlineConfig } from 'vite'

const source = fileURLToPath(new URL('.', import.meta.url)).replaceAll('\\', '/')
const src = fileURLToPath(new URL('../../../src/', import.meta.url)).replaceAll('\\', '/')
const projection = `${source}projection.ts`
const broadcaster = `${source}projection-push.ts`

export function projectionBundle(outDir: string): InlineConfig {
    return {
        configFile: false,
        resolve: {
            alias: [
                {
                    find: /^(?:\.\/broadcaster|@\/electron\/service-bridge\/broadcaster)$/,
                    replacement: broadcaster
                },
                { find: /^@\//, replacement: src }
            ]
        },
        build: {
            target: 'node22',
            outDir,
            emptyOutDir: false,
            minify: false,
            lib: {
                entry: projection,
                formats: ['cjs'],
                fileName: () => 'projection-module.js'
            },
            rollupOptions: {
                external: ['electron', /^node:/, ...builtinModules],
                output: {
                    // The logger installs process-wide listeners. The remainder
                    // of the projection bundle is evaluated per generation.
                    manualChunks: { 'projection-log': [`${src}shared/utils/log.ts`] },
                    banner: chunk =>
                        chunk.isEntry
                            ? `
module.exports.createProjectionRealm = function (client, push) {
    'use strict';
    const exports = {};
    const timers = new Set();
    function setTimeout(callback, delay, ...args) {
        const timer = globalThis.setTimeout(() => {
            timers.delete(timer);
            callback(...args);
        }, delay);
        timers.add(timer);
        return timer;
    }
    function clearTimeout(timer) {
        timers.delete(timer);
        globalThis.clearTimeout(timer);
    }
`
                            : '',
                    footer: chunk =>
                        chunk.isEntry
                            ? `
    const projection = exports.createOverviewProjection(client, push);
    return { ...projection, retire() {
        for (const timer of timers) clearTimeout(timer);
        projection.retire();
    } };
};
`
                            : ''
                }
            }
        }
    }
}
