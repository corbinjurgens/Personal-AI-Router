#!/usr/bin/env node
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Developer commands for the Personal AI Router monorepo, the same on Windows,
 * macOS and Linux:
 *
 *   node scripts/dev.mjs <command>
 *
 * Run `node scripts/dev.mjs help` for the list. Each command composes the
 * existing desktop npm scripts and Go commands; it does not reimplement them.
 * Only Node.js built-ins are used, so the file runs before `npm ci`.
 */

import { spawnSync } from 'node:child_process'
import { existsSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs'
import { devNull } from 'node:os'
import { dirname, isAbsolute, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const DESKTOP = join(ROOT, 'desktop')
const SERVICES = join(ROOT, 'services')
const WINDOWS = process.platform === 'win32'

// Minimums documented in docs/building.mdx. Node's comes from desktop/package.json.
const MIN_GO = '1.25'
const MIN_NODE = JSON.parse(readFileSync(join(DESKTOP, 'package.json'), 'utf8')).engines.node.replace(/^>=/, '')

class CommandError extends Error {}

function fail(message) {
    throw new CommandError(message)
}

// npm is a .cmd shim on Windows, which Node starts only through a shell. Run
// the npm that ships beside this Node.js instead, so no shell is involved and
// npm matches the Node.js version that tools() checked.
function invocation(program, args) {
    if (program !== 'npm' || !WINDOWS) return [program, args]
    const cli = join(dirname(process.execPath), 'node_modules', 'npm', 'bin', 'npm-cli.js')
    if (!existsSync(cli)) fail(`npm-cli.js is missing beside ${process.execPath}`)
    return [process.execPath, [cli, ...args]]
}

// Runs a program with inherited output and fails on a nonzero exit.
function run(program, args, cwd = ROOT) {
    console.log(`\n> ${program} ${args.join(' ')}${cwd === ROOT ? '' : `  (in ${cwd})`}`)
    const [file, argv] = invocation(program, args)
    const result = spawnSync(file, argv, { cwd, stdio: 'inherit' })
    if (result.error) fail(`${program} could not start: ${result.error.message}`)
    if (result.status !== 0) fail(`${program} ${args.join(' ')} exited with ${result.status}`)
}

// Returns trimmed stdout, or null when the program is missing or fails.
function capture(program, args, cwd = ROOT) {
    const [file, argv] = invocation(program, args)
    const result = spawnSync(file, argv, { cwd, encoding: 'utf8' })
    if (result.error || result.status !== 0) return null
    return result.stdout.trim()
}

function npm(script, extra = []) {
    run('npm', ['run', script, ...(extra.length ? ['--', ...extra] : [])], DESKTOP)
}

function goModules() {
    return readdirSync(SERVICES)
        .map(name => join(SERVICES, name))
        .filter(dir => existsSync(join(dir, 'go.mod')))
        .sort()
}

function atLeast(version, minimum) {
    const a = version.split('.').map(Number)
    const b = minimum.split('.').map(Number)
    for (let i = 0; i < Math.max(a.length, b.length); i++) {
        if ((a[i] ?? 0) !== (b[i] ?? 0)) return (a[i] ?? 0) > (b[i] ?? 0)
    }
    return true
}

function requireDependencies() {
    if (!existsSync(join(DESKTOP, 'node_modules'))) fail('desktop/node_modules is missing; run: node scripts/dev.mjs setup')
}

function gitStatus() {
    const status = capture('git', ['status', '--porcelain'])
    if (status === null) fail('git status failed; run the command inside a Git checkout')
    return status
}

function requireCleanTree(when) {
    const status = gitStatus()
    if (status) {
        fail(
            `the checkout has changes ${when}, so the Go binaries would be stamped vcs.modified=true:\n${status}\n` +
                'Commit or discard them. A generated file listed here (for example THIRD_PARTY_NOTICES.md) is out of date: ' +
                'regenerate it with its npm script and commit it.'
        )
    }
}

// --- commands -------------------------------------------------------------

function tools() {
    const missing = ['go', 'node', 'npm', 'git'].filter(tool => capture(tool, tool === 'go' ? ['version'] : ['--version']) === null)
    if (missing.length) fail(`missing: ${missing.join(', ')}. See docs/building.mdx for the prerequisites.`)
    // Inside a module, so the toolchain its go.mod selects is the one reported.
    const goVersion = capture('go', ['env', 'GOVERSION'], join(SERVICES, 'nvpair-service')).replace(/^go/, '')
    const nodeVersion = process.versions.node
    console.log(`go ${goVersion}, node ${nodeVersion}, npm ${capture('npm', ['--version'])}`)
    const problems = []
    if (!atLeast(nodeVersion, MIN_NODE)) problems.push(`Node.js ${nodeVersion} is older than the required ${MIN_NODE}`)
    if (!atLeast(goVersion.split(/[^0-9.]/)[0], MIN_GO)) problems.push(`Go ${goVersion} is older than the required ${MIN_GO}`)
    if (problems.length) fail(`${problems.join('; ')}. Put a newer version first on PATH.`)
}

// `go mod download` rewrites go.mod indirect requirements, which desynchronizes
// the sibling modules that pin them. Compiling to the null device fills the
// module and build caches instead and writes no binaries. The license
// generator later reads that cache with GOPROXY=off.
function setup() {
    tools()
    run('npm', ['ci'], DESKTOP)
    for (const module of goModules()) run('go', ['build', '-o', devNull, './...'], module)
    console.log('\nDependencies installed. Start the desktop app with: node scripts/dev.mjs start')
}

function build() {
    requireDependencies()
    npm('build:modular-binaries')
    npm('build')
}

function start() {
    requireDependencies()
    run('npm', ['start'], DESKTOP)
}

function check() {
    requireDependencies()
    run('node', ['scripts/spdx-headers.mjs'])
    npm('verify:build-scripts')
    npm('lint')
    npm('typecheck')
    npm('service-contracts:check')
    npm('dead-code:check')
    npm('test:unit')
}

function testAll() {
    requireDependencies()
    npm('test:unit')
    for (const module of goModules()) run('go', ['test', './...'], module)
}

function services() {
    run(WINDOWS ? 'cmd.exe' : 'sh', WINDOWS ? ['/d', '/c', 'build.bat'] : ['./build.sh'], SERVICES)
}

function headersFix() {
    run('node', ['scripts/spdx-headers.mjs', '--fix'])
}

function clean() {
    const paths = ['cli-bin', 'out', 'dist', 'release', 'coverage'].map(name => join(DESKTOP, name))
    paths.push(join(SERVICES, 'build'), join(SERVICES, 'dist'))
    for (const file of readdirSync(DESKTOP)) if (file.endsWith('.tsbuildinfo')) paths.push(join(DESKTOP, file))
    for (const module of goModules()) {
        const name = module.split(/[\\/]/).pop()
        paths.push(join(module, name), join(module, `${name}.exe`))
    }
    for (const path of paths) rmSync(path, { recursive: true, force: true })
    console.log('Removed build output. Dependencies in desktop/node_modules are untouched.')
}

const PLATFORMS = { win32: 'win', darwin: 'mac', linux: 'linux' }

function findPackagedService(dir) {
    const name = WINDOWS ? 'nvpair-service.exe' : 'nvpair-service'
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
        const path = join(dir, entry.name)
        if (entry.isDirectory()) {
            const found = findPackagedService(path)
            if (found) return found
        } else if (entry.name === name && dir.endsWith('cli-bin')) return path
    }
    return null
}

// Builds the unpacked desktop app for this machine's platform and
// architecture, without an installer, from a clean checkout, and checks that
// the packaged service binaries carry this commit with a clean stamp.
function packageApp(args) {
    const resultIndex = args.indexOf('--result')
    const resultFile = resultIndex === -1 ? null : args[resultIndex + 1]
    if (resultIndex !== -1 && (!resultFile || !isAbsolute(resultFile))) fail('--result needs an absolute file path')
    const platform = PLATFORMS[process.platform]
    if (!platform) fail(`unsupported platform ${process.platform}`)
    if (!['x64', 'arm64'].includes(process.arch)) fail(`unsupported architecture ${process.arch}`)

    tools()
    requireDependencies()
    requireCleanTree('before the build')
    const revision = capture('git', ['rev-parse', 'HEAD'])
    npm('build')
    requireCleanTree('after the generated files were refreshed')
    npm(`build:electron:${platform}:${process.arch}`, ['--dir'])

    const version = JSON.parse(readFileSync(join(DESKTOP, 'package.json'), 'utf8')).version
    const release = join(DESKTOP, 'release', version)
    const service = existsSync(release) ? findPackagedService(release) : null
    if (!service) fail(`no packaged nvpair-service under ${release}`)
    const metadata = capture('go', ['version', '-m', service])
    if (metadata === null) fail(`go version -m could not read ${service}`)
    if (!metadata.includes(`vcs.revision=${revision}`)) fail(`${service} does not carry vcs.revision=${revision}`)
    if (!metadata.includes('vcs.modified=false')) fail(`${service} is stamped vcs.modified=true`)
    requireCleanTree('after packaging')

    if (resultFile) writeFileSync(resultFile, `${JSON.stringify({ buildRoot: ROOT, outputs: [release] })}\n`)
    console.log(`\nPackaged ${revision.slice(0, 8)} for ${platform}-${process.arch}: ${dirname(dirname(dirname(service)))}`)
    console.log(`Service stamp: vcs.revision=${revision}, vcs.modified=false, built ${statSync(service).mtime.toISOString()}`)
}

const COMMANDS = {
    help: ['List the commands', () => help()],
    tools: ['Check the Go, Node.js, npm and Git versions', tools],
    setup: ['Install the npm packages and fill the Go module cache', setup],
    build: ['Build the service binaries and the desktop bundles', build],
    start: ['Start the desktop app in development mode', start],
    package: ['Build the unpacked desktop app for this platform from a clean checkout and verify its stamp (--result FILE writes the output path as JSON)', packageApp],
    check: ['Run the gates: SPDX headers, build scripts, lint, typecheck, contracts, dead code, desktop unit tests', check],
    test: ['Run the desktop unit tests and go test in every services module', testAll],
    services: ['Stage the standalone services bundle in services/build/bin', services],
    'headers-fix': ['Insert the SPDX header into any file that is missing one', headersFix],
    clean: ['Remove built binaries, bundles and packages', clean],
}

function help() {
    console.log('Personal AI Router development commands\n\nUsage: node scripts/dev.mjs <command>\n')
    for (const [name, [description]] of Object.entries(COMMANDS)) console.log(`  ${name.padEnd(12)}${description}`)
}

const [name = 'help', ...rest] = process.argv.slice(2)
const command = COMMANDS[name]
if (!command) {
    console.error(`unknown command: ${name}\n`)
    help()
    process.exitCode = 2
} else {
    try {
        command[1](rest)
    } catch (error) {
        if (!(error instanceof CommandError)) throw error
        console.error(`\ndev.mjs ${name}: ${error.message}`)
        process.exitCode = 1
    }
}
