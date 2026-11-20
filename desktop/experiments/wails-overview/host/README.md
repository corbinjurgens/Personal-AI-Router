<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Wails Overview experiment host

The host opens the shared Overview in replay mode. Its tray menu has **Show
Overview** and **Quit**. Closing the window uses Wails' regular native close
path. The tray stays active and **Show Overview** creates a new window after
close. The replay fixtures run in the webview. The Go host never connects to
PAIR services or starts a local network listener.

## Build on macOS

Use Go 1.26 or later and Xcode command-line tools. This host pins Wails
`v3.0.0-alpha2.122`, the highest v3 alpha tag returned by
`go list -m -versions github.com/wailsapp/wails/v3` during implementation.
The validation machine used Go `go1.27.1 darwin/arm64`, macOS 27.2.1
(26B131), and WebKit framework build `22625.2.8.11.9`.

Use the shared `dist/renderer/` and `dist/asset-manifest.json` from
`npm run build:experiment:overview` in `desktop/` with Node 25.5 or later.
From this directory, stage those bytes under the Go module for `go:embed`
and build the app. `Info.plist` is the only additional bundle input.

```sh
(
  set -eu
  app=build/bin/PAIROverviewWails.app
  mkdir -p build/overview/renderer "$app/Contents/MacOS"
  rsync -a --delete ../dist/renderer/ build/overview/renderer/
  cp ../dist/asset-manifest.json build/overview/asset-manifest.json
  cp Info.plist "$app/Contents/Info.plist"
  go build -tags production -trimpath -o "$app/Contents/MacOS/PAIROverviewWails" .
  codesign --force --sign - "$app"
  "$app/Contents/MacOS/PAIROverviewWails" -verify-assets
)
```

The build uses the local Go toolchain and requires no Wails CLI. The
`production` tag disables the Wails development server override. The
host validates both the file list and each SHA-256 before launching. The
custom webview origin serves only the renderer files. Staged input and
build outputs under `build/` are ignored by Git.

Launch the app through Launch Services. Quit the running instance before
launching a different scene.

```sh
open -n "$PWD/build/bin/PAIROverviewWails.app" --args -mode=replay -scene=running-13
```

Use `-scene=idle` (default), `-scene=shared-node`, `-scene=running-12`, or
`-scene=running-13`. Pass `-order=nodes-first` to switch columns. The
bundle ID is `com.nvidia.pair.experiment.wails-overview`. On macOS, WebKit
stores website data under `~/Library/WebKit/<bundle-id>/WebsiteData` and
caches under `~/Library/Caches/<bundle-id>/WebKit`. The experimental Windows
WebView2 configuration uses `<user-cache>/<bundle-id>/WebView2`. Windows
runtime validation is separate from this macOS build.

After staging the assets, run `go vet ./...` and `go test ./...` from this
directory. For native validation, use the close button, wait ten seconds,
reopen from the tray, then use **Quit**. WebKit XPC helpers can have launchd
as their parent, so compare PID and start time with a pre-launch snapshot.

On the validation Mac, the initial WebContent process exited by the
ten-second sample. The Networking and GPU helpers remained at ten seconds
and were still present at sixty seconds in a separate run. Reopening from
the tray created a fresh WebContent process and rendered the fixture.
Ordinary tray **Quit** removed the app and the associated helpers.
