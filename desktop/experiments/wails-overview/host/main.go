// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime"

	"github.com/wailsapp/wails/v3/pkg/application"
	"github.com/wailsapp/wails/v3/pkg/events"
	"github.com/wailsapp/wails/v3/pkg/icons"
)

const (
	appName    = "PAIR Overview Wails"
	bundleID   = "com.nvidia.pair.experiment.wails-overview"
	windowName = "overview"
)

func main() {
	mode := flag.String("mode", "replay", "Overview input mode")
	scene := flag.String("scene", "idle", "Replay scene: idle, shared-node, running-12, running-13")
	order := flag.String("order", "", "Replay order: leave empty for workloads first or pass nodes-first")
	verify := flag.Bool("verify-assets", false, "Verify the embedded renderer against its manifest and exit")
	flag.Parse()

	startURL, assets, err := newOverviewBridge(*mode, *scene, *order)
	if err != nil {
		log.Fatal(err)
	}
	if *verify {
		fmt.Println("Embedded Overview assets match the SHA-256 manifest.")
		return
	}

	var windowsDataPath string
	if runtime.GOOS == "windows" {
		cacheDir, err := os.UserCacheDir()
		if err != nil {
			log.Fatal(err)
		}
		windowsDataPath = filepath.Join(cacheDir, bundleID, "WebView2")
	}

	app := application.New(application.Options{
		Name:   appName,
		Assets: assets,
		Mac: application.MacOptions{
			ActivationPolicy: application.ActivationPolicyAccessory,
		},
		Windows: application.WindowsOptions{
			DisableQuitOnLastWindowClosed: true,
			WebviewUserDataPath:           windowsDataPath,
		},
		Linux: application.LinuxOptions{
			DisableQuitOnLastWindowClosed: true,
			ProgramName:                   bundleID,
		},
	})

	windowOptions := application.WebviewWindowOptions{
		Name:                       windowName,
		Title:                      appName,
		URL:                        startURL,
		Width:                      1200,
		Height:                     900,
		DefaultContextMenuDisabled: true,
	}

	showOverview := func() {
		application.InvokeSync(func() {
			showOverviewWindow(app.Window, windowOptions)
		})
	}
	menu := app.NewMenu()
	menu.Add("Show Overview").OnClick(func(*application.Context) { showOverview() })
	menu.Add("Quit").OnClick(func(*application.Context) { app.Quit() })
	tray := app.SystemTray.New()
	tray.SetTooltip(appName)
	if runtime.GOOS == "darwin" {
		tray.SetTemplateIcon(icons.SystrayMacTemplate)
	} else {
		tray.SetIcon(icons.SystrayLight)
	}
	tray.SetMenu(menu)

	app.Event.OnApplicationEvent(events.Common.ApplicationStarted, func(*application.ApplicationEvent) {
		showOverview()
	})
	if err := app.Run(); err != nil {
		log.Fatal(err)
	}
}

type overviewWindowManager interface {
	GetByName(string) (application.Window, bool)
	NewWithOptions(application.WebviewWindowOptions) *application.WebviewWindow
}

// Wails owns destruction and removes closed windows from its registry. A close
// can already have destroyed the native window while its registry entry is
// still present, so check the native handle before reusing that entry.
func showOverviewWindow(windows overviewWindowManager, options application.WebviewWindowOptions) {
	if current, ok := windows.GetByName(options.Name); ok && current.NativeWindow() != nil {
		current.UnMinimise()
		current.Focus()
		return
	}
	windows.NewWithOptions(options)
}
