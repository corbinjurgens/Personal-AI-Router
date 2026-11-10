// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import Foundation

/// Shared between the privileged daemon (`PrivilegedHelper`) and the user-context
/// control tool (`HelperControl`). Both binaries compile this file in, so the
/// XPC interface and the security constants can never drift between the two
/// sides of the connection.
enum HelperConstants {
    /// launchd `Label` of the daemon and the basename of its LaunchDaemon plist.
    static let daemonLabel = "com.nvidia.nvpair.helper"
    static let daemonPlistName = "com.nvidia.nvpair.helper.plist"

    /// Mach service the daemon vends and the control tool connects to. Must match
    /// the `MachServices` key in the LaunchDaemon plist.
    static let machServiceName = "com.nvidia.nvpair.helper.xpc"

    // Code-signing requirements are generated into `HelperSigning.swift` by
    // scripts/build/macos/build-helper.ts from scripts/build/macos/helper-signing.ts,
    // pinned to the signer the build was configured for (NVIDIA's Developer ID
    // team by default). Each carries the signer's anchor: `anchor apple generic`,
    // the signing class's intermediate and leaf certificate markers, and the
    // leaf's Team ID OU.
    //
    // - `signedBinaryRequirement` (no identifier pin): each `cli-bin`
    //   executable, re-verified immediately before `socketfilterfw`.
    // - `ctlCodeSigningRequirement`: the daemon pins it on every **inbound**
    //   connection, so only `nvpair-helper-ctl` may drive privileged operations.
    // - `daemonCodeSigningRequirement`: the control tool pins it on its
    //   **outbound** connection, against a hijacked Mach service name.
    // - `appCodeSigningRequirement`: the host `.app` bundle the daemon configures.
}

/// Fixed command surface exposed over XPC. No arbitrary shell: every method maps
/// to one validated privileged operation.
@objc(PAIRHelperProtocol)
protocol PAIRHelperProtocol {
    /// Returns the daemon's build version (stamped to equal the app version).
    func getVersion(reply: @escaping (String) -> Void)

    /// Adds + unblocks the bundled networked binaries in the macOS Application
    /// Firewall. `appPath` is **ignored** (retained only for wire compatibility):
    /// the daemon derives the target bundle from the *verified* identity of the
    /// connecting process, so a caller can only ever configure its own bundle —
    /// never point the daemon at an arbitrary NVIDIA-signed `.app`.
    func configureFirewall(appPath: String, reply: @escaping (Bool, String?) -> Void)

    /// Removes the bundled networked binaries from the Application Firewall.
    /// `appPath` is ignored (see `configureFirewall`).
    func removeFirewall(appPath: String, reply: @escaping (Bool, String?) -> Void)
}
