// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// pipecommand is a hermetic command whose child inherits and retains stdout.
package main

import (
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"time"
)

func main() {
	if len(os.Args) != 4 {
		os.Exit(2)
	}
	mode, parentFile, childFile := os.Args[1], os.Args[2], os.Args[3]
	if mode != "child" {
		child := exec.Command(os.Args[0], "child", parentFile, childFile)
		child.Stdout, child.Stderr = os.Stdout, os.Stderr
		if err := child.Start(); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(3)
		}
		if err := os.WriteFile(parentFile, []byte(strconv.Itoa(os.Getpid())), 0600); err != nil {
			os.Exit(4)
		}
		if mode == "exit" {
			return
		}
	} else if err := os.WriteFile(childFile, []byte(strconv.Itoa(os.Getpid())), 0600); err != nil {
		os.Exit(5)
	}
	time.Sleep(time.Hour)
}
