// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"nvpair-shared/noderec"
)

// Check the argv at the launch boundary, including custom broker and service
// starts. A reserved plan that never reaches exec is not isolation.
func validateFixtureBrokerArgs(args []string) error {
	if err := validateFixtureMDNSArgs(args, "scanner-"); err != nil {
		return err
	}
	values := make(map[string]int)
	required := (fixtureServicePorts{}).args()
	for i := 0; i < len(args); i++ {
		if !strings.HasPrefix(args[i], "-") || args[i] == "--" || args[i] == "-" {
			return fmt.Errorf("fixture broker has a positional argument %q before all flags are parsed", args[i])
		}
		name, value, hasValue := strings.Cut(strings.TrimLeft(args[i], "-"), "=")
		if !hasValue && name != "scanner-mdns" && name != "version" {
			i++
			if i == len(args) {
				return fmt.Errorf("fixture flag --%s has no value", name)
			}
			value = args[i]
		}
		if name == "manual-ollama-port" || name == "manual-lmstudio-port" || name == "manual-node-info-port" {
			parsed, err := strconv.ParseInt(value, 0, strconv.IntSize)
			if err != nil {
				return fmt.Errorf("fixture flag --%s: %w", name, err)
			}
			port := int(parsed)
			if port != 0 {
				if err := validateFixturePort(port); err != nil {
					return fmt.Errorf("--%s: %w", name, err)
				}
			}
		}
		for j := 0; j < len(required); j += 2 {
			if name != strings.TrimLeft(required[j], "-") {
				continue
			}
			parsed, err := strconv.ParseInt(value, 0, strconv.IntSize)
			if err != nil {
				return fmt.Errorf("fixture flag --%s: %w", name, err)
			}
			port := int(parsed)
			values[name] = port
			break
		}
	}
	seen := make(map[int]string)
	for i := 0; i < len(required); i += 2 {
		name := strings.TrimLeft(required[i], "-")
		port, ok := values[name]
		if !ok {
			return fmt.Errorf("fixture broker omits --%s and would resolve a default service port", name)
		}
		if err := validateFixturePort(port); err != nil {
			return fmt.Errorf("--%s: %w", name, err)
		}
		if previous := seen[port]; previous != "" {
			return fmt.Errorf("fixture flags --%s and --%s share port %d", name, previous, port)
		}
		seen[port] = name
	}
	return nil
}

// Match flag.Bool's bare/equals syntax and last-value-wins behavior. Brokers
// and direct scanner commands prepend multicast-off before caller overrides.
func validateFixtureMDNSArgs(args []string, prefix string) error {
	enabled := false
	namespace := noderec.MDNSNamespace{Domain: noderec.Domain}
	for i := 0; i < len(args); i++ {
		if !strings.HasPrefix(args[i], "-") || args[i] == "--" {
			return fmt.Errorf("fixture positional argument %q", args[i])
		}
		name, value, hasValue := strings.Cut(strings.TrimLeft(args[i], "-"), "=")
		if !hasValue {
			if name == prefix+"mdns" || name == "version" {
				value = "true"
			} else {
				i++
				if i == len(args) {
					return fmt.Errorf("fixture flag --%s has no value", name)
				}
				value = args[i]
			}
		}
		switch name {
		case prefix + "mdns":
			parsed, err := strconv.ParseBool(value)
			if err != nil {
				return err
			}
			enabled = parsed
		case prefix + "mdns-service":
			namespace.Service = value
		}
	}
	if enabled {
		return validateFixtureMDNSNamespace(namespace)
	}
	return nil
}

func fixtureScannerCommand(binary string, args ...string) (*exec.Cmd, error) {
	if err := validateFixtureMDNSArgs(args, ""); err != nil {
		return nil, err
	}
	return exec.Command(binary, append([]string{"--mdns=false"}, args...)...), nil
}

func TestFixtureLaunchRejectsProductionMulticast(t *testing.T) {
	namespace := newFixtureMDNSNamespace(t)
	plan := fixtureServicePorts{24001, 24002, 24003, 24004, 24005, 24006}
	for _, extra := range [][]string{
		{"--scanner-mdns=true"}, {"--scanner-mdns"},
		{"--scanner-mdns=true", "--scanner-mdns-service", "_nvpair-node._tcp"},
		append(fixtureMDNSBrokerArgs(namespace), "-scanner-mdns-service=_NVPAIR-NODE._TCP."),
	} {
		if cmd, err := fixtureBrokerCommand("unused", append(plan.args(), extra...)...); err == nil || cmd != nil {
			t.Fatalf("unsafe multicast argv produced command: %v, %v", cmd, err)
		}
	}
	if cmd, err := fixtureBrokerCommand("unused", append(plan.args(), fixtureMDNSBrokerArgs(namespace)...)...); err != nil || cmd == nil {
		t.Fatalf("private multicast refused: %v", err)
	}
	if cmd, err := fixtureScannerCommand("unused", "--mdns=true"); err == nil || cmd != nil {
		t.Fatalf("direct scanner production multicast produced command: %v, %v", cmd, err)
	}
	if cmd, err := fixtureScannerCommand("unused", fixtureMDNSScannerArgs(namespace)...); err != nil || cmd == nil {
		t.Fatalf("private direct scanner refused: %v", err)
	}
}

func fixtureBrokerDefaults() []string {
	return []string{"--scanner-mdns=false", "--manual-ollama-port=0", "--manual-lmstudio-port=0", "--manual-node-info-port=0"}
}

func fixtureBrokerCommand(binary string, args ...string) (*exec.Cmd, error) {
	if err := validateFixtureBrokerArgs(args); err != nil {
		return nil, err
	}
	return exec.Command(binary, append(fixtureBrokerDefaults(), args...)...), nil
}

func fixtureServiceCommand(binary string, args ...string) (*exec.Cmd, error) {
	if len(args) > 0 && (args[0] == "status" || args[0] == "stop" || args[0] == "call") {
		return exec.Command(binary, args...), nil
	}
	for i, arg := range args {
		if arg != "--" {
			continue
		}
		if err := validateFixtureBrokerArgs(args[i+1:]); err != nil {
			return nil, err
		}
		serviceArgs := append([]string(nil), args[:i+1]...)
		serviceArgs = append(serviceArgs, fixtureBrokerDefaults()...)
		return exec.Command(binary, append(serviceArgs, args[i+1:]...)...), nil
	}
	return nil, fmt.Errorf("fixture service has no explicit broker arguments")
}

func TestFixtureLaunchRejectsDefaultServiceResolution(t *testing.T) {
	plan := fixtureServicePorts{24001, 24002, 24003, 24004, 24005, 24006}
	for _, seam := range []struct {
		name    string
		command func([]string) (*exec.Cmd, error)
	}{
		{"broker helper and custom broker", func(args []string) (*exec.Cmd, error) { return fixtureBrokerCommand("unused-broker", args...) }},
		{"service helper and custom service", func(args []string) (*exec.Cmd, error) {
			return fixtureServiceCommand("unused-service", append([]string{"--endpoint", "unused", "--"}, args...)...)
		}},
	} {
		t.Run(seam.name, func(t *testing.T) {
			if cmd, err := seam.command(plan.args()); err != nil || cmd == nil {
				t.Fatalf("private plan rejected: %v", err)
			}
			for i := 0; i < len(plan.args()); i += 2 {
				args := plan.args()
				name := args[i]
				t.Run("omits "+name, func(t *testing.T) {
					args = append(args[:i], args[i+2:]...)
					if cmd, err := seam.command(args); err == nil || cmd != nil {
						t.Fatalf("missing %s produced a command: %v, %v", name, cmd, err)
					}
				})
				for port := range installedServicePorts {
					t.Run(name+"="+strconv.Itoa(port), func(t *testing.T) {
						args := plan.args()
						args[i+1] = strconv.Itoa(port)
						if cmd, err := seam.command(args); err == nil || cmd != nil {
							t.Fatalf("default service target produced a command: %v, %v", cmd, err)
						}
					})
				}
			}
		})
	}
}

func TestFixtureLaunchRejectsPortFallbackAndArgumentBypasses(t *testing.T) {
	plan := fixtureServicePorts{24001, 24002, 24003, 24004, 24005, 24006}
	for _, tc := range []struct {
		name  string
		extra []string
	}{
		{"zero listener", []string{"--workload-manager-port=0"}},
		{"single dash default override", []string{"-workload-manager-port", "14320"}},
		{"equals default override", []string{"--workload-manager-port=14320"}},
		{"truncated port flag", []string{"--workload-manager-port"}},
		{"positional stops Go flag parsing", []string{"unexpected", "--workload-manager-port=24003"}},
		{"manual default probe override", []string{"--manual-ollama-port=11434"}},
		{"octal default listener", []string{"--workload-manager-port=033760"}},
		{"octal default probe", []string{"--manual-ollama-port=026252"}},
		{"hexadecimal default listener", []string{"--workload-manager-port=0x37f0"}},
		{"hexadecimal default probe", []string{"--manual-ollama-port=0x2caa"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if cmd, err := fixtureBrokerCommand("unused", append(plan.args(), tc.extra...)...); cmd != nil || err == nil {
				t.Fatalf("unsafe broker argv produced a command: %v, %v", cmd, err)
			}
		})
	}
}

func TestFixtureLaunchAcceptsPrivatePortsWithGoFlagSyntax(t *testing.T) {
	plan := fixtureServicePorts{24001, 24002, 24003, 24004, 24005, 24006}
	for _, value := range []string{"060651", "0o60651", "0x61a9", "0b110000110101001", "25_001"} {
		t.Run(value, func(t *testing.T) {
			args := append(plan.args(), "--workload-manager-port="+value, "--manual-ollama-port="+value)
			if cmd, err := fixtureBrokerCommand("unused", args...); cmd == nil || err != nil {
				t.Fatalf("private Go flag port rejected: %v, %v", cmd, err)
			}
		})
	}
}

// Keep direct broker launches and both custom helper seams behind the same
// pre-exec check. The runtime tests above exercise that check without starting
// a product or touching a default port.
func TestFixtureBrokerLaunchSeamsRemainGuarded(t *testing.T) {
	files, err := filepath.Glob("*_test.go")
	if err != nil {
		t.Fatal(err)
	}
	required := map[string]string{
		"startInteropProcess": "fixtureBrokerCommand",
		"startTask1Process":   "fixtureBrokerCommand",
		"startChild":          "fixtureServiceCommand",
	}
	for _, path := range files {
		file, err := parser.ParseFile(token.NewFileSet(), path, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		for _, decl := range file.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok {
				continue
			}
			guard := required[fn.Name.Name]
			found := false
			foundScanner := false
			ast.Inspect(fn, func(node ast.Node) bool {
				call, ok := node.(*ast.CallExpr)
				if !ok {
					return true
				}
				if name, ok := call.Fun.(*ast.Ident); ok && name.Name == guard {
					found = true
				}
				if name, ok := call.Fun.(*ast.Ident); ok && name.Name == "fixtureScannerCommand" {
					foundScanner = true
				}
				selector, ok := call.Fun.(*ast.SelectorExpr)
				if !ok || (selector.Sel.Name != "Command" && selector.Sel.Name != "CommandContext") {
					return true
				}
				pkg, ok := selector.X.(*ast.Ident)
				if !ok || pkg.Name != "exec" {
					return true
				}
				index := 0
				if selector.Sel.Name == "CommandContext" {
					index = 1
				}
				if len(call.Args) <= index {
					return true
				}
				switch target := call.Args[index].(type) {
				case *ast.Ident:
					if target.Name == "brokerBin" || target.Name == "serviceBin" || target.Name == "scannerBin" {
						t.Errorf("%s: %s bypasses the fixture broker/service/scanner launch guard", path, fn.Name.Name)
					}
				case *ast.SelectorExpr:
					if target.Sel.Name == "brokerPath" || target.Sel.Name == "servicePath" {
						t.Errorf("%s: %s bypasses the copied service launch guard", path, fn.Name.Name)
					}
				}
				return true
			})
			if fn.Name.Name == "startTask1Process" && !foundScanner {
				t.Errorf("%s: direct scanner launch no longer guarded", path)
			}
			if guard != "" {
				if !found {
					t.Errorf("%s: %s no longer calls %s", path, fn.Name.Name, guard)
				}
				delete(required, fn.Name.Name)
			}
		}
	}
	if len(required) != 0 {
		t.Fatalf("fixture launch seams not inspected: %v", required)
	}
}
