// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"os/exec"
	"strconv"
	"strings"
	"sync"

	"nvpair-shared/engines"
)

// LocalCredential answers engines.LocalCredentialMethod for the broker.
func (e *Executor) LocalCredential(engine string) (engines.LocalCredential, error) {
	st, err := e.state(engine)
	if err != nil {
		return engines.LocalCredential{}, err
	}
	c, running, err := e.credentialForRunning(context.Background(), st)
	if err != nil {
		return engines.LocalCredential{}, err
	}
	if !running || c.key == "" {
		return engines.LocalCredential{}, nil
	}
	return engines.LocalCredential{Credential: c.key, Port: c.port}, nil
}

// ActionHTTPError preserves the status of a failed HTTP action. Body is the
// bounded, whitespace-trimmed response excerpt used by the action diagnostic.
type ActionHTTPError struct {
	Action     string
	StatusCode int
	Body       string
}

// The request enters client.Do while the projection gate is held. Hand it off
// at transport entry so neither network I/O nor the response holds that gate.
// Once handed off, the existing registered context owns its cancellation.
type activationDispatchTransport struct {
	next    http.RoundTripper
	entered func()
}

func (t activationDispatchTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	t.entered()
	if err := vetoCause(req.Context(), req.Context().Err()); err != nil {
		return nil, err
	}
	return t.next.RoundTrip(req)
}

func (e *ActionHTTPError) Error() string {
	return fmt.Sprintf("action %q: engine returned HTTP %d: %s", e.Action, e.StatusCode, e.Body)
}

// Action runs a manifest-declared action against the engine and, when the
// action declares restart_after, restarts the engine on success.
//
// An action the manifest binds to the delete or pull operation changes the
// engine's installed models: it takes the engine's content-mutation guard
// (model_mutation.go), so the broker withdraws that engine's model identity
// from the destination before the action's first engine-visible effect and the
// change runs only once the withdrawal is confirmed.
func (e *Executor) Action(ctx context.Context, engine, action string, params json.RawMessage) (json.RawMessage, error) {
	st, err := e.state(engine)
	if err != nil {
		return nil, err
	}
	act, ok := st.manifest.Actions[action]
	if !ok {
		return nil, fmt.Errorf("engine %q has no action %q", engine, action)
	}
	if err := st.manifest.validateActionModelParams(action, params, true); err != nil {
		return nil, err
	}
	if action == pullModelAction {
		return e.PullModelStream(ctx, engine, "", params)
	}
	if err := e.checkServingLoad(st, action, params); err != nil {
		return nil, err
	}
	if act.RestartAfter {
		done, err := e.admitMutation(engine)
		if err != nil {
			return nil, err
		}
		defer done()
	}
	run := func() (json.RawMessage, error) {
		loadCtx, releaseServingLoad, err := e.guardServingLoad(ctx, st, action, params)
		if err != nil {
			return nil, err
		}
		// Unlock before admission cleanup and before leaving an outer content
		// mutation. A multiply-bound action never acquires contentMu under opMu.
		defer releaseServingLoad()
		res, err := e.dispatchAction(loadCtx, st, engine, action, act, params)
		err = vetoCause(loadCtx, err)
		// Authorization and dispatch share opMu. The existing restart owner
		// takes it itself, so release the load guard before that step.
		releaseServingLoad()
		if err != nil {
			return nil, err
		}
		// The restart gets the caller's context, not the action's: it is a stop
		// plus a readiness-probed start, which the action timeout does not
		// budget for.
		if act.RestartAfter {
			if err := e.restartAfterAction(ctx, st, engine, action); err != nil {
				return nil, err
			}
		}
		return res, nil
	}
	if !st.manifest.mutatesModelContent(action) {
		return run()
	}
	// The action deletes or pulls models: the withdrawal must be confirmed
	// before the action starts, and the restart that makes the change visible
	// belongs to the same mutation.
	var res json.RawMessage
	if err := e.withModelMutation(ctx, engine, func() error {
		ctx = context.WithValue(ctx, modelContentWriterKey{}, engine)
		var runErr error
		res, runErr = run()
		if runErr == nil {
			if binding, ok := st.manifest.operationBinding(engines.OperationDelete); ok && binding.Action == action {
				e.resolveDeletedDownload(ctx, st, binding.modelParam(), params, res)
			}
		}
		return runErr
	}); err != nil {
		return nil, err
	}
	return res, nil
}

// restartAfterAction restarts a running engine after one of its actions
// succeeded (see Action.RestartAfter). A stopped engine has no serving state to
// reconcile, so it stays stopped. The restart failing is reported as the
// action failing: the action's effect is only observable once the engine comes
// back, and leaving it down silently would look like a successful no-op.
//
// This deliberately does not call Restart. Two reasons:
//
//   - Restart takes opMu itself, so the "is it running?" check would have to sit
//     outside the lock, and a Stop landing in that window would be undone — the
//     deletion would start an engine the user had just turned off. Holding opMu
//     across both makes the two orderings the only possible ones: a Stop either
//     wins the lock first (we observe stopped and leave it down) or waits and
//     then stops the engine we brought back.
//   - Keep the stop/start pair under this lifecycle lock.
func (e *Executor) restartAfterAction(ctx context.Context, st *engineState, engine, action string) error {
	done, err := e.admitMutation(engine)
	if err != nil {
		return err
	}
	defer done()
	if err := prepareStrataState(ctx, st); err != nil {
		return err
	}
	st.opMu.Lock()
	defer st.opMu.Unlock()
	st.mu.Lock()
	running := st.running
	st.mu.Unlock()
	if !running {
		return nil
	}
	slog.Info("restarting engine after action", "engine", engine, "action", action)
	if err := e.doStop(st, engine); err != nil {
		return fmt.Errorf("action %q: engine %q failed to restart: %w", action, engine, err)
	}
	if err := e.doStart(ctx, st, engine, startOpts{}); err != nil {
		return fmt.Errorf("action %q: engine %q failed to restart: %w", action, engine, err)
	}
	return nil
}

// dispatchAction invokes the action itself: a guarded filesystem removal, a CLI
// command, or an HTTP call against the engine's loopback control API with the
// caller's params in the manifest-declared location.
func (e *Executor) dispatchAction(ctx context.Context, st *engineState, engine, action string, act Action, params json.RawMessage) (json.RawMessage, error) {
	ctx, cancel := context.WithTimeout(ctx, e.actionTimeout)
	defer cancel()
	st.mu.Lock()
	port := st.port
	st.mu.Unlock()

	// CLI actions (act.Cmd) drive the engine's control binary, which
	// commonly works while the server is down (e.g. listing or pulling
	// models), so they are intentionally NOT gated on running. HTTP
	// actions hit the engine's loopback control API and therefore require
	// it to be up.
	if act.RemovePath != nil {
		return e.runRemovePathAction(ctx, st, act, params)
	}
	if len(act.Cmd) > 0 {
		return e.runCmdAction(ctx, st, act, port, params)
	}
	credential, running, err := e.credentialForRunning(ctx, st)
	if err != nil {
		return nil, err
	}
	if !running {
		return nil, fmt.Errorf("engine %q is not running", engine)
	}
	port = credential.port

	path, err := resolvePlaceholders(act.HTTP.Path, map[string]string{"port": strconv.Itoa(port)})
	if err != nil {
		return nil, err
	}
	requestURL := fmt.Sprintf("http://127.0.0.1:%d%s", port, path)

	var body io.Reader
	if act.HTTP.ParamsIn == actionHTTPParamsQuery {
		parsedURL, err := url.Parse(requestURL)
		if err != nil {
			return nil, fmt.Errorf("action %q: invalid HTTP URL: %w", action, err)
		}
		queryParams, err := actionQueryParams(params)
		if err != nil {
			return nil, fmt.Errorf("action %q: %w", action, err)
		}
		query := parsedURL.Query()
		for name, values := range queryParams {
			for _, value := range values {
				query.Add(name, value)
			}
		}
		parsedURL.RawQuery = query.Encode()
		requestURL = parsedURL.String()
	} else if len(params) > 0 && string(params) != "null" {
		body = bytes.NewReader(params)
	}
	req, err := http.NewRequestWithContext(ctx, strings.ToUpper(act.HTTP.Method), requestURL, body)
	if err != nil {
		if credential.key != "" {
			return nil, errCredentialRequest
		}
		return nil, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	req.Header.Set(engineIdentityProbeHeader, "1")
	// An action a model operation binds with a longer response budget gets that
	// budget on every call, engine:action included.
	client := e.clientFor(st.manifest.actionResponseBudget(action))
	observation := action == "list_models" || action == "loaded_models"
	if observation {
		// Proof must come from the requested URL's 200, never a redirect
		// destination. Copy the client to preserve other actions' policy.
		observationClient := *client
		observationClient.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
		client = &observationClient
	}
	client, err = credentialHTTPClient(client, req, credential)
	if err != nil {
		return nil, err
	}
	finishDispatch, err := beginActivationEffect(ctx)
	if err != nil {
		if err == context.Canceled || err == context.DeadlineExceeded {
			// The final gate prevents dispatch, while retaining the action's
			// ordinary HTTP cancellation envelope. Typed vetoes remain intact.
			return nil, fmt.Errorf("action %q: %w", action, &url.Error{
				Op: req.Method[:1] + strings.ToLower(req.Method[1:]), URL: req.URL.String(), Err: err,
			})
		}
		return nil, err
	}
	if _, guarded := ctx.Value(activationDispatchKey{}).(activationDispatch); guarded {
		finishDispatch = sync.OnceFunc(finishDispatch)
		defer finishDispatch() // client validation can fail before transport entry
		dispatchClient := *client
		transport := client.Transport
		if transport == nil {
			transport = http.DefaultTransport
		}
		dispatchClient.Transport = activationDispatchTransport{next: transport, entered: finishDispatch}
		client = &dispatchClient
	} else {
		finishDispatch()
	}
	resp, err := client.Do(req)
	if err != nil {
		if credential.key != "" {
			return nil, errCredentialHTTP
		}
		return nil, fmt.Errorf("action %q: %w", action, err)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(io.LimitReader(resp.Body, 8*1024*1024))
	if err != nil {
		if credential.key != "" {
			return nil, errCredentialHTTP
		}
		return nil, fmt.Errorf("action %q: read response body: %w", action, err)
	}
	// Inventory reads are observation evidence only on exactly 200. Apply
	// this to list_models too: a sweep can share its response with loaded_models.
	// Mutation actions retain their general 2xx success contract (e.g. 204).
	if resp.StatusCode < 200 || resp.StatusCode >= 300 || (observation && resp.StatusCode != http.StatusOK) {
		body := strings.TrimSpace(string(data))
		if credential.key != "" {
			body = "Authenticated engine request was refused."
		}
		return nil, &ActionHTTPError{Action: action, StatusCode: resp.StatusCode, Body: body}
	}
	if len(data) == 0 {
		return json.RawMessage("null"), nil
	}
	if json.Valid(data) {
		if credential.key != "" {
			var value any
			decoder := json.NewDecoder(bytes.NewReader(data))
			decoder.UseNumber()
			if err := decoder.Decode(&value); err != nil {
				return nil, errCredentialHTTP
			}
			clean, err := json.Marshal(redactCredentialJSON(value, credential.key))
			if err != nil {
				return nil, errCredentialHTTP
			}
			return clean, nil
		}
		return json.RawMessage(data), nil
	}
	if credential.key != "" {
		data = []byte(strings.ReplaceAll(string(data), credential.key, "[redacted]"))
	}
	wrapped, _ := json.Marshal(string(data))
	return wrapped, nil
}

func actionQueryParams(params json.RawMessage) (url.Values, error) {
	query := make(url.Values)
	if len(params) == 0 {
		return query, nil
	}
	var values map[string]string
	if err := json.Unmarshal(params, &values); err != nil {
		return nil, fmt.Errorf("http query params must be a JSON object with string values: %w", err)
	}
	for name, value := range values {
		query.Add(name, value)
	}
	return query, nil
}

// runRemovePathAction resolves templated path/root placeholders and deletes
// the target when it stays under the declared root.
func (e *Executor) runRemovePathAction(ctx context.Context, st *engineState, act Action, params json.RawMessage) (json.RawMessage, error) {
	if act.RemovePath == nil {
		return nil, fmt.Errorf("remove_path action missing spec")
	}
	vars := st.pathVars()
	if len(params) > 0 {
		var pm map[string]any
		if err := json.Unmarshal(params, &pm); err == nil {
			for k, v := range pm {
				if allowedPlaceholders[k] {
					continue
				}
				vars[k] = fmt.Sprint(v)
			}
		}
	}
	root, err := resolvePlaceholders(act.RemovePath.Root, vars)
	if err != nil {
		return nil, err
	}
	root = expandPath(root)

	target, err := resolvePlaceholders(act.RemovePath.Path, vars)
	if err != nil {
		return nil, err
	}
	target = expandPath(target)
	if err := safeRemoveUnderRoot(root, target); err != nil {
		return nil, err
	}
	return json.RawMessage("null"), nil
}

// runCmdAction executes a CLI action, templating the action's params as
// placeholders (e.g. {model}), and returns the command's stdout as the
// result (parsed as JSON when it is valid JSON, else a JSON string).
func (e *Executor) runCmdAction(ctx context.Context, st *engineState, act Action, port int, params json.RawMessage) (json.RawMessage, error) {
	// A caller param can never override a runner-owned placeholder
	// ({bin},{port},{download},{install_dir}): those keys are dropped
	// from params, then seeded from trusted values — so a malicious param
	// can't hijack argv[0]. Param values are also not env-expanded.
	vars := map[string]string{}
	if len(params) > 0 {
		var pm map[string]any
		if err := json.Unmarshal(params, &pm); err == nil {
			for k, v := range pm {
				if allowedPlaceholders[k] {
					continue
				}
				vars[k] = fmt.Sprint(v)
			}
		}
	}
	vars["port"] = strconv.Itoa(port)
	for name, value := range st.pathVars() {
		vars[name] = value
	}

	argv, err := resolveArgs(act.Cmd, vars)
	if err != nil {
		return nil, err
	}
	out, err := e.runCommandOutput(ctx, argv)
	if err != nil {
		return nil, fmt.Errorf("action command failed: %w", err)
	}
	out = strings.TrimSpace(out)
	if out == "" {
		return json.RawMessage("null"), nil
	}
	if json.Valid([]byte(out)) {
		return json.RawMessage(out), nil
	}
	wrapped, _ := json.Marshal(out)
	return wrapped, nil
}

// runCommandOutput runs argv and returns its stdout; on failure it
// returns the error with stderr attached for diagnostics.
func (e *Executor) runCommandOutput(ctx context.Context, argv []string) (string, error) {
	if len(argv) == 0 {
		return "", nil
	}
	cmd := exec.CommandContext(ctx, argv[0], argv[1:]...)
	cmd.Env = childEnv(nil)
	configureSysProcAttr(cmd)
	// CommandContext's default Cancel kills only the command. Bound pipe
	// draining too: descendants may survive and retain the output pipes.
	cmd.WaitDelay = stopGrace(Runtime{})
	if _, guarded := ctx.Value(activationDispatchKey{}).(activationDispatch); !guarded {
		out, err := cmd.Output()
		if err != nil {
			if ee, ok := err.(*exec.ExitError); ok {
				return "", fmt.Errorf("%v: %s", err, strings.TrimSpace(string(ee.Stderr)))
			}
			return "", err
		}
		return string(out), nil
	}
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	finishDispatch, err := beginActivationEffect(ctx)
	if err != nil {
		return "", err
	}
	err = cmd.Start()
	finishDispatch()
	if err == nil {
		err = cmd.Wait()
	}
	if err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			return "", fmt.Errorf("%v: %s", ee, strings.TrimSpace(stderr.String()))
		}
		return "", err
	}
	return stdout.String(), nil
}
