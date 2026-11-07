// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"nvpair-shared/modelconfig"
)

const (
	modelDiagnosticGenerateTokens = 128
	modelDiagnosticExerciseLimit  = 20 * time.Minute
	modelDiagnosticPromptBytes    = 8 << 20
	modelDiagnosticResponseBytes  = 8 << 20
	modelDiagnosticEventBytes     = 256 << 10
	modelDiagnosticCorpus         = " The amber river crosses a quiet valley. Seven blue stones mark the path. Continue describing this imaginary landscape in plain words."
)

var (
	errModelDiagnosticEngineExited = errors.New("engine-exited")
	errModelDiagnosticIdentity     = errors.New("engine-identity-changed")
)

// The operation supplies the captured owned child's port, exact native model
// name, run ID and exit signal. Check must revalidate that same child/address,
// load and configuration, without changing them. It must honor its context and
// be safe for concurrent calls. This works for ExecutionKey loads too: a finite
// ServingEnvelope is deliberately not required by this HTTP-only exercise.
type modelDiagnosticExerciseChild struct {
	Port    int
	ModelID string
	RunID   string
	Done    <-chan struct{}
	Check   func(context.Context) error
}

type modelDiagnosticExercisePhase struct {
	Phase              string
	TargetPromptTokens int32
	Requests           int32
	// StartedAt is the operation's low/high-start marker. Zero captures it here.
	StartedAt time.Time
}

func modelDiagnosticTargets(perSlotContext, slots int32) (low, high int32, err error) {
	if perSlotContext < 208 {
		return 0, 0, errors.New("context-too-small")
	}
	if slots < 1 || slots > 8 {
		return 0, 0, errors.New("unsupported-slot-count")
	}
	high = perSlotContext - modelDiagnosticGenerateTokens - 16
	return min(int32(512), high/2), high, nil
}

// No lifecycle action or RPC registration happens here. The caller retains its
// reservation through return, then confirms unload/stop through the lifecycle
// owner: closing an HTTP request alone does not prove native work has stopped.
// The operation supplies the whole-run deadline, shared by both phases.
// Coverage is provisional. The broker sets OverlapWindowSampled and the final
// coverage after correlating fresh memory samples with these marker offsets.
func (e *Executor) modelDiagnosticExercise(ctx context.Context, child modelDiagnosticExerciseChild, target modelDiagnosticExercisePhase) (modelconfig.MeasuredPhase, error) {
	phase := modelconfig.MeasuredPhase{Coverage: "partial"}
	if child.Port < 1 || child.Port > 65535 || child.ModelID == "" || len(child.ModelID) > 256 || child.RunID == "" || len(child.RunID) > 256 || child.Done == nil || child.Check == nil {
		return phase, errModelDiagnosticIdentity
	}
	if target.TargetPromptTokens < 1 || target.Requests < 1 || target.Requests > 8 || (target.Phase != "low" && target.Phase != "high") || (target.Phase == "low" && target.Requests != 1) {
		return phase, errors.New("invalid-exercise-target")
	}
	started := target.StartedAt
	if started.IsZero() {
		started = time.Now()
	}
	if started.After(time.Now()) {
		return phase, errors.New("invalid-exercise-marker")
	}
	bounded, stop := context.WithTimeout(ctx, modelDiagnosticExerciseLimit)
	defer stop()
	work, cancel := context.WithCancelCause(bounded)
	watchDone := make(chan struct{})
	go func() {
		defer close(watchDone)
		select {
		case <-child.Done:
			cancel(errModelDiagnosticEngineExited)
		case <-work.Done():
		}
	}()
	defer func() { cancel(context.Canceled); <-watchDone }()
	client := *e.clientFor(modelDiagnosticExerciseLimit)
	client.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	client.Timeout = modelDiagnosticExerciseLimit
	runner := modelDiagnosticExerciseHTTP{client: &client, child: child}
	phase.Requests = make([]modelconfig.MeasuredRequest, target.Requests)
	prompts := make([]modelDiagnosticChatRequest, target.Requests)
	for i := range phase.Requests {
		phase.Requests[i].TargetPromptTokens = target.TargetPromptTokens
	}
	// Count every high prompt before dispatching any generation request. Thus
	// tokenization cannot serialize the concurrent full-context exercise.
	for i := range prompts {
		prompt, err := runner.prompt(work, target, i)
		if err != nil {
			phase.WallMS = time.Since(started).Milliseconds()
			return phase, runner.failure(work, err)
		}
		prompts[i] = prompt
	}
	var wg sync.WaitGroup
	var failure error
	var failureOnce sync.Once
	for i := range prompts {
		wg.Add(1)
		go func() {
			defer wg.Done()
			request, err := runner.complete(work, prompts[i], target.TargetPromptTokens, started)
			phase.Requests[i] = request
			if err != nil {
				failureOnce.Do(func() { failure = err; cancel(err) })
			}
		}()
	}
	wg.Wait() // all response bodies and request goroutines are joined on every exit
	wall := time.Since(started)
	phase.WallMS = wall.Milliseconds()
	if failure != nil {
		return phase, runner.failure(work, failure)
	}
	if err := runner.check(work); err != nil {
		return phase, err
	}
	phase.Coverage = "complete"
	var generated int64
	allGenerated := true
	latestFirst, earliestEnd := 0.0, math.Inf(1)
	overlap := true
	for _, request := range phase.Requests {
		if request.PromptTokens == nil || int64(*request.PromptTokens)*100 < int64(request.TargetPromptTokens)*95 || request.CachedTokens == nil || *request.CachedTokens != 0 {
			phase.Coverage = "partial"
		}
		if request.GeneratedTokens == nil {
			allGenerated = false
		} else {
			generated += int64(*request.GeneratedTokens)
		}
		if request.FirstTokenMS == nil || request.EndMS == nil {
			overlap = false
		} else {
			latestFirst = max(latestFirst, *request.FirstTokenMS)
			earliestEnd = min(earliestEnd, *request.EndMS)
		}
	}
	if target.Phase == "high" {
		overlap = overlap && latestFirst < earliestEnd
		phase.Overlapped = &overlap
		if !overlap {
			phase.Coverage = "partial"
		}
		if allGenerated && wall > 0 {
			rate := float64(generated) / wall.Seconds()
			phase.CombinedGeneratedPerSecond = &rate
		}
	}
	return phase, nil
}

type modelDiagnosticExerciseHTTP struct {
	client *http.Client
	child  modelDiagnosticExerciseChild
}

func (r modelDiagnosticExerciseHTTP) check(ctx context.Context) error {
	select {
	case <-r.child.Done:
		return errModelDiagnosticEngineExited
	default:
	}
	if ctx.Err() != nil {
		return context.Cause(ctx)
	}
	if r.child.Check(ctx) != nil {
		if ctx.Err() != nil {
			return context.Cause(ctx)
		}
		return errModelDiagnosticIdentity
	}
	return nil
}

func (r modelDiagnosticExerciseHTTP) failure(ctx context.Context, err error) error {
	if identity := r.check(ctx); identity != nil {
		return identity
	}
	return err
}

type modelDiagnosticChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type modelDiagnosticChatRequest struct {
	Model       string                       `json:"model"`
	Messages    []modelDiagnosticChatMessage `json:"messages"`
	MaxTokens   int                          `json:"max_tokens"`
	Temperature int                          `json:"temperature"`
	Stream      bool                         `json:"stream"`
	// Pinned 7fe450e tools/server/server-schema.cpp L27-L28 accepts this;
	// server-context.cpp L2999-L3001 clears prior tokens when it is false.
	// The chat parser forwards native parameters in server-common.cpp L1321-L1328.
	// https://github.com/ggml-org/llama.cpp/blob/7fe450e19305b828c199d602c23a8337aaa1f03b/tools/server/server-schema.cpp#L27-L28
	CachePrompt bool `json:"cache_prompt"`
}

func (r modelDiagnosticExerciseHTTP) post(ctx context.Context, path string, body modelDiagnosticChatRequest) (*http.Response, error) {
	if err := r.check(ctx); err != nil {
		return nil, err
	}
	data, err := json.Marshal(body)
	if err != nil || len(data) > modelDiagnosticPromptBytes+4096 {
		return nil, errors.New("exercise-request-bound")
	}
	query := url.Values{"autoload": {"false"}, "model": {r.child.ModelID}}
	address := fmt.Sprintf("http://127.0.0.1:%d%s?%s", r.child.Port, path, query.Encode())
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, address, bytes.NewReader(data))
	if err != nil {
		return nil, errors.New("exercise-request-invalid")
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set(engineIdentityProbeHeader, "1")
	resp, err := r.client.Do(req)
	if err != nil {
		return nil, r.failure(ctx, errors.New("exercise-transport-failed"))
	}
	if err := r.check(ctx); err != nil {
		resp.Body.Close()
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		return nil, errors.New("exercise-http-failed")
	}
	return resp, nil
}

func modelDiagnosticReadJSON(body io.Reader, value any) error {
	data, err := io.ReadAll(io.LimitReader(body, modelDiagnosticResponseBytes+1))
	if err != nil || len(data) > modelDiagnosticResponseBytes || json.Unmarshal(data, value) != nil {
		return errors.New("exercise-response-invalid")
	}
	return nil
}

// The pinned server applies the same chat template for counting and generation:
// tools/server/README.md L1434-L1445; server-context.cpp L5019-L5039.
// https://github.com/ggml-org/llama.cpp/blob/7fe450e19305b828c199d602c23a8337aaa1f03b/tools/server/server-context.cpp#L5019-L5039
func (r modelDiagnosticExerciseHTTP) prompt(ctx context.Context, target modelDiagnosticExercisePhase, index int) (modelDiagnosticChatRequest, error) {
	prefix := fmt.Sprintf("%s/%d/%s\n", target.Phase, index, r.child.RunID)
	request := modelDiagnosticChatRequest{Model: r.child.ModelID, MaxTokens: modelDiagnosticGenerateTokens, Stream: target.Phase == "high", Messages: []modelDiagnosticChatMessage{{Role: "user"}}}
	count := func(length int) (int32, error) {
		request.Messages[0].Content = prefix + strings.Repeat(modelDiagnosticCorpus, length/len(modelDiagnosticCorpus)) + modelDiagnosticCorpus[:length%len(modelDiagnosticCorpus)]
		resp, err := r.post(ctx, "/v1/chat/completions/input_tokens", request)
		if err != nil {
			return 0, err
		}
		defer resp.Body.Close()
		var result struct {
			InputTokens *int32 `json:"input_tokens"`
		}
		if err := modelDiagnosticReadJSON(resp.Body, &result); err != nil {
			return 0, r.failure(ctx, err)
		}
		if err := r.check(ctx); err != nil {
			return 0, err
		}
		if result.InputTokens == nil || *result.InputTokens < 1 {
			return 0, errors.New("exercise-token-count-invalid")
		}
		return *result.InputTokens, nil
	}
	base, err := count(0)
	if err != nil {
		return request, err
	}
	if base > target.TargetPromptTokens {
		return request, errors.New("exercise-prompt-target-unreachable")
	}
	best, bestCount := request.Messages[0].Content, base
	lo, hi := int64(0), min(int64(target.TargetPromptTokens)*4, int64(modelDiagnosticPromptBytes-len(prefix)))
	// Exponential bracketing followed by bounded byte-prefix bisection. Counted
	// values, never a bytes/token assumption, decide the retained prompt.
	for attempts := 0; attempts < 64 && bestCount != target.TargetPromptTokens; attempts++ {
		n, err := count(int(hi))
		if err != nil {
			return request, err
		}
		if n <= target.TargetPromptTokens {
			if n > bestCount {
				best, bestCount = request.Messages[0].Content, n
			}
			lo = hi
			if hi == int64(modelDiagnosticPromptBytes-len(prefix)) {
				break
			}
			hi = min(hi*2, int64(modelDiagnosticPromptBytes-len(prefix)))
			continue
		}
		// hi is now an over-target bound. Tokenizers can have plateaus;
		// always retain the largest observed count that fits the target.
		for lo+1 < hi && attempts < 63 {
			attempts++
			mid := lo + (hi-lo)/2
			n, err = count(int(mid))
			if err != nil {
				return request, err
			}
			if n <= target.TargetPromptTokens {
				lo = mid
				if n > bestCount {
					best, bestCount = request.Messages[0].Content, n
				}
			} else {
				hi = mid
			}
			if bestCount == target.TargetPromptTokens {
				break
			}
		}
		break
	}
	if int64(bestCount)*100 < int64(target.TargetPromptTokens)*95 {
		return request, errors.New("exercise-prompt-target-unreachable")
	}
	request.Messages[0].Content = best
	return request, nil
}

type modelDiagnosticTimings struct {
	PromptN            *int32   `json:"prompt_n"`
	CacheN             *int32   `json:"cache_n"`
	PromptMS           *float64 `json:"prompt_ms"`
	PromptPerSecond    *float64 `json:"prompt_per_second"`
	PredictedN         *int32   `json:"predicted_n"`
	PredictedMS        *float64 `json:"predicted_ms"`
	PredictedPerSecond *float64 `json:"predicted_per_second"`
}

type modelDiagnosticCompletion struct {
	Timings *modelDiagnosticTimings `json:"timings"`
	Error   json.RawMessage         `json:"error"`
	Choices []struct {
		FinishReason *string `json:"finish_reason"`
		Delta        struct {
			Content          string          `json:"content"`
			ReasoningContent string          `json:"reasoning_content"`
			ToolCalls        json.RawMessage `json:"tool_calls"`
		} `json:"delta"`
	} `json:"choices"`
}

func (r modelDiagnosticExerciseHTTP) complete(ctx context.Context, body modelDiagnosticChatRequest, target int32, started time.Time) (modelconfig.MeasuredRequest, error) {
	result := modelconfig.MeasuredRequest{TargetPromptTokens: target}
	resp, err := r.post(ctx, "/v1/chat/completions", body)
	if err != nil {
		return result, err
	}
	defer resp.Body.Close()
	if body.Stream {
		err = modelDiagnosticReadStream(resp.Body, &result, started)
	} else {
		var completion modelDiagnosticCompletion
		err = modelDiagnosticReadJSON(resp.Body, &completion)
		if err == nil {
			if len(completion.Error) > 0 && string(completion.Error) != "null" {
				err = errors.New("exercise-completion-failed")
			} else {
				modelDiagnosticApplyTimings(&result, completion.Timings)
			}
		}
		end := float64(time.Since(started)) / float64(time.Millisecond)
		result.EndMS = &end
	}
	if err != nil {
		return result, r.failure(ctx, err)
	}
	return result, r.check(ctx)
}

func modelDiagnosticReadStream(body io.Reader, result *modelconfig.MeasuredRequest, started time.Time) error {
	limited := &io.LimitedReader{R: body, N: modelDiagnosticResponseBytes + 1}
	scanner := bufio.NewScanner(limited)
	scanner.Buffer(make([]byte, 4096), modelDiagnosticEventBytes)
	var event strings.Builder
	finished := false
	done := false
	consume := func() error {
		data := strings.TrimSpace(event.String())
		event.Reset()
		if data == "" {
			return nil
		}
		if done {
			return errors.New("exercise-stream-invalid")
		}
		if data == "[DONE]" {
			done = true
			return nil
		}
		var chunk modelDiagnosticCompletion
		if json.Unmarshal([]byte(data), &chunk) != nil || len(chunk.Error) > 0 && string(chunk.Error) != "null" {
			return errors.New("exercise-stream-invalid")
		}
		for _, choice := range chunk.Choices {
			if result.FirstTokenMS == nil && (choice.Delta.Content != "" || choice.Delta.ReasoningContent != "" || len(choice.Delta.ToolCalls) > 0 && string(choice.Delta.ToolCalls) != "null" && string(choice.Delta.ToolCalls) != "[]") {
				first := float64(time.Since(started)) / float64(time.Millisecond)
				result.FirstTokenMS = &first
			}
			if choice.FinishReason != nil {
				finished = true
				end := float64(time.Since(started)) / float64(time.Millisecond)
				result.EndMS = &end
			}
		}
		if finished && chunk.Timings != nil {
			modelDiagnosticApplyTimings(result, chunk.Timings)
		}
		return nil
	}
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			if err := consume(); err != nil {
				return err
			}
		} else if strings.HasPrefix(line, "data:") {
			if event.Len()+len(line) > modelDiagnosticEventBytes {
				return errors.New("exercise-stream-bound")
			}
			event.WriteString(strings.TrimPrefix(line, "data:"))
			event.WriteByte('\n')
		}
	}
	if scanner.Err() != nil || limited.N <= 0 || event.Len() != 0 || !done || !finished {
		return errors.New("exercise-stream-incomplete")
	}
	return nil
}

// Mirrors nvpair-proxy/performance_extract.go's llama semantics without its
// private stream helpers: generated tokens are predicted_n, but decode work is
// max(predicted_n-1, 0). Keep the reported rate, never recompute it as n/ms.
func modelDiagnosticApplyTimings(result *modelconfig.MeasuredRequest, timings *modelDiagnosticTimings) {
	if timings == nil {
		return
	}
	count := func(value *int32) *int32 {
		if value == nil || *value < 0 {
			return nil
		}
		return value
	}
	number := func(value *float64) *float64 {
		if value == nil || *value < 0 || math.IsNaN(*value) || math.IsInf(*value, 0) {
			return nil
		}
		return value
	}
	result.PromptTokens, result.CachedTokens, result.GeneratedTokens = count(timings.PromptN), count(timings.CacheN), count(timings.PredictedN)
	result.PromptMS, result.PromptPerSecond = number(timings.PromptMS), number(timings.PromptPerSecond)
	result.DecodeMS, result.DecodePerSecond = number(timings.PredictedMS), number(timings.PredictedPerSecond)
	if result.DecodeMS == nil || *result.DecodeMS == 0 || result.GeneratedTokens == nil {
		result.DecodePerSecond = nil
		return
	}
	steps := max(*result.GeneratedTokens-1, 0)
	if result.DecodePerSecond != nil {
		expected := float64(steps) * 1000 / *result.DecodeMS
		// Native timings carry floating milliseconds. The proxy permits the
		// same relative rounding tolerance for its nanosecond projection.
		if math.Abs(*result.DecodePerSecond-expected) > max(expected*1e-6, 1e-9) {
			result.DecodeMS, result.DecodePerSecond = nil, nil
		}
	}
}
