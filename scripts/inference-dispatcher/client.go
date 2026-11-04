// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"sort"
	"strconv"
	"strings"
	"time"
)

const maxResponseBytes = 16 << 20

type RegisteredModel struct {
	Name         string   `json:"name"`
	Capabilities []string `json:"capabilities,omitempty"`
	Type         string   `json:"type,omitempty"`
}

type backendClient struct {
	cfg    Config
	client *http.Client
	base   *url.URL
}

func newBackendClient(cfg Config) (*backendClient, error) {
	base := &url.URL{
		Scheme: "http",
		Host:   net.JoinHostPort(defaultHost, strconv.Itoa(effectivePort(cfg))),
	}
	return &backendClient{
		cfg: cfg,
		client: &http.Client{
			Timeout: time.Duration(cfg.TimeoutSeconds * float64(time.Second)),
		},
		base: base,
	}, nil
}

func (c *backendClient) endpoint(path string) string {
	copy := *c.base
	copy.Path = strings.TrimRight(copy.Path, "/") + path
	return copy.String()
}

func (c *backendClient) request(ctx context.Context, method, path string, payload any) ([]byte, error) {
	var body io.Reader
	if payload != nil {
		data, err := json.Marshal(payload)
		if err != nil {
			return nil, fmt.Errorf("encode request: %w", err)
		}
		body = bytes.NewReader(data)
	}
	request, err := http.NewRequestWithContext(ctx, method, c.endpoint(path), body)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	request.Header.Set("Accept", "application/json")
	request.Header.Set("User-Agent", "inference-dispatcher/"+Version)
	if payload != nil {
		request.Header.Set("Content-Type", "application/json")
	}
	response, err := c.client.Do(request)
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) {
			return nil, fmt.Errorf("timed out after %g seconds", c.cfg.TimeoutSeconds)
		}
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer response.Body.Close()

	data, err := io.ReadAll(io.LimitReader(response.Body, maxResponseBytes+1))
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}
	if len(data) > maxResponseBytes {
		return nil, fmt.Errorf("response exceeded %d bytes", maxResponseBytes)
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		// The body is described, never quoted. An OpenAI-compatible endpoint
		// echoes the offending request back in its error envelope, so relaying
		// the text would put prompt content into the result log and the debug
		// error log — both durable and unrotated.
		return nil, fmt.Errorf(
			"HTTP %s (%d bytes, %s)",
			response.Status, len(data), digest(string(data)),
		)
	}
	return data, nil
}

func decodeObject(data []byte, target any) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	if err := decoder.Decode(target); err != nil {
		return fmt.Errorf("invalid JSON from server: %w", err)
	}
	return nil
}

func (c *backendClient) listModels(ctx context.Context) ([]RegisteredModel, error) {
	data, err := c.request(ctx, http.MethodGet, "/v1/models", nil)
	if err != nil {
		return nil, fmt.Errorf("query %s models: %w", c.cfg.Backend, err)
	}
	models, err := parseOpenAIModels(data)
	if err != nil {
		return nil, err
	}
	if len(models) == 0 {
		return nil, fmt.Errorf("no registered %s models were found", c.cfg.Backend)
	}
	sort.SliceStable(models, func(i, j int) bool {
		return strings.ToLower(models[i].Name) < strings.ToLower(models[j].Name)
	})
	return models, nil
}

func parseOpenAIModels(data []byte) ([]RegisteredModel, error) {
	var response struct {
		Data []struct {
			ID    string `json:"id"`
			Model string `json:"model"`
		} `json:"data"`
	}
	if err := decodeObject(data, &response); err != nil {
		return nil, err
	}
	models := make([]RegisteredModel, 0, len(response.Data))
	for _, raw := range response.Data {
		name := strings.TrimSpace(raw.ID)
		if name == "" {
			name = strings.TrimSpace(raw.Model)
		}
		if name != "" {
			models = append(models, RegisteredModel{Name: name})
		}
	}
	return models, nil
}

func supportsGeneration(model RegisteredModel) bool {
	if model.Type != "" {
		return model.Type == "llm"
	}
	if len(model.Capabilities) == 0 {
		return true
	}
	for _, capability := range model.Capabilities {
		if capability == "completion" || capability == "chat" || capability == "generate" {
			return true
		}
	}
	return false
}

func (c *backendClient) resolveModel(ctx context.Context) (string, []RegisteredModel, error) {
	if model := strings.TrimSpace(c.cfg.Model); model != "" && !strings.EqualFold(model, "auto") {
		return model, nil, nil
	}
	models, err := c.listModels(ctx)
	if err != nil {
		return "", nil, err
	}
	for _, model := range models {
		if c.cfg.AllowNonGenerativeModels || supportsGeneration(model) {
			return model.Name, models, nil
		}
	}
	return "", models, errors.New("no available model advertises text-generation support")
}

func (c *backendClient) infer(ctx context.Context, model, prompt string) (string, error) {
	payload := map[string]any{
		"model":    model,
		"messages": []map[string]string{{"role": "user", "content": prompt}},
		"stream":   false,
	}
	if c.cfg.MaxTokens > 0 {
		payload["max_tokens"] = c.cfg.MaxTokens
	}
	if c.cfg.Temperature != nil {
		payload["temperature"] = *c.cfg.Temperature
	}
	data, err := c.request(ctx, http.MethodPost, "/v1/chat/completions", payload)
	if err != nil {
		return "", err
	}
	return parseOpenAIResponse(data)
}

func parseOpenAIResponse(data []byte) (string, error) {
	var response struct {
		Choices []struct {
			Message struct {
				Content any `json:"content"`
			} `json:"message"`
			Text string `json:"text"`
		} `json:"choices"`
	}
	if err := decodeObject(data, &response); err != nil {
		return "", err
	}
	if len(response.Choices) == 0 {
		return "", errors.New("OpenAI-compatible response contained no choices")
	}
	choice := response.Choices[0]
	switch content := choice.Message.Content.(type) {
	case string:
		return strings.TrimSpace(content), nil
	case []any:
		var builder strings.Builder
		for _, item := range content {
			switch part := item.(type) {
			case string:
				builder.WriteString(part)
			case map[string]any:
				if text, ok := part["text"].(string); ok {
					builder.WriteString(text)
				}
			}
		}
		return strings.TrimSpace(builder.String()), nil
	}
	return strings.TrimSpace(choice.Text), nil
}
