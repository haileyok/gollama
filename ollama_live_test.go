package gollama

import (
	"net/http"
	"testing"
	"time"
)

const ollamaLiveModel = "gemma4:26b"

// liveOllamaClient builds a client against a local Ollama daemon, skipping the
// test when the daemon is not reachable. Ollama is routed through the
// OpenAI-compatible /v1 endpoint, so the base URL includes "/v1".
func liveOllamaClient(t *testing.T) *Client {
	t.Helper()
	probe := &http.Client{Timeout: 750 * time.Millisecond}
	resp, err := probe.Get("http://localhost:11434/api/tags")
	if err != nil {
		t.Skipf("local Ollama not reachable (%v); skipping live Ollama test", err)
	}
	resp.Body.Close()
	c := NewClient("http://localhost:11434/v1")
	c.SetMaxRetries(0)
	if c.Backend() != BackendOllama {
		t.Fatalf("expected BackendOllama for localhost:11434 URL, got %v", c.Backend())
	}
	return c
}

// TestOllamaThinkingLive_Reasoning checks that a turn with thinking enabled (and
// an effort level that Ollama cannot express, which must be harmlessly ignored)
// is accepted and returns reasoning content normalized into Message.Thinking.
func TestOllamaThinkingLive_Reasoning(t *testing.T) {
	c := liveOllamaClient(t)
	resp, err := c.Turn(RequestOptions{
		Model:    ollamaLiveModel,
		Messages: []Message{{Role: "user", Content: "How many times does the letter r appear in 'strawberry'? Think it through, then give the number."}},
		Thinking: "adaptive", // turns Ollama's think bool on
		Effort:   "high",     // inexpressible on Ollama; must be ignored, not error
		Options:  &Options{MaxTokens: 2048},
	})
	if err != nil {
		t.Fatalf("turn: %v", err)
	}
	m := resp.Choices[0].Message
	t.Logf("thinking_len=%d answer=%q", len(m.Thinking), m.Content)
	if m.Thinking == "" {
		t.Errorf("expected non-empty Message.Thinking with think on")
	}
	if m.Reasoning != "" {
		t.Errorf("Reasoning should be normalized into Thinking and cleared, got %q", m.Reasoning)
	}
}

// TestOllamaThinkingLive_ToolRoundTrip runs a tool-using turn with thinking on
// and then replays the assistant turn plus the tool result, asserting the whole
// exchange does not error. If the model rejects tools, it falls back to a
// two-turn history-replay smoke (still with thinking on) and logs which path ran.
func TestOllamaThinkingLive_ToolRoundTrip(t *testing.T) {
	c := liveOllamaClient(t)

	tools := []ToolParam{{
		Type: "function",
		Function: &ToolFunction{
			Name:        "get_weather",
			Description: "Get the current weather for a location.",
			Parameters: ToolFunctionParams{
				Type: "object",
				Properties: map[string]any{
					"location": map[string]any{"type": "string", "description": "City name"},
				},
				Required: []string{"location"},
			},
		},
	}}

	base := func(ms []Message, withTools bool) RequestOptions {
		o := RequestOptions{
			Model:    ollamaLiveModel,
			Messages: ms,
			Thinking: "adaptive",
			Options:  &Options{MaxTokens: 2048},
		}
		if withTools {
			o.Tools = tools
		}
		return o
	}

	msgs := []Message{{Role: "user", Content: "What's the weather in Paris? Use the get_weather tool."}}
	resp, err := c.Turn(base(msgs, true))
	if err != nil {
		// Some Ollama models reject tools with a 400. Fall back to a plain
		// two-turn thinking-on history replay and prove that does not error.
		t.Logf("tool turn errored (%v); falling back to plain two-turn thinking replay", err)
		runOllamaHistoryReplaySmoke(t, c, base)
		return
	}

	m := resp.Choices[0].Message
	t.Logf("turn1 tool_calls=%d thinking_len=%d text=%q", len(m.ToolCalls), len(m.Thinking), m.Content)
	if len(m.ToolCalls) == 0 {
		// Model chose not to call the tool; still exercise a thinking-on replay.
		t.Logf("no tool call returned; running plain two-turn thinking replay instead")
		runOllamaHistoryReplaySmoke(t, c, base)
		return
	}

	tc := m.ToolCalls[0]
	msgs = append(msgs, m, Message{Role: "tool", ToolCallID: tc.ID, Content: `{"tempC":18,"summary":"partly cloudy"}`})
	resp2, err := c.Turn(base(msgs, true))
	if err != nil {
		t.Fatalf("turn 2 (tool round-trip with thinking on): %v", err)
	}
	t.Logf("turn2 text=%q", resp2.Choices[0].Message.Content)
}

// runOllamaHistoryReplaySmoke runs two sequential thinking-on turns where the
// second replays the first assistant message, asserting neither errors.
func runOllamaHistoryReplaySmoke(t *testing.T, c *Client, base func([]Message, bool) RequestOptions) {
	t.Helper()
	msgs := []Message{{Role: "user", Content: "Think briefly, then say hello."}}
	r1, err := c.Turn(base(msgs, false))
	if err != nil {
		t.Fatalf("replay turn 1: %v", err)
	}
	msgs = append(msgs, r1.Choices[0].Message, Message{Role: "user", Content: "Now think briefly and say goodbye."})
	if _, err := c.Turn(base(msgs, false)); err != nil {
		t.Fatalf("replay turn 2 (assistant history with thinking on): %v", err)
	}
}
