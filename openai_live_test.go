package gollama

import (
	"os"
	"strings"
	"testing"
)

const openaiLiveReasoningModel = "gpt-5.1"

// liveOpenAIClient builds a client against the real OpenAI API, skipping the
// test when no key is available. Set OPENAI_API_KEY to run.
func liveOpenAIClient(t *testing.T) *Client {
	t.Helper()
	key := os.Getenv("OPENAI_API_KEY")
	if key == "" {
		t.Skip("OPENAI_API_KEY not set; skipping live OpenAI test")
	}
	c := NewClient("https://api.openai.com/v1")
	c.SetBearerToken(key)
	if c.Backend() != BackendOpenAI {
		t.Fatalf("expected BackendOpenAI, got %v", c.Backend())
	}
	return c
}

// TestOpenAIReasoningEffortLive checks that a reasoning model accepts the
// reasoning_effort request field (translated from RequestOptions.Effort) and
// returns an answer. Guarded by OPENAI_API_KEY; skips when absent.
func TestOpenAIReasoningEffortLive(t *testing.T) {
	c := liveOpenAIClient(t)
	resp, err := c.Turn(RequestOptions{
		Model:    openaiLiveReasoningModel,
		Messages: []Message{{Role: "user", Content: "How many times does the letter r appear in 'strawberry'? Reason it through, then give the number."}},
		Effort:   "low",
	})
	if err != nil {
		t.Fatalf("turn (reasoning_effort accepted?): %v", err)
	}
	m := resp.Choices[0].Message
	t.Logf("answer=%q reasoning_len=%d", m.Content, len(m.Thinking))
	if strings.TrimSpace(m.Content) == "" {
		t.Errorf("expected a non-empty answer")
	}
}
