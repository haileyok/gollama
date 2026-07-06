package gollama

import (
	"strings"
	"testing"
)

// TestTurnStreamAnthropicLive is a live smoke test for native Anthropic SSE
// streaming. It requests a multi-sentence answer and asserts that TurnStream
// delivers multiple growing snapshots whose last value equals the final
// assistant text, with a stop reason set. Set ANTHROPIC_API_KEY to run.
func TestTurnStreamAnthropicLive(t *testing.T) {
	c := liveAnthropicClient(t)

	var snaps []string
	resp, err := c.TurnStream(RequestOptions{
		Model:    liveThinkingModel,
		System:   "You are concise.",
		Messages: []Message{{Role: "user", Content: "In about 150 words, explain what a hash map is and why lookups are fast."}},
		Options:  &Options{MaxTokens: 1024},
	}, func(text string) {
		snaps = append(snaps, text)
	})
	if err != nil {
		t.Fatalf("TurnStream: %v", err)
	}

	if len(snaps) < 2 {
		t.Fatalf("expected >=2 streamed snapshots, got %d", len(snaps))
	}
	for i := 1; i < len(snaps); i++ {
		if !strings.HasPrefix(snaps[i], snaps[i-1]) {
			t.Errorf("snapshot %d is not a growing prefix of the previous", i)
		}
	}
	final := resp.Choices[0].Message.Content
	if final == "" {
		t.Fatal("expected non-empty final content")
	}
	if snaps[len(snaps)-1] != final {
		t.Errorf("last snapshot != final content\n last: %q\nfinal: %q", snaps[len(snaps)-1], final)
	}
	if resp.StopReason == "" {
		t.Errorf("expected a stop reason on the streamed turn")
	}
	t.Logf("snapshots=%d final_len=%d stop=%s", len(snaps), len(final), resp.StopReason)
}
