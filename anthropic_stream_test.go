package gollama

import (
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
)

// richAnthropicStream is a scripted Anthropic Messages SSE body exercising a
// thinking block (thinking_delta + signature_delta), several text_delta chunks,
// and a tool_use whose input JSON is split across multiple input_json_delta
// chunks, finishing with stop_reason "tool_use".
const richAnthropicStream = `event: message_start
data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-x","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1,"cache_creation_input_tokens":3,"cache_read_input_tokens":2}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Let me "}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"think."}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig123"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Hello"}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":", "}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"world!"}}

event: content_block_stop
data: {"type":"content_block_stop","index":1}

event: content_block_start
data: {"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"location\":"}}

event: content_block_delta
data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"\"San Francisco\","}}

event: content_block_delta
data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"\"units\":\"celsius\"}"}}

event: content_block_stop
data: {"type":"content_block_stop","index":2}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":25}}

event: message_stop
data: {"type":"message_stop"}

`

// equivalent non-streaming JSON body that must convert to the identical result.
const richAnthropicNonStreaming = `{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-x","content":[{"type":"thinking","thinking":"Let me think.","signature":"sig123"},{"type":"text","text":"Hello, world!"},{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"location":"San Francisco","units":"celsius"}}],"stop_reason":"tool_use","stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":25,"cache_creation_input_tokens":3,"cache_read_input_tokens":2}}`

func sseServer(t *testing.T, body string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		io.WriteString(w, body)
	}))
}

func anthropicTestClient(t *testing.T, url string) *Client {
	t.Helper()
	c := NewClient(url)
	c.SetAnthropicMode(true)
	c.SetAPIKey("test-key")
	c.SetMaxRetries(0)
	return c
}

// TestTurnStreamAnthropic_RichStream verifies that a streamed turn assembles a
// final message byte-equivalent to the non-streaming shape, that onDelta
// snapshots are monotonically growing prefixes ending at the final content, and
// that usage is merged from message_start + message_delta.
func TestTurnStreamAnthropic_RichStream(t *testing.T) {
	srv := sseServer(t, richAnthropicStream)
	defer srv.Close()
	c := anthropicTestClient(t, srv.URL)

	var snaps []string
	result, err := c.TurnStream(RequestOptions{
		Model:    "claude-sonnet-4-x",
		Messages: []Message{{Role: "user", Content: "hi"}},
	}, func(text string) {
		snaps = append(snaps, text)
	})
	if err != nil {
		t.Fatalf("TurnStream: %v", err)
	}

	// Byte-equivalence: same converter fed the equivalent non-streaming body.
	want, err := parseAnthropicResponse(&http.Response{Body: io.NopCloser(strings.NewReader(richAnthropicNonStreaming))})
	if err != nil {
		t.Fatalf("parseAnthropicResponse: %v", err)
	}
	if !reflect.DeepEqual(result, want) {
		t.Fatalf("streamed result != non-streaming result\n got: %#v\nwant: %#v", result, want)
	}

	// Snapshots: monotonically growing prefixes, last equals final content.
	if len(snaps) < 2 {
		t.Fatalf("expected >=2 text snapshots, got %d: %v", len(snaps), snaps)
	}
	for i := 1; i < len(snaps); i++ {
		if !strings.HasPrefix(snaps[i], snaps[i-1]) {
			t.Errorf("snapshot %d %q is not a prefix-extension of %q", i, snaps[i], snaps[i-1])
		}
	}
	final := result.Choices[0].Message.Content
	if snaps[len(snaps)-1] != final {
		t.Errorf("last snapshot %q != final content %q", snaps[len(snaps)-1], final)
	}
	if final != "Hello, world!" {
		t.Errorf("final content = %q, want %q", final, "Hello, world!")
	}

	// Usage merged: input+cache from message_start, output from message_delta.
	u := result.Usage
	if u.PromptTokens != 10 || u.CompletionTokens != 25 || u.TotalTokens != 35 {
		t.Errorf("usage tokens = %+v, want prompt=10 completion=25 total=35", u)
	}
	if u.CacheCreationInputTokens != 3 || u.CacheReadInputTokens != 2 {
		t.Errorf("usage cache = creation:%d read:%d, want 3/2", u.CacheCreationInputTokens, u.CacheReadInputTokens)
	}

	// Tool call round-trips with the assembled input JSON.
	if len(result.Choices[0].Message.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(result.Choices[0].Message.ToolCalls))
	}
	tc := result.Choices[0].Message.ToolCalls[0]
	if tc.ID != "toolu_1" || tc.Function.Name != "get_weather" {
		t.Errorf("unexpected tool call: %+v", tc)
	}
	if !strings.Contains(tc.Function.Arguments, "San Francisco") || !strings.Contains(tc.Function.Arguments, "celsius") {
		t.Errorf("tool args missing accumulated input: %q", tc.Function.Arguments)
	}
	if result.StopReason != "tool_use" {
		t.Errorf("stop_reason = %q, want tool_use", result.StopReason)
	}
}

// TestTurnStreamFallback verifies the turnStreamFallback helper (used by
// TurnStream for backends without a native SSE path, i.e. Bedrock): it runs a
// blocking turn and delivers the whole text as exactly one snapshot delta, with
// a result identical to Turn. It is exercised against a mock /chat/completions
// server because Bedrock itself needs AWS credentials and is not mockable
// offline; the routing line (BackendBedrock -> turnStreamFallback) is covered by
// inspection.
func TestTurnStreamFallback(t *testing.T) {
	const respJSON = `{"model":"gpt-x","choices":[{"index":0,"message":{"role":"assistant","content":"the whole answer"}}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("unexpected path %q (expected fallback to blocking /chat/completions)", r.URL.Path)
		}
		io.WriteString(w, respJSON)
	}))
	defer srv.Close()

	c := NewClient(srv.URL) // default OpenAI-compatible backend
	c.SetMaxRetries(0)

	opts := RequestOptions{Model: "gpt-x", Messages: []Message{{Role: "user", Content: "hi"}}}

	var snaps []string
	streamed, err := c.turnStreamFallback(opts, func(text string) { snaps = append(snaps, text) })
	if err != nil {
		t.Fatalf("turnStreamFallback: %v", err)
	}
	if len(snaps) != 1 {
		t.Fatalf("expected exactly 1 fallback delta, got %d: %v", len(snaps), snaps)
	}
	if snaps[0] != "the whole answer" {
		t.Errorf("fallback delta = %q, want full text", snaps[0])
	}

	blocking, err := c.Turn(opts)
	if err != nil {
		t.Fatalf("Turn: %v", err)
	}
	if !reflect.DeepEqual(streamed, blocking) {
		t.Errorf("fallback streamed result != Turn result\n got: %#v\nwant: %#v", streamed, blocking)
	}
}

// TestTurnStreamAnthropic_ErrorEvent verifies that an error SSE event mid-stream
// surfaces as an error from TurnStream.
func TestTurnStreamAnthropic_ErrorEvent(t *testing.T) {
	const body = `event: message_start
data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-x","content":[],"usage":{"input_tokens":1,"output_tokens":1}}}

event: error
data: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}

`
	srv := sseServer(t, body)
	defer srv.Close()
	c := anthropicTestClient(t, srv.URL)

	_, err := c.TurnStream(RequestOptions{
		Model:    "claude-sonnet-4-x",
		Messages: []Message{{Role: "user", Content: "hi"}},
	}, nil)
	if err == nil {
		t.Fatal("expected an error from a mid-stream error event")
	}
	if !strings.Contains(err.Error(), "overloaded_error") || !strings.Contains(err.Error(), "Overloaded") {
		t.Errorf("error should carry type and message; got %v", err)
	}
}
