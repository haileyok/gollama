package gollama

import (
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
)

// sseChunks joins a set of chat.completion.chunk JSON payloads into an SSE body
// (each on its own "data:" line, terminated by a blank line), followed by the
// terminal [DONE] sentinel — the exact wire shape an OpenAI-compatible server
// emits for stream:true.
func sseChunks(chunks ...string) string {
	var b strings.Builder
	for _, c := range chunks {
		b.WriteString("data: ")
		b.WriteString(c)
		b.WriteString("\n\n")
	}
	b.WriteString("data: [DONE]\n\n")
	return b.String()
}

// richOpenAIStream exercises multi-chunk text, reasoning deltas, a tool call
// whose arguments are split across three fragments, a second tool call at index
// 1, a finish_reason, and a final usage-only chunk with an empty choices array.
var richOpenAIStream = sseChunks(
	`{"id":"c1","object":"chat.completion.chunk","model":"gpt-x","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}`,
	`{"id":"c1","object":"chat.completion.chunk","model":"gpt-x","choices":[{"index":0,"delta":{"content":", world!"},"finish_reason":null}]}`,
	`{"id":"c1","object":"chat.completion.chunk","model":"gpt-x","choices":[{"index":0,"delta":{"reasoning":"Let me "},"finish_reason":null}]}`,
	`{"id":"c1","object":"chat.completion.chunk","model":"gpt-x","choices":[{"index":0,"delta":{"reasoning":"think."},"finish_reason":null}]}`,
	`{"id":"c1","object":"chat.completion.chunk","model":"gpt-x","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"location\":"}}]},"finish_reason":null}]}`,
	`{"id":"c1","object":"chat.completion.chunk","model":"gpt-x","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"San Francisco\","}}]},"finish_reason":null}]}`,
	`{"id":"c1","object":"chat.completion.chunk","model":"gpt-x","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"units\":\"celsius\"}"}}]},"finish_reason":null}]}`,
	`{"id":"c1","object":"chat.completion.chunk","model":"gpt-x","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_2","type":"function","function":{"name":"get_time","arguments":"{\"tz\":\"UTC\"}"}}]},"finish_reason":null}]}`,
	`{"id":"c1","object":"chat.completion.chunk","model":"gpt-x","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
	`{"id":"c1","object":"chat.completion.chunk","model":"gpt-x","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":25,"total_tokens":35}}`,
)

// equivalent non-streaming body that must decode to the identical result.
const richOpenAINonStreaming = `{"model":"gpt-x","choices":[{"index":0,"message":{"role":"assistant","content":"Hello, world!","reasoning":"Let me think.","tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"San Francisco\",\"units\":\"celsius\"}"}},{"id":"call_2","type":"function","function":{"name":"get_time","arguments":"{\"tz\":\"UTC\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":25,"total_tokens":35}}`

// sseCaptureServer serves an SSE body and records the last request body it
// received (decoded into captured).
func sseCaptureServer(t *testing.T, body string, captured *map[string]any) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if captured != nil {
			b, _ := io.ReadAll(r.Body)
			var m map[string]any
			_ = json.Unmarshal(b, &m)
			*captured = m
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		io.WriteString(w, body)
	}))
}

// TestTurnStreamOpenAI_RichStream verifies native OpenAI-compatible streaming:
// the assembled final message is byte-equivalent to the non-streaming decode,
// onDelta snapshots are monotonically growing prefixes ending at the final
// content, and the request carried stream:true + stream_options.include_usage.
func TestTurnStreamOpenAI_RichStream(t *testing.T) {
	var reqBody map[string]any
	srv := sseCaptureServer(t, richOpenAIStream, &reqBody)
	defer srv.Close()

	c := NewClient(srv.URL)
	c.SetMaxRetries(0)
	if c.Backend() != BackendOpenAI {
		t.Fatalf("expected OpenAI backend, got %v", c.Backend())
	}

	opts := RequestOptions{Model: "gpt-x", Messages: []Message{{Role: "user", Content: "hi"}}}

	var snaps []string
	result, err := c.TurnStream(opts, func(text string) { snaps = append(snaps, text) })
	if err != nil {
		t.Fatalf("TurnStream: %v", err)
	}

	// Byte-equivalence with the non-streaming decode of the same content.
	nsSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, richOpenAINonStreaming)
	}))
	defer nsSrv.Close()
	nsClient := NewClient(nsSrv.URL)
	nsClient.SetMaxRetries(0)
	want, err := nsClient.Turn(opts)
	if err != nil {
		t.Fatalf("non-streaming Turn: %v", err)
	}
	if !reflect.DeepEqual(result, want) {
		t.Fatalf("streamed result != non-streaming result\n got: %#v\nwant: %#v", result, want)
	}

	// Snapshots: monotonically growing prefixes; last equals final content.
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

	// Reasoning folded into Thinking, Reasoning cleared.
	msg := result.Choices[0].Message
	if msg.Thinking != "Let me think." || msg.Reasoning != "" {
		t.Errorf("reasoning normalization: thinking=%q reasoning=%q", msg.Thinking, msg.Reasoning)
	}

	// Tool calls assembled in index order with concatenated arguments.
	if len(msg.ToolCalls) != 2 {
		t.Fatalf("expected 2 tool calls, got %d", len(msg.ToolCalls))
	}
	if msg.ToolCalls[0].ID != "call_1" || msg.ToolCalls[0].Function.Name != "get_weather" ||
		msg.ToolCalls[0].Function.Arguments != `{"location":"San Francisco","units":"celsius"}` {
		t.Errorf("tool call 0 mis-assembled: %#v", msg.ToolCalls[0])
	}
	if msg.ToolCalls[1].ID != "call_2" || msg.ToolCalls[1].Function.Arguments != `{"tz":"UTC"}` {
		t.Errorf("tool call 1 mis-assembled: %#v", msg.ToolCalls[1])
	}

	// Usage carried from the final usage-only chunk.
	if result.Usage.PromptTokens != 10 || result.Usage.CompletionTokens != 25 || result.Usage.TotalTokens != 35 {
		t.Errorf("usage = %+v, want 10/25/35", result.Usage)
	}
	if result.Choices[0].FinishReason != "tool_calls" {
		t.Errorf("finish_reason = %q, want tool_calls", result.Choices[0].FinishReason)
	}

	// Request body: stream:true and stream_options.include_usage:true.
	sent := reqBody
	if sent == nil {
		t.Fatalf("no request body captured")
	}
	if sent["stream"] != true {
		t.Errorf("request stream = %v, want true", sent["stream"])
	}
	so, ok := sent["stream_options"].(map[string]any)
	if !ok || so["include_usage"] != true {
		t.Errorf("request stream_options = %v, want {include_usage:true}", sent["stream_options"])
	}
}

// TestTurnStreamOpenAI_ToolOnly verifies a tool-only turn: onDelta is never
// called (no content deltas), and the tool call is assembled correctly.
func TestTurnStreamOpenAI_ToolOnly(t *testing.T) {
	body := sseChunks(
		`{"model":"gpt-x","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_9","type":"function","function":{"name":"lookup","arguments":"{\"q\":"}}]},"finish_reason":null}]}`,
		`{"model":"gpt-x","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"weather\"}"}}]},"finish_reason":null}]}`,
		`{"model":"gpt-x","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
	)
	srv := sseCaptureServer(t, body, nil)
	defer srv.Close()

	c := NewClient(srv.URL)
	c.SetMaxRetries(0)

	called := 0
	result, err := c.TurnStream(RequestOptions{Model: "gpt-x", Messages: []Message{{Role: "user", Content: "hi"}}},
		func(text string) { called++ })
	if err != nil {
		t.Fatalf("TurnStream: %v", err)
	}
	if called != 0 {
		t.Errorf("onDelta called %d times for a tool-only turn, want 0", called)
	}
	msg := result.Choices[0].Message
	if msg.Content != "" {
		t.Errorf("content = %q, want empty", msg.Content)
	}
	if len(msg.ToolCalls) != 1 || msg.ToolCalls[0].ID != "call_9" ||
		msg.ToolCalls[0].Function.Arguments != `{"q":"weather"}` {
		t.Errorf("tool call mis-assembled: %#v", msg.ToolCalls)
	}
}

// TestTurnStreamOpenAI_ErrorEvent verifies that a mid-stream error object
// surfaces as an error from TurnStream.
func TestTurnStreamOpenAI_ErrorEvent(t *testing.T) {
	body := sseChunks(
		`{"model":"gpt-x","choices":[{"index":0,"delta":{"role":"assistant","content":"partial"},"finish_reason":null}]}`,
		`{"error":{"message":"upstream exploded","type":"server_error","code":"500"}}`,
	)
	srv := sseCaptureServer(t, body, nil)
	defer srv.Close()

	c := NewClient(srv.URL)
	c.SetMaxRetries(0)

	_, err := c.TurnStream(RequestOptions{Model: "gpt-x", Messages: []Message{{Role: "user", Content: "hi"}}}, nil)
	if err == nil {
		t.Fatal("expected an error from a mid-stream error object")
	}
	if !strings.Contains(err.Error(), "server_error") || !strings.Contains(err.Error(), "upstream exploded") {
		t.Errorf("error should carry type and message; got %v", err)
	}
}

// TestTurnStreamOllama_RequestShape verifies the Ollama branch of the shared
// request builder: think:true is sent (from opts.Think) and reasoning_effort is
// NOT (Ollama has no effort levels), while stream_options.include_usage is set
// for streaming. Ollama is detected by the ":11434" base URL.
func TestTurnStreamOllama_RequestShape(t *testing.T) {
	c := NewClient("http://127.0.0.1:11434/v1")
	c.SetMaxRetries(0)
	if c.Backend() != BackendOllama {
		t.Fatalf("expected BackendOllama, got %v", c.Backend())
	}

	opts := RequestOptions{
		Model:    "gemma",
		Messages: []Message{{Role: "user", Content: "hi"}},
		Think:    true,
		Effort:   "high", // inexpressible on Ollama; must be dropped
		Stream:   true,
	}
	body, err := c.buildOpenAIRequest(opts)
	if err != nil {
		t.Fatalf("buildOpenAIRequest: %v", err)
	}
	raw, _ := json.Marshal(body)
	var sent map[string]any
	if err := json.Unmarshal(raw, &sent); err != nil {
		t.Fatalf("decode request: %v", err)
	}
	if sent["think"] != true {
		t.Errorf("think = %v, want true", sent["think"])
	}
	if _, present := sent["reasoning_effort"]; present {
		t.Errorf("reasoning_effort must not be sent to Ollama; got %v", sent["reasoning_effort"])
	}
	if sent["stream"] != true {
		t.Errorf("stream = %v, want true", sent["stream"])
	}
	so, ok := sent["stream_options"].(map[string]any)
	if !ok || so["include_usage"] != true {
		t.Errorf("stream_options = %v, want {include_usage:true}", sent["stream_options"])
	}
}

// TestTurnStreamOllama_NativeStream drives a full streamed turn against an
// Ollama-detected client (base URL literally containing :11434), asserting
// reasoning is folded into Thinking. It binds a listener on 127.0.0.1:11434 so
// Backend() sniffs Ollama; if that port is unavailable (e.g. a real Ollama is
// running) the streaming portion is skipped — the request-shape branch is
// already covered by TestTurnStreamOllama_RequestShape.
func TestTurnStreamOllama_NativeStream(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:11434")
	if err != nil {
		t.Skipf("cannot bind 127.0.0.1:11434 (%v); request shape covered elsewhere", err)
	}
	body := sseChunks(
		`{"model":"gemma","choices":[{"index":0,"delta":{"role":"assistant","reasoning":"Hmm. "},"finish_reason":null}]}`,
		`{"model":"gemma","choices":[{"index":0,"delta":{"content":"Hi there"},"finish_reason":null}]}`,
		`{"model":"gemma","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		`{"model":"gemma","choices":[],"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6}}`,
	)
	srv := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, body)
	}))
	srv.Listener.Close()
	srv.Listener = ln
	srv.Start()
	defer srv.Close()

	c := NewClient("http://127.0.0.1:11434/v1")
	c.SetMaxRetries(0)
	if c.Backend() != BackendOllama {
		t.Fatalf("expected BackendOllama, got %v", c.Backend())
	}

	var snaps []string
	result, err := c.TurnStream(RequestOptions{
		Model:    "gemma",
		Messages: []Message{{Role: "user", Content: "hi"}},
		Think:    true,
	}, func(text string) { snaps = append(snaps, text) })
	if err != nil {
		t.Fatalf("TurnStream: %v", err)
	}
	msg := result.Choices[0].Message
	if msg.Content != "Hi there" {
		t.Errorf("content = %q, want %q", msg.Content, "Hi there")
	}
	if msg.Thinking != "Hmm. " || msg.Reasoning != "" {
		t.Errorf("reasoning fold: thinking=%q reasoning=%q", msg.Thinking, msg.Reasoning)
	}
	// Reasoning is not streamed to onDelta: only the content snapshot is.
	if len(snaps) != 1 || snaps[0] != "Hi there" {
		t.Errorf("snapshots = %v, want [\"Hi there\"]", snaps)
	}
}

// TestTurnStreamOpenAI_NonSSEFallback verifies the compatibility fallback: an
// OpenAI-compatible endpoint that ignores stream:true and answers with a plain
// JSON completion must still decode correctly through TurnStream, with the full
// text delivered as one onDelta snapshot, instead of returning the empty
// response the SSE reader would produce.
func TestTurnStreamOpenAI_NonSSEFallback(t *testing.T) {
	var streamed bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		var m map[string]any
		_ = json.Unmarshal(b, &m)
		if v, ok := m["stream"].(bool); ok {
			streamed = v
		}
		// Deliberately application/json, not text/event-stream: this is the
		// wire shape of an endpoint that ignored stream:true.
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, richOpenAINonStreaming)
	}))
	defer srv.Close()

	c := NewClient(srv.URL)
	c.SetMaxRetries(0)

	var snaps []string
	result, err := c.TurnStream(RequestOptions{
		Model:    "gpt-x",
		Messages: []Message{{Role: "user", Content: "hi"}},
	}, func(text string) { snaps = append(snaps, text) })
	if err != nil {
		t.Fatalf("TurnStream: %v", err)
	}
	if !streamed {
		t.Error("request did not carry stream:true")
	}
	msg := result.Choices[0].Message
	if msg.Content != "Hello, world!" {
		t.Errorf("content = %q, want %q", msg.Content, "Hello, world!")
	}
	if msg.Thinking != "Let me think." {
		t.Errorf("reasoning fold: thinking=%q", msg.Thinking)
	}
	if len(msg.ToolCalls) != 2 || msg.ToolCalls[0].Function.Name != "get_weather" {
		t.Errorf("tool calls not decoded: %+v", msg.ToolCalls)
	}
	if len(snaps) != 1 || snaps[0] != "Hello, world!" {
		t.Errorf("snapshots = %v, want exactly one full-content snapshot", snaps)
	}
}
