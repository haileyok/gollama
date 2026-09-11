package gollama

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"
)

// rewriteTransport redirects every request to a fixed target host/scheme while
// leaving the path intact. It lets a client carry a base URL that looks like an
// Ollama endpoint (contains ":11434", so Backend() detects Ollama) while the
// traffic actually reaches an httptest server on a random port.
type rewriteTransport struct {
	target *url.URL
}

func (t rewriteTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	req.URL.Scheme = t.target.Scheme
	req.URL.Host = t.target.Host
	return http.DefaultTransport.RoundTrip(req)
}

// ollamaBackedClient returns a client whose base URL contains ":11434" (so it is
// detected as the Ollama backend) but whose requests are transparently routed to
// the given test-server URL.
func ollamaBackedClient(t *testing.T, serverURL string) *Client {
	t.Helper()
	target, err := url.Parse(serverURL)
	if err != nil {
		t.Fatalf("parse server URL: %v", err)
	}
	c := NewClient("http://127.0.0.1:11434/v1")
	c.SetMaxRetries(0)
	c.httpClient.Transport = rewriteTransport{target: target}
	return c
}

// captureServer records the last request body it received and returns a fixed
// OpenAI-compatible response body.
func captureServer(t *testing.T, respBody string, captured *map[string]any) *httptest.Server {
	t.Helper()
	if respBody == "" {
		respBody = `{"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`
	}
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var m map[string]any
		_ = json.Unmarshal(body, &m)
		*captured = m
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		io.WriteString(w, respBody)
	}))
}

func openaiTestClient(t *testing.T, url string) *Client {
	t.Helper()
	c := NewClient(url)
	c.SetBearerToken("test-key")
	c.SetMaxRetries(0)
	return c
}

// TestOpenAIReasoningEffort verifies that RequestOptions.Effort is translated
// into the OpenAI reasoning_effort request field, that "max" clamps to "xhigh"
// for OpenAI but passes through when EffortPassthrough is set, that an empty
// Effort omits the key, and that the Ollama-only "think" key is never present
// for the OpenAI backend.
func TestOpenAIReasoningEffort(t *testing.T) {
	cases := []struct {
		name        string
		effort      string
		passthrough bool
		wantEffort  string // "" means the key must be absent
	}{
		{"low passes through", "low", false, "low"},
		{"medium passes through", "medium", false, "medium"},
		{"high passes through", "high", false, "high"},
		{"xhigh passes through", "xhigh", false, "xhigh"},
		{"max clamps to xhigh", "max", false, "xhigh"},
		{"max passes through when allowed", "max", true, "max"},
		{"passthrough leaves other levels alone", "high", true, "high"},
		{"empty omits key", "", false, ""},
		{"empty omits key with passthrough", "", true, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var got map[string]any
			srv := captureServer(t, "", &got)
			defer srv.Close()
			c := openaiTestClient(t, srv.URL)
			if c.Backend() != BackendOpenAI {
				t.Fatalf("expected BackendOpenAI, got %v", c.Backend())
			}

			if _, err := c.Turn(RequestOptions{
				Model:             "gpt-5.1",
				Messages:          []Message{{Role: "user", Content: "hi"}},
				Effort:            tc.effort,
				EffortPassthrough: tc.passthrough,
				Thinking:          "adaptive", // must NOT produce think on OpenAI
			}); err != nil {
				t.Fatalf("Turn: %v", err)
			}

			re, present := got["reasoning_effort"]
			if tc.wantEffort == "" {
				if present {
					t.Errorf("reasoning_effort should be absent, got %v", re)
				}
			} else {
				if !present {
					t.Fatalf("reasoning_effort missing; body=%v", got)
				}
				if re != tc.wantEffort {
					t.Errorf("reasoning_effort = %v, want %q", re, tc.wantEffort)
				}
			}
			if _, hasThink := got["think"]; hasThink {
				t.Errorf("think key must never be present on OpenAI backend; body=%v", got)
			}
		})
	}
}

// TestOllamaThinkTranslation verifies that on the Ollama backend a non-empty
// Thinking (or Think) sets "think":true, that Effort never produces a
// reasoning_effort key (levels are inexpressible on Ollama), and that a response
// carrying message.reasoning is normalized into Message.Thinking.
func TestOllamaThinkTranslation(t *testing.T) {
	// The Ollama backend is detected from the :11434 port in the base URL, so we
	// point the client at an httptest server on that address.
	resp := `{"choices":[{"index":0,"message":{"role":"assistant","content":"answer","reasoning":"let me think"},"finish_reason":"stop"}]}`

	cases := []struct {
		name      string
		opts      RequestOptions
		wantThink bool
	}{
		{
			name:      "thinking adaptive turns think on",
			opts:      RequestOptions{Thinking: "adaptive", Effort: "high"},
			wantThink: true,
		},
		{
			name:      "think bool turns think on",
			opts:      RequestOptions{Think: true},
			wantThink: true,
		},
		{
			name:      "effort alone does not turn think on",
			opts:      RequestOptions{Effort: "high"},
			wantThink: false,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var got map[string]any
			srv := captureServer(t, resp, &got)
			defer srv.Close()

			c := ollamaBackedClient(t, srv.URL)
			if c.Backend() != BackendOllama {
				t.Fatalf("expected BackendOllama, got %v", c.Backend())
			}

			opts := tc.opts
			opts.Model = "gemma4:26b"
			opts.Messages = []Message{{Role: "user", Content: "hi"}}
			out, err := c.Turn(opts)
			if err != nil {
				t.Fatalf("Turn: %v", err)
			}

			think, hasThink := got["think"]
			if tc.wantThink {
				if !hasThink || think != true {
					t.Errorf("think = %v (present=%v), want true; body=%v", think, hasThink, got)
				}
			} else if hasThink && think == true {
				t.Errorf("think should not be true; body=%v", got)
			}
			if _, hasEffort := got["reasoning_effort"]; hasEffort {
				t.Errorf("reasoning_effort must never be sent to Ollama; body=%v", got)
			}

			// Response normalization: message.reasoning -> Message.Thinking.
			m := out.Choices[0].Message
			if m.Thinking != "let me think" {
				t.Errorf("Thinking = %q, want %q", m.Thinking, "let me think")
			}
			if m.Reasoning != "" {
				t.Errorf("Reasoning should be cleared after normalization, got %q", m.Reasoning)
			}
		})
	}
}
