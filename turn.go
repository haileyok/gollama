package gollama

import "strings"

// Backend identifies which provider/transport a Client is configured to talk to.
type Backend int

const (
	BackendOpenAI Backend = iota // OpenAI-compatible /chat/completions (also serves Ollama's /v1 endpoint)
	BackendAnthropic
	BackendBedrock
	BackendOllama
)

func (b Backend) String() string {
	switch b {
	case BackendAnthropic:
		return "anthropic"
	case BackendBedrock:
		return "bedrock"
	case BackendOllama:
		return "ollama"
	default:
		return "openai"
	}
}

// Backend reports the provider/transport this client will use. Bedrock and
// Anthropic are detected from explicit configuration (AWS auth / anthropic mode
// or URL). Ollama is heuristically detected from its default port; otherwise the
// client is treated as an OpenAI-compatible endpoint.
//
// Note: Turn routes Ollama through the OpenAI-compatible path, so an Ollama
// endpoint should be configured with a base URL that includes "/v1". The logical
// backend is best tracked by the caller (e.g. ycc's model registry); this accessor
// is a convenience for introspection.
func (c *Client) Backend() Backend {
	switch {
	case c.IsBedrockAPI():
		return BackendBedrock
	case c.IsAnthropicAPI():
		return BackendAnthropic
	case strings.Contains(c.baseURL, ":11434"):
		return BackendOllama
	default:
		return BackendOpenAI
	}
}

// Turn is the canonical, backend-agnostic entry point for a single
// (non-streaming) model turn. It routes to the correct provider based on the
// client's configuration and returns a normalized ResponseMessageGenerate whose
// Choices[0].Message carries the assistant text and any tool calls in a single
// shape regardless of backend.
//
// It delegates to ChatCompletion, which already dispatches Anthropic, Bedrock,
// and OpenAI-compatible endpoints; Turn exists so callers have one stable method
// to depend on and never have to branch per provider. Streaming is always
// disabled (the agent loop consumes whole turns).
func (c *Client) Turn(opts RequestOptions) (*ResponseMessageGenerate, error) {
	opts.Stream = false
	return c.ChatCompletion(opts)
}

// TurnStream is the streaming counterpart to Turn: it runs a single model turn
// while delivering the assistant's text incrementally through onDelta, then
// returns the same normalized ResponseMessageGenerate that Turn would return for
// the same options. Callers never have to branch per provider — a turn that
// cannot stream natively still returns a correct final message.
//
// CONTRACT: onDelta receives SNAPSHOTS — the full accumulated assistant text so
// far — NOT increments. It is invoked serially (never concurrently with itself)
// for a single TurnStream call, and may be called zero times (e.g. a turn that
// produces only tool calls). Snapshot semantics make lossy/throttled delivery
// safe: a consumer can drop intermediate snapshots and simply replace its live
// tail with the latest one. onDelta may be nil.
//
// Backend support: Anthropic streams natively via the Messages SSE API. Every
// other backend (OpenAI-compatible, Ollama, Bedrock) falls back gracefully — it
// runs a blocking Turn and, if that yields non-empty assistant text, delivers
// the whole text as a single snapshot delta. Native streaming for those
// backends is a follow-up; the fallback keeps callers uniform in the meantime.
func (c *Client) TurnStream(opts RequestOptions, onDelta func(text string)) (*ResponseMessageGenerate, error) {
	if c.Backend() == BackendAnthropic {
		return c.chatCompletionAnthropicStream(opts, onDelta)
	}

	// Fallback: no native streaming for this backend. Run a blocking turn and
	// deliver the whole assistant text as a single snapshot so callers need not
	// branch.
	resp, err := c.Turn(opts)
	if err != nil {
		return nil, err
	}
	if onDelta != nil && len(resp.Choices) > 0 {
		if text := resp.Choices[0].Message.Content; text != "" {
			onDelta(text)
		}
	}
	return resp, nil
}
