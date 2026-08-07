package gollama

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestContextAwareTurnsCancelHTTP(t *testing.T) {
	for _, anthropic := range []bool{false, true} {
		for _, stream := range []bool{false, true} {
			name := "openai_blocking"
			if anthropic {
				name = "anthropic_blocking"
			}
			if stream {
				name += "_stream"
			}
			t.Run(name, func(t *testing.T) {
				started := make(chan struct{}, 1)
				serverCanceled := make(chan struct{}, 1)
				srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.Header().Set("Content-Type", "text/event-stream")
					w.WriteHeader(http.StatusOK)
					w.(http.Flusher).Flush()
					started <- struct{}{}
					<-r.Context().Done()
					serverCanceled <- struct{}{}
				}))
				defer srv.Close()

				client := NewClient(srv.URL)
				client.SetMaxRetries(0)
				client.SetAnthropicMode(anthropic)
				ctx, cancel := context.WithCancel(context.Background())
				done := make(chan error, 1)
				go func() {
					var err error
					if stream {
						_, err = client.TurnStreamCtx(ctx, RequestOptions{Model: "test"}, nil)
					} else {
						_, err = client.TurnCtx(ctx, RequestOptions{Model: "test"})
					}
					done <- err
				}()

				select {
				case <-started:
				case <-time.After(time.Second):
					t.Fatal("request did not reach server")
				}
				cancel()
				select {
				case err := <-done:
					if !errors.Is(err, context.Canceled) {
						t.Fatalf("turn error = %v, want context.Canceled", err)
					}
				case <-time.After(time.Second):
					t.Fatal("turn did not return promptly after cancellation")
				}
				select {
				case <-serverCanceled:
				case <-time.After(time.Second):
					t.Fatal("server did not observe request context cancellation")
				}
			})
		}
	}
}

func TestRetryBackoffIsContextAware(t *testing.T) {
	attempted := make(chan struct{}, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempted <- struct{}{}
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer srv.Close()

	client := NewClient(srv.URL)
	client.SetMaxRetries(1)
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		_, err := client.TurnCtx(ctx, RequestOptions{Model: "test"})
		done <- err
	}()

	select {
	case <-attempted:
	case <-time.After(time.Second):
		t.Fatal("request did not reach server")
	}
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("turn error = %v, want context.Canceled", err)
		}
	case <-time.After(time.Second):
		t.Fatal("retry sleep did not stop promptly after cancellation")
	}
}
