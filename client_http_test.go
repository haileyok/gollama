package gollama

import (
	"net/http"
	"testing"
	"time"
)

func TestSetHTTPClient(t *testing.T) {
	client := NewClient("https://example.invalid")
	custom := &http.Client{Timeout: 7 * time.Second}
	client.SetHTTPClient(custom)
	if client.httpClient != custom {
		t.Fatal("SetHTTPClient did not install the provided client")
	}
	client.SetHTTPClient(nil)
	if client.httpClient != custom {
		t.Fatal("SetHTTPClient(nil) replaced the existing client")
	}
}
