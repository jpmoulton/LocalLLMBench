"""Backend contract tests. All server state and responses are in-memory fakes."""

import io
import unittest

from llmbench.backends.http import SSEStream, TransportError, UrllibHTTPTransport


class SSETests(unittest.TestCase):
    def test_multiline_events_comments_crlf_and_usage(self):
        response = io.BytesIO(b': keepalive\r\nevent: message\r\ndata: {"usage":\r\ndata: {"completion_tokens":5}}\r\n\r\ndata: [DONE]\n\n')
        events = list(SSEStream(response))
        self.assertEqual(events, [("message", {"usage": {"completion_tokens": 5}}), ("done", "[DONE]")])
        self.assertTrue(response.closed)

    def test_malformed_stream_fails_without_repair(self):
        with self.assertRaises(TransportError):
            list(SSEStream(io.BytesIO(b'data: {"bad":\n\n')))

    def test_oversized_stream_event_rejected(self):
        with self.assertRaises(TransportError):
            list(SSEStream(io.BytesIO(b'data: {"content":"long text"}\n\n'), max_event_bytes=10))

    def test_eof_does_not_invent_done_or_token_counts(self):
        events = list(SSEStream(io.BytesIO(b'data: {"choices": []}\n\n')))
        self.assertEqual(events, [("message", {"choices": []})])

    def test_unsafe_transport_origins_rejected_without_connection(self):
        for origin in ("http://localhost:1234/v1", "http://secret@localhost:1234", "https://external.example"):
            with self.assertRaises(ValueError):
                UrllibHTTPTransport(origin)


class _FakeResponse(io.BytesIO):
    headers = {"Content-Type": "application/json"}


class _FakeOpener:
    """Replaces the urllib opener; no socket is ever created."""

    def __init__(self, *, body=b"{}", error_status=None):
        self.body, self.error_status, self.requests = body, error_status, []

    def open(self, request, timeout=None):
        from urllib.error import HTTPError
        self.requests.append((request.get_method(), request.full_url))
        if self.error_status is not None:
            raise HTTPError(request.full_url, self.error_status, "error", {}, io.BytesIO(self.body))
        return _FakeResponse(self.body)


class TransportEvidenceTests(unittest.TestCase):
    def _transport(self, **kwargs):
        transport = UrllibHTTPTransport("http://127.0.0.1:8080")
        transport._opener = _FakeOpener(**kwargs)
        return transport

    def test_transport_error_keeps_status_and_bounded_body_outside_message(self):
        secret = b'{"error":{"type":"exceed_context_size_error","prompt":"SECRET"}}'
        transport = self._transport(body=secret + b"x" * (80 * 1024), error_status=400)
        with self.assertRaises(TransportError) as failure:
            transport.request_json("POST", "/completion", {"prompt": [1]})
        error = failure.exception
        self.assertEqual(str(error), "HTTP 400 from the inference server /completion")
        self.assertNotIn("SECRET", str(error))
        self.assertEqual(error.status, 400)
        self.assertEqual(len(error.body), 64 * 1024)
        self.assertTrue(error.body.startswith(secret))

    def test_transport_error_defaults_and_constructor_bound(self):
        plain = TransportError("Malformed JSON response")
        self.assertIsNone(plain.status)
        self.assertIsNone(plain.body)
        self.assertEqual(str(plain), "Malformed JSON response")
        self.assertEqual(len(TransportError("x", status=500, body=b"y" * 70_000).body), 64 * 1024)

    def test_request_any_returns_lists_and_request_json_still_rejects_them(self):
        transport = self._transport(body=b'[{"id": 0, "is_processing": false}]')
        self.assertEqual(transport.request_any("GET", "/slots"), [{"id": 0, "is_processing": False}])
        with self.assertRaisesRegex(TransportError, "must be an object"):
            transport.request_json("GET", "/slots")
        self.assertEqual(self._transport(body=b'{"status": "ok"}').request_json("GET", "/health"), {"status": "ok"})

    def test_request_any_keeps_size_and_json_bounds(self):
        with self.assertRaisesRegex(TransportError, "Malformed"):
            self._transport(body=b"not json").request_any("GET", "/slots")
        transport = UrllibHTTPTransport("http://127.0.0.1:8080", max_response_bytes=4)
        transport._opener = _FakeOpener(body=b"[1, 2, 3]")
        with self.assertRaisesRegex(TransportError, "byte limit"):
            transport.request_any("GET", "/slots")


if __name__ == "__main__":
    unittest.main()
