"""Exercise startup polling against real local HTTP connections, without Docker/GPU."""

from http.client import RemoteDisconnected
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import unittest
from unittest.mock import patch

from orchestrator.runner import DockerBackend


class ReadinessTests(unittest.TestCase):
    def serve(self, responses):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                action = responses[min(len(requests), len(responses) - 1)]
                requests.append(self.path)
                if action == "disconnect":
                    self.close_connection = True
                    return  # Accept TCP, then close without any HTTP response.
                body = json.dumps({"status": "ok" if action == "ready" else "loading"}).encode()
                self.send_response(200 if action == "ready" else 503)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                self.do_GET()

            def log_message(self, format, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=.01), daemon=True)
        thread.start()

        def close():
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.addCleanup(close)
        return f"http://127.0.0.1:{server.server_port}", requests

    def test_startup_disconnect_then_loading_waits_for_healthy_response(self):
        url, requests = self.serve(["disconnect", "loading", "ready"])
        backend = DockerBackend()
        with patch.object(backend, "docker_json", return_value=[{"State": {"Running": True}}]):
            backend.wait_ready("owned-inference", url, timeout=3)
        self.assertEqual(requests, ["/health", "/health", "/health"])

    def test_persistent_disconnects_end_at_startup_deadline(self):
        url, requests = self.serve(["disconnect"])
        backend = DockerBackend()
        with patch.object(backend, "docker_json", return_value=[{"State": {"Running": True}}]):
            with self.assertRaises(TimeoutError):
                backend.wait_ready("owned-inference", url, timeout=.15)
        self.assertEqual(requests, ["/health"])

    def test_container_exit_after_disconnect_still_fails(self):
        url, requests = self.serve(["disconnect", "ready"])
        backend = DockerBackend()
        states = [[{"State": {"Running": True}}], [{"State": {"Running": False}}]]
        with patch.object(backend, "docker_json", side_effect=states):
            with self.assertRaises(RuntimeError):
                backend.wait_ready("owned-inference", url, timeout=3)
        self.assertEqual(requests, ["/health"])

    def test_healthy_response_does_not_hide_container_exit(self):
        url, requests = self.serve(["ready"])
        backend = DockerBackend()
        states = [[{"State": {"Running": True}}], [{"State": {"Running": False}}]]
        with patch.object(backend, "docker_json", side_effect=states):
            with self.assertRaises(RuntimeError):
                backend.wait_ready("owned-inference", url, timeout=3)
        self.assertEqual(requests, ["/health"])

    def test_generation_probe_disconnect_is_not_retried(self):
        url, requests = self.serve(["disconnect"])
        with self.assertRaises(RemoteDisconnected):
            DockerBackend().probe(url, timeout=1)
        self.assertEqual(requests, ["/v1/chat/completions"])


if __name__ == "__main__":
    unittest.main()
