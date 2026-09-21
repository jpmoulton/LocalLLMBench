"""Speed evidence conversion checks using preserved, entirely synthetic events."""

import unittest

from llmbench.backends import StreamEvent
from llmbench.backends.measurement_bridge import observation_from_events


def event(at, kind, data):
    return StreamEvent("request-1", at, kind, data)


def observe(events, **kwargs):
    return observation_from_events(events, started_at=100, finished_at=110,
                                    requested_output_tokens=512, expected_input_tokens=1024, **kwargs)


class SpeedBridgeTests(unittest.TestCase):
    def test_sse_usage_cannot_invent_native_decode_time_or_speed(self):
        observation = observe([
            event(101, "message", {"choices": [{"delta": {"content": "many tokens in one chunk"}}]}),
            event(108, "message", {"choices": [{"delta": {"content": "tail"}, "finish_reason": "length"}]}),
            event(109, "message", {"usage": {"completion_tokens": 512, "prompt_tokens": 1024}}),
            event(110, "done", "[DONE]"),
        ])
        self.assertEqual(observation.status, "completed")
        self.assertTrue(observation.accepted_tokens_verified)
        self.assertIsNone(observation.native_generation_seconds)
        self.assertIsNone(observation.metrics()["native_tokens_per_second"])
        self.assertEqual(observation.metrics()["maximum_delivery_gap_seconds"], 7)

    def test_partial_stream_is_interrupted_not_completed(self):
        observation = observe([event(101, "message", {"choices": [{"delta": {"content": "partial"}}]})])
        self.assertEqual(observation.status, "interrupted")
        self.assertIsNone(observation.output_tokens)

    def test_cancellation_keeps_partial_usage_but_never_qualifies_complete(self):
        observation = observe([
            event(101, "message", {"usage": {"completion_tokens": 60}}),
            event(105, "cancelled", {"server_cancellation_verified": False}),
            event(110, "done", "[DONE]"),
        ])
        self.assertEqual(observation.status, "cancelled")
        self.assertEqual(observation.output_tokens, 60)
        self.assertFalse(observation.cancellation_acknowledged)

    def test_mixed_requests_or_impossible_timestamps_rejected(self):
        with self.assertRaises(ValueError):
            observe([event(101, "message", {}), StreamEvent("other-request", 102, "message", {})])
        with self.assertRaises(ValueError):
            observe([event(111, "message", {})])
        with self.assertRaises(ValueError):
            observe([event(106, "message", {}), event(102, "message", {})])

    def test_inconsistent_usage_invalidates_evidence(self):
        observation = observe([event(107, "message", {"usage": {"completion_tokens": 30}}),
                               event(109, "message", {"usage": {"completion_tokens": 512},
                                                      "choices": [{"delta": {}, "finish_reason": "length"}]}),
                               event(110, "done", "[DONE]")])
        self.assertEqual(observation.status, "failed")
        self.assertFalse(observation.accepted_tokens_verified)

    def test_timeout_not_promoted_by_last_done_event(self):
        observation = observe([event(109, "message", {"choices": [{"delta": {"content": "x"}, "finish_reason": "stop"}]}),
                               event(110, "done", "[DONE]")], terminal_status="timed_out")
        self.assertEqual(observation.status, "timed_out")


if __name__ == "__main__":
    unittest.main()
