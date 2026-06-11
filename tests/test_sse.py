import unittest

from app.sse import iter_sse_events


class SSEParserTests(unittest.TestCase):
    def test_parses_log_and_result_events(self):
        events = list(
            iter_sse_events(
                [
                    "event: log",
                    'data: {"message":"working"}',
                    "",
                    "event: result",
                    'data: {"answer":"done"}',
                    "",
                ]
            )
        )
        self.assertEqual(events[0]["event"], "log")
        self.assertEqual(events[1]["event"], "result")
        self.assertIn('"answer":"done"', events[1]["data"])

    def test_accepts_byte_lines(self):
        events = list(
            iter_sse_events(
                [
                    b"event: result",
                    b'data: {"answer":"weather"}',
                    b"",
                ]
            )
        )
        self.assertEqual(events[0]["event"], "result")


if __name__ == "__main__":
    unittest.main()
