import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiment_io import CachedJev


class FakeHttpResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, *_):
        return json.dumps({
            "model": "jev-latest",
            "answers": {"answer": {"type": "noul", "noul": 0.73}},
            "usage": {"input_tokens": 100, "output_tokens": 4},
        }).encode("utf-8")


class JevCacheTests(unittest.TestCase):
    def test_response_and_usage_are_replayed_without_an_api_key(self):
        with tempfile.TemporaryDirectory() as directory:
            client = CachedJev(Path(directory), input_price=0.042)
            with patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-only"}):
                with patch("experiment_io.urlopen", return_value=FakeHttpResponse()) as call:
                    probability, first = client.ask(
                        "sample/direct.json", {"claim": "example"},
                        "Is the claim true?"
                    )
                    self.assertEqual(call.call_count, 1)
            with patch.dict(os.environ, {"TYPESAFE_API_KEY": ""}):
                with patch("experiment_io.urlopen") as call:
                    replayed, second = client.ask(
                        "sample/direct.json", {"claim": "example"},
                        "Is the claim true?"
                    )
                    call.assert_not_called()
            self.assertEqual(probability, 0.73)
            self.assertEqual(replayed, probability)
            self.assertEqual(first["usage"]["input_tokens"], 100)
            self.assertEqual(second["cost_usd"], 100 * 0.042 / 1000000)
            self.assertTrue((Path(directory) / "sample/direct.json").exists())


if __name__ == "__main__":
    unittest.main()
