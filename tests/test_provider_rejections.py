"""Check the DMX refusal variants observed in Qwen benchmark runs."""

import unittest

from benchmark_jev_qbaf import is_provider_content_rejection


class BadRequestError(Exception):
    status_code = 400


class ProviderRejectionTests(unittest.TestCase):
    def test_old_and_new_dmx_content_refusals(self):
        self.assertTrue(is_provider_content_rejection(BadRequestError(
            "Input text data may contain inappropriate content")))
        self.assertTrue(is_provider_content_rejection(BadRequestError(
            "Input data may contain inappropriate content: data_inspection_failed")))
        self.assertFalse(is_provider_content_rejection(BadRequestError(
            "Unrelated invalid parameter")))


if __name__ == "__main__":
    unittest.main()
