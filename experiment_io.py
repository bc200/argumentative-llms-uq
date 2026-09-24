"""Small persistent response cache and measured model calls for QBAF experiments."""

import json
import os
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def cached_call(path, request, perform):
    path = Path(path)
    if path.exists():
        record = read_json(path)
        if record["request"] != request:
            raise ValueError("Cached request differs from current experiment: " + str(path))
        return record
    response, usage, elapsed = perform()
    record = {
        "request": request,
        "response": response,
        "usage": usage,
        "elapsed_seconds": elapsed,
    }
    write_json(path, record)
    return record


class CachedLlmManager:
    """Adapter with the same chat_completion interface used by ArgumentMiner."""

    def __init__(
        self, delegate, model_name, directory, input_price=None, output_price=None,
        cache_config=None,
        currency="USD",
    ):
        self.delegate = delegate
        self.model_name = model_name
        self.directory = Path(directory)
        self.input_price = input_price
        self.output_price = output_price
        self.cache_config = cache_config or {}
        self.currency = currency
        self.records = []
        self.call_number = 0

    def chat_completion(self, message, **kwargs):
        self.call_number += 1
        path = self.directory / ("call_%03d.json" % self.call_number)
        request = {
            "model": self.model_name,
            "model_config": self.cache_config,
            "message": message,
            "kwargs": kwargs,
        }

        def perform():
            start = time.perf_counter()
            quiet_kwargs = dict(kwargs)
            quiet_kwargs["print_result"] = False
            response = self.delegate.chat_completion(message, **quiet_kwargs)
            elapsed = time.perf_counter() - start
            return response, getattr(self.delegate, "last_usage", None), elapsed

        record = cached_call(path, request, perform)
        usage = record["usage"]
        if not self.model_name.startswith("openai/"):
            cost = 0.0
        elif usage is None or self.input_price is None or self.output_price is None:
            cost = None
        else:
            cost = (
                usage["input_tokens"] * self.input_price
                + usage["output_tokens"] * self.output_price
            ) / 1000000
        record["cost_usd"] = cost if self.currency == "USD" else 0.0
        record["cost_cny"] = cost if self.currency == "CNY" else 0.0
        record["_cache_path"] = str(path)
        self.records.append(record)
        return record["response"]


class CachedJev:
    def __init__(self, cache_root, model="jev-latest",
                 endpoint="https://api.typesafe.ai/v1/systemone",
                 input_price=0.042, api_key_env="TYPESAFE_API_KEY"):
        self.cache_root = Path(cache_root)
        self.model = model
        self.endpoint = endpoint
        self.input_price = input_price
        self.api_key_env = api_key_env

    def ask(self, relative_path, state, instructions, criteria=None):
        question = {"type": "noul", "instructions": instructions}
        if criteria is not None:
            question["criteria"] = criteria
        payload = {
            "model": self.model,
            "state": state,
            "questions": {"answer": question},
        }
        path = self.cache_root / relative_path
        request = {"endpoint": self.endpoint, "payload": payload}

        def perform():
            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise RuntimeError(
                    "Missing Jev API key environment variable: " + self.api_key_env
                )
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            http_request = Request(
                self.endpoint,
                data=body,
                headers={
                    "Authorization": "Bearer " + api_key,
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            start = time.perf_counter()
            for attempt in range(4):
                try:
                    with urlopen(http_request, timeout=60) as response:
                        data = json.load(response)
                    return data, data.get("usage"), time.perf_counter() - start
                except HTTPError as error:
                    if error.code not in (429, 529) or attempt == 3:
                        raise
                    time.sleep(2 ** attempt)
                except URLError:
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)
            raise RuntimeError("Jev request failed")

        record = cached_call(path, request, perform)
        usage = record["usage"]
        record["cost_usd"] = (
            usage["input_tokens"] * self.input_price / 1000000
            if usage is not None and "input_tokens" in usage else None
        )
        record["cost_cny"] = 0.0
        record["_cache_path"] = str(path)
        answer = record["response"]["answers"]["answer"]
        if answer["type"] != "noul" or not 0.0 <= answer["noul"] <= 1.0:
            raise ValueError("Jev did not return a valid Noul probability")
        return float(answer["noul"]), record
