from abc import ABC, abstractmethod

import os

from utils import construct_constraint_fun


class LlmManager(ABC):
    """
    An "interface" for various LLM manager objects.
    """

    @abstractmethod
    def chat_completion(
        self,
        prompt,
        print_result=False,
        seed=42,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.0,
    ):
        pass
    
class HuggingFaceLlmManager(LlmManager):
    def __init__(
        self,
        model_name,
        preloaded = None,
        cache_dir="argumentative-llms/cache",
        model_args=None,
        input_device="cuda:1",
        quantization="4bit",
    ):
        super().__init__()
        import torch
        import transformers
        from transformers import BitsAndBytesConfig

        self.last_usage = None
        if quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif quantization == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        elif quantization == "none":
            quantization_config = None
        else:
            raise ValueError(f"Invalid quantization value {quantization}")

        if preloaded is not None:
            self.pipeline = preloaded
        else:
            self.pipeline = transformers.pipeline(
                "text-generation",
                model=model_name,
                device_map=input_device,
                model_kwargs={
                    "torch_dtype": "auto",
                    "quantization_config": quantization_config,
                    "cache_dir": cache_dir,
                },
            )
        self.input_device = input_device

    def chat_completion(
        self,
        message,
        print_result=False,
        seed=42,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.0,
        constraint_prefix=None,
        constraint_options=None,
        constraint_end_after_options=False,
        trim_response=True,
        apply_template=True,
    ):
        import transformers

        self.last_usage = None
        transformers.set_seed(seed)
        messages = [{"role": "user", "content": message}]
        if apply_template:
            prompt = self.pipeline.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = message
        if constraint_prefix is not None or constraint_options is not None:
            prefix_allowed_tokens_fn = construct_constraint_fun(
                self.pipeline.tokenizer,
                prompt,
                force_prefix=constraint_prefix,
                force_options=constraint_options,
                end_after_options=constraint_end_after_options,
            )
        else:
            prefix_allowed_tokens_fn = None
            
        response = self.pipeline(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            )[0]["generated_text"]

        if print_result:
            print(response, flush=True)

        if trim_response:
            response = response.replace(prompt, "").strip()

        return response


class OpenAiLlmManager(LlmManager):
    def __init__(
        self,
        model_name,
        base_url=None,
        api_key_env=None,
        thinking_mode=None,
    ):
        self.model_name = model_name.split("openai/")[1]
        from openai import OpenAI

        self.last_usage = None
        self.thinking_mode = thinking_mode
        if api_key_env:
            api_key = os.environ[api_key_env]
        else:
            api_key = os.environ.get("OPENAI_KEY") or os.environ["OPENAI_API_KEY"]
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def chat_completion(
        self,
        message,
        print_result=False,
        seed=42,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.0,
        constraint_prefix=None,
        constraint_options=None,
        constraint_end_after_options=False,
        trim_response=True,
        apply_template=True,
    ):
        prompt = message

        messages = [{"role": "user", "content": prompt}]
        if constraint_prefix and constraint_options:
            if (constraint_prefix == "Likelihood:" and
                    constraint_options == [f" {value}%" for value in range(101)]):
                format_instruction = (
                    'Respond with exactly one line in the format "Likelihood: N%", '
                    'where N is an integer from 0 to 100. Do not add explanations.'
                )
            else:
                format_instruction = (
                    "Respond with exactly " + constraint_prefix + " followed by one of: "
                    + ", ".join(constraint_options) + "."
                )
                if constraint_end_after_options:
                    format_instruction += " Do not add any other text."
            messages.insert(0, {"role": "system", "content": format_instruction})
        request = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_new_tokens,
            "top_p": top_p,
            "presence_penalty": repetition_penalty,
        }
        if self.thinking_mode is not None:
            request["extra_body"] = {"thinking": {"type": self.thinking_mode}}
        completion = self.client.chat.completions.create(**request)

        self.last_usage = {
            "input_tokens": completion.usage.prompt_tokens,
            "output_tokens": completion.usage.completion_tokens,
        } if completion.usage else None
        response = completion.choices[0].message.content

        if print_result:
            print(response, flush=True)

        if trim_response:
            response = response.replace(prompt, "").strip()

        return response
