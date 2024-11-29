import os
from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple, Union

from lm_eval.api.registry import register_model
from lm_eval.api.api_models import TemplateAPI
from lm_eval.utils import eval_logger


@register_model("djl-invocations")
class DJLInvocationsAPI(TemplateAPI):
    def __init__(
        self,
        base_url="http://127.0.0.1:8080/invocations",
        tokenizer_backend="huggingface",
        tokenized_requests=False,
        **kwargs,
    ):
        super().__init__(
            base_url=base_url, tokenizer_backend=tokenizer_backend, tokenized_requests=tokenized_requests, **kwargs
        )

    def _create_payload(
        self,
        messages: Union[List[List[int]], List[dict], List[str], str],
        generate=False,
        gen_kwargs: Optional[dict] = None,
        seed: int = 1234,
        **kwargs,
    ) -> dict:
        if generate:
            if "max_tokens" in gen_kwargs:
                gen_kwargs["max_new_tokens"] = gen_kwargs.pop("max_tokens")
            else:
                gen_kwargs["max_new_tokens"] = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
            gen_kwargs["temperature"] = gen_kwargs.pop("temperature", 0)
            gen_kwargs["stop"] = gen_kwargs.pop("until", ["<|endoftext|>"])
            gen_kwargs["top_n_tokens"] = gen_kwargs.pop("logprobs", 1)
            gen_kwargs["prompt_logprobs"] = gen_kwargs.pop("prompt_logprobs", 1)
            gen_kwargs["seed"] = seed
            gen_kwargs["details"] = True
            gen_kwargs["decoder_input_details"] = True
            return {
                "inputs": messages,
                "parameters": gen_kwargs,
                "adapters": "adapter1",
            }
        else:
            return {
                "inputs": messages,
                "parameters": {
                    "temperature": 0,
                    "max_new_tokens": 1,
                    "prompt_logprobs": 1,
                    "top_n_tokens": 1,
                    "seed": seed,
                    "details": True,
                    "decoder_input_details": True,
                },
                "adapters": "adapter1",
            }

    @staticmethod
    def parse_logprobs(
        outputs: Union[Dict, List[Dict]],
        tokens: List[List[int]] = None,
        ctxlens: List[int] = None,
        **kwargs,
    ) -> List[Tuple[float, bool]]:
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            # The first entry of prompt_logprobs is None because the model has no previous tokens to condition on.
            prompt_logprobs = out["details"]["prefill"]

            # prompt_logprob = sum(
            #     logprob_dict.get("log_prob") if logprob_dict.get("id") == tok else 0
            #     for tok, logprob_dict in zip(
            #         token[ctxlen:], prompt_logprobs[ctxlen + 1:]
            #     )
            # )
            prompt_logprob = prompt_logprobs[-1]["log_prob"]

            token_logprobs = [tok["log_prob"] for tok in out["details"]["tokens"]]
            top_logprobs = [[t["log_prob"] for t in top] for top in out["details"]["top_tokens"]]
            is_greedy = True
            for tok, top in zip(token_logprobs, top_logprobs):
                if tok != max(top):
                    is_greedy = False
                    break
            res.append((prompt_logprob, is_greedy))
        return res

    @staticmethod
    def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            res.append(out["generated_text"].strip())
        return res
