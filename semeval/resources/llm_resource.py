# mypy: ignore-errors

import gzip
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import typing

import requests
import together
from dagster import ConfigurableResource
from dagster._utils import file_relative_path
from dagster._utils.cached_method import cached_method


class ChatMessageModel:
    def __init__(
        self, system_prompt=None, user_message_history=[], model_replies_history=[]
    ):
        assert len(user_message_history) == len(model_replies_history)

        self.system_prompt = system_prompt
        self.user_messages = user_message_history
        self.model_replies = model_replies_history

    def add_user_message(self, message: str):
        self.user_messages.append(message)
        self._is_valid()

    def add_model_reply(self, reply: str):
        self._is_valid()
        self.model_replies.append(reply)

    def get_user_messages(self, strip=True):
        return [x.strip() for x in self.user_messages] if strip else self.user_messages

    def get_model_replies(self, strip=True):
        return [x.strip() for x in self.model_replies] if strip else self.model_replies

    def _is_valid(self):
        if len(self.user_messages) != len(self.model_replies) + 1:
            raise ValueError(
                "Error: Expected len(user_messages) = len(model_replies) + 1. Add a new user message!"
            )


class PromptTemplateBase(ConfigurableResource):
    @abstractmethod
    def build_prompt(self, chat_model: ChatMessageModel):
        raise Exception("Not implemented")


class LanguageModel(ConfigurableResource):
    prompt_template: PromptTemplateBase
    token_usage: dict = {
        "completion_tokens": 0,
        "prompt_tokens": 0,
    }

    @abstractmethod
    def call(
        self,
        prompt: str,
        max_new_tokens: int,
        candidates_per_step: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ):
        pass

    def init_token_usage(self):
        self.token_usage = {
            "completion_tokens": 0,
            "prompt_tokens": 0,
        }

    def generate_prediction(self, prompt: ChatMessageModel, max_new_tokens):
        return self.call(
            prompt=self.prompt_template.build_prompt(prompt),
            max_new_tokens=max_new_tokens,
            candidates_per_step=1,
            temperature=0.8,
            top_p=0,
            top_k=1,
        )[0]

    def generate_prompts(
        self, prompt: ChatMessageModel, max_new_tokens, candidates_per_step
    ) -> typing.List[str]:
        return self.call(
            prompt=self.prompt_template.build_prompt(prompt),
            max_new_tokens=max_new_tokens,
            candidates_per_step=candidates_per_step,
            temperature=1.2,
            top_p=0.2,
            top_k=50,
        )

    def reformulate(self, prompt: ChatMessageModel, max_new_tokens=1024):
        return self.call(
            prompt=self.prompt_template.build_prompt(prompt),
            max_new_tokens=max_new_tokens,
            candidates_per_step=1,
            temperature=0.8,
            top_p=0,
            top_k=1,
        )[0]


class TogetherPromptModel(LanguageModel):
    model_path: str
    api_key: str

    def call(
        self,
        prompt: str,
        max_new_tokens: int,
        candidates_per_step: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ):
        together.api_key = self.api_key
        output = together.Complete.create(
            prompt=prompt,
            model=self.model_path,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop=["</s>", "[/INST]"],
        )

        if "output" not in output:
            print("Call to together API failed")
            raise Exception("Generation failed")

        self._update_token_usage(output)

        predictions = [choice["text"] for choice in output["output"]["choices"]]

        return predictions

    def _update_token_usage(self, response: dict):
        if not response or "output" not in response:
            return

        output = response["output"]

        if "usage" not in output:
            return None

        usage = response["output"]["usage"]
        self.token_usage["completion_tokens"] += usage["completion_tokens"]
        self.token_usage["prompt_tokens"] += usage["prompt_tokens"]


class MistralPromptTemplate(PromptTemplateBase):
    def build_prompt(self, chat_model: ChatMessageModel):
        if chat_model.system_prompt is not None:
            system_message = f"<s>[INST] <<SYS>>\n{chat_model.system_prompt}\n<</SYS>>"
        else:
            system_message = "<s>[INST] "

        conversation_messages = ""
        for i in range(len(chat_model.user_messages) - 1):
            user_message, model_reply = (
                chat_model.user_messages[i],
                chat_model.model_replies[i],
            )
            conversation_ = f"{user_message}\n[/INST]\n{model_reply}"
            if i != 0:
                conversation_ = "[INST] " + conversation_
            conversation_messages += conversation_

        if len(chat_model.user_messages) == 1:
            conversation_messages += f"{chat_model.user_messages[-1]}\n[/INST]"
        else:
            conversation_messages += (
                f"</s>\n[INST]\n{chat_model.user_messages[-1]}\n[/INST]"
            )

        if len(chat_model.user_messages) == len(chat_model.model_replies):
            conversation_messages += f"\n{chat_model.model_replies[-1]}"

        return system_message + conversation_messages
