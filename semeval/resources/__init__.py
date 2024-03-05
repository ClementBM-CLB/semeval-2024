import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Sequence, Union

from dagster import ConfigurableResource
from dagster import EnvVar
from pydantic import Field
from semeval.resources.llm_resource import MistralPromptTemplate, TogetherPromptModel


llm_client = TogetherPromptModel(
    api_key=EnvVar("TOGETHER_API_KEY"),
    model_path="mistralai/Mistral-7B-Instruct-v0.2",
    prompt_template=MistralPromptTemplate(),
)
