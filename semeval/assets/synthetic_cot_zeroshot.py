# mypy: ignore-errors
import time
import typing
from itertools import islice

from dagster import (
    AssetExecutionContext,
    AssetIn,
    AssetOut,
    AssetsDefinition,
    MaterializeResult,
    Config,
    Output,
    multi_asset,
    MetadataValue,
    asset,
)
from pydantic import Field
from semeval.output_parser import OutputParserConfig
from semeval.prompt_manager import PromptConfig
import pandas as pd

from semeval.prompt_templates.labels import SemEvalLabels
from semeval.resources.llm_resource import ChatMessageModel, TogetherPromptModel
from semeval.assets.data_processing import define_cast_prediction
from semeval.semeval_schemas import SemEvalSample

SYNTHETIC_COT_ZEROSHOT_GROUP = "Synthetic_Chain_Of_Thought"


class CoTPromptConfig(Config):
    system_prompt: str = Field(
        default="",
        description="System prompt",
    )
    clinical_trial_label: str = Field(
        default="Clinical trial report",
        description="unique ctr label",
    )
    primary_clinical_trial_label: str = Field(
        default="Primary clinical trial report",
        description="primary ctr label",
    )
    secondary_clinical_trial_label: str = Field(
        default="Secondary clinical trial report",
        description="secondary ctr label",
    )
    statement_label: str = Field(
        default="Statement",
        description="statement label",
    )
    instruction: str = Field(
        default="Let's generate a chain of thought explanation",
        description="Chain of Thought instruction",
    )
    max_tokens: int = Field(default=512, description="Max tokens generation")


@asset(group_name=SYNTHETIC_COT_ZEROSHOT_GROUP)
def synthetic_cot(
    context: AssetExecutionContext,
    config: CoTPromptConfig,
    llm_client: TogetherPromptModel,
    semeval2024_data: typing.List[SemEvalSample],
) -> typing.Dict[str, ChatMessageModel]:

    semeval_labels = SemEvalLabels(
        clinical_trial=config.clinical_trial_label,
        primary_clinical_trial=config.primary_clinical_trial_label,
        secondary_clinical_trial=config.secondary_clinical_trial_label,
        statement=config.statement_label,
    )

    prompter = PromptConfig(prompt_type="cot").build()
    prompt_arguments = {
        "system_prompt": None,
        "instruction": config.instruction,
        "labels": semeval_labels,
    }

    chat_messages = {}

    for sample in semeval2024_data:
        chat_message = ChatMessageModel()

        message = prompter.build(**(prompt_arguments | {"problem_sample": sample}))
        chat_message.add_user_message(message)
        prediction = llm_client.generate_prediction(
            chat_message, max_new_tokens=config.max_tokens
        )

        time.sleep(3)

        chat_message.add_model_reply(prediction)
        chat_messages[sample.key] = chat_message

    chat_message_df = pd.DataFrame.from_records(
        [x.dict() for x in chat_messages.values()]
    )

    context.add_output_metadata(
        metadata={
            "chat_messages": MetadataValue.md(chat_message_df.head().to_markdown()),
        }
    )

    return chat_messages


SEMEVAL_FORMAT_INSTRUCTION = """
* Return "Entailment" if the hypothesis is supported by the clinical trial report.
* Return "Contradiction" if the hypothesis refuted by the clinical trial report.
* Return "Insufficient Information"  if the clinical trial report does not provide enough information to evaluate the hypothesis.

Answer in JSON format:

{ "answer": "" }

Therefore, the answer is """


class ZeroShotPromptConfig(Config):
    system_prompt: str = Field(
        default="",
        description="System prompt",
    )
    clinical_trial_label: str = Field(
        default="Clinical trial report",
        description="unique ctr label",
    )
    primary_clinical_trial_label: str = Field(
        default="Primary clinical trial report",
        description="primary ctr label",
    )
    secondary_clinical_trial_label: str = Field(
        default="Secondary clinical trial report",
        description="secondary ctr label",
    )
    statement_label: str = Field(
        default="Statement",
        description="statement label",
    )
    instruction: str = Field(
        default=SEMEVAL_FORMAT_INSTRUCTION, description="format instruction"
    )


@asset(
    name="prediction",
    group_name=SYNTHETIC_COT_ZEROSHOT_GROUP,
    key_prefix=[SYNTHETIC_COT_ZEROSHOT_GROUP],
)
def cot_zeroshot_prediction(
    context: AssetExecutionContext,
    config: ZeroShotPromptConfig,
    llm_client: TogetherPromptModel,
    synthetic_cot: typing.Dict[str, ChatMessageModel],
    semeval2024_data: typing.List[SemEvalSample],
) -> typing.Dict[str, ChatMessageModel]:

    prompter = PromptConfig(prompt_type="zeroshot").build()
    semeval_labels = SemEvalLabels(
        clinical_trial=config.clinical_trial_label,
        primary_clinical_trial=config.primary_clinical_trial_label,
        secondary_clinical_trial=config.secondary_clinical_trial_label,
        statement=config.statement_label,
    )

    prompt_arguments = {
        "system_prompt": config.system_prompt,
        "instructions": {
            "Single": config.instruction,
            "Comparison": config.instruction,
        },
        "labels": semeval_labels,
    }

    chat_messages = {}

    for sample in semeval2024_data:
        chat_message = synthetic_cot[sample.key]

        message = prompter.build(**(prompt_arguments | {"problem_sample": sample}))
        chat_message.add_user_message(message)

        prediction = llm_client.generate_prediction(chat_message, max_new_tokens=64)
        time.sleep(3)

        chat_message.add_model_reply(prediction)
        chat_messages[sample.key] = chat_message

    chat_message_df = pd.DataFrame.from_records(
        [x.dict() for x in chat_messages.values()]
    )

    context.add_output_metadata(
        metadata={
            "chat_messages": MetadataValue.md(chat_message_df.head().to_markdown()),
        }
    )

    return chat_messages


cot_zeroshot_cast_prediction = define_cast_prediction(
    name="cast_prediction",
    group_name=SYNTHETIC_COT_ZEROSHOT_GROUP,
)
