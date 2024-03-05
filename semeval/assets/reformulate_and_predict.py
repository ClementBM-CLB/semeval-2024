# mypy: ignore-errors
import time
import typing

from dagster import (
    AssetExecutionContext,
    Config,
    MetadataValue,
    asset,
)
from pydantic import Field
from semeval.assets.data_processing import define_cast_prediction
from semeval.output_parser import OutputParserConfig
from semeval.prompt_manager import PromptConfig
import pandas as pd
from semeval.prompt_templates.labels import SemEvalLabels

from semeval.resources.llm_resource import ChatMessageModel, TogetherPromptModel
from semeval.semeval_schemas import SemEvalSample

REFORMULATION_GROUP = "Reformulate"


class ReformulatePromptConfig(Config):
    system_prompt: str = Field(
        default="",
        description="System prompt",
    )


@asset(group_name=REFORMULATION_GROUP)
def reformulate(
    context: AssetExecutionContext,
    config: ReformulatePromptConfig,
    llm_client: TogetherPromptModel,
    semeval2024_data: typing.List[SemEvalSample],
) -> typing.Dict[str, ChatMessageModel]:
    prompter = PromptConfig(prompt_type="reformulate").build()
    prompt_arguments = {
        "instruction": "Please reformulate the following hypothesis:",
        "format_instruction": """Answer in JSON format:\n\n{ "statement": "" }""",
    }

    chat_messages = {}

    for sample in semeval2024_data:
        chat_message = ChatMessageModel(system_prompt=config.system_prompt)

        message = prompter.build(**(prompt_arguments | {"problem_sample": sample}))

        chat_message.add_user_message(message)
        prediction = llm_client.generate_prediction(chat_message, max_new_tokens=128)
        time.sleep(3)

        chat_message.add_model_reply(prediction)
        chat_messages[sample.key] = chat_message

    chat_message_df = pd.DataFrame.from_records(
        [x.dict() for x in chat_messages.values()]
    )

    context.add_output_metadata(
        metadata={
            "chat messages": MetadataValue.md(chat_message_df.head().to_markdown()),
        },
    )

    return chat_messages


@asset(group_name=REFORMULATION_GROUP)
def cast_reformulation(
    context: AssetExecutionContext,
    reformulate: typing.Dict[str, ChatMessageModel],
    semeval2024_data: typing.List[SemEvalSample],
) -> typing.List[SemEvalSample]:
    output_parser = OutputParserConfig(element_name="statement", format="json").build()

    for sample in semeval2024_data:
        raw_prediction = reformulate[sample.key].get_last_model_reply()
        casted_prediction = output_parser.parse(raw_prediction)

        sample.statement = casted_prediction["statement"]

    sample_df = pd.DataFrame.from_records([x.model_dump() for x in semeval2024_data])

    context.add_output_metadata(
        metadata={
            "samples": MetadataValue.md(sample_df.head().to_markdown()),
        }
    )

    return semeval2024_data


class OpPromptConfig(Config):
    system_prompt: str = Field(
        default="",
        description="System prompt",
    )
    single_instruction: str = Field(
        default="""
Given a clinical trial and a statement, determine whether the statement logically follows from the clinical trial.
If the statement logically follows from the clinical trial, you need to return "Entailment". If not, you need to return "Contradiction".
Do not explain or elaborate and do not mention the term "statement" or "trial". If you are unable to extract the information, write "N-A".
""",
        description="Single instruction",
    )
    comparison_instruction: str = Field(
        default="""
Given two clinical trials and a statement, determine whether the statement logically follows from the clinical trials.
If the statement logically follows from the clinical trials, you need to return "Entailment". If not, you need to return "Contradiction".
Do not explain or elaborate and do not mention the term "statement" or "trial". If you are unable to extract the information, write "N-A".
""",
        description="Comparison instruction",
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


@asset(
    name="prediction",
    group_name=REFORMULATION_GROUP,
    key_prefix=[REFORMULATION_GROUP],
)
def reformulate_zeroshot_prediction(
    context: AssetExecutionContext,
    config: OpPromptConfig,
    llm_client: TogetherPromptModel,
    cast_reformulation: typing.List[SemEvalSample],
) -> typing.Dict[str, ChatMessageModel]:

    prompter = PromptConfig(prompt_type="zeroshot").build()
    semeval_labels = SemEvalLabels(
        clinical_trial=config.clinical_trial_label,
        primary_clinical_trial=config.primary_clinical_trial_label,
        secondary_clinical_trial=config.secondary_clinical_trial_label,
        statement=config.statement_label,
    )

    prompt_arguments = {
        "instructions": {
            "Single": config.single_instruction,
            "Comparison": config.comparison_instruction,
        },
        "labels": semeval_labels,
    }

    chat_messages = {}

    for sample in cast_reformulation:
        chat_message = ChatMessageModel(system_prompt=config.system_prompt)

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


reformulate_zeroshot_cast_prediction = define_cast_prediction(
    name="cast_prediction",
    group_name=REFORMULATION_GROUP,
)
