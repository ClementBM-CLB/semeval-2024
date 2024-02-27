# mypy: ignore-errors
import os
import time
import typing

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
from semeval.semeval_dataset import SemEvalDataset, load_clinical_trials, load_data
from semeval.semeval_schemas import SemEvalSample


class DatasetConfig(Config):
    name: str = Field(
        default="dev", description="Dataset split name: train, dev, or test"
    )


@asset(group_name="llm")
def semeval2024_data(
    context: AssetExecutionContext, config: DatasetConfig
) -> SemEvalDataset:
    ctr_dict = load_clinical_trials()
    data = load_data(name=config.name)

    is_labelized = True
    if config.name == "test":
        is_labelized = False

    dataset = SemEvalDataset(
        dataset=data, clinical_trials=ctr_dict, is_labelized=is_labelized
    )

    samples = pd.DataFrame.from_records(x.dict() for x in dataset)

    context.add_output_metadata(
        metadata={
            "samples": MetadataValue.md(samples.head().to_markdown()),
        }
    )

    return dataset


@multi_asset(
    group_name="llm",
    outs={
        "result_samples": AssetOut(),
        "chat_messages": AssetOut(),
    },
)
def reformulate(
    context: AssetExecutionContext,
    llm_client: TogetherPromptModel,
    semeval2024_data: SemEvalDataset,
):
    prompter = PromptConfig(prompt_type="reformulate").build()
    prompt_arguments = {
        "system_prompt": None,
        "instruction": "Please reformulate the following hypothesis:",
        "format_instruction": """Answer in JSON format:\n\n{ "statement": "" }""",
    }

    output_parser = OutputParserConfig(element_name="statement", format="json").build()

    i = 0
    result_samples = []
    chat_messages = []
    for sample in semeval2024_data:
        chat_message = ChatMessageModel()

        message = prompter.build(**(prompt_arguments | {"problem_sample": sample}))

        chat_message.add_user_message(message)
        prediction = llm_client.generate_prediction(chat_message, max_new_tokens=128)
        time.sleep(3)

        chat_message.add_model_reply(prediction)

        casted_prediction = output_parser.parse(prediction)

        sample.statement = casted_prediction["statement"]

        i += 1
        result_samples.append(sample)
        chat_messages.append(chat_message)

        if i > 5:
            break

    reformulate_samples = pd.DataFrame.from_records(x.dict() for x in result_samples)

    context.add_output_metadata(
        metadata={
            "samples": MetadataValue.md(reformulate_samples.head().to_markdown()),
        },
        output_name="result_samples",
    )

    return result_samples, chat_messages


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
    group_name="llm",
    ins={
        "result_samples": AssetIn(key="result_samples"),
        "chat_messages": AssetIn(key="chat_messages"),
    },
)
def prediction(
    context: AssetExecutionContext,
    config: OpPromptConfig,
    llm_client: TogetherPromptModel,
    result_samples: typing.List[SemEvalSample],
    chat_messages: typing.List[ChatMessageModel],
):
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
            "Single": config.single_instruction,
            "Comparison": config.comparison_instruction,
        },
        "labels": semeval_labels,
    }

    context.log.info(f"Num samples {len(result_samples)}")
    context.log.info(f"Num chat_messages {len(chat_messages)}")

    output_parser = OutputParserConfig(element_name="answer", format="json").build()

    result_predictions = []

    for sample, chat_message in zip(result_samples, chat_messages):

        message = prompter.build(**(prompt_arguments | {"problem_sample": sample}))
        chat_message.add_user_message(message)

        prediction = llm_client.generate_prediction(chat_message, max_new_tokens=64)
        time.sleep(3)

        chat_message.add_model_reply(prediction)

        casted_prediction = output_parser.parse(prediction)

        context.log.info(prediction)

        result_predictions.append(casted_prediction["answer"])

        result_samples.append(sample)

    predictions = pd.DataFrame({"prediction": result_predictions})

    context.add_output_metadata(
        metadata={
            "predictions": MetadataValue.md(predictions.head().to_markdown()),
        }
    )

    return predictions
