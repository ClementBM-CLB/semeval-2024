# mypy: ignore-errors
import time
import typing
import pandas as pd
import mlflow

from dagster import (
    AssetExecutionContext,
    Config,
    MetadataValue,
    asset,
)
from pydantic import Field
from artifacts import EXPERIMENT_FOLDER
from semeval.prompt_manager import PromptConfig

from semeval.prompt_templates.labels import SemEvalLabels
from semeval.resources.llm_resource import ChatMessageModel, TogetherPromptModel
from semeval.assets.data_processing import define_cast_prediction
from semeval.semeval_metrics import PromptMetrics
from semeval.semeval_schemas import SemEvalSample

SYNTHETIC_COT_ZEROSHOT_GROUP = "Synthetic_Chain_Of_Thought"


class PromptAssetConfig(Config):
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


class CoTConfig(PromptAssetConfig):
    instruction: str = Field(
        default="Let's generate a chain of thought explanation",
        description="Chain of Thought instruction",
    )
    max_tokens: int = Field(default=512, description="Max tokens generation")


@asset(group_name=SYNTHETIC_COT_ZEROSHOT_GROUP, compute_kind="llm")
def synthetic_cot(
    context: AssetExecutionContext,
    config: CoTConfig,
    llm_client: TogetherPromptModel,
    semeval2024_data: typing.List[SemEvalSample],
) -> typing.Dict[str, ChatMessageModel]:
    """Generate a chain of thought explanation to resolve the provided problem"""
    semeval_labels = SemEvalLabels(
        clinical_trial=config.clinical_trial_label,
        primary_clinical_trial=config.primary_clinical_trial_label,
        secondary_clinical_trial=config.secondary_clinical_trial_label,
        statement=config.statement_label,
    )

    prompter = PromptConfig(prompt_type="cot").build()
    prompt_arguments = {
        "instruction": config.instruction,
        "labels": semeval_labels,
    }

    chat_messages = {}

    for sample in semeval2024_data:
        chat_message = ChatMessageModel(system_prompt=config.system_prompt)

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


class FormattingConfig(PromptAssetConfig):
    instruction: str = Field(
        default=SEMEVAL_FORMAT_INSTRUCTION,
        description="Chain of Thought instruction",
    )
    max_tokens: int = Field(default=64, description="Max tokens generation")


@asset(
    name="prediction",
    group_name=SYNTHETIC_COT_ZEROSHOT_GROUP,
    key_prefix=[SYNTHETIC_COT_ZEROSHOT_GROUP],
    compute_kind="llm",
)
def cot_zeroshot_prediction(
    context: AssetExecutionContext,
    config: FormattingConfig,
    llm_client: TogetherPromptModel,
    synthetic_cot: typing.Dict[str, ChatMessageModel],
    semeval2024_data: typing.List[SemEvalSample],
) -> typing.Dict[str, ChatMessageModel]:
    """Add formatting instruction and continue generating"""
    chat_messages = {}

    for sample in semeval2024_data:
        chat_message = synthetic_cot[sample.key]
        chat_message.add_user_message(config.instruction)

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


cot_zeroshot_cast_prediction = define_cast_prediction(
    name="cast_prediction",
    group_name=SYNTHETIC_COT_ZEROSHOT_GROUP,
)


# mlflow server --host 127.0.0.1 --port 8894
@asset(
    name="evaluate",
    group_name=SYNTHETIC_COT_ZEROSHOT_GROUP,
    key_prefix=[SYNTHETIC_COT_ZEROSHOT_GROUP],
    compute_kind="mlflow",
)
def evaluate(
    context: AssetExecutionContext,
    cast_prediction: typing.List[dict],
    prediction: typing.Dict[str, ChatMessageModel],
):
    """Evaluate the experiment and log to mlflow"""
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8894")

    # set the experiment id
    mlflow.set_experiment("SemEval2024")

    # asset_provenance = context.get_asset_provenance(context.asset_key)
    # context.asset_key_for_input("cast_prediction")

    # start a run
    with mlflow.start_run():
        result_df = pd.DataFrame.from_records(cast_prediction)
        result_df["is_accurate"] = result_df["label"] == result_df["casted_prediction"]

        for i, row in result_df.iterrows():
            if not EXPERIMENT_FOLDER.exists():
                EXPERIMENT_FOLDER.mkdir()

            file_path = EXPERIMENT_FOLDER / f"{row['is_accurate']}-{row['key']}.txt"
            with open(file_path, mode="w") as file_writer:

                chat_message = prediction[row["key"]]
                file_writer.write(str(chat_message))

                file_writer.write("\n\n# Casted prediction\n\n")
                file_writer.write(row["casted_prediction"])
                file_writer.write("\n\n# True label\n\n")
                file_writer.write(row["label"])

            mlflow.log_artifact(local_path=file_path, artifact_path="prompts")
            mlflow.log_metric(
                key="guid_" + row["key"].replace("-", "_"), value=row["is_accurate"]
            )

        # mlflow.log_input(dataset=dev_dataset, context="training")

        prompt_metrics = PromptMetrics(cast_prediction)

        experiment_scores = prompt_metrics.evaluate()

        mlflow.log_metrics(experiment_scores)
        context.add_output_metadata(metadata=experiment_scores)
        # mlflow.log_metrics(experiment.model.token_usage)

        mlflow.log_params(
            {
                "sample_count": prompt_metrics.total_length,
                # "model": model_path,
                # "max_new_tokens": max_new_tokens,
            }
        )


# config_from_files
