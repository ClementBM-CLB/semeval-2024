# mypy: ignore-errors
import os

from dagster import (
    AssetExecutionContext,
    AssetsDefinition,
    MaterializeResult,
    Config,
    Output,
    asset,
)
from pydantic import Field
from semeval.output_parser import OutputParserConfig
from semeval.prompt_manager import PromptConfig

from semeval.resources.llm_resource import ChatMessageModel, TogetherPromptModel
from semeval.semeval_dataset import SemEvalDataset, load_clinical_trials, load_data


class DatasetConfig(Config):
    name: str = Field(
        default="dev", description="Dataset split name: train, dev, or test"
    )


@asset(group_name="llm")
def semeval2024_data(config: DatasetConfig) -> SemEvalDataset:
    ctr_dict = load_clinical_trials()
    data = load_data(name=config.name)

    dataset = SemEvalDataset(dataset=data, clinical_trials=ctr_dict, is_labelized=True)
    return dataset


@asset(group_name="llm")
def reformulate(
    context: AssetExecutionContext,
    llm_client: TogetherPromptModel,
    semeval2024_data: SemEvalDataset,
):

    context.log.info(f"Number data sample {len(semeval2024_data)}")

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
        prediction = (
            "toto"  # llm_client.generate_prediction(chat_message, max_new_tokens=128)
        )
        chat_message.add_model_reply(prediction)

        casted_prediction = output_parser.parse(prediction)

        sample.statement = "tata"  # casted_prediction["statement"]

        i += 1
        result_samples.append(sample)
        chat_messages.append(chat_message)

        if i > 5:
            break

    return result_samples, chat_messages
