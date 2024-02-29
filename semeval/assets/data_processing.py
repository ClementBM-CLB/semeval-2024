import typing

from dagster import (
    AssetExecutionContext,
    AssetIn,
    AssetKey,
    MetadataValue,
    asset,
)
from semeval.output_parser import OutputParserConfig
import pandas as pd

from semeval.resources.llm_resource import ChatMessageModel
from semeval.semeval_schemas import SemEvalSample


def define_cast_prediction(name, group_name):
    @asset(
        name=name,
        key_prefix=[group_name],
        group_name=group_name,
        ins={"prediction": AssetIn(key_prefix=[group_name])},
    )
    def cast_prediction(
        context: AssetExecutionContext,
        prediction: typing.Dict[str, ChatMessageModel],
        semeval2024_data: typing.List[SemEvalSample],
    ):
        output_parser = OutputParserConfig(element_name="answer", format="json").build()
        predictions = []

        for sample in semeval2024_data:
            raw_prediction = prediction[sample.key].get_last_model_reply()
            casted_prediction = output_parser.parse(raw_prediction)

            predictions.append(
                {
                    "casted_prediction": casted_prediction["answer"],
                    "raw_prediction": raw_prediction,
                }
                | sample.model_dump()
            )

        prediction_df = pd.DataFrame.from_records(predictions)

        context.add_output_metadata(
            metadata={
                "predictions": MetadataValue.md(prediction_df.head().to_markdown()),
            }
        )

        return predictions

    return cast_prediction
