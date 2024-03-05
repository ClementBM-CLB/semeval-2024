import typing
from pydantic import Field
import pandas as pd

from dagster import (
    AssetExecutionContext,
    Config,
    MetadataValue,
    asset,
)

from semeval.semeval_dataset import SemEvalDataset, load_clinical_trials, load_data
from semeval.semeval_schemas import SemEvalSample

DATASET_GROUP = "dataset"


class DatasetConfig(Config):
    name: str = Field(
        default="dev", description="Dataset split name: train, dev, or test"
    )
    section_filter: typing.List[str] = Field(
        default=["Adverse Events", "Eligibility", "Results", "Intervention"],
        description="Filter on section: Adverse Events, Eligibility, Results, Intervention",
    )
    type_filter: typing.List[str] = Field(
        default=["Single", "Comparison"],
        description="Filter on type: Single, Comparison",
    )
    top_k: int = Field(
        default=10, description="Take top k sample from the dataset, -1 for all"
    )


@asset(group_name=DATASET_GROUP)
def semeval2024_data(
    context: AssetExecutionContext,
    config: DatasetConfig,
) -> typing.List[SemEvalSample]:
    ctr_dict = load_clinical_trials()
    data = load_data(name=config.name)

    is_labelized = True
    if config.name == "test":
        is_labelized = False

    dataset = SemEvalDataset(
        dataset=data,
        clinical_trials=ctr_dict,
        is_labelized=is_labelized,
    )

    filtered_dataset = [
        sample
        for sample in dataset
        if sample.type in config.type_filter and sample.section in config.section_filter
    ]

    sample_df = pd.DataFrame.from_records([x.model_dump() for x in filtered_dataset])

    context.add_output_metadata(
        metadata={
            "samples": MetadataValue.md(sample_df.head().to_markdown()),
            "filtered_count": len(filtered_dataset),
        }
    )
    if config.top_k == -1:
        return filtered_dataset
    return filtered_dataset[: config.top_k]
