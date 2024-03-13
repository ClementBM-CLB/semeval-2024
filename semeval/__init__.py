# A job lets you target a selection of assets
# to materialize them together as a single action.
# Assets can belong to multiple jobs.

from dagster import (
    Definitions,
    ScheduleDefinition,
    load_assets_from_modules,
    FilesystemIOManager,  # Update the imports at the top of the file to also include this
)

from semeval.resources import llm_client
from semeval.assets import dataset_asset
from semeval.assets import reformulate_and_predict
from semeval.assets import synthetic_cot_zeroshot

from semeval.jobs import synthetic_cot_job, reformulate_and_predict_job

all_assets = load_assets_from_modules(
    [
        dataset_asset,
        synthetic_cot_zeroshot,
        reformulate_and_predict,
    ]
)
# Addition: a ScheduleDefinition the job it should run and a cron schedule of how frequently to run it
reformulate_and_predict_schedule = ScheduleDefinition(
    job=reformulate_and_predict_job,
    cron_schedule="0 * * * *",  # every hour
)

io_manager = FilesystemIOManager(
    base_dir="artifacts/data",  # Path is built relative to where `dagster dev` is run
)

defs = Definitions(
    assets=all_assets,
    schedules=[
        reformulate_and_predict_schedule
    ],  # Addition: add the job to Definitions object (see below)
    jobs=[reformulate_and_predict_job, synthetic_cot_job],
    resources={
        "io_manager": io_manager,
        "llm_client": llm_client,
    },
)
