# A job lets you target a selection of assets
# to materialize them together as a single action.
# Assets can belong to multiple jobs.

from dagster import (
    AssetSelection,
    Definitions,
    ScheduleDefinition,
    define_asset_job,
    load_assets_from_modules,
    FilesystemIOManager,  # Update the imports at the top of the file to also include this
)

from semeval.resources import llm_client
from semeval.assets import reformulate_and_predict

all_assets = load_assets_from_modules([reformulate_and_predict])

llm_job = define_asset_job(
    name="llm_reformulate",
    selection=AssetSelection.groups("llm"),
)

# Addition: a ScheduleDefinition the job it should run and a cron schedule of how frequently to run it
llm_schedule = ScheduleDefinition(
    job=llm_job,
    cron_schedule="0 * * * *",  # every hour
)

io_manager = FilesystemIOManager(
    base_dir="artifacts/data",  # Path is built relative to where `dagster dev` is run
)

defs = Definitions(
    assets=all_assets,
    schedules=[llm_schedule],  # Addition: add the job to Definitions object (see below)
    jobs=[llm_job],
    resources={
        "io_manager": io_manager,
        "llm_client": llm_client,
    },
)
