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
from semeval.assets import llm_test, hn_news

all_assets = load_assets_from_modules([llm_test, hn_news])

# Addition: define a job that will materialize the assets
hackernews_job = define_asset_job("hackernews_job", selection=AssetSelection.all())

# Addition: a ScheduleDefinition the job it should run and a cron schedule of how frequently to run it
hackernews_schedule = ScheduleDefinition(
    job=hackernews_job,
    cron_schedule="0 * * * *",  # every hour
)

io_manager = FilesystemIOManager(
    base_dir="data",  # Path is built relative to where `dagster dev` is run
)

defs = Definitions(
    assets=all_assets,
    schedules=[
        hackernews_schedule
    ],  # Addition: add the job to Definitions object (see below)
    resources={"io_manager": io_manager, "llm_client": llm_client},
)
