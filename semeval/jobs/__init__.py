from dagster import AssetSelection, define_asset_job
from semeval.assets.dataset_asset import DATASET_GROUP
from semeval.assets.reformulate_and_predict import REFORMULATION_GROUP
from semeval.assets.synthetic_cot_zeroshot import SYNTHETIC_COT_ZEROSHOT_GROUP

synthetic_cot_job = define_asset_job(
    "synthetic_cot_job",
    selection=AssetSelection.groups(SYNTHETIC_COT_ZEROSHOT_GROUP)
    | AssetSelection.groups(DATASET_GROUP),
)


reformulate_and_predict_job = define_asset_job(
    name="reformulate_and_predict",
    selection=AssetSelection.groups(REFORMULATION_GROUP)
    | AssetSelection.groups(DATASET_GROUP),
)
