from pathlib import Path

ARTIFACTS_FOLDER = Path(__file__).parent.absolute()

MODEL_FOLDER = ARTIFACTS_FOLDER / "models"
DATASET_FOLDER = ARTIFACTS_FOLDER / "datasets"
OUTPUT_FOLDER = ARTIFACTS_FOLDER / "outputs"
EXPERIMENT_FOLDER = ARTIFACTS_FOLDER / "experiments"
STATEMENT_FOLDER = ARTIFACTS_FOLDER / "statements"
INDEX_FOLDER = ARTIFACTS_FOLDER / "vector_index"
RESULTS_FOLDER = ARTIFACTS_FOLDER / "results"

DATA_PATH = Path("~/data/data_raw")
