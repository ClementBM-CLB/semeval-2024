from pathlib import Path

DATA_FOLDER = Path(__file__).parent.absolute()

TRAIN_DATA_PATH = DATA_FOLDER / "train.json"
DEV_DATA_PATH = DATA_FOLDER / "dev.json"
CTR_DATA_PATH = DATA_FOLDER / "clinical_trials"
TEST_DATA_PATH = DATA_FOLDER / "test.json"
