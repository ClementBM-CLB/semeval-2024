import json
from torch.utils.data import Dataset
from pydantic import BaseModel

from enum import Enum, unique
from artifacts.semeval import (
    TRAIN_DATA_PATH,
    DEV_DATA_PATH,
    TEST_DATA_PATH,
    CTR_DATA_PATH,
)
from semeval.semeval_schemas import SemEvalSample


def load_data(name):
    """name: `train` or `dev` or `test`"""
    data_path = DEV_DATA_PATH
    if name == "train":
        data_path = TRAIN_DATA_PATH
    if name == "test":
        data_path = TEST_DATA_PATH

    with open(data_path) as json_file:
        dev_data = json.load(json_file)
    return dev_data


def load_clinical_trials():
    ctr_dict = {}
    for ctr_path in CTR_DATA_PATH.iterdir():
        if ctr_path.name == ".DS_Store":
            continue

        with open(ctr_path) as json_file:
            ctr_data = json.load(json_file)

        ctr_dict[ctr_data["Clinical Trial ID"]] = {
            "Intervention": ctr_data["Intervention"],
            "Adverse Events": ctr_data["Adverse Events"],
            "Results": ctr_data["Results"],
            "Eligibility": ctr_data["Eligibility"],
        }

    return ctr_dict


class SemEvalDataset(Dataset):
    def __init__(self, dataset: dict, clinical_trials: dict, is_labelized=True):
        super().__init__()
        dataset_values = {
            item[0]: {"Key": item[0], **item[1]} for item in dataset.items()
        }
        self.clinical_trials = clinical_trials

        self.dataset = list(dataset_values.values())

        self.is_labelized = is_labelized

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> SemEvalSample:
        inference = self.dataset[idx]

        section = inference["Section_id"]

        primary_ctr = self.clinical_trials[inference["Primary_id"]]

        if inference["Type"] == "Comparison":
            secondary_ctr = self.clinical_trials[inference["Secondary_id"]]

        return SemEvalSample(
            **{
                "key": inference["Key"],
                "type": inference["Type"],
                "section": inference["Section_id"],
                "statement": inference["Statement"],
                "primary_ct_section": primary_ctr[section],
                "secondary_ct_section": (
                    secondary_ctr[section] if "Secondary_id" in inference else []
                ),
                "label": inference["Label"] if self.is_labelized else "NA",
            }
        )
