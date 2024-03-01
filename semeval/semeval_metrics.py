import pandas as pd
from sklearn.metrics import f1_score


class PromptMetrics:
    def __init__(self, experiment_result) -> None:
        self.experiment_result = pd.DataFrame.from_records(experiment_result)
        self.experiment_result["is_accurate"] = (
            self.experiment_result["label"]
            == self.experiment_result["casted_prediction"]
        )

        self.total_length = len(self.experiment_result)

    def f1_score(self):
        return f1_score(
            self.experiment_result["label"] == "Entailment",
            self.experiment_result["casted_prediction"] == "Entailment",
            average="macro",
        ).tolist()

    def accuracy(self):
        return sum(self.experiment_result["is_accurate"]) / self.total_length

    def prediction_entailment_ratio(self):
        return (
            sum(self.experiment_result["casted_prediction"] == "Entailment")
            / self.total_length
        )

    def prediction_contradiction_ratio(self):
        return (
            sum(self.experiment_result["casted_prediction"] == "Contradiction")
            / self.total_length
        )

    def label_entailment_ratio(self):
        return sum(self.experiment_result["label"] == "Entailment") / self.total_length

    def label_contradiction_ratio(self):
        return (
            sum(self.experiment_result["label"] == "Contradiction") / self.total_length
        )

    def evaluate(self):
        return {
            "accuracy": self.accuracy(),
            "f1-score": self.f1_score(),
            "prediction_entailment_ratio": self.prediction_entailment_ratio(),
            "prediction_contradiction_ratio": self.prediction_contradiction_ratio(),
            "label_entailment_ratio": self.label_entailment_ratio(),
            "label_contradiction_ratio": self.label_contradiction_ratio(),
        }

    def na_ratio(self):
        pass
