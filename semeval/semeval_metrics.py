import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score


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

    def confusion_matrix(self):
        cm = confusion_matrix(
            self.experiment_result["label"] == "Entailment",
            self.experiment_result["casted_prediction"] == "Entailment",
        )
        t_n, f_p, f_n, t_p = cm.ravel()
        return {
            "true_negative": int(t_n),
            "false_positive": int(f_p),
            "false_negative": int(f_n),
            "true_positive": int(t_p),
            "precision": (
                0 if int(t_p + f_p) == 0 else float(t_p) / (float(t_p) + float(f_p))
            ),
            "recall": (
                0 if int(t_p + f_n) == 0 else float(t_p) / (float(t_p) + float(f_n))
            ),
        }

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
        } | self.confusion_matrix()

    def na_ratio(self):
        pass
