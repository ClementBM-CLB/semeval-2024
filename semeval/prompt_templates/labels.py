from pydantic import BaseModel


class SemEvalLabels(BaseModel):
    statement: str = "Statement"
    clinical_trial: str = "Clinical trial report"

    primary_clinical_trial: str = "First clinical trial report"
    secondary_clinical_trial: str = "Second clinical trial report"
