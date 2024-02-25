from typing import List
from pydantic import BaseModel
from semeval.semeval_schemas import (
    SemEvalInstructionSample,
)
from semeval.prompt_templates.labels import SemEvalLabels


class FewshotsModel(BaseModel):
    demo_samples: List[SemEvalInstructionSample]
    new_problem: SemEvalInstructionSample
    labels: SemEvalLabels
