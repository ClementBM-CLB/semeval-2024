from typing import List
from pydantic import BaseModel

from semeval.semeval_schemas import (
    InstructionScore,
    SemEvalSample,
)
from semeval.prompt_templates.labels import SemEvalLabels


class LanguageModel(BaseModel):
    instruction_scores: List[InstructionScore]
    samples: List[SemEvalSample]
    labels: SemEvalLabels
