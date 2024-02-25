from pydantic import BaseModel
from semeval.semeval_schemas import SemEvalInstructionSample

from semeval.prompt_templates.labels import SemEvalLabels


class ZeroshotModel(BaseModel):
    sample: SemEvalInstructionSample
    labels: SemEvalLabels
