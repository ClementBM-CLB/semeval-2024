from pydantic import BaseModel
from semeval.semeval_schemas import SemEvalSample

from semeval.prompt_templates.labels import SemEvalLabels


class ChainofThoughtsModel(BaseModel):
    sample: SemEvalSample
    labels: SemEvalLabels
    instruction: str = "Let's generate a chain of thought explanation"
