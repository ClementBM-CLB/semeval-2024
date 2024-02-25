from abc import abstractmethod
import typing
from semeval.semeval_schemas import (
    InstructionScore,
    SemEvalInstructionSample,
    SemEvalSample,
)
from pydantic import BaseModel
from semeval.prompt_templates.chainofthoughts import ChainofThoughtsModel
from semeval.prompt_templates.fewshots import FewshotsModel
from semeval.prompt_templates.labels import SemEvalLabels
from semeval.prompt_templates.opro import LanguageModel
from jinja2 import Environment, PackageLoader, select_autoescape

from semeval.prompt_templates.zeroshot import ZeroshotModel


class PromptConfig(BaseModel):
    prompt_type: str

    def build(self):
        match self.prompt_type:
            case "opro":
                return OproPrompt()
            case "fewshot":
                return FewShotsPrompt()
            case "zeroshot":
                return ZeroShotPrompt()
            case "reformulate":
                return ReformulatePrompt()
            case _:
                raise Exception("Unknown prompt type")


class PromptBase:
    def __init__(self, template_name):
        env = Environment(
            loader=PackageLoader("semeval", package_path="prompt_templates"),
            autoescape=select_autoescape(),
        )
        self.template_name = template_name
        self.template = env.get_template(template_name)

    @abstractmethod
    def build(self, *args, **kwargs):
        pass
        # return self.template.render(model=model)


class OproPrompt(PromptBase):
    def __init__(self):
        super().__init__("opro.prompt")

    def create_model(
        self,
        instruction_scores: typing.List[InstructionScore],
        problem_samples: typing.List[SemEvalSample],
        labels: SemEvalLabels,
        **kwargs,
    ):
        return LanguageModel(
            instruction_scores=instruction_scores,
            samples=problem_samples,
            labels=labels,
        )

    def build(self, *args, **kwargs) -> str:
        model = self.create_model(**kwargs)
        return self.template.render(model=model)


class FewShotsPrompt(PromptBase):
    def __init__(self):
        super().__init__("fewshots.prompt")

    def create_model(
        self,
        problem_sample: SemEvalSample,
        demo_problem_samples: typing.List[SemEvalSample],
        labels: SemEvalLabels,
        instruction: str,
        **kwargs,
    ):
        return FewshotsModel(
            demo_samples=[
                SemEvalInstructionSample(
                    sample=demo_problem_sample,
                    instruction=instruction,
                )
                for demo_problem_sample in demo_problem_samples
            ],
            new_problem=SemEvalInstructionSample(
                sample=problem_sample,
                instruction=instruction,
            ),
            labels=labels,
        )

    def build(self, *args, **kwargs):
        model = self.create_model(**kwargs)
        return self.template.render(model=model)


class ZeroShotPrompt(PromptBase):
    def __init__(self):
        super().__init__("zeroshot.prompt")

    def create_model(
        self,
        problem_sample: SemEvalSample,
        labels: SemEvalLabels,
        instructions: dict,
        **kwargs,
    ):
        return ZeroshotModel(
            sample=SemEvalInstructionSample(
                sample=problem_sample,
                instruction=instructions[problem_sample.type],
            ),
            labels=labels,
        )

    def build(self, *args, **kwargs):
        model = self.create_model(**kwargs)
        return self.template.render(model=model)


class ReformulatePrompt(PromptBase):
    def __init__(self):
        super().__init__("reformulation.prompt")

    def create_model(
        self,
        problem_sample: SemEvalSample,
        instruction: str,
        format_instruction: str,
        **kwargs,
    ):
        """
        Please reformulate the statement enclosed in triple quotes, using different words to convey the same meaning. If there are any technical terms, please provide explanations for them
        Instruction: Please reformulate the following statement.
        Reformulate the text delimited by triple quote, express it in different words while maintaining its original meaning, while explaining technical terms when possible:
        Please reformulate the statement enclosed in triple quotes, using different words to convey the same meaning. If there are any technical terms, please provide explanations for them.
        """
        return {
            "model": problem_sample,
            "instruction": instruction,
            "format_instruction": format_instruction,
        }

    def build(self, *args, **kwargs):
        model = self.create_model(**kwargs)
        return self.template.render(
            model=model["model"],
            instruction=model["instruction"],
            format_instruction=model["format_instruction"],
        )


class PromptManager:
    def __init__(
        self,
        instruction: str,
        labels: SemEvalLabels,
    ) -> None:
        self.instruction = instruction
        self.labels = labels

        env = Environment(
            loader=PackageLoader("semeval", package_path="prompt_templates"),
            autoescape=select_autoescape(),
        )

        self.opro_template = env.get_template("opro.prompt")
        self.fewshots_template = env.get_template("fewshots.prompt")
        self.zeroshot_template = env.get_template("zeroshot.prompt")
        self.reformulation_template = env.get_template("reformulation.prompt")
        self.chainofthoughts_template = env.get_template("chainofthought.prompt")

    def build_opro_prompt(
        self,
        instruction_scores: typing.List[InstructionScore],
        problem_samples: typing.List[SemEvalSample],
    ):
        opro_model = LanguageModel(
            instruction_scores=instruction_scores,
            samples=problem_samples,
            labels=self.labels,
        )

        return self.opro_template.render(model=opro_model)

    def build_fewshots_prompt(
        self,
        new_problem_sample: SemEvalSample,
        demo_problem_samples: typing.List[SemEvalSample],
    ):
        fewshots_model = FewshotsModel(
            demo_samples=[
                SemEvalInstructionSample(
                    sample=demo_problem_sample,
                    instruction=self.instruction,
                )
                for demo_problem_sample in demo_problem_samples
            ],
            new_problem=SemEvalInstructionSample(
                sample=new_problem_sample, instruction=self.instruction
            ),
            labels=self.labels,
        )

        return self.fewshots_template.render(model=fewshots_model)

    def build_zeroshot_prompt(
        self,
        problem_sample: SemEvalSample,
    ):
        zeroshot_model = ZeroshotModel(
            sample=SemEvalInstructionSample(
                sample=problem_sample,
                instruction=self.instruction,
            ),
            labels=self.labels,
        )

        return self.zeroshot_template.render(model=zeroshot_model)

    def build_chainofthought_prompt(
        self,
        problem_sample: SemEvalSample,
    ):
        chainofthoughts_model = ChainofThoughtsModel(
            sample=problem_sample,
            labels=self.labels,
        )

        return self.chainofthoughts_template.render(model=chainofthoughts_model)

    def build_reformulation_prompt(
        self,
        problem_sample: SemEvalSample,
    ):
        """
        Instruction: Please reformulate the following statement.
        Reformulate the text delimited by triple quote, express it in different words while maintaining its original meaning, while explaining technical terms when possible:
        Please reformulate the statement enclosed in triple quotes, using different words to convey the same meaning. If there are any technical terms, please provide explanations for them.
        """
        return self.reformulation_template.render(
            model=problem_sample,
            instruction="Please reformulate the statement enclosed in triple quotes, using different words to convey the same meaning. If there are any technical terms, please provide explanations for them.",
        )

    def build_opposite_prompt(
        self,
        problem_sample: SemEvalSample,
    ):
        """
        Instruction: Please negate the following statement.
        Please express the following statement in a negative form.
        """
        return self.reformulation_template.render(
            model=problem_sample,
            instruction="Please express the following statement in a negative form.",
        )

    def build_rephrase_prompt(
        self,
        problem_sample: SemEvalSample,
    ):
        """
        Instruction: Rephrase the statement in a positive form while preserving its original meaning.
        """
        return self.reformulation_template.render(
            model=problem_sample,
            instruction="Rephrase the statement in a positive form while preserving its original meaning.",
        )
