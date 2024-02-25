from typing import List
from pydantic import BaseModel


class InstructionScore(BaseModel):
    instruction: str
    score: float

    def __str__(self):
        return f"instruction: {self.instruction}\nscore: {self.score}"


class SemEvalSample(BaseModel):
    statement: str
    primary_ct_section: List[str]
    secondary_ct_section: List[str]
    label: str
    key: str
    type: str
    section: str

    @property
    def primary_ct_section_str(self):
        return _preprocess_section(self.section, self.primary_ct_section)

    @property
    def secondary_ct_section_str(self):
        return _preprocess_section(self.section, self.secondary_ct_section)


class SemEvalSampleOneShot(SemEvalSample):
    oneshot_prompt: str


class SemEvalInstructionSample(BaseModel):
    sample: SemEvalSample
    instruction: str

    @property
    def primary_ct_section_str(self):
        return _preprocess_section(self.sample.section, self.sample.primary_ct_section)

    @property
    def secondary_ct_section_str(self):
        return _preprocess_section(
            self.sample.section, self.sample.secondary_ct_section
        )


def _preprocess_section(section_name, section_lines):
    if section_name not in ["Intervention", "Adverse Events", "Results"]:
        return "\n".join(section_lines)

    adverse_event_label = "Adverse Events "
    result_label = "Results "
    intervention_label = "INTERVENTION "

    lines = []
    for line in section_lines:
        if line[0] == " ":
            lines.append(line)
        elif adverse_event_label in line:
            lines.append(
                "Adverse Events in cohort " + line.replace(adverse_event_label, "")
            )
        elif result_label in line:
            lines.append("Results in cohort " + line.replace(result_label, ""))
        elif intervention_label in line:
            lines.append(
                "INTERVENTION in cohort " + line.replace(intervention_label, "")
            )
        else:
            lines.append(line)

    return "\n".join(lines)
