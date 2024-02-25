import re
from pydantic import BaseModel
import json


class OutputParserConfig(BaseModel):
    element_name: str
    format: str

    def build(self):
        return OutputParser(element_name=self.element_name, format=self.format)


class OutputParser:
    json_pattern = r"({[^\{\}]+})"

    def __init__(self, element_name: str, format="json") -> None:
        self.element_name = element_name
        self.format = format

    def parse(self, output: str):
        if self.format == "json":
            return self.parse_json(output)
        elif self.format == "xml":
            return self.parse_xml(output)
        else:
            raise Exception(f"Unknown format: {self.format}")

    def parse_json(self, output: str):
        try:
            result = re.search(self.json_pattern, output)

            if result is None:
                return "N-A"

            group_result = result.group(1)
            json_dict = json.loads(group_result)

            if self.element_name in json_dict:
                return json_dict
            else:
                return "N-A"
        except Exception as error:
            return str(error)

    def parse_xml(self, output: str):
        chunk = re.split(f"<{self.element_name}>", output, flags=re.IGNORECASE)
        if len(chunk) < 2:
            return None

        if "</ins>" not in chunk[1].lower():
            return None

        chunk = re.split(f"</{self.element_name}>", chunk[1], flags=re.IGNORECASE)
        return chunk[0]


import re
from pydantic import BaseModel
import json


class OutputParserConfig(BaseModel):
    element_name: str
    format: str

    def build(self):
        return OutputParser(element_name=self.element_name, format=self.format)


class OutputParser:
    json_pattern = r"({[^\{\}]+})"

    def __init__(self, element_name: str, format="json") -> None:
        self.element_name = element_name
        self.format = format

    def parse(self, output: str):
        if self.format == "json":
            return self.parse_json(output)
        elif self.format == "xml":
            return self.parse_xml(output)
        else:
            raise Exception(f"Unknown format: {self.format}")

    def parse_json(self, output: str):
        try:
            result = re.search(self.json_pattern, output)

            if result is None:
                return "N-A"

            group_result = result.group(1)
            json_dict = json.loads(group_result)

            if self.element_name in json_dict:
                return json_dict
            else:
                return "N-A"
        except Exception as error:
            return str(error)

    def parse_xml(self, output: str):
        chunk = re.split(f"<{self.element_name}>", output, flags=re.IGNORECASE)
        if len(chunk) < 2:
            return None

        if "</ins>" not in chunk[1].lower():
            return None

        chunk = re.split(f"</{self.element_name}>", chunk[1], flags=re.IGNORECASE)
        return chunk[0]
