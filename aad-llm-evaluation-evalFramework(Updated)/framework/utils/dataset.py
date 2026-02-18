from openai import OpenAI
from typing import Optional
import os
import json
import random


class Dataset:
    @staticmethod
    def get_client(api_key: Optional[str] = None):
        if api_key:
            return OpenAI(api_key=api_key)
        return OpenAI()  
 
    def __init__(self, api_key=None):
        self.client = Dataset.get_client(api_key)

    def _read_lines(self, file_input: str):
        # if os.path.exists(file_input):
        #     with open(file_input, "r", encoding="utf-8") as f:
        #         return f.readlines()

        response = self.client.files.content(file_input)
        content = response.read().decode("utf-8")
        return content.splitlines()

    
    def validate(self, file_input: str, required_keys=None):
        if required_keys is None:
            required_keys = []

        lines = self._read_lines(file_input)

        for line_num, line in enumerate(lines, 1):
            try:
                record = json.loads(line)
            except Exception:
                raise ValueError(f"Invalid JSON at line {line_num}: {line}")

            for key in required_keys:
                value = record
                for part in key.split("."):
                    if part not in value:
                        raise ValueError(
                            f"Missing required key '{key}' at line {line_num}: {record}"
                        )
                    value = value[part]

        return True


    def preview(self, file_input: str, n: int = 5):
        lines = self._read_lines(file_input)
        return [json.loads(line) for line in lines[:n]]

    
    def count(self, file_input: str):
        lines = self._read_lines(file_input)
        return len(lines)

    
    def load(self, file_input: str):
        lines = self._read_lines(file_input)
        return [json.loads(line) for line in lines]

    def get_schema(self, file_input: str, n: int = 20):
        lines = self._read_lines(file_input)
        schema = set()

        for line in lines[:n]:
            record = json.loads(line)

            def explore(node, prefix=""):
                if isinstance(node, dict):
                    for k, v in node.items():
                        path = f"{prefix}.{k}" if prefix else k
                        schema.add(path)
                        explore(v, path)

            explore(record)

        return sorted(schema)
    def get_stats(self, file_input: str):
        lines = self._read_lines(file_input)
        return {
            "total_lines": len(lines),
            "first_line_length": len(lines[0]) if lines else 0,
            "avg_line_length": sum(len(l) for l in lines) // len(lines) if lines else 0,
        }

    
    def shuffle(self, file_input: str):
        records = self.load(file_input)
        random.shuffle(records)
        return records

    def check_duplicates(self, file_input: str):
        lines = self._read_lines(file_input)
        seen = set()
        duplicates = []

        for line in lines:
            if line in seen:
                duplicates.append(json.loads(line))
            else:
                seen.add(line)

        return duplicates
