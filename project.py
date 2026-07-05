import os
from typing import TypedDict

class pipelinestate(TypedDict):
    raw_input: str
    edited_text: str
    script_text: str
    final_output: str