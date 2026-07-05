import os
from typing import TypedDict

class pipelinestate(TypedDict):
    raw_input: str
    edited_text: str
    script_text: str
    final_output: str

from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama3.3-70b-versatile", temperature=0.7)

def editor_node(state: pipelinestate) -> dict:
    """Stage 1 - Cleans up grammer, removes typos and refine the tone"""

    prompt = (
        "you are an expert copyeditor, clean up the following raw text"
        "fix any gramatical error, spelling mistake and smooth ouut the transition flow"
        "while keeping the core message intact, return only the edited text.\n\n"
        f"Text:\n{state['raw_input']}"
    )

    response = llm.invoke(prompt)

    return {"edited_text": response.content.strip()}