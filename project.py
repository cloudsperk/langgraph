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

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

def editor_node(state: pipelinestate) -> dict:
    """Stage 1 - Cleans up grammer, removes typos and refine the tone"""
    print("\n---[Stage-1] Executing editor node---")

    prompt = (
        "you are an expert copyeditor, clean up the following raw text"
        "fix any gramatical error, spelling mistake and smooth ouut the transition flow"
        "while keeping the core message intact, return only the edited text.\n\n"
        f"Text:\n{state['raw_input']}"
    )

    response = llm.invoke(prompt)

    return {"edited_text": response.content.strip()}


def scriptwriter_node(state: pipelinestate) -> dict:
    """Stage 2 - format the clean text in to engaging video script style"""
    print("\n---[Stage-2] Executing scriptwriter node---")

    prompt = (
        "you are a cherismatic youtube content creator, take this edited text and transform"
        "it into a highly engaging, punchy, conversational video script hook. Make it sound"
        "like a real person speaking pasionately. return only the script content\n\n"
        f"Edited Text:\n{state['edited_text']}"
    )

    response = llm.invoke(prompt)

    return {"script_text": response.content.strip()}


def translator_node(state: pipelinestate) -> dict:
    """Stage 3 - Translate the script into natural flowing hinglish"""
    print("\n---[Stage-3] Executing translator node---")

    prompt = (
        "You are an expert content localizer for the Indian market. Take the following script"
        "and convert it into natural, flowing 'Hinglish'. Do not simply translate it sentence by sentence,"
        "or repeat information. Alternate comfortably between Hindi and English phrases like"
        "an intellectual tech educator would speak naturally on a live stream. Keep the energy and clarity."
        "Return only the final Hinglish text.\n\n"
        f"Script:\n{state['script_text']}"
    )

    response = llm.invoke(prompt)

    return {"final_output": response.content.strip()}


from langgraph.graph import StateGraph, START, END

# create graph
graph = StateGraph(pipelinestate)
#add nodes to graph
graph.add_node("editor",editor_node)
graph.add_node("scriptwriter",scriptwriter_node)
graph.add_node("translator",translator_node)

# Format edges
graph.add_edge(START,"editor")
graph.add_edge("editor","scriptwriter")
graph.add_edge("scriptwriter","translator")
graph.add_edge("translator",END)

# Compile the graph
app = graph.compile()

result = app.invoke({
    "raw_input": "In the deep green forest, a curious fox padded silently between the tall pine trees as morning mist curled around the roots. Nearby, a wise owl watched from a branch, while a small deer drank from a crystal-clear stream. The forest hummed with life—birds chirped, leaves rustled, and sunlight filtered through the canopy like golden threads. Every creature moved with quiet purpose, as if the whole woodland were connected by an invisible rhythm. The fox paused, listening to the ancient song of the trees, and for a moment, the forest felt alive with mystery and wonder."
})

print("Here is the Final Result --\n\n")
print(result['final_output'])