import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

load_dotenv()
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

def merge_score_dicts(existing_dict:dict, updated_dict:dict) -> dict:
    if existing_dict is None:
        return updated_dict
    return{**existing_dict, **updated_dict}

class analyzerState(TypedDict):
    raw_text: str
    safety_score: Annotated[dict[str,int], merge_score_dicts]

# Nodes
def toxicity_node(state: analyzerState) -> dict:
    print("\n[Node 1] Analysing toxicity and hate speech")
    prompt = f"""You are a toxicity evaluation assistant. Given the text below, assign a toxicity score from 1 to 100.
Scoring guide:
- 1-20: harmless or neutral
- 21-50: mildly rude, sarcastic, or offensive
- 51-80: clearly toxic, insulting, or aggressive
- 81-100: extremely toxic, hateful, threatening, or harmful
Return only a single integer between 1 and 100.

Text:
{state['raw_text']}"""
    response = llm.invoke(prompt)
    try:
        score = int(response.content.strip())
    except ValueError:
        score=0
    return {"safety_score":{"toxicity_level": score}}

def copyright_node(state: analyzerState) -> dict:
    print("\n[Node 2] Analysing copyright risk")
    prompt = f"""You are a copyright risk evaluation assistant. Given the text below, assign a copyright risk score from 1 to 100.
Scoring guide:
- 1-20: likely original or low risk
- 21-50: somewhat similar to existing content
- 51-80: strong signs of copied or paraphrased material
- 81-100: very high likelihood of copyright infringement
Return only a single integer between 1 and 100.

Text:
{state['raw_text']}"""
    response = llm.invoke(prompt)
    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0
    return {"safety_score": {"copyright_risk": score}}


def cultural_node(state: analyzerState) -> dict:
    print("\n[Node 3] Analysing cultural and regional sensitivity")
    prompt = f"""You are a cultural and regional sensitivity evaluation assistant. Given the text below, assign a cultural sensitivity score from 1 to 100.
Scoring guide:
- 1-20: culturally neutral or low sensitivity
- 21-50: mildly sensitive or potentially region-specific
- 51-80: clearly culturally sensitive or potentially offensive in some regions
- 81-100: highly sensitive, offensive, or inappropriate across many cultures
Return only a single integer between 1 and 100.

Text:
{state['raw_text']}"""
    response = llm.invoke(prompt)
    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0
    return {"safety_score": {"cultural_sensitivity": score}}

# Creating the Graph
mygraph = StateGraph(analyzerState)

# Add nodes to myGraph
mygraph.add_node("toxicity_node",toxicity_node)
mygraph.add_node("copyright_node",copyright_node)
mygraph.add_node("cultural_node",cultural_node)

# Format edges
mygraph.add_edge(START,"toxicity_node")
mygraph.add_edge(START,"copyright_node")
mygraph.add_edge(START,"cultural_node")

mygraph.add_edge("toxicity_node",END)
mygraph.add_edge("copyright_node",END)
mygraph.add_edge("cultural_node",END)

# Compile the graph
app = mygraph.compile()

custom_comment="This is our land and they do not deserve peace."
initial_state = {
    "raw_text": custom_comment,
    "safety_score": {}
}

# Invoke the application
result = app.invoke(initial_state)
print(result['safety_score'])