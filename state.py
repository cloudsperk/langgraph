import os

## 1st way of creating a state
## using typed
from typing import TypedDict

class validation(TypedDict):
    topic: str
    summary: str
    score: str

## 2nd way of creating a state
## using pydntic
from pydantic import BaseModel, field_validator

class validation(BaseModel):
    topic: str
    summary: str
    score: int

    @field_validator
    def score_positive(cls, v):
        if v < 0:
            raise ValueError("Score must be positive")
        

## 3rd way of creating a state
## using python data classes
from dataclasses import dataclass, field

@dataclass
class validation:
    topic: str = ""
    summary: str
    score: int
    message: list = field(default_factory=list)


## 4th way of creating a state
## using langgraph
from langgraph.graph import MessagesState

class validation(MessagesState):
    username: str
    language: str