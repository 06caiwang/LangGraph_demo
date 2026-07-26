import operator
from typing import TypedDict, Annotated

from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import Overwrite

# 历史消息的覆盖
class State(TypedDict):
    message: Annotated[list[str], operator.add]

def add_message(state: State):
    """追加消息"""
    return {
        "message": ["first message"]
    }

def overwrite_message(state: State):
    """覆盖历史消息"""
    # return {
    #     "message": ["second message"]
    # }

    return {
        "message": Overwrite(["overwrite message"])
    }

builder = StateGraph(State)

builder.add_node(add_message)
builder.add_node(overwrite_message)

builder.add_edge(START,"add_message")
builder.add_edge("add_message", "overwrite_message")

graph = builder.compile()

result = graph.invoke(
    {
        "message": [""],
    }
)

print(result)
