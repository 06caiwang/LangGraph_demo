from typing import TypedDict

from langgraph.constants import START, END
from langgraph.graph import StateGraph

class InputState(TypedDict):
    question: str

class OutputState(TypedDict):
    answer: str

class State(InputState, OutputState):
    string: str

def node(state: State):
    """通过问题生成答案"""
    state["string"] = "这个问题的答案"

    return {
        "question": state["question"],
        "answer": f"{state['question']} -> {state['string']}"
    }

builder = StateGraph(
    State,
    input_schema=InputState,    # 输入验证
    output_schema=OutputState)  # 输出过滤

builder.add_node(node)
builder.add_edge(START, "node")
builder.add_edge("node", END)

graph = builder.compile()
result = graph.invoke({
    "question": "i am a question"
})

print(result)