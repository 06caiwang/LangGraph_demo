from dataclasses import dataclass
from typing import TypedDict

from langchain_core.messages import AnyMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore


@dataclass
class ContextSchema:
    user_id: str   # 不要修改
    language: str = "en"

class State(TypedDict):
    messages: list[str]   # 可以修改
    user_name: str    # 可以修改

def node(state: State, runtime: Runtime[ContextSchema]):
    # 静态运行时上下文
    if runtime.context.language == "en":
        greeting = "hello"
    else:
        greeting = "你好"

    # 动态运行时上下文
    user_name = state.get("user_name", "Guest")

    return {
        "messages": [f"{greeting}， {user_name}！"]   # 覆盖
    }

# context 需要添加进图
builder = StateGraph(State, context_schema=ContextSchema)
builder.add_node(node)
builder.add_edge(START, "node")
builder.add_edge("node", END)
graph = builder.compile()

print(graph.invoke(
    {},
    context={"user_id": "1", "language": "en"}
))