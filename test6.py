from typing import TypedDict

from langgraph.constants import START
from langgraph.graph import StateGraph

class State(TypedDict):
    result: str

class Node1OutputState(TypedDict):
    # 隐私数据
    sensitive_data: int

class Node2InputState(TypedDict):
    # 隐私数据
    sensitive_data: int

def node_1(state: State) -> Node1OutputState:
    """1. 获取隐私数据"""
    print("node1: 获取隐私数据 xxx")

    return {
        "sensitive_data": 1
    }

def node_2(state: Node2InputState) -> State:
    """2. 拿到隐私数据进行处理"""
    print("node2: 处理隐私数据")
    state["sensitive_data"] += 1

    return {
        "result": f"{state['sensitive_data']}"
    }

def node_3(state: State):
    """3. 构造返回结果"""
    print("node3: 构造返回结果")

    return {
        "result": f"对结果 {state['result']} 进行包装"
    }

builder = StateGraph(State)

builder.add_sequence([node_1, node_2, node_3])

builder.add_edge(START, "node_1")

graph = builder.compile()

result = graph.invoke({
    "result": ""
})

print(result)
