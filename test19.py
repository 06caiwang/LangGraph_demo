from typing import TypedDict

from langgraph.constants import START
from langgraph.graph import StateGraph


# 子图（需要编译后才能被调用）
class SubState(TypedDict):
    sub_1: str
    sub_2: str    # 输入

def sub_node_1(state: SubState):
    return {"sub_1": "pass sub node 1"}

def sub_node_2(state: SubState):
    return {"sub_2": state["sub_2"] + state["sub_1"]}

sub_builder = StateGraph(SubState)
sub_builder.add_sequence([sub_node_1, sub_node_2])
sub_builder.add_edge(START, "sub_node_1")
sub_graph = sub_builder.compile()
# print(sub_graph.invoke({"sub_2": "hahaha"}))

# 主图
class ParentState(TypedDict):
    parent: str   # 输入

def node_1(state: ParentState):
    return {"parent": "hi! " + state["parent"]}

def node_2(state: ParentState):
    """ 更新parent： 替换为子图中的sub_2参数值 """
    result = sub_graph.invoke({"sub_2": "hahaha"})
    return {"parent": result["sub_2"]}

# def node_3(state: ParentState):
#     """ 希望使用子图中的状态（不调用子图） """
#     # 前面的节点需要将子图状态转换为主图状态，才可以访问



builder = StateGraph(ParentState)
builder.add_sequence([node_1, node_2])
builder.add_edge(START, "node_1")
graph = builder.compile()
# print(graph.invoke({"parent": "小明"}))

# subgraphs=True 支持在流式输出中包含子图的输出
for chunk in graph.stream({"parent": "小明"}, stream_mode="updates", subgraphs=True):
    print(chunk)

print()

for chunk in graph.stream({"parent": "小明"}, stream_mode="values", subgraphs=True):
    print(chunk)