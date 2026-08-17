from langgraph.graph import START, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from typing_extensions import TypedDict

class State(TypedDict):
    foo: str

# 子图
def subgraph_node_1(state: State):
    print("sub_node_1")
    return {}

def subgraph_node_2(state: State):
    print("sub_node_2")
    # 在子图节点中中断
    value = interrupt("输入值:")
    return {"foo": state["foo"] + value}

subgraph_builder = StateGraph(State)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_node(subgraph_node_2)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph_builder.add_edge("subgraph_node_1", "subgraph_node_2")
subgraph = subgraph_builder.compile()

# 主图1（直接将子图构建为节点）
# builder = StateGraph(State)
# builder.add_node("node_1", subgraph)
# builder.add_edge(START, "node_1")
#
# graph = builder.compile(checkpointer=InMemorySaver())

# 主图2（通过主图节点调通子图）
def node_1(state: State):
    # 只要子图中断再恢复，调用子图的主图节点也会再次执行一次
    print("node1")
    # 调用子图后返回的结果是子图的最终state(foo)，和主图中的state(foo)互不影响
    result = subgraph.invoke({"foo": state["foo"]})
    return {"foo": result["foo"]}

builder = StateGraph(State)
builder.add_node(node_1)
builder.add_edge(START, "node_1")
graph = builder.compile(checkpointer=InMemorySaver())

# 执行主图，设置config
config = {"configurable": {"thread_id": "1"}}
print(graph.invoke({"foo": "hahaha"}, config=config))

# 恢复：
# 1. 接将子图构建为主图节点
# 2. 主图节点调通子图
print(graph.invoke(Command(resume="000"), config=config))