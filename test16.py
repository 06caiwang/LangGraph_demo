from dataclasses import dataclass

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.constants import START, END
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolRuntime, ToolNode, tools_condition

class State(MessagesState):
    user_name: str

@dataclass
class Context:
    user_id: str

@tool
def search(runtime: ToolRuntime[Context]):
    """搜索天气的工具"""
    user_id = runtime.context.user_id
    user_name = runtime.state["user_name"]

    print(f"日志记录： user_id:{user_id}, user_name:{user_name}")
    return f"user_id:{user_id}, user_name:{user_name} 查询天气：晴天，15-20度"  # 模拟调用

model_with_tool = init_chat_model("deepseek-v4-flash").bind_tools([search])
def llm_call(state: State):
    return {
        "messages": [
            model_with_tool.invoke([SystemMessage(content="你支持调用工具去查询天气")] + state["messages"])
        ]
        # AIMessage(tool_calls)
    }

builder = StateGraph(State, context_schema=Context)
builder.add_node(llm_call)   # 决定是否调用工具
builder.add_node("tool_node", ToolNode([search]))   # 执行工具

builder.add_edge(START, "llm_call")
builder.add_conditional_edges(
    "llm_call",
    tools_condition,
    {
        "tools": "tool_node",
        "__end__": END
    }
)

# H、A(t)、T、A
builder.add_edge("tool_node", "llm_call")

graph = builder.compile()

for chunk in graph.stream(
    {
        "messages": [HumanMessage(content="今天西安天气如何")],
        "user_name": "小明"
    },
    context={"user_id": "1"}
):
    for node, update in chunk.items():
        print(f"节点：{node}更新的最新消息如下")
        update["messages"][-1].pretty_print()
