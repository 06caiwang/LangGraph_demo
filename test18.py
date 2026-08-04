import time
from dataclasses import dataclass

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.config import get_stream_writer
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
    # 流式写入器
    writer = get_stream_writer()
    user_id = runtime.context.user_id
    user_name = runtime.state["user_name"]

    writer(
        {
            "type": "search_tool",
            "status": "start",
            "user_id": user_id,
            "user_name": user_name,
        }
    )

    search_steps = [
        {"name": "搜索1", "time": 1, "result": "晴天，"},
        {"name": "搜索2", "time": 2, "result": "15-20度"}
    ]

    all_result = "查询天气："
    for i, step in enumerate(search_steps, 1):
        writer(
            {
                "type": "search_tool",
                "status": "searching",
                "cur_step": i,
                "all_step": len(search_steps),
                "step": step["name"],
                "user_id": user_id,
                "user_name": user_name,
            }
        )
        time.sleep(step["time"])
        all_result += step["result"]

        writer(
            {
                "type": "search_tool",
                "status": "end",
                "user_id": user_id,
                "user_name": user_name,
                "result": all_result
            }
        )
    return all_result  # 模拟调用

model_with_tool = init_chat_model(
    model="deepseek-v4-flash",
    extra_body={"thinking": {"type": "disabled"}}
).bind_tools([search])

def llm_call(state: State):
    writer = get_stream_writer()
    writer(
        {
            "type": "llm_call",
            "status": "start",
            "message": "开始调用LLM",
            "content": state["messages"][-1].content
        }
    )

    result = model_with_tool.invoke([SystemMessage(content="你支持调用工具去查询天气")] + state["messages"])

    writer(
        {
            "type": "llm_call",
            "status": "end",
            "message": "调用LLM结束",
        }
    )
    return {
        "messages": [result]
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

print("【开始思考】\n图开始执行：")
for chunk in graph.stream(
    {
        "messages": [HumanMessage(content="徐州今天的天气")],
        "user_name": "小明"
    },
    context={"user_id": "1"},
    stream_mode=["custom", "values"]
):
    # print(chunk)
    if chunk[0] == "custom":
        info = chunk[-1]
        if info.get("type") == "search_tool":
            status = info.get("status")
            if status == "start":
                print(f"用户id:{info["user_id"]}, 用户名称：{info["user_name"]}开始调用工具...")
            elif status == "searching":
                print(f"[{info["cur_step"]}/{info["all_step"]}] 正在处理：{info["step"]}")   # [1/2] 正在处理：步骤1
            elif status == "end":
                print(f"搜索完成！结果：{info["result"]}")
        elif info.get("type") == "llm_call":
            pass
    elif chunk[0] == "values":
        info = chunk[-1]
        # 最后AIMessage：不包含tool_calls
        if isinstance(info["messages"][-1], AIMessage) and not info["messages"][-1].tool_calls:
            print("【最终结果】")
            print(info["messages"][-1].content)