import operator
from typing import TypedDict, Annotated

from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage, AIMessage, SystemMessage, ToolMessage, HumanMessage
from langchain_tavily import TavilySearch
from langgraph.constants import START, END
from langgraph.graph import StateGraph

search = TavilySearch(max_results=1)
tools = [search]
model = init_chat_model("deepseek-v4-pro", temperature=0)
model_with_tools = model.bind_tools(tools)

# 1. 状态定义
class MessageState(TypedDict):
    # 消息列表 -- 会话记忆，上下文维护
    message: Annotated[list[AnyMessage], operator.add]
    # 调用大模型次数
    llm_call_count: int

# 2. 节点定义
def llm_call(state: MessageState):
    """LLM决定是否调用工具"""
    message = state["message"]
    result = model_with_tools.invoke(
        [
            SystemMessage(content="你是一个搜索引擎，支持工具调用"),
        ]
        + message
    )
    return {
        "message": [result],
        "llm_call_count": state.get("llm_call_count", 0) + 1
    }

# 将工具转化为字典格式
tools_by_name = {tool.name: tool for tool in tools}

def tool_node(state: MessageState):
    """执行工具的调用"""
    result = []
    for tool_call in state["message"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        obs = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=obs, tool_call_id=tool_call["id"]))

    return {
        "message": result
    }

# 3. 定义图
agent_builder = StateGraph(MessageState)
agent_builder.add_node(llm_call)
agent_builder.add_node(tool_node)

# 4. 添加边
agent_builder.add_edge(START, "llm_call")

# 选择边函数定义
def should_continue(state: MessageState):
    last_message = state["message"][-1]
    if last_message.tool_calls:
        return "tool_node"

    return END

agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)

agent_builder.add_edge("tool_node", "llm_call")

# 5. 编译图
agent_search = agent_builder.compile()

# （可选）6.生成图样式
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
#
# try:
#     # 生成 Mermaid 图表并保存为图片
#     mermaid_code = agent_search.get_graph(xray=True).draw_mermaid()
#     # print(mermaid_code)
#     # 保存文件
#     png_data = agent_search.get_graph(xray=True).draw_mermaid_png()
#     with open("jpg/test2.png", "wb") as f:
#         f.write(png_data)
#
#     # 使用 matplotlib 显示图像
#     img = mpimg.imread("jpg/test2.png")
#     plt.imshow(img)  # 显示图片
#     plt.axis('off')  # 关闭坐标轴
#     plt.show()  # 弹出窗口显示图片
# except Exception as e:
#     print(f"An error occurred: {e}")

# 7. 执行图
result = agent_search.invoke({
    "message": [HumanMessage(content="南京今天的天气？")]
    # "message": [HumanMessage(content="你好")]
})
# result 是最终的状态结果
print(f"一共调用了{result['llm_call_count']}次LLM")
for msg in result["message"]:
    msg.pretty_print()
