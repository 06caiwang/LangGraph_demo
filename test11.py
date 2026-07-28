import operator
from typing import TypedDict, Annotated

from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage, AIMessage, SystemMessage, ToolMessage, HumanMessage
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph

search = TavilySearch(max_results=1)
tools = [search]
model = init_chat_model(
    model="deepseek-v4-pro",
    temperature=0,
    extra_body={"thinking": {"type": "disabled"}}
)
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
            SystemMessage(
                content="你是一个搜索引擎，支持工具调用，每次询问，如果需要工具，"
                        "只需要一次调用工具即可，遇到结果直接返回即可，不必再次调用结果"
            ),
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

# 内存记忆
# checkpointer = InMemorySaver()

# 5. 编译图
# agent_search = agent_builder.compile(checkpointer=checkpointer)

# 线程配置
# config = {"configurable": {"thread_id": "123"}}

DB_URI = "postgresql://postgres:123456@127.0.0.1:5432/postgres"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()

    agent_search = agent_builder.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "123456"}}

    result = agent_search.invoke(
        {"message": [HumanMessage(
            content="查询今天南京的天气，不必查未来几日的，简单的概括即可，同时不用每个地区的，南京整个地方大概的天气即可")]},
        config=config
    )
    # result["message"][-1].pretty_print()

    # 第二次执行
    result = agent_search.invoke(
        {"message": [HumanMessage(content="刚才我们聊了什么？")]},
        config=config)

    for r in result["message"]:
        r.pretty_print()
