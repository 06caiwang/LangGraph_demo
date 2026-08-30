# 动态工具
# 1. 运行时，根据条件筛选已存在的工具
# Agent[tool1, tool2, tool3] -> [tool2]
# 2. 运行时，新增工具
# Agent[tool1]  ->  [tool1, tool2]

from typing import Callable
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse, wrap_tool_call
from langchain.tools import tool

@tool
def public_search(query: str) -> str:
    """公开搜索：无需认证即可使用，返回基础信息。"""
    print(f"[公开搜索结果] 关于 '{query}' 的基础信息：这是公开可获取的内容。")
    return f"[公开搜索结果] 关于 '{query}' 的基础信息：这是公开可获取的内容。"

@tool
def private_search(query: str) -> str:
    """私有搜索：仅已认证用户可用，返回敏感或个性化数据。"""
    print(f"[私有搜索结果] 关于 '{query}' 的私密数据：仅限认证用户查看。")
    return f"[私有搜索结果] 关于 '{query}' 的私密数据：仅限认证用户查看。"

@tool
def advanced_search(query: str) -> str:
    """高级搜索：提供深度分析。"""
    print(f"[高级搜索结果] 关于 '{query}' 的深度分析报告：包含详细统计和趋势。")
    return f"[高级搜索结果] 关于 '{query}' 的深度分析报告：包含详细统计和趋势。"

class State(AgentState):
    auth: bool

@wrap_model_call(state_schema=State)
def state_based_tools(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """根据条件筛选已经存在的工具，例如根据状态筛选"""
    state = request.state
    is_auth = state.get("auth", False)

    # 未认证用户只能使用public_的工具
    if not is_auth:
        tools = [t for t in request.tools if t.name.startswith("public_")]
        request = request.override(tools=tools)
    else:
        tools = [t for t in request.tools if t.name.startswith("private_")]
        request = request.override(tools=tools)

    return handler(request)

agent = create_agent(
    model="deepseek-v4-flash",
    tools=[public_search, private_search, advanced_search],
    system_prompt="你是一位乐于助人的客服助手。根据用户的问题，选择合适的工具来提供答案",
    middleware=[state_based_tools]
)

response = agent.invoke(
    {
        "messages": [{
            "role": "user",
            "content": "北京的天气如何"
        }],
        "auth": False,
    }
)
print(response)