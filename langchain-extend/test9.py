from typing import NotRequired, Any, Callable

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import after_model, wrap_model_call, ModelRequest, ModelResponse, ExtendedModelResponse
from langgraph.runtime import Runtime
from langgraph.types import Command
from langchain.tools import tool


@tool
def get_weather_for_location(city: str) -> str:
    """获取指定城市的天气信息。"""
    return f"在{city}总是阳光明媚！"

class TrackingState(AgentState):
    model_call_count: NotRequired[int]


class UsageTrackingState(AgentState):
    """追踪令牌使用的情况"""
    last_call_tokens: NotRequired[int]


@after_model(state_schema=TrackingState)
def add_counter(state: TrackingState, runtime: Runtime) -> dict[str, Any] | None:
    # 返回的字典就是更新的状态。
    # 类似于LangGraph中讲解的节点的返回类型
    return {"model_call_count": state.get("model_call_count", 0) + 1}


@wrap_model_call(state_schema=UsageTrackingState)
def track_usage(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],  # 执行LLM的方法(回调)
) -> ExtendedModelResponse:
    response = handler(request)   # llm.invoke()
    return ExtendedModelResponse(
        model_response=response,
        command=Command(
            update={"last_call_tokens":
                        response.result[-1].response_metadata['token_usage']
                                                             ['completion_tokens']})
    )

# 定义 agent
agent = create_agent(
    model="deepseek-v4-flash",
    tools=[get_weather_for_location],
    system_prompt="你是一位乐于助人的助手。",
    middleware=[add_counter, track_usage],
)


response = agent.invoke(
    {"messages": [
        {
            "role": "user",
            "content": "北京的天气如何"}]
    }
)
print(f"模型调用次数：{response.get("model_call_count")}")
print(f"最新ai消息消耗token数：{response.get("last_call_tokens")}")