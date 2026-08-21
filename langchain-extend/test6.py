from typing import Any, Callable

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_agent, after_agent, before_model, after_model, wrap_model_call, \
    wrap_tool_call, ModelRequest, ModelResponse
from langchain.tools import tool
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command


@tool
def get_weather_for_location(city: str) -> str:
    """获取指定城市的天气信息。"""
    return f"在{city}总是阳光明媚！"


@before_agent
def log_before_agent(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print("即将执行Agent")
    return None

@after_agent
def log_after_agent(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print("Agent执行完成")
    return None

@before_model
def log_before_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print("即将执行LLM")
    return None

@after_model
def log_after_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print("LLM执行完成")
    return None

@wrap_model_call
def retry_model(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],  # 执行LLM的方法(回调)
) -> ModelResponse:

    for t in range(3):
        print(f"【wrap model】最新消息：{request.messages[-1].content}")
        try:
            result = handler(request)  # 调用llm，可能会出现错误，如网络连接超时
            print("【wrap model】模型调用完成")
            return result
        except Exception as e:
            if t == 2:
                raise
            print(f"【wrap model】模型调用出现错误，将重试{t+1}次/3次，错误信息为：{e}")

@wrap_tool_call
def retry_tool(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],  # 执行工具的方法(回调)
) -> ToolMessage | Command:
    print(f"【wrap tool】执行工具：{request.tool_call['name']}")
    print(f"【wrap tool】参数：{request.tool_call['args']}")
    try:
        result = handler(request)  # 执行工具
        print("【wrap tool】工具调用完成")
        return result
    except Exception as e:
        print(f"【wrap tool】工具调用出现错误：{e}")


# 定义 agent
agent = create_agent(
    model="deepseek-v4-flash",
    tools=[get_weather_for_location],
    system_prompt="你是一位乐于助人的助手。",
    middleware=[log_before_agent, log_after_agent, log_before_model, log_after_model
                , retry_model, retry_tool],
)


response = agent.invoke(
    {"messages": [
        {
            "role": "user",
            "content": "北京的天气如何"}]
    }
)
print(response["messages"][-1].content)