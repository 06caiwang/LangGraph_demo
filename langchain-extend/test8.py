from typing import Any, Callable

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_agent, after_agent, before_model, after_model, wrap_model_call, \
    wrap_tool_call, ModelRequest, ModelResponse, AgentMiddleware
from langchain.tools import tool
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command


@tool
def get_weather_for_location(city: str) -> str:
    """获取指定城市的天气信息。"""
    return f"在{city}总是阳光明媚！"


class LoggingMiddleware(AgentMiddleware):

    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print("【1】即将执行Agent")
        return None

    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print("【1】Agent执行完成")
        return None


    def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],  # 执行LLM的方法(回调)
    ) -> ModelResponse:

        for t in range(3):
            print(f"【1】【wrap model】最新消息：{request.messages[-1].content}")
            try:
                result = handler(request)  # 调用llm，可能会出现错误，如网络连接超时
                print("【1】【wrap model】模型调用完成")
                return result
            except Exception as e:
                if t == 2:
                    raise
                print(f"【1】【wrap model】模型调用出现错误，将重试{t+1}次/3次，错误信息为：{e}")



class Logging2Middleware(AgentMiddleware):

    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print("【2】即将执行Agent")
        return None

    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print("【2】Agent执行完成")
        return None


    def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],  # 执行LLM的方法(回调)
    ) -> ModelResponse:

        for t in range(3):
            print(f"【2】【wrap model】最新消息：{request.messages[-1].content}")
            try:
                result = handler(request)  # 调用llm，可能会出现错误，如网络连接超时
                print("【2】【wrap model】模型调用完成")
                return result
            except Exception as e:
                if t == 2:
                    raise
                print(f"【2】【wrap model】模型调用出现错误，将重试{t+1}次/3次，错误信息为：{e}")


# 定义 agent
agent = create_agent(
    model="deepseek-v4-flash",
    tools=[get_weather_for_location],
    system_prompt="你是一位乐于助人的助手。",
    middleware=[LoggingMiddleware(), Logging2Middleware()],
)


response = agent.invoke(
    {"messages": [
        {
            "role": "user",
            "content": "北京的天气如何"}]
    }
)
print(response["messages"])