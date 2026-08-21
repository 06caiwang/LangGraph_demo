from typing import Any

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import before_agent, after_agent, before_model, after_model
from langchain.tools import tool
from langgraph.runtime import Runtime


# 定义工具
@tool
def get_weather(city: str) -> str:
    """获取城市的天气"""
    if city == "上海":
        return f"{city}的天气为晴天，25-30℃"
    elif city == "北京":
        return f"{city}的天气为小雨，10-15℃"
    elif city == "徐州":
        return f"{city}的天气为雨夹雪，-4~-2℃"
    else:
        return f"暂未获取到改{city}的天气"

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

agent = create_agent(
    model="deepseek-v4-flash",
    tools=[get_weather],
    system_prompt="你是一个查询天气的助手",
    middleware=[log_before_agent, log_after_agent, log_before_model, log_after_model],
)

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "徐州的天气怎么样？"
            }
        ]
    }
)

print(response["messages"][-1].content)