from typing import Callable

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """获取天气的工具"""

    return f"在{city}总是阳光明媚！"

model_1 = init_chat_model(model="deepseek-v4-flash")
model_2 = init_chat_model(model="deepseek-v4-pro")

@wrap_model_call
def dynamic_model_selection(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """根据对话的复杂程度进行模型的选择"""
    count = len(request.state["messages"])
    if count > 1:
        final_model = model_1
    else:
        final_model = model_2

    return handler(request.override(model=final_model))

agent = create_agent(
    model=model_1,
    tools=[get_weather],
    system_prompt="你是一位乐于助人的助手。",
    middleware=[dynamic_model_selection],
)

response = agent.invoke(
    {"messages": [
        {
            "role": "user",
            "content": "北京的天气如何"}]
    }
)

print(response)

# HumanMessage(content='北京的天气如何', additional_kwargs={}, response_metadata={}, id='8725cbec-82fb-4955-b9f2-0f92d695f605'),

# AIMessage(     model_name': 'deepseek-v4-pro',      'system_fingerprint': 'a307abda487cd1b463329ccb945ce396', 'id': 'f5263dce-158d-42c4-8be9-c335d3c334f9', 'finish_reason': 'tool_calls', 'logprobs': None}, id='lc_run--01a02f19-c417-7210-a755-95dd98e0eb42-0', tool_calls=[{'name': 'get_weather', 'args': {'city': '北京'}, 'id': 'call_00_jxIzuVT8GhlWICvrakWh1623', 'type': 'tool_call'}], invalid_tool_calls=[], usage_metadata={'input_tokens': 359, 'output_tokens': 62, 'total_tokens': 421, 'input_token_details': {'cache_read': 256}, 'output_token_details': {'reasoning': 17}}),

# ToolMessage(content='在北京总是阳光明媚！', name='get_weather', id='2ce1b12d-33c4-438b-a18f-1ea462d50a26', tool_call_id='call_00_jxIzuVT8GhlWICvrakWh1623'),

# AIMessage(     model_name': 'deepseek-v4-flash',     'system_fingerprint': 'a26a7955944dc5c60445bff77fac9c8e', 'id': 'bd88b545-0653-437b-a88b-e4f9a7aea0b0', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--01a02f19-ca0b-7672-93d7-a80440a5052c-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 438, 'output_tokens': 27, 'total_tokens': 465, 'input_token_details': {'cache_read': 256}, 'output_token_details': {'reasoning': 0}})
