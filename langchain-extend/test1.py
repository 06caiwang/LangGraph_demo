from langchain.agents import create_agent
from langchain.tools import tool


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

# 构造agent
agent = create_agent(
    model="deepseek-v4-flash",
    tools=[get_weather],
    system_prompt="你是一个乐于助人的助手"
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "徐州的天气怎么样"}]}
)

print(result["messages"][-1].content)

# HumanMessage(content='北京的天气怎么样', additional_kwargs={}, response_metadata={}, id='9425d869-fa6e-478e-a1b2-6a4b34b1ac37')
# AIMessage(content='', additional_kwargs={'refusal': None, 'reasoning_content': 'The user is asking about the weather in Beijing. I should call the get_weather tool for Beijing.'}, response_metadata={'token_usage': {'completion_tokens': 66, 'prompt_tokens': 359, 'total_tokens': 425, 'completion_tokens_details': {'accepted_prediction_tokens': None, 'audio_tokens': None, 'reasoning_tokens': 21, 'rejected_prediction_tokens': None}, 'prompt_tokens_details': {'audio_tokens': None, 'cache_write_tokens': None, 'cached_tokens': 256}, 'prompt_cache_hit_tokens': 256, 'prompt_cache_miss_tokens': 103}, 'model_provider': 'deepseek', 'model_name': 'deepseek-v4-flash', 'system_fingerprint': 'a26a7955944dc5c60445bff77fac9c8e', 'id': 'e39be8c3-7868-4d59-896b-bc8c657ae0c0', 'finish_reason': 'tool_calls', 'logprobs': None}, id='lc_run--01a00f86-93ce-7c00-a29a-b459612bd57b-0', tool_calls=[{'name': 'get_weather', 'args': {'city': '北京'}, 'id': 'call_00_QEWTUugDtKqaxL4WPBdB6771', 'type': 'tool_call'}], invalid_tool_calls=[], usage_metadata={'input_tokens': 359, 'output_tokens': 66, 'total_tokens': 425, 'input_token_details': {'cache_read': 256}, 'output_token_details': {'reasoning': 21}})
# ToolMessage(content='北京的天气为小雨，10-15℃', name='get_weather', id='2c496629-b9f4-4d20-b70b-3ec9f0950ab5', tool_call_id='call_00_QEWTUugDtKqaxL4WPBdB6771')
# AIMessage(content='北京目前是小雨天气，气温在 **10℃ 到 15℃** 之间。出门记得带伞，注意保暖哦！', additional_kwargs={'refusal': None, 'reasoning_content': ''}, response_metadata={'token_usage': {'completion_tokens': 30, 'prompt_tokens': 446, 'total_tokens': 476, 'completion_tokens_details': {'accepted_prediction_tokens': None, 'audio_tokens': None, 'reasoning_tokens': 0, 'rejected_prediction_tokens': None}, 'prompt_tokens_details': {'audio_tokens': None, 'cache_write_tokens': None, 'cached_tokens': 384}, 'prompt_cache_hit_tokens': 384, 'prompt_cache_miss_tokens': 62}, 'model_provider': 'deepseek', 'model_name': 'deepseek-v4-flash', 'system_fingerprint': 'a26a7955944dc5c60445bff77fac9c8e', 'id': 'dee79221-64fe-4774-a972-620721668fb0', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--01a00f86-9943-7090-8ef9-4d599de0bb84-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 446, 'output_tokens': 30, 'total_tokens': 476, 'input_token_details': {'cache_read': 384}, 'output_token_details': {'reasoning': 0}})]}