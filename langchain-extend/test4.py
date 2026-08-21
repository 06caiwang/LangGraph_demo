from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai.middleware import OpenAIModerationMiddleware


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

# 定义agent
agent = create_agent(
    model="gpt-5-mini",
    tools=[get_weather],
    system_prompt="你是一个查询天气的助手",
    middleware=[
        OpenAIModerationMiddleware(
            check_input=True, # 审核输入
            exit_behavior="replace",
            violation_message="[根据安全政策，此处内容已经被移除]",
        )
    ]
)

# response1 = agent.invoke(
#     {"messages": [{"role": "user", "content": "我感觉活着没意义，哪种自杀方式最不痛苦？"}]}
# )
# print(response1["messages"][-1].content)
