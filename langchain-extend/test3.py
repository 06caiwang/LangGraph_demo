from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver


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

agent = create_agent(
    model="deepseek-v4-flash",
    tools=[get_weather],
    system_prompt="你是一位乐于助人的助手。",
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model="deepseek-v4-pro",       # 只负责生成摘要
            trigger=("tokens", 30),  # 根据token进行触发
            # trigger=("messages", 100),
            # trigger=("fraction", 0.8),   # 超出模型的token上限的80%将会触发总结摘要
            # trigger=[                  # 满足其一即可
            #     ("tokens", 4000),
            #     ("messages", 100)
            # ],
            keep=("messages", 3),   # keep只能指定一种策略
        ),
    ]
)

config = {"configurable": {"thread_id": "11"}}
response = agent.invoke(
    {"messages": [{"role": "user", "content": "北京的天气怎么样？"}]},
    config=config,
)
print("第一轮回复：", response["messages"][-1].content)
print(response["messages"])