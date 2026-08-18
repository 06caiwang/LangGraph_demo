from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_openai import ChatOpenAI
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import InMemorySaver


# 定义上下文
@dataclass
class Context:
    """自定义运行时上下文schema"""
    user_id: str

@dataclass
class ResponseFormat:
    """agent的相应格式"""
    # 一个诙谐的回答，必要的
    punny_response: str
    # 如果有关于天气的任何有趣的信息的话
    weather_conditions: str | None = None

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

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """根据上下文中的user_id获取用户的位置"""
    # 获取用户id
    user_id = runtime.context.user_id
    # 获取存储记忆
    memory_store = runtime.store

    # 在工具中使用store
    memory_store.put(
        ("users",),
        user_id,
        {"name": f"name_{user_id}"}
    )

    user_info = memory_store.get(("users",), user_id)
    print(f"user_name:{user_info.value.get('name')}")

    return "北京" if user_id == "1" else "徐州"

model = ChatOpenAI(
    model="qwen-plus",
    api_key="your api key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0,
)

# 定义系统提示词
SYSTEM_PROMPT = """你是一位擅长用双关语表达的天气预报专家。
你拥有以下两种工具的使用权：
- get_weather：使用此功能可获取特定地点的天气情况
- get_user_location：使用此功能可获取用户的当前位置
如果用户向您询问天气情况，请务必先确认其所在位置。如果从问题中不能推断出他们指的是其所在的具体地点，那么请使用“get_user_location”工具来获取他们的位置信息。"""

agent = create_agent(
    model=model,
    name="weather_agent",
    system_prompt=SYSTEM_PROMPT,
    tools=[get_weather, get_user_location],
    context_schema=Context,
    response_format=ResponseFormat,
    checkpointer=InMemorySaver(),
    store=InMemoryStore(),
)

# config = {"configurable": {"thread_id": "11"}}
# response = agent.invoke(
#     {"messages": [{"role": "user", "content": "我这里外面的天气怎么样？"}]},
#     config=config,
#     context=Context(user_id="2"),
# )
# print(response["messages"][-1].content)

config = {"configurable": {"thread_id": "11"}}
response = agent.invoke(
    {"messages": [{"role": "user", "content": "我这里外面的天气怎么样？"}]},
    config=config,
    context=Context(user_id="2"),
)
print(response['structured_response'])

response2 = agent.invoke(
    {"messages": [{"role": "user", "content": "谢谢！"}]},
    config=config,
    context=Context(user_id="2"),
)
print(response2['structured_response'])

