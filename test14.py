from typing import TypedDict

from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START
from langgraph.graph import StateGraph

class State(TypedDict):
    topic: str
    joke: str

model = init_chat_model(
    model="deepseek-v4-pro",
    extra_body={"thinking": {"type": "disabled"}}
)

def generate_topic(state: State):
    """生成joke主题"""
    topic = model.invoke("生成一个搞笑的笑话主题，仅生成5个字以内的主题").content
    return {
        "topic": topic
    }

def generate_joke(state: State):
    """生成joke内容"""
    joke = model.invoke(f"写一个关于{state["topic"]}的笑话").content
    return {
        "joke": joke
    }

builder = StateGraph(State)

builder.add_sequence([generate_topic, generate_joke])

builder.add_edge(START, "generate_topic")

graph = builder.compile(checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "1"}}

# 1. 执行一次工作流，产生历史记录
print(graph.invoke({}, config=config))

# 2. 获取历史记录，找到要修改的状态快照
states = list(graph.get_state_history(config))
print(states)
# states list 顺序：按照时间排序
update = states[1]
# update.values 是当前的state
print(update.values["topic"])
print(update.config)

# 3. 更新状态
# update.config 中包含：线程id、状态快照id
new_config = graph.update_state(update.config, values={"topic": "夏日的趣事"})

# 4. 使用新的状态重放
print(graph.invoke(None, config=new_config))