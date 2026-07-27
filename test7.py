from typing import TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.constants import START
from langgraph.graph import StateGraph

model = init_chat_model("deepseek-v4-pro")

class InputState(TypedDict):
    topic: str

class OutputState(TypedDict):
    final_content: str

class State(InputState, OutputState):
    outline: str
    draft: str
    polished_draft: str

PROMPT_1 = (
    "根据主题生成文章大纲。\n"
    "主题：{topic}\n"
    "要求："
    "1.只需两个最核心标题"
    "2.不用进行说明，只返回最终大纲"
)

def generate_outline(state: State):
    """根据主题生成大纲"""
    print("*" * 50)
    print(f"内容大纲生成中...\n")
    topic = state["topic"]
    prompt = PROMPT_1.format(topic=topic)
    result = model.invoke([HumanMessage(prompt)])
    outline = result.content
    print(f"大纲已生成：\n{outline}\n")

    return {
        "outline": outline
    }

PROMPT_2 = (
    "根据以下内容生成文章完整初稿。\n"
    "主题：{topic}\n"
    "大纲: "
    "{outline}\n"
    "要求："
    "1.每个标题下，最多使用三句话的内容即可"
    "2.不用进行说明，只返回最终结果"
)

def generate_draft(state: State):
    """根据大纲生成稿子"""
    print("*" * 50)
    print(f"初稿生成中...\n")
    topic = state["topic"]
    outline = state["outline"]
    prompt = PROMPT_2.format(topic=topic, outline=outline)
    result = model.invoke([HumanMessage(prompt)])
    draft = result.content
    print(f"初稿已生成：\n{draft}\n")

    return {
        "draft": draft
    }

PROMPT_3 = (
    "根据文章初稿进行润色。\n"
    "主题：{topic}\n"
    "初稿: "
    "{draft}\n"
    "要求："
    "1.润色后，文章不能太长"
)

def generate_polished_draft(state: State):
    """根据稿子进行润色"""
    print("*" * 50)
    print(f"初稿润色中...\n")
    prompt = PROMPT_3.format(
        topic=state["topic"],
        draft=state["draft"]
    )
    result = model.invoke([HumanMessage(prompt)])
    print(f"初稿润色完成：\n{result.content}\n")

    return {
        "polished_draft": result.content
    }

PROMPT_4 = (
    "根据润色版文章，生成文章终稿。\n"
    "主题：{topic}\n"
    "大纲: "
    "{outline}\n"
    "润色版文章: "
    "{polished_draft}\n"
)

def generate_final_article(state: State):
    """根据润色后的文章进行定稿"""
    print("*" * 50)
    print(f"终稿生成中...\n")
    prompt = PROMPT_4.format(
        topic=state["topic"],
        outline=state["outline"],
        polished_draft = state["polished_draft"]
    )
    result = model.invoke([HumanMessage(prompt)])

    return {
        "final_article": result.content
    }

builder = StateGraph(State)

nodes = [generate_outline, generate_draft, generate_polished_draft, generate_final_article]
builder.add_sequence(nodes)

builder.add_edge(START, "generate_outline")

graph = builder.compile()

result = graph.invoke({"topic": "人工智能的未来发展"})
print(result)