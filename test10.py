import operator
from typing import TypedDict, Annotated

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import Send
from pydantic import BaseModel

class Section(BaseModel):
    name: str
    description: str

class Sections(BaseModel):
    sections: list[Section]

class State(TypedDict):
    topic: str
    section: Section
    sections: list
    completed_sections: Annotated[list, operator.add]
    final_report: str

model = init_chat_model(
    model="deepseek-v4-pro",
    extra_body={"thinking": {"type": "disabled"}}
)
planner = model.with_structured_output(Sections)

def orchestrator(state: State):
    """协调者：分析任务并制定计划"""
    print("协调者进行任务拆分...")

    result = planner.invoke(
        [
            HumanMessage(content=f"为主题{state['topic']}制定报告大纲，要包含两个章节")
        ]
    )

    return {
        "sections": result.sections
    }

def worker_1(state: State):
    """工作者：根据分配的章节生成内容"""

    print("工作者正在生成内容...")
    section = state["section"]
    result = model.invoke(
        [
            HumanMessage(
                content=f"编写报告章节：{section.name}, 内容要求：{section.description}"
            )
        ]
    )

    return {
        "completed_sections": [result.content]
    }

def synthesizer(state: State):
    """汇总工作者的结果"""
    print("正在汇总内容...")
    completed_sections = state["completed_sections"]
    final_report = "\n\n ---- \n\n".join(completed_sections)

    return {
        "final_report": final_report
    }

builder = StateGraph(State)

builder.add_node(orchestrator)
builder.add_node(worker_1)
builder.add_node(synthesizer)

builder.add_edge(START, "orchestrator")

def assign_workers(state: State):
    """为每个任务分配工作者"""
    worker_tasks = []
    for section in state["sections"]:
        worker_tasks.append(
            Send("worker_1", {"section": section})
        )

    return worker_tasks

builder.add_conditional_edges(
    "orchestrator",
    assign_workers,
)

builder.add_edge("worker_1", "synthesizer")
builder.add_edge("synthesizer", END)

graph = builder.compile()

print(graph.invoke({
    "topic": "中国近代史"
}))
