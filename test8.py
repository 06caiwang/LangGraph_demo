from typing import TypedDict

from langgraph.constants import START, END
from langgraph.graph import StateGraph

class State(TypedDict):
    concept: str
    market: str
    competitor: str
    tech: str
    report: str

def market_task(state: State):
    """市场分析"""
    return {"market": "用户关注续航、重量、防盗，对骑行社交有兴趣..."}

def competitor_task(state: State):
    """竞品分析"""
    return {"competitor": "传统品牌智能化不足，互联网品牌续航和售后差..."}

def tech_task(state: State):
    """技术分析"""
    return {"tech": "轻量化电池车身、GPS防盗、社交App集成..."}

# 汇总结果
def combine_results(state: State):
    """生成最终报告"""
    report = f"产品分析报告\n\n"
    report += f"市场分析：\n{state['market']}\n\n"
    report += f"竞品分析：\n{state['competitor']}\n\n"
    report += f"技术分析：\n{state['tech']}\n\n"
    report += "建议：聚焦续航、防盗、社交功能的平衡发展"
    return {"report": report}

builder = StateGraph(State)

builder.add_node(market_task)
builder.add_node(competitor_task)
builder.add_node(tech_task)
builder.add_node(combine_results)

builder.add_edge(START, "market_task")
builder.add_edge(START, "competitor_task")
builder.add_edge(START, "tech_task")
builder.add_edge("market_task", "combine_results")
builder.add_edge("competitor_task", "combine_results")
builder.add_edge("tech_task", "combine_results")
builder.add_edge("combine_results", END)

graph = builder.compile()

# print(graph.get_graph(xray=True).draw_mermaid())

print(graph.invoke({"concept": "智能电动自行车"})["report"])