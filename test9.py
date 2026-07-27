from typing import TypedDict, Literal

from langchain.chat_models import init_chat_model
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field


# 状态
class State(TypedDict):
    input: str
    decision: str   # 路由决策
    output: str

class Route(BaseModel):
    decision: Literal["pre_sale", "after_sale", "technical"] = Field(
        description="根据用户问题类型决定路由到售前、售后还是技术处理"
    )

# 节点
def model_call_router(state: State):
    model = init_chat_model(
        model="deepseek-v4-pro",
        extra_body={"thinking": {"type": "disabled"}}
    )
    result = model.with_structured_output(Route).invoke(state["input"])
    print(f"分析出问题类型为:{result.decision}")
    return {
        "decision": result.decision
    }

def pre_sale(state: State):
    """处理售前咨询"""
    return {"output": "已处理售前咨询...."}

def after_sale(state: State):
    """处理售后咨询"""
    return {"output": "已处理售后咨询...."}

def technical(state: State):
    """处理技术咨询"""
    return {"output": "已处理技术问题...."}

route_builder = StateGraph(State)
route_builder.add_node(model_call_router)
route_builder.add_node(pre_sale)
route_builder.add_node(after_sale)
route_builder.add_node(technical)
route_builder.add_edge(START, "model_call_router")

def route_decision(state: State):
    if state["decision"] == "pre_sale":
        return "pre_sale"
    elif state["decision"] == "after_sale":
        return "after_sale"
    elif state["decision"] == "technical":
        return "technical"
    else:
        return "pre_sale"

route_builder.add_conditional_edges(
    "model_call_router",
    route_decision,
    ["pre_sale", "after_sale", "technical"]
)

route_builder.add_edge("pre_sale", END)
route_builder.add_edge("after_sale", END)
route_builder.add_edge("technical", END)
route = route_builder.compile()

# 测试
test_cases = [
        "我想了解一下你们产品的价格和功能",  # 售前咨询
        "我购买的产品有质量问题，需要退货",  # 售后问题
        "这个软件安装后无法正常运行，报错代码0x80070005",  # 技术问题
        "请问你们的售后服务政策是什么",  # 售前咨询
        "我的订单已经发货但还没收到",  # 售后问题
        "如何配置数据库连接参数"  # 技术问题
]
for test_case in test_cases:
    print("*" * 50)
    result = route.invoke({"input": test_case})
    print(f"用户问题：{test_case}\n{result['output']}")