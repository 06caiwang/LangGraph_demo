# 动态工具
# 1. 运行时，根据条件筛选已存在的工具
# Agent[tool1, tool2, tool3] -> [tool2]
# 2. 运行时，新增工具
# Agent[tool1]  ->  [tool1, tool2]

from typing import Callable
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse, wrap_tool_call, AgentMiddleware, \
    ExtendedModelResponse
from langchain.agents.middleware.types import ResponseT
from langchain.tools import tool
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command
from langgraph.typing import ContextT


# 定义工具
@tool
def get_weather_for_location(city: str) -> str:
    """获取指定城市的天气信息"""
    return f"在{city}总是阳光明媚！"


# 该工具将在运行时动态添加的工具
@tool
def calculate_tip(bill_amount: float, tip_percentage: float = 20.0) -> str:
    """计算一笔账单的小费金额。"""
    print("小费计算中...")
    # 80 * 20%
    tip = bill_amount * (tip_percentage / 100)
    return f"小费: {tip:.2f}元, 一共: {bill_amount + tip:.2f}元"

class DynamicToolMiddleware(AgentMiddleware):
    """能够注册并处理动态工具的中间件。"""

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        # 将新的工具添加进来
        updated_request = request.override(tools=[*request.tools, calculate_tip])
        return handler(updated_request)  # llm调用(选择工具)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        # 处理动态工具的执行过程
        # 走到此处已经把tool选好了，只是知道要调用哪个工具和参数

        try:
            if request.tool_call["name"] == "calculate_tip":
                return handler(request.override(tool=calculate_tip))  # calculate_tip.invoke()

            return handler(request) # tool.invoke()
        except Exception as e:
            # print(e)
            # 自定义错误消息返回 ToolMessage(content="工具错误:xxx")
            return ToolMessage(
                content=f"工具错误：请检查您的输入并重新尝试。（{str(e)}）",
                tool_call_id=request.tool_call["id"]
            )


agent = create_agent(
    model="deepseek-v4-flash",
    tools=[get_weather_for_location],  # 只是绑定了一个天气工具
    system_prompt="你是一位乐于助人的客服助手。根据用户的问题，选择合适的工具来提供答案",
    middleware=[DynamicToolMiddleware()]
)

response = agent.invoke(
    {
        "messages": [{
            "role": "user",
            "content": "计算80元账单的小费是多少？"  # 80 * 20%
        }]
    }
)

print(response["messages"][-1].content)
