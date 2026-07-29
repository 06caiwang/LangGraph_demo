import operator
import uuid
from typing import TypedDict, Annotated, Optional

from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage, AIMessage, SystemMessage, ToolMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: Optional[str] = Field(default=None, description="人的姓名")
    age: Optional[int] = Field(default=None, description="人的年龄")
    height: Optional[str] = Field(default=None, description="人的身高")
    food: Optional[list[str]] = Field(default=None, description="喜欢的食物")

search = TavilySearch(max_results=1)
tools = [search]
model = init_chat_model(
    model="deepseek-v4-pro",
    temperature=0,
    extra_body={"thinking": {"type": "disabled"}}
)
model_with_tools =  model.bind_tools(tools)
model_with_structured = model.with_structured_output(Person)

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

def get_person_info(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
    """通过LLM获取人物信息"""
    people_info = model_with_structured.invoke(
        [SystemMessage(content="你是一个提取信息的专家，只从文本中提取我的相关信息，不能提取别人的信息。"
                               "如果你不知道要提取的属性的值，属性值返回null")]
        + state["messages"]
    )

    # 保存提取出来的信息
    user_id = config["configurable"]["user_id"]
    namespace1 = (user_id, "info")

    # put之前应先查询有没有
    store.put(
        namespace1,
        str(uuid.uuid4()),
        {
            "name": people_info.name,
            "age": people_info.age,
            "height": people_info.height
        }
    )

    namespace2 = (user_id, "preferences")
    store.put(
        namespace2,
        str(uuid.uuid4()),
        {
            "food": people_info.food,
        }
    )

    return {
        "llm_calls": state.get("llm_calls", 0) + 1
    }

def llm_call(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
    """LLM决定是否调用工具"""
    user_id = config["configurable"]["user_id"]
    namespace1 = (user_id, "info")
    namespace2 = (user_id, "preferences")

    info_result = store.search(namespace1, limit=1)
    prefs_result = store.search(namespace2, limit=1)

    print(info_result)
    print(prefs_result)

    messages = state["messages"]

    result = model_with_tools.invoke(
        [
            SystemMessage(content="你是一个乐于助人的助手，支持调用工具进行地方天气搜索")
        ] + [HumanMessage(content=f"必须参考以下信息："
                                  f"1. 用户基本情况：{info_result[0].value}"
                                  f"2. 用户偏好情况：{prefs_result[0].value}")]
        + messages
    )

    return {
        "messages": [result],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

tools_by_name = {tool.name: tool for tool in tools}
def tool_node(state: MessagesState):
    """执行工具调用"""
    # result 就是 ToolMessage

    result = []
    # 当前最新的消息就是带有tool_calls的AImessage
    for tool_call in state["messages"][-1].tool_calls:
        # 就可以获取到tool_call的name,args,id...
        # 要根据tool_call知道，去执行哪个工具
        tool = tools_by_name[tool_call["name"]]
        obs = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=obs, tool_call_id=tool_call["id"]))

    return {
        "messages": result,
    }

# 3. 定义图，添加节点和边
agent_builder = StateGraph(MessagesState)

agent_builder.add_node(llm_call)
agent_builder.add_node(tool_node)
agent_builder.add_node(get_person_info)
agent_builder.add_edge(START, "get_person_info")
agent_builder.add_edge("get_person_info", "llm_call")


def should_continue(state: MessagesState):
    # 最新消息是AIMessage，要判断它是否带有tool_calls
    # 带有tool_calls：要走tool_node
    # 不带：END

    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool_node"

    return END

agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)

agent_builder.add_edge("tool_node", "get_person_info")

# 4. 编译图
agent_search = agent_builder.compile(checkpointer=InMemorySaver(), store=InMemoryStore())

# 5. 执行图
# 模拟第一次执行
config1 = {"configurable": {"thread_id": "1111", "user_id": "user_123"}}

result1 = agent_search.invoke({
    "messages": [HumanMessage(content="我叫小明，今年18岁，身高174cm，我喜欢吃川菜里的回锅肉")]
}, config1)

for msg in result1["messages"]:
    msg.pretty_print()

# 模拟第二次执行，新开对话
config2 = {"configurable": {"thread_id": "2222", "user_id": "user_123"}}

result2 = agent_search.invoke({
    "messages": [HumanMessage(content="我爱吃什么")]
}, config2)
for msg in result2["messages"]:
    msg.pretty_print()
