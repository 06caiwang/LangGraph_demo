from langgraph.graph import StateGraph, START


# 定义状态结构
class State(dict):
    topic: str
    joke: str


# 创建节点函数
def refine_topic(state):
    return {"topic": state["topic"] + "和猫"}


def generate_joke(state):
    return {"joke": f"这是一个关于{state['topic']}的笑话"}


# 构建图
graph = (
    StateGraph(State)
    .add_node(refine_topic)
    .add_node(generate_joke)
    .add_edge(START, "refine_topic")
    .add_edge("refine_topic", "generate_joke")
    .compile()
)

# 流式输出状态更新
# for chunk in graph.stream(
#         {"topic": "冰激凌"},
#         stream_mode="updates"  # 只看更新部分
# ):
#     print(chunk)

for chunk in graph.stream(
        {"topic": "冰激凌"},
        stream_mode="values"  # 每一步的完整状态
):
    print(chunk)