from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.messages import HumanMessage, filter_messages
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_classic.tools.retriever import create_retriever_tool
from langgraph.constants import START, END
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field


# 嵌入模型的定义
model = init_chat_model("deepseek-v4-pro")
embeddings = OllamaEmbeddings(model="bge-m3")

# 加载文档列表
paths = [
    "Docs/markdown/企业介绍.md",
    "Docs/markdown/C++开发方向.md",
    "Docs/markdown/Java开发方向.md",
    "Docs/markdown/测试开发方向.md"
]

docs = [UnstructuredMarkdownLoader(path).load() for path in paths]
docs_list = [item for sublist in docs for item in sublist]

# 文本分割
text_splitter = RecursiveCharacterTextSplitter().from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=1000,
    chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs_list)

# 使用内存中向量存储和 OpenAI 嵌入
vectorstore = InMemoryVectorStore.from_documents(
    documents=doc_splits,
    embedding=embeddings
)

# 使用 LangChain 的预构建 create_retriever_tool 创建检索器工具：
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 创建检索工具并绑定模型
retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_test",
    "搜索并返回有关比特就业课的信息。",
)
model_with_tools = model.bind_tools([retriever_tool])

# 1. 状态使用langgraph.graph包中自带的 MessagesState
# 2. 节点
def generate_query_or_respond(state: MessagesState):
    """调用模型，基于当前状态生成响应。
    给定问题，它将决定使用检索工具进行检索，或者简单的生成用户响应"""
    message = state.get("messages")
    result = model_with_tools.invoke(message)

    return {
        "messages": [result]
    }

# 工具节点
retriever_node = ToolNode([retriever_tool])

REWRITE_PROMPT = (
    "查看输入并尝试推断潜在的语义意图/含义。\n"
    "这是最初的问题："
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "提出一个改进后的问题："
)
def rewrite_question(state: MessagesState):
    """重写用户问题"""
    question = state["messages"][0]
    prompt = REWRITE_PROMPT.format(question=question)
    result = model.invoke([HumanMessage(content=prompt)])

    return {
        "messages": [HumanMessage(content=result.content)]
    }

GENERATE_PROMPT = (
    "你是负责回答问题的助手。 "
    "使用以下检索到的上下文片段来回答问题。 "
    "如果你不知道答案，就说你不知道。 "
    "最多只用三句话，回答要简明扼要。\n"
    "Question: {question} \n"
    "Context: {context}"
)

def generate_answer(state: MessagesState):
    """生成答案"""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    result = model.invoke([HumanMessage(content=prompt)])
    return {
        "messages": [result]
    }

# 3. 图、边、节点
workflow = StateGraph(MessagesState)

workflow.add_node(generate_query_or_respond)
workflow.add_node("retrieve", retriever_node)
workflow.add_node(rewrite_question)
workflow.add_node(generate_answer)

workflow.add_edge(START, "generate_query_or_respond")
workflow.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,  # 判断最后一条AI消息是否包含工具调用
    {
        "tools": "retrieve",
        "__end__": END,
    }
)

GRADE_PROMPT = (
    "你是一个评分员，评估检索到的文档与用户问题的相关性。 \n "
    "以下是检索到的文档： \n\n {context} \n\n"
    "以下是用户的问题： {question} \n"
    "如果文档包含与用户问题相关的关键字或语义，则将其评为相关。 \n"
    "给出一个二元分数“yes”或“no”，以表明该文档是否与问题相关。"
)

class GradeDocuments(BaseModel):
    score: str = Field(description="相关性评分：如果相关则为“yes”, 不相关则为“no”")

def grade_documents(state: MessagesState) -> Literal["rewrite_question", "generate_answer"]:
    """确定检索到的文档是否与问题有关"""
    user_messages = filter_messages(
        state["messages"],
        include_types="human"
    )
    question = user_messages[-1].content
    tool_message = state["messages"][-1]
    context = tool_message.content
    prompt = GRADE_PROMPT.format(question=question, context=context)
    result = model.with_structured_output(GradeDocuments).invoke(
        [HumanMessage(content=prompt)],
        extra_body={"thinking": {"type": "disabled"}}  # 针对结构化输出关闭思考模式
    )

    if result.score == "yes":
        return "generate_answer"
    else:
        return "rewrite_question"

workflow.add_conditional_edges(
    "retrieve",
    grade_documents,
    ["generate_answer", "rewrite_question"]
)
workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

# 4. 编译图
graph = workflow.compile()

# 5. 执行图
for chunk in graph.stream(
    {
        "messages": [HumanMessage(content="测试开发方向的主线课程有哪些")]
    }
):
    for node, update in chunk.items():
        print(f"由节点{node}更新消息")
        update["messages"][-1].pretty_print()
        print("\n\n")