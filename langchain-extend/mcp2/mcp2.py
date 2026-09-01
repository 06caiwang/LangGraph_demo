# client、agent
import asyncio

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient


async def main():

    # client 实际上是异步的
    client = MultiServerMCPClient(
        {   # 要连接哪些mcp服务器
            "DocumentStore": {  # 服务器配置
                "transport": "stdio",  # 本地进程通信
                "command": "python",
                "args": ["D:/agent_demo/LangGraph_demo/langchain-extend/mcp2/mcp1.py"]
            }
        }
    )

    # 获取资源
    blobs = await client.get_resources("DocumentStore", uris=["file:///help/faq.json"])

    # 将资源内容转换成一个工具（方便agent按需检索）
    @tool
    def search_aq_docs(query: str) -> str:
        """搜索重置密码相关文档"""
        results = []
        for blob in blobs:
            if blob.mimetype == "text/plain":
                content = blob.as_string()
                print(content)
                results.append(content)
        return "\n\n".join(results) if results else "未找到相关文档"

    agent = create_agent("deepseek-v4-flash", tools=[search_aq_docs], system_prompt="务必调用工具")
    response = await agent.ainvoke({
        "messages": [{"role": "user", "content": "如何重置密码？请查阅文档"}]
    })
    print(response)


if  __name__  == "__main__":
    asyncio.run(main())