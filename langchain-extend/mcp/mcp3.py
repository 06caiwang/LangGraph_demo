import asyncio

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools


async def main():
    client = MultiServerMCPClient(
        {  # 要连接哪些mcp服务器
            "math": {  # 服务器配置
                "transport": "stdio",  # 本地进程通信
                "command": "python",
                "args": ["D:/agent_demo/LangGraph_demo/langchain-extend/mcp/mcp1.py"]
            },
            "weather": {
                "transport": "streamable-http",  # 基于HTTP的远程调用
                "url": "http://127.0.0.1:8000/mcp"
            }
        }
    )

    # 获取连接的服务器的所有的工具（无状态）
    # tools = await client.get_tools()

    async with client.session("weather") as session:  # 持久会话
        tools = await load_mcp_tools(session)  # 从该会话中加载工具

        # 绑定工具给Agent
        agent = create_agent("deepseek-v4-flash", tools, system_prompt="你是一个乐于助人的助手，回答问题必须使用工具完成！")
        # 该Agent的所有工具调用将复用一个session
        math_result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "(3 + 4) * 5 等于多少"}]}
        )
        weather_result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "徐州的天气怎么样？"}]}
        )
        # LLM 决策工具调用
        # 工具调用
        print(math_result)
        print(weather_result)


# client 是异步的，因此需要启动时使用asyncio.run()
if __name__ == "__main__":
    asyncio.run(main())