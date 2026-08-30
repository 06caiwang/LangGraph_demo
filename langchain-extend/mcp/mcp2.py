# MCP server: 查询天气的工具
from fastmcp import FastMCP

mcp = FastMCP("weather")

@mcp.tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息"""
    return f"在{city}总是阳光明媚！"

if __name__ == "__main__":
    # 启动一个以http方式调用的mcp服务
    mcp.run(transport="streamable-http", port=8000)