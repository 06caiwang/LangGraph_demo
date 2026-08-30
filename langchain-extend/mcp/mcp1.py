# MCP server: 计算方法
from fastmcp import FastMCP

mcp = FastMCP("Math")

@mcp.tool
def add(a: int, b: int) -> int:
    """两个整数相加"""
    return a + b

@mcp.tool
def multiply(a: int, b: int) -> int:
    """两个整数乘法"""
    return a * b

if __name__ == "__main__":
    # 通过本地进程通信的方式启动MCP服务
    mcp.run(transport="stdio")