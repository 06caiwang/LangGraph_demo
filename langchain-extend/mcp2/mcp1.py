# Agent： LLM -> tool
# 加入MCP(工具、资源、提示词)
# 资源：只读，更适合LLM来查询信息（给LLM提供一些上下文）
# 1. 获取资源（QA文档、使用说明文档）
# 2. 定义工具：获取某方面资源的工具
# 3. 绑定工具给Agent，问问题：“给我说一下xxx的操作流程” -》 决策工具  -》  执行工具（获取 MCP 资源） -》 输出结果
import json

from fastmcp import FastMCP

# doc_server.py

mcp = FastMCP("DocumentStore")

@mcp.resource("file:///help/guide.txt")
def get_guide() -> str:
    return """# 用户指南
1. 首先登录系统
2. 点击“新建项目”
3. 输入项目名称
"""

@mcp.resource("file:///help/faq.json")
def get_faq() -> str:
    return json.dumps(
        {
            "q1": "如何重置密码？",
            "a1": "请点击“忘记密码”链接。"
        }
    )

if __name__ == "__main__":
    mcp.run(transport="stdio")