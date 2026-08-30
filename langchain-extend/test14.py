from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command


def write_file():
    """写文件工具"""

def read_data():
    """读数据工具"""

def execute_sql(sql: str) -> str:
    """执行sql的工具"""
    return f"执行sql成功{sql}"

agent = create_agent(
    model="deepseek-v4-flash",
    tools=[write_file, execute_sql, read_data],
    middleware=[
        HumanInTheLoopMiddleware(
            #设置允许的操作
            interrupt_on={
                "write_file": True, # True表示允许approve\edit\reject三个操作
                "read_data": False, # False代表自动批准，可以理解为不使用HITL
                "execute_sql": {"allowed_decisions": ["approve", "edit"]}
            },
            description_prefix="工具执行尚待批准" # 中断提示前缀
        )
    ],
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": "11"}}

# 中断
response1 = agent.invoke(
    {"messages": [{"role": "user", "content": "删除数据库中data表中id=1的旧数据"}]},
    config=config,
    version="v2",  # LangChain v1.1后
)

print(response1.interrupts)

# 测试场景：选择操作：审批通过、修改后通过
# 恢复: approve

# response2 = agent.invoke(
#     Command(resume={"decisions": [{"type": "approve"}]}),
#     config=config,
#     version="v2"
# )

# 修改
response2 = agent.invoke(
    Command(
        resume={
            "decisions": [{
                "type": "edit",
                "edited_action": {
                    "name": "execute_sql",
                    "args": {'sql': 'DELETE FROM data WHERE id = 2;'},
                }
            }],
        }
    ),
    config=config,
    version="v2"
)

for i in response2.value["messages"]:
    print(i.content)