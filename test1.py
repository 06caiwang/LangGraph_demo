import operator
from typing import TypedDict, Annotated

from langgraph.constants import START, END
from langgraph.graph import StateGraph


# 1. 状态定义 -- 贯穿整个流程
class PackageStatus(TypedDict):
    # 包裹状态
    # 1. 包裹id
    package_id: str
    # 2. 包裹出发地
    package_origin: str
    # 3. 包裹目的地
    package_destination: str

    # 配送状态 -- “待揽收”、“已揽收”、“运输中”、“派送中”、“已签收”
    status: str

    # 流转历史
    history: Annotated[list[str], operator.add]

    # 总里程
    total_distance: Annotated[int, operator.add]

    # 配送模式 -- 加急、普通
    mode: str

# 2.节点定义 -- 函数
def receive_package(status: PackageStatus):
    """揽收站"""
    origin = status['package_origin']

    return {
        "status": "已揽收",
        "history": [f"在{origin}处揽收"]
    }

def sort_package(status: PackageStatus):
    """分拣中心"""
    destination = status['package_destination']
    next = None
    if "北京" in destination:
        next = "北京分拣中心"
    elif "上海" in destination:
        next = "上海分拣中心"
    else:
        next = "其他分拣中心"

    return {
        "status": "已分拣",
        "history": [f"分拣至{next}"]
    }

def final_delivery(status: PackageStatus):
    """派送站"""
    destination = status['package_destination']
    return {
        "status": "已签收",
        "history": [f"已送达{destination}"]
    }

def standard_delivery(status: PackageStatus):
    """标准配送"""
    return {
        "status": "正在运输中",
        "history": ["标准运输"],
        "total_distance": 300
    }

def express_delivery(status: PackageStatus):
    """加急配送"""
    return {
        "status": "正在加急运输中",
        "history": ["空运加急"],
        "total_distance": 1000
    }

# 3.定义图 -- 大框架
delivery = StateGraph(PackageStatus)

# 4.添加节点 -- 流程中的每个节点
delivery.add_node("揽收站", receive_package)
delivery.add_node("分拣中心", sort_package)
delivery.add_node("派送站", final_delivery)
delivery.add_node("标准配送", standard_delivery)
delivery.add_node("加急配送", express_delivery)

# 5.添加边 -- 流程的流向
delivery.add_edge(START, "揽收站")
delivery.add_edge("揽收站", "分拣中心")

def select_delivery(state: PackageStatus):
    mode = state["mode"]
    if mode == "加急":
        return "备注加急"   # 返回的是字符串，不是节点
    else:
        return "无备注"   # 返回的是字符串，不是节点

# 添加条件边
delivery.add_conditional_edges(
    "分拣中心",   # 条件的起始节点
    select_delivery,   # path：确定下一个节点可调节对象
    {
        "备注加急": "加急配送",
        "无备注": "标准配送"
    }
)

delivery.add_edge("加急配送", "派送站")
delivery.add_edge("标准配送", "派送站")
delivery.add_edge("派送站", END)
# 6.编译图
delivery_system = delivery.compile()

# 7.执行图
test_packages = [
{
        "package_id": "P001",
        "package_origin": "北京",
        "package_destination": "上海",
        "mode": "普通",
        "history": [],
        "total_distance": 0
    },
    {
        "package_id": "P002",
        "package_origin": "广州",
        "package_destination": "乌鲁木齐",
        "mode": "加急",
        "history": [],
        "total_distance": 0
    }
]

for package in test_packages:
    print(f"\n配送包裹: {package['package_id']}")
    # 执行图，发一遍快递
    result = delivery_system.invoke(package)
    print("最终状态:", result["status"])
    print("配送历史:", result["history"])
    print("总里程:", result["total_distance"])