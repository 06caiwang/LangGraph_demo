import uuid

from langchain.embeddings import init_embeddings
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres import PostgresStore

# 1. 内存级存储
store = InMemoryStore(
    index={
        "embed": init_embeddings("ollama:bge-m3"),
        "dims": 1024,
        "fileds": ["$"]
    }
)

# 2. 定义命名空间
# user_id = str("user_1")
# namespace_1 = (user_id, "prefs", "food")
# namespace_2 = (user_id, "prefs", "music")
#
# # 3. 存入一条记忆
# memory_id_1 = str(uuid.uuid4())
# memory_value_1 = {"eat_hobby": "pizza"}
#
# memory_id_2 = str(uuid.uuid4())
# memory_value_2 = {"china": "夜曲"}
#
# # 4. 记忆存储
# store.put(namespace=namespace_1, key=memory_id_1, value=memory_value_1)
# store.put(namespace=namespace_2, key=memory_id_2, value=memory_value_2)

# 5. 记忆读取
# store.get(namespace_1, memory_id_1)
# all = store.search((user_id, "prefs", ))
# for mem in all:
#     print(mem.dict())   # 将记忆对象转成字典查看

# all = store.search((user_id, "prefs", ), query="用户喜欢的中国音乐", limit=1)
# for mem in all:
#     print(mem.dict())

# 2. Postgres store
DB_URI = "postgresql://postgres:123456@127.0.0.1:5432/postgres"
with (
    PostgresSaver.from_conn_string(DB_URI) as checkpointer,
    PostgresStore.from_conn_string(DB_URI) as store,
):
    # 第一次 setup
    # store.setup()
    user_id = "user_123"
    namespace_1 = (user_id, "prefs", "food")
    memory_id_1 = "abb27bb2-de39-478a-8ccb-e7fdf78f83fc"
    memory_value_1 = {"eat_hobby": "披萨"}
    # store.put(namespace=namespace_1, key=memory_id_1, value=memory_value_1)
    # 记忆读取
    print(store.get(namespace_1, memory_id_1))