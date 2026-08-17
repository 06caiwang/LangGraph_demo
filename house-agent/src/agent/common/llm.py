from langchain.chat_models import init_chat_model

model = init_chat_model(
    model="deepseek-v4-flash",
    temperature=0,
    extra_body={"thinking": {"type": "disabled"}}
)