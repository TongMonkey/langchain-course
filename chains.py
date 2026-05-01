from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import AzureChatOpenAI;

# 反思角色：像评审一样批评 tweet，并给出具体修改建议。
reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet."
            "Always provide detailed recommendations, including requests for length, virality, style, etc.",
        ),
        # 运行时把完整对话历史插入这里，让模型看到前面的生成和反馈。
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# 生成角色：负责写 tweet；如果收到 critique，就基于反馈改写。
generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assistant tasked with writing excellent twitter posts."
            " Generate the best twitter post possible for the user's request."
            " If the user provides critique, respond with a revised version of your previous attempts.",
        ),
        # 与上面同名，invoke 时通过 {"messages": ...} 填充。
        MessagesPlaceholder(variable_name="messages"),
    ]
)


llm = AzureChatOpenAI(
    model="gpt-5.4-mini",
    api_version="2024-12-01-preview",
    temperature=0,
)

# LCEL 管道：prompt 先格式化输入，再交给 LLM 调用。
generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm