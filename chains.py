import datetime
from dotenv import load_dotenv

from schemas import AnswerQuestion, ReviseAnswer

load_dotenv()

from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser
)

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import AzureChatOpenAI;



llm = AzureChatOpenAI(
    model="gpt-5.4-mini",
    api_version="2024-12-01-preview",
    temperature=0,
)
# 输出解析器
parser = JsonOutputToolsParser(return_id=True)
# 这个解析器会从 LLM 拿到 response, 会去调用 function calling, 然后解析返回值，再 transform 成 AnswerQuestion 对象
parse_pydantic = PydanticToolsParser(tools=[AnswerQuestion])


# 这个是 actor 的 prompt 模板，会根据不同的 first_instruction 来生成不同的 prompt
# 这个模板最后会拼成：上面是 SystemMessage:... 下面是从外面传进来的 messages"最终发给 LLM:
# [
#   SystemMessage("You are expert researcher..."),
#   HumanMessage("Write about AI-Powered SOC...")
# ]
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        # "system" 是 SystemMessage 类型，是给 LLM 的最高指令，后面是最高指令的内容
        (
            "system",
            """You are expert researcher.
            Current time: {time}

            1. {first_instruction}
            2. Reflect and critique your answer. Be severe to maximize improvement.
            3. Recommend search queries to research information and improve your answer."""
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat()
).partial(
    first_instruction=lambda: revise_instructions
)

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer."
)

first_responder = first_responder_prompt_template | llm.bind_tools(
    # tool_choice="AnswerQuestion"：强制接口返回名为 AnswerQuestion 的 tool_calls（JSON 参数），
    # 用作结构化输出；不会由此自动执行任何本地 Python 工具函数。
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)

# 这个是要插入到 actor_prompt_template 中的 {first_instruction} 中的
revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
    - Add a "References" section to the bottom of your answer ( which does not count towards the word limitations)
      - [1] https://example.com
      - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your anser and make sure it is not more than 250 words.
"""

revisor_prompt_template = actor_prompt_template.partial(
    first_instruction=revise_instructions
)
revisor = revisor_prompt_template | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")

if __name__ == "__main__":
    human_message = HumanMessage(
        content="Write about AI-Powered SOC / autonomous soc problem domain,"
        " list startups that do that and raised capital."
    )
    chain = (
        first_responder_prompt_template 
        # tool_choice="AnswerQuestion"：强制接口返回名为 AnswerQuestion 的 tool_calls（JSON 参数），
        # 用作结构化输出；不会由此自动执行任何本地 Python 工具函数。
        | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion") 
        # 使用 PydanticToolsParser 解析器，将 LLM 的输出解析为 AnswerQuestion 对象
        | parse_pydantic
    )
    response = chain.invoke({
        "messages": [human_message]
    })
    print(response)
    