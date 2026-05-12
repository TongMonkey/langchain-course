import datetime
from dotenv import load_dotenv

from schemas import AnswerQuestion

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



actor_prompt_template = ChatPromptTemplate.from_messages(
    [
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
)

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer."
)

first_responder = first_responder_prompt_template | llm.bind_tools(
    # 通过提供一个 tool_choice 参数，当提供的是 AnswerQuestion 时，永远调用 AnswerQuestion tool
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)

if __name__ == "__main__":
    human_message = HumanMessage(
        content="Write about AI-Powered SOC / autonomous soc problem domain,"
        " list startups that do that and raised capital."
    )
    chain = (
        first_responder_prompt_template 
        | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion") 
        | parse_pydantic
    )
    response = chain.invoke({
        "messages": [human_message]
    })
    print(response)
    