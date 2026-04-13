import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

# ToDelete-begin
# 待办 ToDelete：本机学习用（绕过 TLS 等）。当你说「删除所有 ToDelete」时，移除本 begin/end 之间全部内容，
# 并在 main() 里 ToDelete 区块移除后写回无 http_client 的 ChatOpenAI(temperature=..., model=..., openai_api_key=..., openai_api_base=...)。
import warnings

import httpx


def _insecure_ssl_enabled() -> bool:
    return os.getenv("ZAI_INSECURE_SSL", "").strip().lower() in ("1", "true", "yes")


def _optional_debug_http_client() -> httpx.Client | None:
    """Corporate proxies often break TLS verification; use only for local debugging."""
    if not _insecure_ssl_enabled():
        return None
    warnings.warn(
        "TLS certificate verification is disabled (ZAI_INSECURE_SSL). "
        "Do not use in production.",
        UserWarning,
        stacklevel=2,
    )
    return httpx.Client(verify=False)


# ToDelete-end


def main():
    print("Hello from langchain-course!")
    information = """
    Elon Reeve Musk (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a businessman and entrepreneur known for his leadership of Tesla, SpaceX, X, and xAI. Musk has been the wealthiest person in the world since 2025; as of April 2026, Forbes estimates his net worth to be US$809 billion.

Born into a wealthy family in Pretoria, South Africa, Musk emigrated in 1989 to Canada; he has Canadian citizenship since his mother was born there. He received bachelor's degrees in 1997 from the University of Pennsylvania before moving to California to pursue business ventures. In 1995, Musk co-founded the software company Zip2. Following its sale in 1999, he co-founded X.com, an online payment company that later merged to form PayPal, which was acquired by eBay in 2002. Musk also became an American citizen in 2002.

In 2002, Musk founded the space technology company SpaceX, becoming its CEO and chief engineer; the company has since led innovations in reusable rockets and commercial spaceflight. Musk joined the automaker Tesla as an early investor in 2004 and became its CEO and product architect in 2008; it has since become a leader in electric vehicles. In 2015, he co-founded OpenAI to advance artificial intelligence (AI) research, but later left; growing discontent with the organization's direction and leadership in the AI boom in the 2020s led him to establish xAI, which became a subsidiary of SpaceX in 2026. In 2022, he acquired the social network Twitter, implementing significant changes, and rebranding it as X in 2023. His other businesses include the neurotechnology company Neuralink, which he co-founded in 2016, and the tunneling company the Boring Company, which he founded in 2017. In November 2025, a Tesla pay package worth $1 trillion for Musk was approved, which he is to receive over 10 years if he meets specific goals.

Musk is a supporter of global far-right figures, causes, and political parties. He was the largest donor in the 2024 U.S. presidential election, where Musk supported Donald Trump. After Trump was inaugurated as president in January 2025, Musk served as Senior Advisor to the President and as the de facto head of the Department of Government Efficiency (DOGE). Shortly before a public feud with Trump, Musk left the Trump administration in May 2025 and returned to managing his companies.

His political activities, views, and statements have made Musk a polarizing figure. Musk has been criticized for COVID-19 misinformation, promoting conspiracy theories, and affirming antisemitic, racist, and transphobic comments. His acquisition of Twitter was controversial due to a subsequent increase in hate speech and the spread of misinformation on the service, following his pledge to decrease censorship. His role in the second Trump administration attracted public backlash, particularly in response to DOGE. The emails Musk sent to Jeffrey Epstein are included in the Epstein files, which were published in 2025 and 2026 and became a topic of worldwide debate.
    """

    summary_template = """
    Summarize the following {information} about a given person:
    1. A short summary
    2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )

    # ToDelete-begin
    llm_kwargs: dict = {
        "temperature": 0.6,
        "model": "glm-4.6",
        "openai_api_key": os.getenv("ZAI_API_KEY"),
        "openai_api_base": "https://open.bigmodel.cn/api/paas/v4/",
    }
    http_client = _optional_debug_http_client()
    if http_client is not None:
        llm_kwargs["http_client"] = http_client
    llm = ChatOpenAI(**llm_kwargs)
    # ToDelete-end

    # We are using something which is called the link chain expression language or LCL
    # In this LCL syntax we create a chain by composing two components together, a prompt template and a language model and a large language model.
    # The template is going to format input variables into a prompt string witch will eventually be propagated into the language model.
    # And the llm variable is a chat open AI object, which takes in an input prompt string and generates a text response.
    # And this pipe operator | is going to create a new runnable chain object.
    # So this is a new term runnable by connecting the output of the left component as an input to the right component.
    chain = summary_prompt_template | llm
    response = chain.invoke({"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
