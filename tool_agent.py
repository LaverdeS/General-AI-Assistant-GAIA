from langchain_core.messages import HumanMessage

import gradio_ui_langgraph
import json
import time

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.callbacks.manager import get_openai_callback
from langchain.globals import set_llm_cache, get_llm_cache
from langchain_community.cache import InMemoryCache

from langgraph.prebuilt import create_react_agent

from basic_agent import print_conversation, generate_final_answer

from dotenv import load_dotenv

from openai import RateLimitError

from langchain.globals import set_debug

from prompts import REFINED_SYS_PROMPT

from tools import (
    smart_read_file,
    search_and_extract,
    search_and_extract_from_wikipedia,
    aggregate_information,
    extract_clean_text_from_url,
    youtube_search_tool,
    load_youtube_transcript,
    get_audio_from_youtube,
    image_query_tool,
    transcribe_audio,
)

set_debug(False)


load_dotenv()
# set_llm_cache(InMemoryCache())

#"""
llm_cache = get_llm_cache()
if llm_cache:
    llm_cache.clear()

set_llm_cache(InMemoryCache())
#"""


def conversation_step(query:str, tool_agent_executor:AgentExecutor, metadata:dict[str, str]|None = None):
    """Perform a single step in the conversation with the tool agent executor."""
    if metadata is None:
        metadata = {}

    with_attachments = False
    query_message = HumanMessage(content=query)

    if "image_path" in metadata:

        # Create a HumanMessage with image content
        query_message = HumanMessage(
            content=[
                {"type": "text", "text": query},
                {"type": "text", "text": f"image_path: {metadata['image_path']}"},
            ]
        )
        with_attachments = True

    user_message = {'role': 'user', 'content': query if not with_attachments else query_message}
    print_conversation([user_message])

    response = tool_agent_executor.invoke({
        "query": query if not with_attachments else query_message,
    #    "tools_info": tools_info,
    })
    response_message = {'role': 'assistant', 'content': response}
    print_conversation([response_message])

    final_answer = generate_final_answer({
        'query': query,
        'response': response,
    })
    final_answer_message = {'role': 'Final Answer', 'content': final_answer}
    print_conversation([final_answer_message])


def conversation_step_demo(q_data, agent_executor, cb):
    metadata = q_data.get('Attachments', None)
    conversation_step(q_data['Question'], agent_executor, metadata)
    time.sleep(3)

    token_use_message = {
        'role': 'Token Use',
        'content': {
            "prompt_tokens": cb.prompt_tokens,
            "completion_tokens": cb.completion_tokens,
            "total_tokens": cb.total_tokens,
            "successful_requests": cb.successful_requests,
            "total_cost": f"${cb.total_cost:.6f}"
        }
    }
    print_conversation([token_use_message])
    print()


if __name__ == "__main__":
    MODE = "demo"

    with get_openai_callback() as cb:
        chatgpt = ChatOpenAI(model="gpt-4o", temperature=0, max_retries=5)
        tools = [
            smart_read_file,
            search_and_extract,
            search_and_extract_from_wikipedia,
            aggregate_information,
            extract_clean_text_from_url,
            youtube_search_tool,
            load_youtube_transcript,
            get_audio_from_youtube,
            image_query_tool,
            transcribe_audio,
        ]
        tools_info = '\n\n'.join([f'{tool.name}: {tool.description}: {tool.args}' for tool in tools])
        print(f"Tools available: {tools_info}\n")
        chatgpt_with_tools = chatgpt.bind_tools(tools)

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", REFINED_SYS_PROMPT),
                MessagesPlaceholder(variable_name="history", optional=True),
                ("human", "{query}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        agent = create_tool_calling_agent(chatgpt, tools, prompt_template)
        agent_executor = AgentExecutor(agent=agent,
                                       tools=tools,
                                       early_stopping_method='force',
                                       max_iterations=10,)

        if MODE == "conversational":
            gradio_ui_langgraph.GradioUI(agent_executor).launch()

        elif MODE == "demo":
            with open("gaia_default_results_log.json", "r") as f:
                gaia_data = json.load(f)

            for q_data in gaia_data:
                try:
                    conversation_step_demo(q_data, agent_executor, cb)
                except RateLimitError:
                    error_message = {'role': 'Rate-limit-hit', 'content': 'Rate limit error encountered. Retrying after a short pause...'}
                    print_conversation([error_message])
                    time.sleep(5)
                    try:
                        conversation_step_demo(q_data, agent_executor, cb)
                    except RateLimitError:
                        error_message = {'role': 'Rate-limit-hit', 'content': 'Rate limit error encountered again. Skipping this query.'}
                        print_conversation([error_message])
                        continue
                break



        last_query = "Answer this two queries: What tools are available for you to use as an agent? What have I asked you so far in our whole conversation?"
        conversation_step(last_query, agent_executor)