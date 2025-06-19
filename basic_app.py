import json

from weasyprint.css.validation.properties import background_attachment

import gradio_ui_langgraph

from basic_agent import BasicOpenAIAgentWorkflow
from tools import search_web_extract_info, search_and_extract

from dotenv import load_dotenv


MODE = "demo"  # "conversational" or "demo"
load_dotenv()

tools = [
    search_and_extract,
]
tools_info = '\n\n'.join([f'{tool.name}: {tool.description}: {tool.args}' for tool in tools])

SYS_PROMPT = f"""Act as a helpful assistant.
                You run in a loop of Thought, Action, PAUSE, Observation.
                At the end of the loop, you output an Answer.
                Use Thought to describe your thoughts about the question you have been asked.
                Use Action to run one of the actions available to you - then return PAUSE.
                Observation will be the result of running those actions.
                Repeat till you get to the answer for the given user query.

                Use the following workflow format:
                  Question: the input task you must solve
                  Thought: you should always think about what to do
                  Action: the action to take which can be any of the following:
                            - break it into smaller steps if needed
                            - see if you can answer the given task with your trained knowledge
                            - call the most relevant tools at your disposal mentioned below in case you need more information
                  Action Input: the input to the action
                  Observation: the result of the action
                  ... (this Thought/Action/Action Input/Observation can repeat N times)
                  Thought: I now know the final answer
                  Final Answer: the final answer to the original input question

                Tools at your disposal to perform tasks as needed:

                {tools_info}
             """


basic_tool_agent = BasicOpenAIAgentWorkflow(
    tools=tools,
    backstory=SYS_PROMPT
)
basic_tool_agent.create_basic_tool_use_agent_state_graph()


if __name__ == "__main__":
    if MODE == "conversational":
        gradio_ui_langgraph.GradioUI(basic_tool_agent).launch()

    elif MODE == "demo":
        with open("gaia_default_results_log.json", "r") as f:
            gaia_data = json.load(f)

        # basic_tool_agent.chat_batch([q_data["Question"] for q_data in gaia_data], only_final_answer=False)
        for q_data in gaia_data:
            query = q_data["Question"]
            _ = basic_tool_agent.chat(query, verbose=1, only_final_answer=True)


# todo: memory is persisted, but better to store last_k interactions (or None), use a SQL db or vector store.
# todo: stream output