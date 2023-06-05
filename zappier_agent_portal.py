from contextlib import redirect_stdout
from io import StringIO
from os import environ
from re import compile as re_compile

import streamlit as st
from langchain.agents import initialize_agent
from langchain.agents.agent import AgentExecutor
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.agents.agent_types import AgentType
from langchain.llms import OpenAI
from langchain.utilities.zapier import ZapierNLAWrapper


@st.cache_resource
def setup_agent(openai_api_key: str, zapier_api_key: str) -> AgentExecutor:
    # set api keys as environmental variables
    environ['OPENAI_API_KEY'] = openai_api_key
    environ['ZAPIER_NLA_API_KEY'] = zapier_api_key

    llm = OpenAI(temperature=0)
    zapier = ZapierNLAWrapper()
    toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)
    agent = initialize_agent(toolkit.get_tools(),
                             llm,
                             agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                             verbose=True)

    st.sidebar.markdown("**Available Tools based on your Zapier API key**")
    for tool in toolkit.get_tools():
        tool_expander = st.sidebar.expander(tool.name)
        tool_expander.write(tool.description)

    return agent


def remove_ansi_escape_codes(text: str) -> str:
    ansi_escape_pattern = re_compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape_pattern.sub('', text)


def main():
    col1, col2 = st.columns(2)
    col1.title("Welcome to your AI's central command! 🤖")
    col1.markdown("*:lightgrey[where you can use the power of ChatGPT to read and create emails, "
                  "Slack messages, reminders, and more]*")
    col2.image("robot_logo.png")

    openai_api_key: str = st.sidebar.text_input("OPENAI API KEY", type="password")
    zapier_api_key: str = st.sidebar.text_input("ZAPIER NLA API KEY", type="password")

    if openai_api_key != '' and zapier_api_key != '':
        # initialize agents
        agent: AgentExecutor = setup_agent(openai_api_key, zapier_api_key)

        user_prompt: str = col1.text_area("", placeholder='Enter Prompt', label_visibility='collapsed')

        if user_prompt.strip() not in ["", "Enter Prompt"]:
            stdout = StringIO()
            try:
                with redirect_stdout(stdout):
                    agent.run(user_prompt)
            except Exception as e:
                st.write(f"Failure while executing: {e}")
            finally:
                # get output and remove ansi escape characters used for coloring of text in langchain
                output_string: str = remove_ansi_escape_codes(stdout.getvalue())

                # add new lines to make output easier to read
                output_string = output_string.replace('...', '...\n\n')
                output_string = output_string.replace('Action', '\n**Action**')
                output_string = output_string.replace('Observation', '\n**Observation**')
                output_string = output_string.replace('Thought', '\n**Thought**')
                output_string = output_string.replace('Final Answer', '\n**Final Answer**')

                st.markdown(output_string, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
