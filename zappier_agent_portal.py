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

    main_tab, setup_tab, docs_tab = st.tabs(["Main", "How to Setup", "Other Notes and Docs"])

    with main_tab:
        openai_api_key: str = st.sidebar.text_input("OPENAI API KEY", type="password")
        zapier_api_key: str = st.sidebar.text_input("ZAPIER NLA API KEY", type="password")

        if openai_api_key != '' and zapier_api_key != '':
            # initialize agents
            agent: AgentExecutor = setup_agent(openai_api_key, zapier_api_key)

            user_prompt: str = col1.text_area("Enter Prompt", placeholder='Enter Prompt', label_visibility='collapsed')

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

    with setup_tab:
        st.markdown("""
        In order to use this tool, you need an OpenAI API key, and a Zapier NLA API key:\n\n
        - **OpenAI**: In order to use ChatGPT capabilities, you need to have an account with OpenAI and have an API key,
        which you can get from [here](https://openai.com/blog/openai-api). Note that the keys are entered as passwords, 
        and are not saved here. However, if you don't feel comfortable entering your OpenAI API key, I will add other 
        open-source LLM options in the future.\n
        - **Zapier**: This toolkit also uses the [Zapier NLA](https://nla.zapier.com/docs/) (Natural Language Actions) 
        API. Once you create a Zapier account, you can set up action settings for the API 
        [here](https://nla.zapier.com/providers/). Your action settings determine what your API key will have access to, 
        such as your email, Slack, and 5000+ other applications made available through Zapier. The portal's sidebar will 
        inform you what you have access to once you input your key. 
        """, unsafe_allow_html=True)

    with docs_tab:
        st.markdown("""
        This portal is made for fun and personal use by [Yara Mohajerani](https://twitter.com/yara_mo_), and all the 
        code is accessible in the associated Github repo: 
        [https://github.com/yaramohajerani/ai_agent](https://github.com/yaramohajerani/ai_agent)
        
        \n\nAnd here is a disclaimer made by GPT-4 because why not...\n
        
        **Disclaimer**\n

        This website and its services (collectively, the "Website") are provided "as is," without warranty of any kind, 
        either express or implied. The use of this Website is entirely at your own risk.\n

        The hosting service (the "Website Owner") and the content creator (the "Creator") assume no responsibility or 
        liability for any harm, damage, or inconvenience that may arise from your use of the Website, including but not 
        limited to direct, indirect, incidental, consequential, or any other form of damages. The Website Owner and the 
        Creator cannot and will not be held liable for any loss, damage, or harm resulting from any situation related 
        to the Website.\n

        The Website Owner and the Creator are not responsible for any user-generated content and accounts. Content 
        submitted expresses the views of their author only.\n

        Your use of this Website is intended for personal, non-commercial purposes only. Any misuse or inappropriate 
        use of this Website is strictly prohibited.\n

        This Website may contain content generated by Artificial Intelligence (AI). The accuracy, completeness, or 
        usefulness of such content is not guaranteed, and the Website Owner and the Creator are not responsible for 
        any consequences resulting from the use or interpretation of AI-generated content.\n

        You agree to indemnify and hold harmless the Website Owner, the Creator, their contractors, and their 
        licensors, and their respective directors, officers, employees, and agents from and against any and all claims 
        and expenses, including attorneys’ fees, arising out of your use of the Website, including but not limited to 
        your violation of this Agreement.\n

        This Agreement constitutes the entire agreement between you and the Website operator and supersedes and 
        replaces any prior agreements, oral or otherwise, regarding the Website.\n

        By using the Website, you agree to be bound by these terms. If you do not agree to be bound by these terms, 
        please do not use the Website.\n

        This Agreement is subject to change by the Website operator at any time at their sole discretion. Your 
        continued use of this Website following the posting of any changes to this Agreement constitutes your 
        acceptance of those changes.\n
        """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
