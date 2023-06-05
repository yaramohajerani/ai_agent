from contextlib import redirect_stdout
from io import StringIO
from os import environ
from re import compile as re_compile
from typing import Pattern

import streamlit as st
from htbuilder import HtmlElement, div, hr, a, p, styles
from htbuilder.units import percent, px
from langchain.agents import initialize_agent
from langchain.agents.agent import AgentExecutor
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.agents.agent_types import AgentType
from langchain.llms import OpenAI
from langchain.utilities.zapier import ZapierNLAWrapper


@st.cache_resource
def setup_agent(openai_api_key: str, zapier_api_key: str) -> AgentExecutor:
    """
    Set up agents using the Zapier NLA (Natural Language Actions) toolkit

    :param openai_api_key: OpenAI API key provided by user
    :param zapier_api_key: Zapier NLA API key provided by user
    :return:
    """
    # set api keys as environmental variables
    environ['OPENAI_API_KEY'] = openai_api_key
    environ['ZAPIER_NLA_API_KEY'] = zapier_api_key

    llm: OpenAI = OpenAI(temperature=0)
    zapier: ZapierNLAWrapper = ZapierNLAWrapper()
    toolkit: ZapierToolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)
    agent: AgentExecutor = initialize_agent(toolkit.get_tools(),
                                            llm,
                                            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                            verbose=True)

    st.sidebar.markdown("**Available Tools based on your Zapier API key**")
    for tool in toolkit.get_tools():
        tool_expander = st.sidebar.expander(tool.name)
        tool_expander.write(tool.description)

    return agent


def footer_layout(*args) -> None:
    """
    Layout for footer

    :param args: footer args passed as a list (text and links html blocks as separate argument)
    :return: None
    """
    style = """
    <style>
      MainMenu {visibility: visible;}
      footer {visibility: visible;}
    </style>
    """

    style_div: str = styles(
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        text_align="center",
        height="60px",
        opacity=0.6
    )

    style_hr: str = styles()

    body = p()
    foot = div(style=style_div)(hr(style=style_hr), body)

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)
        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)


def add_footer() -> None:
    """
    Add footer to page

    :return: None
    """
    footer_args = [
        "<b>Yara Mohajerani &copy; 2023 | Contact ",
        a(_href="https://twitter.com/yara_mo_", _target="_blank")("@yara_mo_")
    ]
    footer_layout(*footer_args)


def remove_ansi_escape_codes(text: str) -> str:
    """
    Langchain output includes ANSI escape codes for colors. This function removes them.

    :param text: Input text with ANSI escape codes
    :return: cleaned text
    """
    ansi_escape_pattern: Pattern = re_compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape_pattern.sub('', text)


def main():
    col1, col2 = st.columns(2)
    col1.title("Welcome to your AI's central command! 🤖")
    col1.markdown("*:lightgrey[where you can use the power of ChatGPT to read and create emails, "
                  "Slack messages, reminders and calendars events, interact with users and documents, and more...]*")
    col2.image("images.dir/robot_logo.png")

    st.video('https://youtu.be/DPoGo01QO5Y', format="video/mp4", start_time=0)

    main_tab, setup_tab, docs_tab = st.tabs(["Main", "How to Setup", "Other Notes and Docs"])

    with main_tab:
        openai_api_key: str = st.sidebar.text_input("OPENAI API KEY", type="password")
        zapier_api_key: str = st.sidebar.text_input("ZAPIER NLA API KEY", type="password")

        if openai_api_key.strip() != '' and zapier_api_key.strip() != '':
            # initialize agents
            agent: AgentExecutor = setup_agent(openai_api_key, zapier_api_key)

            user_prompt: str = st.text_area("Enter Prompt", placeholder='Enter Prompt', label_visibility='collapsed')

            if user_prompt.strip() not in ["", "Enter Prompt"]:
                stdout: StringIO = StringIO()
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

                stdout.close()
        else:
            st.write("👈 Provide your API keys on the left to see the toolkit 🙂")

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
        
        I recommend adding actions for your daily tools like gmail, calendar, docs, sheets, and Slack to get the 
        most out of your AI assistant!
        """, unsafe_allow_html=True)

        # add screenshots of Zapier actiosn
        image_col1, image_col2 = st.columns(2)
        image_col1.image('images.dir/gmail_actions.png')
        image_col2.image('images.dir/calendar_actions.png')
        image_col1.image('images.dir/slack_actions.png')
        image_col2.image('images.dir/docs_actions.png')

    with docs_tab:
        st.markdown("""
        This portal is made for fun and personal use by [Yara Mohajerani](https://twitter.com/yara_mo_), and all the 
        code is accessible in the associated Github repo: 
        [https://github.com/yaramohajerani/ai_agent](https://github.com/yaramohajerani/ai_agent)
        
        \n\nAnd here is an important disclaimer made by GPT-4 because why not...\n
        
        **Disclaimer**\n

        This website, its content, and its services (collectively, the "Website") are provided "as is," without 
        warranty of any kind, either express or implied. The use of this Website is entirely at your own risk.\n

        The owner and operator of this Website assumes no responsibility or liability for any harm, damage, or 
        inconvenience that may arise from your use of the Website, including but not limited to direct, indirect, 
        incidental, consequential, or any other form of damages. The owner and operator of the Website cannot and will 
        not be held liable for any loss, damage, or harm resulting from any situation related to the Website.

        The owner and operator of this Website are not responsible for any user-generated content and accounts. Content 
        submitted expresses the views of their author only.\n

        Your use of this Website is intended for personal, non-commercial purposes only. Any misuse or inappropriate 
        use of this Website is strictly prohibited.\n

        This Website may contain content generated by Artificial Intelligence (AI). The accuracy, completeness, or 
        usefulness of such content is not guaranteed, and the owner and operator of this Website are not responsible for 
        any consequences resulting from the use or interpretation of AI-generated content.\n

        You agree to indemnify and hold harmless the owner and operator of this Website, their contractors, and their 
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
    add_footer()
