from os import environ
from langchain.llms import OpenAI
from langchain.agents import initialize_agent
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.agents.agent_types import AgentType
from langchain.agents.agent import AgentExecutor
from langchain.utilities.zapier import ZapierNLAWrapper
import streamlit as st


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

    return agent


def page_footer() -> None:
    """
    display page footer

    :return: None
    """
    st.markdown("""---""")
    st.markdown("<center><p>Yara Mohajerani &copy; 2023. All Rights Reserved.</p></center>",
                unsafe_allow_html=True)


def check_password() -> bool:
    """
    Returns `True` if the user had the correct password.

    :return: boolean for whether entered password is correct
    """

    def password_entered() -> None:
        """
        Checks whether a password entered by the user is correct.

        :return: None
        """
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        page_footer()
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("Incorrect Password")
        page_footer()
        return False
    else:
        # Password correct.
        return True


def main():
    openai_api_key: str = st.sidebar.text_input("OPENAI API KEY", type="password")
    zapier_api_key: str = st.sidebar.text_input("ZAPIER NLA API KEY", type="password")

    if openai_api_key != '' and zapier_api_key != '':
        # initialize agents
        agent: AgentExecutor = setup_agent(openai_api_key, zapier_api_key)

        user_prompt: str = st.text_input("Enter Prompt")

        if user_prompt.strip() != "":
            st.info(agent.run(user_prompt))

    page_footer()


if __name__ == '__main__':
    col1, col2 = st.columns([3, 2])
    col1.title("Yara's Central AI Command 👹🤖")
    col1.caption("BETA")
    col2.image(
        "DALL·E 2023-06-04 14.20.46 - High resolution realistic image of a command room where a humanoid robot "
        "sitting behind a large desk controls the world on a giant wall-to-wall screen.png")
    if check_password():
        main()
