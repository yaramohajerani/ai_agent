"""
Simple command-line version of the toolkit
Note API keys have to be added to environmental variables
"""
from langchain.agents import initialize_agent
from langchain.agents.agent import AgentExecutor
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.agents.agent_types import AgentType
from langchain.llms import OpenAI
from langchain.utilities.zapier import ZapierNLAWrapper


def main():
    llm: OpenAI = OpenAI(temperature=0)
    zapier: ZapierNLAWrapper = ZapierNLAWrapper()
    toolkit: ZapierToolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)
    agent: AgentExecutor = initialize_agent(toolkit.get_tools(),
                                            llm,
                                            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                            verbose=True)

    # display available tools
    for tool in toolkit.get_tools():
        print(tool.name)
        print(tool.description)
        print("\n\n")

    # run prompts until user exists
    while True:
        user_prompt: str = input("Enter prompt or type 'exit' to end session.")
        if user_prompt.strip().lower() == "exit":
            break

        agent.run(user_prompt)


if __name__ == "__main__":
    main()
