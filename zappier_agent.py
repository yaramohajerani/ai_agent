from langchain.llms import OpenAI
from langchain.agents import initialize_agent
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.agents.agent_types import AgentType
from langchain.utilities.zapier import ZapierNLAWrapper


def main():
    llm = OpenAI(temperature=0)
    zapier = ZapierNLAWrapper()
    toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)
    agent = initialize_agent(toolkit.get_tools(),
                             llm,
                             agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                             verbose=True)

    for tool in toolkit.get_tools():
        print(tool.name)
        print(tool.description)
        print("\n\n")

    while True:
        user_prompt: str = input("Enter prompt for agent or type-in 'exit' to quit.")
        if user_prompt.strip().lower() == "exit":
            break

        agent.run(user_prompt)


if __name__ == "__main__":
    main()
