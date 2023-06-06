"""
Simple command-line version of the toolkit
Note API keys have to be added to environmental variables
"""
from argparse import ArgumentParser, Namespace
from sys import exit as sys_exit
from typing import Optional, Union

from langchain.agents import initialize_agent
from langchain.agents.agent import AgentExecutor
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.agents.agent_types import AgentType
from langchain.llms import OpenAI, HuggingFaceHub
from langchain.utilities.zapier import ZapierNLAWrapper


def main():
    # set up path references
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--llm', help='choose LLM to use (OpenAI, Hugging Face)',
                        type=str, required=True)
    parser.add_argument('--model_name', help='Name of specific model to use.',
                        type=str, required=False, default=None)
    args: Namespace = parser.parse_args()

    llm_type: str = args.llm.strip().lower()
    model_name: Optional[str] = args.model_name

    # initialize llm
    llm: Optional[Union[OpenAI, HuggingFaceHub]] = None
    if llm_type == 'openai':
        if model_name:
            llm = OpenAI(temperature=0, model_name=model_name)
        else:
            llm = OpenAI(temperature=0, model_name="text-davinci-003")
    elif llm_type == 'hugging face':
        if model_name:
            llm = HuggingFaceHub(repo_id=model_name, model_kwargs={"temperature": 0})
        else:
            llm = HuggingFaceHub(repo_id='tiiuae/falcon-40b', model_kwargs={"temperature": 0})
    else:
        print("LLM must be either openai or hugging face")
        sys_exit()

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
