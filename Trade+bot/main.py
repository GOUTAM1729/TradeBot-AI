import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from router import route_query
from stock_research_agent import stock_research_agent_executor
from general_agent import create_general_agent

def main():
    """
    Main function to run the multi-agent application.
    """
    general_agent = create_general_agent()

    while True:
        query = input("Enter your query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break

        agent_name = route_query(query)

        if agent_name == "stock_research":
            print("\n--- Running Stock Research Agent ---")
            result = stock_research_agent_executor.invoke({
                "input": query
            })
            print(result.get("output", "Error: No output from agent."))
            print("-------------------------------------\n")
        elif agent_name == "general":
            print("\n--- Running General Agent ---")
            result = general_agent.invoke({
                "question": query
            })
            print(result.get("text", "Error: No output from agent."))
            print("-----------------------------\n")
        else:
            print("Unknown agent.")

if __name__ == "__main__":
    main()

