from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

class GeneralAgent:
    """Wrapper class to maintain compatibility with the old LLMChain interface."""
    
    def __init__(self, chain):
        self.chain = chain
    
    def invoke(self, input_dict):
        """Invoke the chain and return result in the expected format."""
        result = self.chain.invoke(input_dict)
        # Extract text content from AIMessage
        if hasattr(result, 'content'):
            return {"text": result.content}
        return {"text": str(result)}

def create_general_agent():
    """
    Creates a conversational agent that can answer general questions.
    """
    # 1. Create a Gemini model
    # Ensure GOOGLE_API_KEY is set in your environment
    gemini_model = ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        temperature=0,
    )

    # 2. Create a prompt template
    template = """
    You are a helpful assistant. Answer the following question to the best of your ability.

    Question: {question}
    Answer:
    """
    prompt = PromptTemplate.from_template(template)

    # 3. Create a chain using LCEL (LangChain Expression Language)
    chain = prompt | gemini_model
    
    # 4. Wrap in compatibility class
    agent = GeneralAgent(chain)

    return agent
