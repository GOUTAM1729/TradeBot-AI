# Trading AI Bot

A multi-agent AI trading assistant built with LangChain and Ollama that provides intelligent stock research and general question answering capabilities.

## Features

- **Stock Research Agent**: Specialized agent for analyzing stocks, retrieving financial data, and calculating technical indicators
  - Get current stock prices and basic information
  - Retrieve financial statements (revenue, net income, assets, debt)
  - Calculate technical indicators (SMA, RSI, trend signals)
  
- **General Agent**: Conversational AI agent for answering general questions

- **Intelligent Routing**: Automatically routes queries to the appropriate agent based on keywords

## Requirements

- Python 3.13+
- Ollama installed and running
- llama3 model downloaded in Ollama

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd "Trade+bot"
   ```

2. **Create a virtual environment (if not already created):**
   ```bash
   python3 -m venv venv
   ```

3. **Activate the virtual environment:**
   ```bash
   # On macOS/Linux:
   source venv/bin/activate
   
   # On Windows:
   venv\Scripts\activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Install and set up Ollama:**
   - Download Ollama from [https://ollama.ai](https://ollama.ai)
   - Install and start Ollama service
   - Pull the llama3 model:
     ```bash
     ollama pull llama3
     ```

## Usage

1. **Activate the virtual environment** (if not already activated):
   ```bash
   source venv/bin/activate
   ```

2. **Run the application:**
   ```bash
   python main.py
   ```

3. **Enter your queries:**
   - For stock-related queries (e.g., "What's the price of AAPL?", "Get financials for TSLA"), the Stock Research Agent will be used
   - For general questions, the General Agent will handle them
   - Type `exit` to quit the application

### Example Queries

**Stock Research:**
- "What's the current price of AAPL?"
- "Get financial statements for TSLA"
- "Show me technical indicators for MSFT"
- "What are the financials for GOOGL?"

**General Questions:**
- "What is machine learning?"
- "Explain quantum computing"
- "Tell me about Python"

## Project Structure

```
Trade+bot/
├── main.py                    # Main application entry point
├── router.py                   # Query routing logic
├── general_agent.py           # General conversational agent
├── stock_research_agent.py     # Stock research agent with tools
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Components

### Main Application (`main.py`)
- Initializes both agents
- Handles user input loop
- Routes queries to appropriate agents
- Displays results

### Router (`router.py`)
- Analyzes queries for keywords
- Routes to "stock_research" or "general" agent
- Keywords for stock research: "stock", "price", "financials", "technical", "market"

### General Agent (`general_agent.py`)
- Uses LangChain LCEL (LangChain Expression Language)
- Powered by Ollama's llama3 model
- Handles general conversational queries

### Stock Research Agent (`stock_research_agent.py`)
- Uses LangChain Classic ReAct agent framework
- Equipped with specialized tools:
  - `get_stock_price`: Retrieves current stock price and basic info
  - `get_financial_statements`: Gets revenue, net income, assets, debt
  - `get_technical_indicators`: Calculates SMA, RSI, and trend signals
- Uses yfinance for real-time stock data

## Configuration

### Changing the Model

To use a different Ollama model, edit the model name in:
- `general_agent.py` (line 11)
- `stock_research_agent.py` (line 16)

Example:
```python
ollama_model = ChatOllama(
    model="llama3.2",  # Change to your preferred model
    temperature=0,
)
```

### Adjusting Temperature

Modify the `temperature` parameter in the model initialization:
- `0` = More deterministic, focused responses
- `1` = More creative, varied responses

## Troubleshooting

### Error: "model 'llama3' not found"
- Ensure Ollama is running: `ollama list`
- Pull the model: `ollama pull llama3`

### Error: "does not support tools"
- The stock research agent uses `langchain_classic` which is compatible with models that don't support native tool calling
- This is already configured correctly in the code

### Import Errors
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`

### Ollama Connection Issues
- Verify Ollama service is running: `ollama list`
- Check if Ollama is accessible: `curl http://localhost:11434/api/tags`

## Dependencies

- **langchain**: Core LangChain framework
- **langchain-community**: Community integrations
- **langchain-ollama**: Ollama integration for LangChain
- **langchain-core**: Core LangChain components
- **yfinance**: Yahoo Finance API for stock data
- **gradio**: (Optional) For future web UI
- **lark**: (Optional) Parsing library

## License

This project is for educational and personal use.

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## Notes

- The application requires an active internet connection for stock data retrieval
- Stock data is provided by Yahoo Finance via yfinance
- Model responses depend on the Ollama model's capabilities
- For production use, consider adding error handling, logging, and rate limiting

