# Trading AI Bot

A multi-agent AI trading assistant built with LangChain and Google Gemini API that provides intelligent stock research and general question answering capabilities.

## Features

- **Stock Research Agent**: Specialized agent for analyzing stocks, retrieving financial data, and calculating technical indicators
  - Get current stock prices and basic information
  - Retrieve financial statements (revenue, net income, assets, debt)
  - Calculate technical indicators (SMA, RSI, trend signals)
  - **New**: Powered by Gemini 2.0 Flash for faster and smarter analysis
  
- **General Agent**: Conversational AI agent for answering general questions

- **Intelligent Routing**: Automatically routes queries to the appropriate agent based on keywords

## Requirements

- Python 3.13+
- Google API Key (Get one from [Google AI Studio](https://aistudio.google.com/))

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

5. **Set up Google API Key:**
   - Create a `.env` file in the root directory:
     ```env
     GOOGLE_API_KEY=your_api_key_here
     ```
   - Or export it in your shell:
     ```bash
     export GOOGLE_API_KEY=your_api_key_here
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
- "Recommend some stocks to buy" (Uses LLM analysis)

**General Questions:**
- "Who is the CEO of Google?"
- "Explain quantum computing"

## Project Structure

```
Trade+bot/
├── main.py                    # Main application entry point
├── router.py                   # Query routing logic
├── general_agent.py           # General conversational agent
├── stock_research_agent.py     # Stock research agent with tools
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables (API Key)
└── README.md                  # This file
```

## Troubleshooting

### Error: "404 NOT_FOUND" or "Model not found"
- This usually means the specified model (e.g., `gemini-1.5-pro`) is not available to your API key or in your region.
- The code is currently configured to use `gemini-flash-latest`.

### Error: "429 RESOURCE_EXHAUSTED"
- You are hitting the rate limit of the API (Free tier is often 5 RPM or 15 RPM).
- Wait a minute and try again.

### Import Errors
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`

## Dependencies

- **langchain-google-genai**: Google Gemini integration for LangChain
- **yfinance**: Yahoo Finance API for stock data
- **python-dotenv**: For loading environment variables


