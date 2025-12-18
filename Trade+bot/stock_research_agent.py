from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import yfinance as yf
import logging

from langchain_core.tools import tool
from typing import Dict, Optional, cast
import pandas as pd
import json

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
# 1. Create a Gemini model
# Ensure GOOGLE_API_KEY is set in your environment
gemini_model = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    temperature=0,
)


@tool
def get_stock_price(symbol: str) -> str:
    """Get current stock price and basic information."""
    logging.info(f"[TOOL] Fetching stock price for: {symbol}")
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        hist = stock.history(period="1d")
        if hist.empty:
            logging.error("No historical data found")
            return json.dumps({"error": f"Could not retrieve data for {symbol}"})
            
        current_price = hist['Close'].iloc[-1]
        result = {
            "symbol": symbol,
            "current_price": round(current_price, 2),
            "company_name": info.get('longName', symbol),
            "market_cap": info.get('marketCap', 0),
            "pe_ratio": info.get('trailingPE', 'N/A'),
            "52_week_high": info.get('fiftyTwoWeekHigh', 0),
            "52_week_low": info.get('fiftyTwoWeekLow', 0)
        }
        logging.info(f"[TOOL RESULT] {result}")
        return json.dumps(result, indent=2)

    except Exception as e:
        logging.exception("Exception in get_stock_price")
        return json.dumps({"error": str(e)})

@tool
def get_financial_statements(symbol: str) -> str:
    """Retrieve key financial statement data."""
    try:
        stock = yf.Ticker(symbol)
        financials = stock.financials
        balance_sheet = stock.balance_sheet
        
        latest_year = financials.columns[0]
        
        period_source = getattr(latest_year, "year", latest_year)
        period_label = str(period_source)
        
        return json.dumps({
            "symbol": symbol,
            "period": period_label,
            "revenue": float(financials.loc['Total Revenue', latest_year]) if 'Total Revenue' in financials.index else 'N/A',
            "net_income": float(financials.loc['Net Income', latest_year]) if 'Net Income' in financials.index else 'N/A',
            "total_assets": float(balance_sheet.loc['Total Assets', latest_year]) if 'Total Assets' in balance_sheet.index else 'N/A',
            "total_debt": float(balance_sheet.loc['Total Debt', latest_year]) if 'Total Debt' in balance_sheet.index else 'N/A'
        }, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def get_technical_indicators(symbol: str, period: str = "3mo") -> str:
    """Calculate key technical indicators."""
    try:
        stock = yf.Ticker(symbol)
        hist = cast(pd.DataFrame, stock.history(period=period))
        
        if hist.empty:
            return f"Error: No historical data for {symbol}"
        
        close_series = cast(pd.Series, hist['Close'])
        hist['SMA_20'] = close_series.rolling(window=20).mean()
        hist['SMA_50'] = close_series.rolling(window=50).mean()
        
        delta = close_series.diff()
        gain = cast(pd.Series, delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = cast(pd.Series, (-delta.where(delta < 0, 0))).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        latest = hist.iloc[-1]
        latest_rsi = float(cast(pd.Series, rsi).iloc[-1])
        
        return json.dumps({
            "symbol": symbol,
            "current_price": round(latest['Close'], 2),
            "sma_20": round(latest['SMA_20'], 2),
            "sma_50": round(latest['SMA_50'], 2),
            "rsi": round(latest_rsi, 2),
            "volume": int(latest['Volume']),
            "trend_signal": "bullish" if latest['Close'] > latest['SMA_20'] > latest['SMA_50'] else "bearish"
        }, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_stock_recommendation(symbol: str) -> str:
    """Provide a simplified buy/hold/sell view or general best-idea list."""
    symbol = symbol.strip().upper()
    if not symbol:
        return json.dumps({"error": "Symbol is required."})
    
    if symbol == "BEST":
        # Use the LLM to generate recommendations
        prompt = (
            "Recommend 5 distinct stocks to buy right now. "
            "Return a valid JSON object with a single key 'recommendations'. "
            "The value of 'recommendations' must be a list of objects, each with 'symbol' and 'thesis' keys. "
            "Example: {\"recommendations\": [{\"symbol\": \"AAPL\", \"thesis\": \"Strong ecosystem.\"}]} "
            "Do not include any other text, markdown, or code blocks."
        )
        try:
            response = gemini_model.invoke(prompt)
            content = response.content
            logging.debug(f"LLM Response Content: {content}")
            
            # clean up potential markdown code blocks if present
            content = content.replace("```json", "").replace("```", "").strip()
            
            # Attempt to find the JSON object
            start_idx = content.find('{')
            end_idx = content.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                content = content[start_idx:end_idx+1]
            
            data = json.loads(content)
            
            return json.dumps({
                "mode": "best_ideas",
                "recommendations": data.get("recommendations", []),
                "note": "Generated by LLM based on general market knowledge."
            }, indent=2)
        except Exception as e:
            logging.error(f"Failed to get LLM recommendations: {e}")
            # Fallback if LLM fails or parsing fails
            return json.dumps({
                "error": "Could not generate recommendations at this time."
            })
    
    try:
        stock = yf.Ticker(symbol)
        hist = cast(pd.DataFrame, stock.history(period="6mo"))
        info = stock.info
        
        if hist.empty:
            return json.dumps({"error": f"No six-month data for {symbol}."})
        
        closing: pd.Series = cast(pd.Series, hist["Close"])
        latest_price = float(closing.iloc[-1])
        start_price = float(closing.iloc[0])
        six_month_return = ((latest_price - start_price) / start_price) * 100
        
        sma_50_series = cast(pd.Series, closing.rolling(window=50).mean())
        sma_50 = float(sma_50_series.iloc[-1])
        sma_200_series = cast(pd.Series, closing.rolling(window=200).mean())
        sma_200: Optional[float] = None
        non_null_200 = sma_200_series.dropna()
        if not non_null_200.empty:
            sma_200 = float(non_null_200.iloc[-1])
        
        delta: pd.Series = closing.diff()
        positive_delta = cast(pd.Series, delta.where(delta > 0, 0.0))
        negative_delta = cast(pd.Series, -delta.where(delta < 0, 0.0))
        gain = cast(pd.Series, positive_delta.rolling(window=14).mean())
        loss = cast(pd.Series, negative_delta.rolling(window=14).mean())
        rs_series = cast(pd.Series, gain / loss)
        valid_rs = rs_series.dropna()
        rsi: Optional[float] = None
        if not valid_rs.empty:
            last_rs = valid_rs.iloc[-1]
            if last_rs != 0:
                rsi = float(100 - (100 / (1 + last_rs)))
        
        profit_margin = info.get("profitMargins")
        pe_ratio = info.get("trailingPE")
        revenue_growth = info.get("revenueGrowth")
        
        score = 0
        rationale = []
        
        if six_month_return > 8:
            score += 1
            rationale.append("Positive momentum over the last six months.")
        elif six_month_return < -8:
            score -= 1
            rationale.append("Negative six-month momentum.")
        
        if rsi is not None:
            if rsi < 35:
                score += 1
                rationale.append("RSI indicates the stock is oversold.")
            elif rsi > 70:
                score -= 1
                rationale.append("RSI indicates the stock is overbought.")
        
        if latest_price > sma_50:
            score += 1
            rationale.append("Price is trading above the 50-day moving average.")
        else:
            score -= 1
            rationale.append("Price is below the 50-day moving average.")
        
        if sma_200:
            if latest_price > sma_200:
                score += 1
                rationale.append("Price is above the 200-day trend support.")
            else:
                score -= 1
                rationale.append("Price is below the 200-day trend support.")
        
        if profit_margin and profit_margin > 0.1:
            score += 1
            rationale.append("Healthy profit margins (>10%).")
        elif profit_margin and profit_margin < 0:
            score -= 1
            rationale.append("Negative profit margins.")
        
        if revenue_growth and revenue_growth > 0.05:
            score += 1
            rationale.append("Revenue growth is accelerating (>5%).")
        
        if pe_ratio and pe_ratio > 35:
            score -= 1
            rationale.append("Rich valuation (PE>35).")
        
        if score >= 2:
            recommendation = "buy"
        elif score <= -2:
            recommendation = "sell"
        else:
            recommendation = "hold"
        
        summary = f"Score {score}: {recommendation.upper()} based on trend, momentum, and profitability."
        
        return json.dumps({
            "mode": "single_symbol",
            "symbol": symbol,
            "latest_price": round(latest_price, 2),
            "six_month_return_pct": round(six_month_return, 2),
            "rsi": round(rsi, 2) if rsi is not None else "N/A",
            "sma_50": round(sma_50, 2),
            "sma_200": round(sma_200, 2) if sma_200 is not None else "N/A",
            "profit_margin": profit_margin if profit_margin is not None else "N/A",
            "revenue_growth": revenue_growth if revenue_growth is not None else "N/A",
            "pe_ratio": pe_ratio if pe_ratio is not None else "N/A",
            "score": score,
            "recommendation": recommendation,
            "summary": summary,
            "rationale": rationale
        }, indent=2)
    except Exception as exc:
        logging.exception("Exception in get_stock_recommendation")
        return json.dumps({"error": str(exc)})


# Main research instructions
# Using standard ReAct format - matching LangChain's default format exactly
template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
prompt = PromptTemplate.from_template(template)

# Define all tools
tools = [
    get_stock_price,
    get_financial_statements, 
    get_technical_indicators,
    get_stock_recommendation
]

# Create the agent (LangChain classic ReAct agent)
# Using standard ReAct prompt format
agent = create_react_agent(
    llm=gemini_model,
    tools=tools,
    prompt=prompt
)

# Create the agent executor
stock_research_agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)
