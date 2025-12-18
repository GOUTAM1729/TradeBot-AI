def route_query(query: str) -> str:
    """
    Routes a query to the appropriate agent based on keywords.
    """
    query = query.lower()
    if any(keyword in query for keyword in ["stock", "price", "financials", "technical", "market"]):
        return "stock_research"
    else:
        return "general"
