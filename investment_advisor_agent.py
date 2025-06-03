"""
Investment Robot Advisor using Strands Agent SDK

This agent provides investment advice by:
1. Querying market data from Yahoo Finance
2. Querying market news from Yahoo Finance
3. Writing CFA analyst reports
4. Estimating portfolio performance under different market scenarios
"""

import os
from typing import List, Dict, Any, Optional
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from strands import Agent, tool
from strands.models import BedrockModel




# Define tools for the investment advisor agent
"""Tool to query market data from Yahoo Finance."""
@tool
def get_ticker_data(ticker: str, period: str = "1mo", interval: str = "1d") -> Dict[str, Any]:
    """
    Get historical market data for a specific ticker.

    Args:
        ticker: The stock ticker symbol (e.g., AAPL, MSFT)
        period: Time period to fetch data for (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)

    Returns:
        Dictionary containing historical market data
    """
    try:
        ticker_data = yf.Ticker(ticker)
        hist = ticker_data.history(period=period, interval=interval)

        # Convert DataFrame to dictionary for easier serialization
        result = {
            "ticker": ticker,
            "info": ticker_data.info,
            "history": hist.reset_index().to_dict(orient="records"),
            "current_price": hist["Close"].iloc[-1] if not hist.empty else None,
            "period": period,
            "interval": interval
        }
        return result
    except Exception as e:
        return {"error": str(e)}

@tool
def get_multiple_tickers_data(tickers: List[str], period: str = "1mo") -> Dict[str, Any]:
    """
    Get market data for multiple tickers.

    Args:
        tickers: List of stock ticker symbols
        period: Time period to fetch data for

    Returns:
        Dictionary containing data for all requested tickers
    """
    result = {}
    for ticker in tickers:
        result[ticker] = get_ticker_data(ticker, period)
    return result


@tool
def get_ticker_news(ticker: str, limit: int = 10) -> Dict[str, Any]:
    """
    Get recent news for a specific ticker.

    Args:
        ticker: The stock ticker symbol
        limit: Maximum number of news items to return

    Returns:
        Dictionary containing news data
    """
    try:
        ticker_data = yf.Ticker(ticker)
        news = ticker_data.news

        # Limit the number of news items and format them
        limited_news = news[:limit] if news else []
        formatted_news = []

        for item in limited_news:
            formatted_news.append({
                "title": item.get("title", ""),
                "publisher": item.get("publisher", ""),
                "link": item.get("link", ""),
                "published": datetime.fromtimestamp(item.get("providerPublishTime", 0)).strftime("%Y-%m-%d %H:%M:%S"),
                "summary": item.get("summary", "")
            })

        return {
            "ticker": ticker,
            "news_count": len(formatted_news),
            "news": formatted_news
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def generate_analyst_report(ticker: str, market_data: Dict[str, Any], news_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a CFA-style analyst report for a specific ticker.

    Args:
        ticker: The stock ticker symbol
        market_data: Market data from YahooFinanceMarketDataTool
        news_data: News data from YahooFinanceNewsTool

    Returns:
        Dictionary containing the analyst report
    """
    # In a real implementation, this would use an LLM to generate the report
    # For this example, we'll create a structured template

    try:
        info = market_data.get("info", {})
        history = market_data.get("history", [])
        news = news_data.get("news", [])

        # Calculate some basic metrics
        current_price = market_data.get("current_price")
        previous_close = info.get("previousClose")
        target_mean_price = info.get("targetMeanPrice")

        # Price change calculation
        price_change = 0
        price_change_percent = 0
        if current_price and previous_close:
            price_change = current_price - previous_close
            price_change_percent = (price_change / previous_close) * 100

        # Recommendation
        recommendation = "HOLD"  # Default
        if target_mean_price and current_price:
            upside_potential = ((target_mean_price - current_price) / current_price) * 100
            if upside_potential > 15:
                recommendation = "BUY"
            elif upside_potential < -5:
                recommendation = "SELL"

        # Create the report structure
        report = {
            "ticker": ticker,
            "company_name": info.get("shortName", ""),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "report_date": datetime.now().strftime("%Y-%m-%d"),
            "current_price": current_price,
            "price_change": {
                "value": price_change,
                "percent": price_change_percent
            },
            "target_price": {
                "low": info.get("targetLowPrice"),
                "mean": target_mean_price,
                "high": info.get("targetHighPrice")
            },
            "recommendation": recommendation,
            "financial_highlights": {
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "dividend_yield": info.get("dividendYield", 0) * 100 if info.get("dividendYield") else 0,
                "eps": info.get("trailingEps"),
                "revenue": info.get("totalRevenue"),
                "profit_margin": info.get("profitMargins", 0) * 100 if info.get("profitMargins") else 0
            },
            "analysis_summary": f"Based on current market data and news analysis, we are issuing a {recommendation} recommendation for {info.get('shortName', ticker)}.",
            "recent_developments": [item.get("title") for item in news[:3]],
            "risk_factors": [
                "Market volatility",
                "Industry competition",
                "Regulatory changes"
            ],
            "disclaimer": "This report is generated by an AI system and should not be considered as financial advice. Always consult with a qualified financial advisor before making investment decisions."
        }

        return report
    except Exception as e:
        return {"error": str(e)}

@tool
def estimate_performance(portfolio: Dict[str, float], scenario: str, time_horizon: str = "1y") -> Dict[str, Any]:
    """
    Estimate portfolio performance under different market scenarios.

    Args:
        portfolio: Dictionary mapping ticker symbols to their weights in the portfolio (should sum to 1.0)
        scenario: Market scenario to simulate ('bull', 'bear', 'neutral', 'recession', 'inflation')
        time_horizon: Time horizon for the simulation ('6m', '1y', '3y', '5y', '10y')

    Returns:
        Dictionary containing estimated portfolio performance
    """
    # Define scenario parameters (in a real implementation, these would be more sophisticated)
    scenario_params = {
        "bull": {
            "mean_annual_return": 0.15,
            "volatility": 0.18,
            "description": "Strong economic growth, low unemployment, rising corporate profits"
        },
        "bear": {
            "mean_annual_return": -0.10,
            "volatility": 0.25,
            "description": "Economic contraction, rising unemployment, declining corporate profits"
        },
        "neutral": {
            "mean_annual_return": 0.07,
            "volatility": 0.12,
            "description": "Moderate economic growth, stable employment, steady corporate profits"
        },
        "recession": {
            "mean_annual_return": -0.15,
            "volatility": 0.30,
            "description": "Severe economic contraction, high unemployment, significant decline in corporate profits"
        },
        "inflation": {
            "mean_annual_return": 0.04,
            "volatility": 0.20,
            "description": "Rising prices, higher interest rates, pressure on profit margins"
        }
    }

    # Convert time horizon to years
    time_horizon_years = {
        "6m": 0.5,
        "1y": 1,
        "3y": 3,
        "5y": 5,
        "10y": 10
    }.get(time_horizon, 1)

    try:
        # Get scenario parameters
        params = scenario_params.get(scenario.lower(), scenario_params["neutral"])

        # Fetch historical data for each ticker to estimate correlations
        tickers = list(portfolio.keys())
        weights = list(portfolio.values())

        # In a real implementation, we would use actual historical data and correlations
        # For this example, we'll use a simplified Monte Carlo simulation

        # Number of simulations
        num_simulations = 1000

        # Generate random returns based on scenario parameters
        annual_return = params["mean_annual_return"]
        annual_volatility = params["volatility"]

        # Daily parameters (assuming 252 trading days per year)
        daily_return = annual_return / 252
        daily_volatility = annual_volatility / np.sqrt(252)

        # Number of trading days in simulation
        num_days = int(252 * time_horizon_years)

        # Initialize array for final portfolio values
        final_values = np.zeros(num_simulations)

        # Run simulations
        for i in range(num_simulations):
            # Simulate each ticker's performance
            ticker_final_values = []
            for ticker in tickers:
                # Generate random daily returns
                daily_returns = np.random.normal(daily_return, daily_volatility, num_days)

                # Calculate cumulative returns
                cumulative_returns = np.cumprod(1 + daily_returns)

                # Final value of $1 invested
                ticker_final_values.append(cumulative_returns[-1])

            # Calculate weighted portfolio return
            portfolio_final_value = sum(v * w for v, w in zip(ticker_final_values, weights))
            final_values[i] = portfolio_final_value

        # Calculate statistics
        mean_final_value = np.mean(final_values)
        median_final_value = np.median(final_values)
        percentile_5 = np.percentile(final_values, 5)
        percentile_95 = np.percentile(final_values, 95)

        # Calculate annualized return
        annualized_return = (mean_final_value ** (1 / time_horizon_years)) - 1

        return {
            "scenario": scenario,
            "scenario_description": params["description"],
            "time_horizon": time_horizon,
            "portfolio": portfolio,
            "expected_performance": {
                "mean_final_value": mean_final_value,
                "median_final_value": median_final_value,
                "annualized_return": annualized_return * 100,  # Convert to percentage
                "percentile_5": percentile_5,
                "percentile_95": percentile_95
            },
            "risk_metrics": {
                "volatility": annual_volatility * 100,  # Convert to percentage
                "max_drawdown": (1 - percentile_5) * 100 if percentile_5 < 1 else 0  # Convert to percentage
            },
            "disclaimer": "These projections are based on simplified models and historical data. Actual results may vary significantly."
        }
    except Exception as e:
        return {"error": str(e)}


# Create the Investment Advisor Agent
class InvestmentAdvisorAgent:
    def __init__(self):
        # Create a BedrockModel
        bedrock_model = BedrockModel(
            model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            region_name='us-east-1',
            temperature=0.3,
        )
        
        # Create the agent
        self.agent = Agent(
            model=bedrock_model,
            system_prompt="You are professional CFA to provides market data, news, analyst reports, and portfolio performance estimates under different scenarios.",
            tools=[get_ticker_data, get_multiple_tickers_data, get_ticker_news, generate_analyst_report, estimate_performance]
        )
    
    def run(self):
        """Run the agent in interactive mode."""
        print("Investment Advisor Agent is running. Type 'exit' to quit.")
        
        while True:
            user_input = input("\nWhat would you like to know about investments? ")
            
            if user_input.lower() in ["exit", "quit"]:
                print("Thank you for using the Investment Advisor. Goodbye!")
                break
            
            # Process the user's message
            response = self.agent(user_input)
            
            # Print the agent's response
            print("\nInvestment Advisor:")
            print(response)


if __name__ == "__main__":
    advisor = InvestmentAdvisorAgent()
    advisor.run()
