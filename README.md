# AI Investment Robot Advisor by AWS Strands Agent SDK

An AI-powered investment advisor built using the Strands Agent SDK that provides financial analysis and portfolio recommendations.

## Features

- Query market data from Yahoo Finance
- Retrieve market news from Yahoo Finance
- Generate CFA-style analyst reports
- Estimate portfolio performance under different market scenarios

## Installation

1. Clone this repository
2. Create a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Command Line Interface

Run the investment advisor agent in the terminal:

```bash
python investment_advisor_agent.py
```

### Web Interface (Streamlit)

Run the Streamlit web application:

```bash
# Make sure your virtual environment is activated
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Run Streamlit using the Python module approach to ensure the correct environment is used
python -m streamlit run app.py
```

This will start a local web server and open the application in your default web browser.

> **Note:** If you encounter module import errors with Streamlit, always use the `python -m streamlit run app.py` command instead of just `streamlit run app.py` to ensure the correct Python environment is used.

## Example Interactions

Here are some example queries you can ask the agent:

- "Get market data for AAPL"
- "What's the latest news about Tesla?"
- "Generate an analyst report for MSFT"
- "Estimate how my portfolio would perform in a recession scenario"
- "Compare the performance of AAPL, MSFT, and GOOGL over the last month"
- "What's your recommendation for tech stocks in the current market?"

## Portfolio Analysis

To analyze a portfolio, specify the tickers and their weights:

```
Analyze this portfolio: AAPL (40%), MSFT (30%), GOOGL (20%), AMZN (10%) in a bull market scenario
```

## Troubleshooting

### AWS Connection Issues

If you encounter AWS connection timeouts like:
```
EventLoopException: AWSHTTPSConnectionPool(host='bedrock-runtime.us-east-1.amazonaws.com', port=443): Read timed out.
```

Try the following:
1. Check your AWS credentials and permissions
2. Verify your network connection
3. Try a different AWS region if available
4. Check the AWS Service Health Dashboard for any service disruptions

### Streamlit Import Errors

If you see "No module named 'strands'" errors when running Streamlit:
1. Make sure you're using the virtual environment where strands-agents is installed
2. Run Streamlit using the Python module approach: `python -m streamlit run app.py`

## Customization

You can extend the agent by adding more tools or enhancing the existing ones:

- Add more data sources beyond Yahoo Finance
- Implement more sophisticated portfolio analysis algorithms
- Integrate with trading platforms
- Add support for cryptocurrency analysis

## Disclaimer

This tool is for educational purposes only. The financial analysis and recommendations provided should not be considered as financial advice. Always consult with a qualified financial advisor before making investment decisions.
