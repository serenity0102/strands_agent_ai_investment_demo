"""
Streamlit Web GUI for Investment Robot Advisor
"""

import streamlit as st
import pandas as pd
import json
import re
from investment_advisor_agent import InvestmentAdvisorAgent

# Set page configuration - THIS MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="AI Investment Robot Advisor",
    page_icon="💹",
    layout="wide",
)

# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    
if 'input_value' not in st.session_state:
    st.session_state.input_value = ""

# Function to handle example query selection
def set_example_query(query):
    st.session_state.input_value = query

# Initialize the investment advisor agent
@st.cache_resource
def get_advisor():
    return InvestmentAdvisorAgent()

advisor = get_advisor()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .disclaimer {
        font-size: 0.8rem;
        color: #64748B;
        font-style: italic;
    }
    .stButton>button {
        background-color: #1E3A8A;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .user-message {
        background-color: #E2E8F0;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .advisor-message {
        background-color: #F1F5F9;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 5px solid #1E3A8A;
    }
    .example-query {
        cursor: pointer;
        padding: 0.5rem;
        background-color: #F1F5F9;
        border-radius: 5px;
        margin-bottom: 0.5rem;
        border: 1px solid #CBD5E1;
    }
    .example-query:hover {
        background-color: #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">AI Investment Robot Advisor</div>', unsafe_allow_html=True)
st.markdown('<div>Your CFA-powered financial analysis assistant</div>', unsafe_allow_html=True)

# Main layout with tabs
tab1, tab2 = st.tabs(["Chat with Advisor", "Portfolio Analysis"])

with tab1:
    # Chat interface
    st.markdown('<div class="sub-header">Chat with Investment Advisor</div>', unsafe_allow_html=True)
    
    # Example queries
    with st.expander("Example Queries (Click to use)"):
        example_queries = [
            "Get market data for AMZN",
            "What's the latest news about Tesla?",
            "Generate an analyst report for AMZN",
            "Compare the performance of Nvidia, Amazon, and Tesla over the last month",
            "What's your recommendation for tech stocks in the current market?"
        ]
        
        for query in example_queries:
            if st.button(query, key=f"example_{query}"):
                set_example_query(query)
                st.rerun()
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f'<div class="user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
            else:
                # Try to format JSON responses nicely
                content = message['content']
                try:
                    # Check if the content is a JSON string
                    if isinstance(content, str) and (content.startswith('{') or content.startswith('[')):
                        json_obj = json.loads(content)
                        content = f"```json\n{json.dumps(json_obj, indent=2)}\n```"
                except:
                    pass
                
                st.markdown(f'<div class="advisor-message"><strong>Investment Advisor:</strong> {content}</div>', unsafe_allow_html=True)
    
    # User input
    user_input = st.text_input(
        "What would you like to know about investments?", 
        value=st.session_state.input_value,
        key="user_input"
    )
    
    # Send button
    if st.button("Send", key="send_button"):
        if user_input:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Get response from agent
            with st.spinner("Investment Advisor is thinking..."):
                response = advisor.agent(user_input)
            
            # Add advisor response to chat history
            st.session_state.chat_history.append({"role": "advisor", "content": response})
            
            # Clear the input for next time
            st.session_state.input_value = ""
            
            # Rerun to update the UI
            st.rerun()

with tab2:
    # Portfolio Analysis
    st.markdown('<div class="sub-header">Portfolio Analysis</div>', unsafe_allow_html=True)
    st.write("Analyze how your portfolio would perform under different market scenarios.")
    
    # Portfolio input
    portfolio_input = st.text_area(
        "Enter your portfolio (format: TICKER (WEIGHT%), e.g., AMZN (40%), MSFT (30%), GOOGL (30%))",
        height=100,
        placeholder="Example: AMZN (40%), MSFT (30%), GOOGL (30%)"
    )
    
    # Scenario selection
    scenario = st.selectbox(
        "Select market scenario",
        ["bull", "bear", "neutral", "recession", "inflation"],
        format_func=lambda x: {
            "bull": "Bull Market (Strong economic growth)",
            "bear": "Bear Market (Economic contraction)",
            "neutral": "Neutral Market (Moderate growth)",
            "recession": "Recession (Severe economic contraction)",
            "inflation": "Inflation (Rising prices, higher interest rates)"
        }[x]
    )
    
    # Time horizon
    time_horizon = st.selectbox(
        "Select time horizon",
        ["6m", "1y", "3y", "5y", "10y"],
        format_func=lambda x: {
            "6m": "6 Months",
            "1y": "1 Year",
            "3y": "3 Years",
            "5y": "5 Years",
            "10y": "10 Years"
        }[x],
        index=1
    )
    
    if st.button("Analyze Portfolio"):
        if portfolio_input:
            # Parse portfolio input
            try:
                # Extract ticker and weight pairs using regex
                pattern = r'([A-Z]+)\s*\(\s*(\d+(?:\.\d+)?)\s*%?\s*\)'
                matches = re.findall(pattern, portfolio_input)
                
                if not matches:
                    st.error("Invalid portfolio format. Please use the format: TICKER (WEIGHT%), e.g., AAPL (40%), MSFT (30%)")
                else:
                    portfolio = {}
                    for ticker, weight in matches:
                        portfolio[ticker] = float(weight) / 100  # Convert percentage to decimal
                    
                    # Normalize weights if they don't sum to 1
                    total_weight = sum(portfolio.values())
                    if abs(total_weight - 1.0) > 0.01:  # Allow small rounding errors
                        for ticker in portfolio:
                            portfolio[ticker] /= total_weight
                    
                    # Generate query for the agent
                    query = f"Estimate performance for this portfolio: {portfolio} in a {scenario} market scenario with {time_horizon} time horizon"
                    
                    with st.spinner("Analyzing portfolio..."):
                        response = advisor.agent(query)
                    
                    # Display results
                    st.subheader("Analysis Results")
                    
                    try:
                        # Try to parse the response as JSON
                        if isinstance(response, str) and (response.startswith('{') or response.startswith('[')):
                            result = json.loads(response)
                            
                            # Create a more user-friendly display
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Portfolio Composition**")
                                portfolio_df = pd.DataFrame({
                                    'Ticker': list(portfolio.keys()),
                                    'Weight': [f"{w*100:.1f}%" for w in portfolio.values()]
                                })
                                st.dataframe(portfolio_df)
                                
                                st.write("**Scenario Details**")
                                st.write(f"**Scenario:** {scenario.capitalize()}")
                                if 'scenario_description' in result:
                                    st.write(f"**Description:** {result['scenario_description']}")
                                st.write(f"**Time Horizon:** {time_horizon}")
                            
                            with col2:
                                st.write("**Expected Performance**")
                                if 'expected_performance' in result:
                                    perf = result['expected_performance']
                                    st.metric("Annualized Return", f"{perf.get('annualized_return', 0):.2f}%")
                                    st.metric("Final Value (of $1 invested)", f"${perf.get('mean_final_value', 0):.2f}")
                                    
                                    st.write("**Risk Metrics**")
                                    if 'risk_metrics' in result:
                                        risk = result['risk_metrics']
                                        st.metric("Volatility", f"{risk.get('volatility', 0):.2f}%")
                                        st.metric("Max Drawdown", f"{risk.get('max_drawdown', 0):.2f}%")
                            
                            # Show full JSON response in an expander
                            with st.expander("View Full Analysis Details"):
                                st.json(result)
                        else:
                            # If not JSON, just display the text response
                            st.write(response)
                    except:
                        # If parsing fails, just display the text response
                        st.write(response)
            except Exception as e:
                st.error(f"Error analyzing portfolio: {str(e)}")
        else:
            st.error("Please enter your portfolio details")

# Disclaimer
st.markdown('<div class="disclaimer">Disclaimer: This tool is for educational purposes only. The financial analysis and recommendations provided should not be considered as financial advice. Always consult with a qualified financial advisor before making investment decisions.</div>', unsafe_allow_html=True)
