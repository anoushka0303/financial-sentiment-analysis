import streamlit as st
import requests

BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Financial Sentiment Analysis", page_icon="📈")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .sidebar .stButton button {
        width: 100%;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for authentication
with st.sidebar:
    st.header("🔐 Authentication")
    st.markdown("---")
    if st.button("📝 Register", key="register"):
        st.info("Registration feature coming soon!")
    if st.button("🔑 Login", key="login"):
        st.info("Login feature coming soon!")
    st.markdown("---")
    st.markdown("**Note:** Authentication is optional for demo purposes.")

st.markdown('<h1 class="main-header">📈 Financial Analysis Tool</h1>', unsafe_allow_html=True)

# Fetch companies
try:
    response = requests.get(f"{BASE_URL}/api/companies")
    if response.status_code == 200:
        companies = response.json()["companies"]
        company_options = {f"{c['name']} ({c['ticker']})": c['ticker'] for c in companies}
    else:
        st.error("Failed to load companies")
        company_options = {}
except Exception as e:
    st.error(f"Error: {e}")
    company_options = {}

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    selected_company_display = st.selectbox("🏢 Select Company", list(company_options.keys()))

with col2:
    query = st.text_input("🔍 Enter your financial query")

selected_ticker = company_options.get(selected_company_display, "")

# Analyze button
if st.button("🚀 Analyze", type="primary"):
    if query and selected_ticker:
        payload = {
            "query": query,
            "company": selected_ticker,
            "session_id": "streamlit_session"
        }
        try:
            response = requests.post(f"{BASE_URL}/api/analyze", json=payload)
            if response.status_code == 200:
                result = response.json()
                st.success("✅ Analysis completed successfully!")

                # Display results in a styled box
                with st.container():
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.markdown(f"**🔍 Query:** {result['query']}")
                    st.markdown(f"**🏢 Company:** {result['company']}")
                    st.markdown("**📊 Analysis:**")
                    st.write(result['analysis_result'])
                    st.markdown(f"**🎯 Confidence:** {result['confidence']:.2%}")
                    if result.get('predicted_price'):
                        st.markdown(f"**💰 Predicted Price:** ₹{result['predicted_price']:.2f}")
                    if result.get('current_price'):
                        st.markdown(f"**📈 Current Price:** ₹{result['current_price']:.2f}")
                    if result.get('sentiment_score'):
                        st.markdown(f"**😊 Sentiment Score:** {result['sentiment_score']:.2f}")
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error(f"❌ Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"❌ Request failed: {e}")
    else:
        st.warning("Please enter a query and select a company")