import streamlit as st
import requests

BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Financial Sentiment Analysis", page_icon="📈")

st.title("Financial Sentiment Analysis Chatbot")

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

selected_company_display = st.selectbox("Select Company", list(company_options.keys()))
selected_ticker = company_options.get(selected_company_display, "")

# Query input
query = st.text_input("Enter your financial query")

if st.button("Analyze"):
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
                st.success("Analysis Result:")
                st.write(f"**Query:** {result['query']}")
                st.write(f"**Company:** {result['company']}")
                st.write(f"**Analysis:** {result['analysis_result']}")
                st.write(f"**Confidence:** {result['confidence']}")
                if result.get('predicted_price'):
                    st.write(f"**Predicted Price:** {result['predicted_price']}")
                if result.get('current_price'):
                    st.write(f"**Current Price:** {result['current_price']}")
                if result.get('sentiment_score'):
                    st.write(f"**Sentiment Score:** {result['sentiment_score']}")
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")
    else:
        st.warning("Please enter a query and select a company")

# Chat section
st.header("Chat Interface")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_message = st.text_input("Your message", key="chat_input")

if st.button("Send Message"):
    if user_message:
        payload = {
            "session_id": "streamlit_chat",
            "message": user_message
        }
        try:
            response = requests.post(f"{BASE_URL}/api/chat", json=payload)
            if response.status_code == 200:
                chat_data = response.json()
                st.session_state.chat_history = chat_data["messages"]
            else:
                st.error(f"Chat error: {response.status_code}")
        except Exception as e:
            st.error(f"Chat request failed: {e}")

# Display chat history
for msg in st.session_state.chat_history:
    if msg["type"] == "user":
        st.write(f"**You:** {msg['message']}")
    elif msg["type"] == "assistant":
        st.write(f"**Assistant:** {msg['analysis']}")