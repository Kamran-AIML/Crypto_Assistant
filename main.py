from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait as wait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver import ActionChains

import csv
import logging
import uuid
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import time
import requests
from io import StringIO
from deep_translator import GoogleTranslator
from datetime import datetime

# Import your custom tools
from Bitcoin_Prediction_Tool import btc_predict
from Crypto_Latest_News_Scrap_Tool_Version_2 import crypto_latest_news, detail_crypto_latest_news

# LangChain imports
from langchain.agents import Tool, initialize_agent
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

# Streamlit imports
import streamlit as st
from streamlit_lottie import st_lottie
import requests
import json

#------------------------------------------

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OpenAI API key is not set in environment variables.")
    st.stop()

# Initialize LLM
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)

# Define tools
tools = [
    Tool(name="Crypto_Latest_News", func=crypto_latest_news, 
         description="Latest crypto news provider"),
    Tool(name="Detail_Crypto_Latest_News", func=detail_crypto_latest_news, 
         description="Fetch full crypto news article content by URL"),
    Tool(name="BTC_Predict", func=btc_predict, 
         description="Predict Bitcoin price for the next N days. Input: number of days. Output: Price forecast."),
]

# Initialize conversation memory
memory = ConversationBufferMemory(memory_key="chat_history")

# Initialize agent with memory
if 'agent' not in st.session_state:
    st.session_state.agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent="conversational-react-description",  # Changed for conversation support
        verbose=True,
        memory=memory,
        handle_parsing_errors=True
    )
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your crypto assistant. How can I help you today?"}]

# #---------------------------------------------------------


# with open("assets/crypto_animation_2.json", "r") as f:
#     lottie_json = json.load(f)

# st_lottie(lottie_json, speed=0.1, width=300, height=300, key="crypto_anim")

# #---------------------------------------------------------

# st.title("💰 Crypto Assistant")
# st.caption("Ask about crypto news, price predictions, or get detailed articles")


#-----------------------------------------------------------------

# Row with two animations at left and right
col_left, col_right = st.columns([1, 1])

with col_left:
    with open("assets/crypto_animation_2.json", "r") as f:
        lottie_json = json.load(f)
    st_lottie(lottie_json, speed=0.1, width=300, height=300, key="crypto_anim_left")

with col_right:
    with open("assets/crypto_animation_2.json", "r") as f:
        lottie_json = json.load(f)
    st_lottie(lottie_json, speed=0.1, width=300, height=300, key="crypto_anim_right")

# Below animations: full width title + caption
st.title("💰 Crypto Assistant")
st.caption("Ask about crypto news, price predictions, or get detailed articles")
#-----------------------------------------------------------------

def send_message(user_input):
    try:
        response = st.session_state.agent.run(input=user_input)
        return response
    except Exception as e:
        return f"Error processing your request: {str(e)}"

# Display chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Chat input
if prompt := st.chat_input("Ask about crypto..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    st.chat_message("user").write(prompt)
    
    # Get and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = send_message(prompt)
            st.write(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})