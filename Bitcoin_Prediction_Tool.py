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

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import time
import os
import requests
from io import StringIO

from deep_translator import GoogleTranslator
from datetime import datetime

from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options

from tensorflow.keras.models import load_model
import joblib
from datetime import datetime, date, time, timedelta




def btc_predict(human_prompt):
    """ 
    This tool is used for predicting Bitcoin (BTC) prices.
    It uses a trained LSTM model for prediction.
    """

    print("... Running btc_predict--")

    model_path = 'Trained_Model/btc_lstm_model.h5'
    scaler_path = 'Trained_Model/btc_scaler.save'

    # -----------------------------
    # SELENIUM CODE

    # Set Firefox options for headless mode (for Streamlit Cloud)
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    # Create service without hardcoded path — geckodriver will be in PATH if installed via packages.txt
    service = Service()

    # Initialize driver
    page_main = webdriver.Firefox(service=service, options=options)

    # Open CoinGecko BTC page
    page_main.get("https://www.coingecko.com/en/coins/bitcoin")
    page_main.implicitly_wait(3)

    # Get today's BTC price
    todays_price = page_main.find_element(
        By.XPATH,
        '//*[@class="tw-font-bold tw-text-gray-900 dark:tw-text-moon-50 tw-text-3xl md:tw-text-4xl tw-leading-10"]'
    ).text
    todays_price = todays_price.split('$')[1].replace(',', '')

    # Close browser
    page_main.quit()

    # -----------------------------
    # PREDICTION CODE

    start_price = float(todays_price)
    days = 10  # Fixed prediction length

    today = date.today()

    # Load trained model & scaler
    model = load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)

    current_price = np.array([[start_price]])  # 2D input
    days_pred_list = []

    print(f"\n📈 Predicting next {days} days of BTC opening prices...\n")

    for i in range(days):
        scaled_input = scaler.transform(current_price)
        model_input = scaled_input.reshape(1, 1, 1)
        pred_scaled = model.predict(model_input, verbose=0)
        pred_price = scaler.inverse_transform(pred_scaled)
        print(f"Day {i+1}: {pred_price[0][0]:.2f} USD")
        current_price = pred_price
        days_pred_list.append(current_price)

    # Format results
    result = f"📈 BTC Price Prediction starting from {today} using price ${start_price:.2f}:\n\n"
    for i, pred in enumerate(days_pred_list):
        pred_date = today + timedelta(days=i + 1)
        result += f"{pred_date}: ${pred[0][0]:.2f} USD\n"

    return {
        "summary": result.strip(),
        "todays_bitcoin_price": f"📈 Today's - ({today}) BTC price is ${start_price:.2f} USD"
    }
    
