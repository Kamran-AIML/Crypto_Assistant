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




from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

def btc_predict(human_prompt):
    """
    Predicts future Bitcoin prices using a trained LSTM model.
    Scrapes current BTC price using headless Chromium via Selenium.
    """
    print("... Running btc_predict -- (using headless Chromium)")

    model_path = 'Trained_Model/btc_lstm_model.h5'
    scaler_path = 'Trained_Model/btc_scaler.save'

    # Streamlit Cloud-compatible Selenium setup
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    service = Service()
    driver = webdriver.Chrome(service=service, options=options)

    driver.get("https://www.coingecko.com/en/coins/bitcoin")
    driver.implicitly_wait(3)

    price_elem = WebDriverWait(driver, 30).until(
        EC.visibility_of_element_located((
            By.XPATH,
            "//span[contains(@class, 'tw-font-bold') and " +
            "contains(@class, 'tw-text-gray-900') and " +
            "contains(@class, 'dark:tw-text-moon-50') and " +
            "contains(@class, 'tw-text-3xl') and " +
            "contains(@class, 'tw-leading-10')]"
        ))
    )
    
    price_text = price_elem.text
    
    driver.quit()

    # Extract numeric price
    todays_price = float(price_text.split('$')[1].replace(',', ''))

    # Load model and scaler
    model = load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)

    days = 10
    today = date.today()
    current_price = np.array([[todays_price]])
    days_pred_list = []

    print(f"\n📈 Predicting next {days} days of BTC prices...\n")
    for i in range(days):
        scaled = scaler.transform(current_price)
        pred_scaled = model.predict(scaled.reshape(1, 1, 1), verbose=0)
        pred_price = scaler.inverse_transform(pred_scaled)
        print(f"Day {i+1}: ${pred_price[0][0]:.2f} USD")
        current_price = pred_price
        days_pred_list.append(current_price)

    # Build summary text
    summary = f"📈 BTC Price Prediction from {today} starting at ${todays_price:.2f}:\n\n"
    for i, pred in enumerate(days_pred_list):
        pred_date = today + timedelta(days=i + 1)
        summary += f"{pred_date}: ${pred[0][0]:.2f} USD\n"

    return {
        "summary": summary.strip(),
        "todays_bitcoin_price": f"📈 Today's BTC price ({today}): ${todays_price:.2f} USD"
    }
