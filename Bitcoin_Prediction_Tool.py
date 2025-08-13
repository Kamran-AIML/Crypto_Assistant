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

# from selenium.webdriver.firefox.service import Service
# from selenium.webdriver.firefox.options import Options

from tensorflow.keras.models import load_model
import joblib
from datetime import datetime, date, time, timedelta





from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from tensorflow.keras.models import load_model
import joblib
import numpy as np
from datetime import date, timedelta

def get_chrome_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")  # Use new headless mode argument
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.binary_location = "/usr/bin/chromium"
    driver_path = "/usr/lib/chromium/chromedriver"
    service = Service(driver_path)
    return webdriver.Chrome(service=service, options=chrome_options)

def get_btc_price():
    driver = get_chrome_driver()
    driver.get("https://www.coingecko.com/en/coins/bitcoin")
    
    # Optionally wait for element visibility if needed
    
    todays_price = driver.find_element(
        By.XPATH,
        '//*[@class="tw-font-bold tw-text-gray-900 dark:tw-text-moon-50 tw-text-3xl md:tw-text-4xl tw-leading-10"]'
    ).text
    
    driver.quit()
    
    # Clean price string "$" and comma
    price_clean = todays_price.split('$')[1].replace(',', '')
    return float(price_clean)

def btc_predict():
    print("... Running btc_predict--")
    model_path = 'Trained_Model/btc_lstm_model.h5'
    scaler_path = 'Trained_Model/btc_scaler.save'
    
    start_price = get_btc_price()
    days = 10
    today = date.today()
    
    model = load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)
    
    current_price = np.array([[start_price]])
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
        
    result = f"📈 BTC Price Prediction starting from {today} using price ${start_price:.2f}:\n\n"
    for i, pred in enumerate(days_pred_list):
        pred_date = today + timedelta(days=i + 1)
        result += f"{pred_date}: ${pred[0][0]:.2f} USD\n"
        
    return {
        "summary": result.strip(),
        "todays_bitcoin_price": f"📈 Today's - ({today}) BTC price is ${start_price:.2f} USD"
    }

