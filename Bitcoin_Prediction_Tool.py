from selenium import webdriver
# from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
# from selenium.webdriver.chrome.service import Service
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

#--------------------------------
import os
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
#--------------------------------

def btc_predict(human_prompt):

    """ 
    This tool is used for predicting Bitcoin (BTC) prices. It uses a trained LSTM model for prediction
    """

    print("... Running btc_predict--")

    model_path = 'Trained_Model/btc_lstm_model.h5'
    scaler_path = 'Trained_Model/btc_scaler.save'

    #-----------------------------
    geckodriver_path = os.path.join(os.path.dirname(__file__), 'geckodriver.exe')  # Adjust for your actual executable
    service = Service(executable_path=geckodriver_path)

    options = Options()
    options.add_argument("--headless")

    driver = webdriver.Firefox(service=service, options=options)
    driver.get("https://www.coingecko.com/en/coins/bitcoin")

    price = driver.find_element(
        By.XPATH,
        '//*[@class="tw-font-bold tw-text-gray-900 dark:tw-text-moon-50 tw-text-3xl md:tw-text-4xl tw-leading-10"]'
    ).text

    driver.quit()
    print("BTC Price:", price)
        
        
    todays_price = page_main.find_element(By.XPATH, '//*[@class="tw-font-bold tw-text-gray-900 dark:tw-text-moon-50 tw-text-3xl md:tw-text-4xl tw-leading-10"]').text
    todays_price = todays_price.split('$')[1].replace(',','')
    
    # page_main.close()

    #-----------------------------

    start_price = float(todays_price)
    # days = int(input("Enter Prediction for how many days from today (eg:-3) : "))

    days = 10

    # Today's date
    today = date.today()
    # print(f"Today's date: {today}")
    
    
    

    # Load model without compiling
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

    # ✅ Correct formatting for return
    result = f"📈 BTC Price Prediction starting from {today} using price ${start_price:.2f}:\n\n"
    for i, pred in enumerate(days_pred_list):
        pred_date = today + timedelta(days=i + 1)
        result += f"{pred_date}: ${pred[0][0]:.2f} USD\n"

    # return result.strip()

        # Return both result and today's price (can be used by LLM)
    return {
        "summary": result.strip(),
        "todays_bitcoin_price": f"📈 Today's - ({today}) BTC price is ${start_price:.2f} USD"
    }
    
