
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
import re

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

import os
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options





# START_CODE :------------------------------


# Tool_1

def crypto_latest_news(human_prompt):
    """ 
    Creates a tool to fetch the latest Crypto Currency News within past 24 hours and present it point wise.
    """

    print("... Running crypto_latest_news--")
    print("\n Hold on, this might take some time \n")

    # Path to geckodriver
    fname = os.path.join(os.getcwd(), 'geckodriver.exe')

    # Setup Firefox driver
    service = Service(executable_path=fname)
    options = Options()
    options.add_argument("--headless")
    page_main =  webdriver.Firefox(options=options)

    page_main.maximize_window()
    page_main.get('https://crypto.news/')
    # print('Site Opened')
    page_main.implicitly_wait(10)

    try:
        page_main.find_element(By.CSS_SELECTOR, "#onesignal-slidedown-dialog > div > div:nth-child(2) > button:nth-child(2)").click()
        page_main.find_element(By.CSS_SELECTOR, "#cookie-consent-button").click()
    except Exception as e:
        # print("Notification or cookie popup skipped:", e)
        pass

    try:
        news_list = page_main.find_elements(By.CSS_SELECTOR, ".home-latest-news__list > a")
    except Exception as e:
        print("Could not find news list:", e)
        return []

    latest_news_list = []
    for news in news_list:
        title = news.text.strip()
        url = news.get_attribute('href')

        # Format using Markdown-style link
        formatted = f"{title} [Read more]({url})"
        latest_news_list.append(formatted)

    page_main.quit()  # ✅ Important: Free up the browser

    return latest_news_list
    

# Tool_2

def extract_url(prompt):
    match = re.search(r'(https?://[^\s]+)', prompt)
    return match.group(0) if match else None



def detail_crypto_latest_news(human_prompt):

    """ 
    Creates a tool to fetch the URL of detailed Crypto Currency News from prompt and then get the detailed news.
    """

    print("... Running detail_crypto_latest_news--")
    print("\n Hold on, this might take some time \n")

    
    # Path to geckodriver (you already set fname)
    fname = os.path.join(os.getcwd(), 'geckodriver.exe')
    
    # Create Service with driver path
    service = Service(executable_path=fname)
    ## (Optional) configure browser options
    options = Options()
    # Set Firefox options for headless mode
    options.add_argument("--headless")  # ✅ key line for headless
    
    # Correct way to initialize
    page_details = webdriver.Firefox(options=options)

    url = extract_url(human_prompt)
    if not url:
        return "No valid URL found in the prompt."

    # Now you can proceed
    print(f"Opening URL: {url}")
    page_details.maximize_window()
    page_details.get(url)
    print('Site Opened')
    page_details.implicitly_wait(10)
    
    try:
        notification_click = page_details.find_element(By.CSS_SELECTOR, "#onesignal-slidedown-dialog > div > div:nth-child(2) > button:nth-child(2)").click()
        cookies_click = page_details.find_element(By.CSS_SELECTOR, "#cookie-consent-button").click()
    except Exception as e:
        # print("An error occurred:", e)
        pass
        
    
    try:
        news_list = page_details.find_elements(By.XPATH, '//*[@class="post-detail__content blocks"]//p')
    except Exception as e:
        print("An error occurred:", e)
        pass

    news_test = ''
    latest_news_list = []
    for news_lines in news_list:
        news_test += news_lines.text + '\n'

    return news_test





    

