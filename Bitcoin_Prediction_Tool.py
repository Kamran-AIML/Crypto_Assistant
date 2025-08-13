from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import os

def get_chrome_driver():

    """ 
    This tool is used for predicting Bitcoin (BTC) prices. It uses a trained LSTM model for prediction
    """

    print("... Running btc_predict--")

    model_path = 'Trained_Model/btc_lstm_model.h5'
    scaler_path = 'Trained_Model/btc_scaler.save'
    
    # Set Chrome options for headless execution
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Headless mode
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")

    # Tell Selenium where Chromium is installed on Streamlit Cloud
    chrome_options.binary_location = "/usr/bin/chromium"

    # Path to ChromiumDriver
    driver_path = "/usr/lib/chromium/chromedriver"
    service = Service(driver_path)

    return webdriver.Chrome(service=service, options=chrome_options)

# Example usage
def get_btc_price():
    driver = get_chrome_driver()
    driver.get("https://www.coingecko.com/en/coins/bitcoin")
    price = driver.find_element(
        "xpath",
        '//*[@class="tw-font-bold tw-text-gray-900 dark:tw-text-moon-50 tw-text-3xl md:tw-text-4xl tw-leading-10"]'
    ).text
    driver.quit()
    return price

if __name__ == "__main__":
    print("Current BTC Price:", get_btc_price())
