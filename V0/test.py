from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from lxml import html
import time
import pandas as pd
from tqdm import tqdm
import undetected_chromedriver as uc

path = "../chromedriver/chromedriver"
browserpath = "/opt/google/chrome/google-chrome"

options = webdriver.ChromeOptions()
options.binary_location = browserpath
options.add_argument('--headless')
options.add_argument('window-size=1200x900')

service = Service(path)
# driver = webdriver.Chrome(service=service, options=options)
url = "https://opensea.io/rankings"

driver = uc.Chrome(service=service,options=options)
driver.get(url)

x_path = '//*[@id="main"]/div/div[3]/button[2]'
elems = WebDriverWait(driver, 10).until(EC.visibility_of_all_elements_located((By.XPATH, x_path)))[0]
elems.click()
driver.close()

exit()
