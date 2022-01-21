from constantly import Names
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from lxml import html
import time
import pandas as pd
from tqdm import tqdm

path = "../chromedriver/chromedriver"
browserpath = "/opt/google/chrome/google-chrome"

options = webdriver.ChromeOptions()
options.binary_location = browserpath
options.add_argument('headless')
options.add_argument('window-size=1200x900')

service = Service(path)
driver = webdriver.Chrome(service=service, options=options)
url = "https://dappradar.com/nft/collections"
url = "https://cryptoslam.io/nfts"

time.sleep(1)
driver.get(url=url)

x_path = '//*[@id="table"]/tbody'
elems = WebDriverWait(driver, 10).until(EC.visibility_of_all_elements_located((By.XPATH, x_path)))

for elem in elems:
    i = 1
    rows = elem.find_elements(By.XPATH,'//*[@class="odd"]') #Don't forget the even class
    for row in rows: #tqdm(rows):
        if row.text:

            # To search relative to a particular element, 
            # you should prepend the expression with . instea

            # hrefs = row.find_elements(By.XPATH, './/*[@href]')
            # for href in hrefs:
            #     if "twitter" in href.get_attribute("href"):
            #         print(href.get_attribute("href"))
            #     elif "discord" in href.get_attribute("href"):
            #         print(href.get_attribute("href"))

            collection_name = []
            names = row.text.split(" ")[3:]
            for name in names:
                if (not name[0].isdigit()) and (name != "-"):
                    collection_name.append(name)
                else:
                    break
            collection_name = "".join(collection_name).lower()
            print(collection_name)
            #break


# action = '//*[@id="table_next"]/a'
# filter_page = WebDriverWait(driver, 10).until(EC.visibility_of_all_elements_located((By.XPATH, action)))[0]
# try:
#     filter_page.click()
#     time.sleep(10)
# except Exception as e:
#     print(e)

# x_path = '//*[@id="table"]/tbody'
# elems = WebDriverWait(driver, 10).until(EC.visibility_of_all_elements_located((By.XPATH, x_path)))

# for elem in elems:
#     rows = elem.find_elements(By.XPATH,'//*[@class="odd"]')
#     for row in tqdm(rows):
#         if row.text:

#             # To search relative to a particular element, 
#             # you should prepend the expression with . instea

#             # hrefs = row.find_elements(By.XPATH, './/*[@href]')
#             # for href in hrefs:
#             #     if "twitter" in href.get_attribute("href"):
#             #         print(href.get_attribute("href"))
#             #     elif "discord" in href.get_attribute("href"):
#             #         print(href.get_attribute("href"))
            
#             collection_name = row.text.split(" ")[3]
#             print(collection_name)
#             break
        
driver.close()