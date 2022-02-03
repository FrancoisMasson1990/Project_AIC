from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
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

columns = ["Collection","Website","Twitter","Discord"]
df = pd.DataFrame(columns=columns)
x_path = '//*[@id="table"]/tbody'

elems = driver.find_elements(By.XPATH, x_path)
while not elems:
    driver.close()
    driver.quit()

    time.sleep(1)
    service = Service(path)
    driver = webdriver.Chrome(service=service, options=options)
    driver.get(url=url)
    elems = driver.find_elements(By.XPATH, x_path)

    print("Failed. Retry starting service")

for elem in elems:
    rows = elem.find_elements(By.XPATH,'//*[@class="odd"]')
    for row in tqdm(rows):
        if row.text:
            # To search relative to a particular element, 
            # you should prepend the expression with . instea

            hrefs = row.find_elements(By.XPATH, './/*[@href]')
            for href in hrefs:
                if "twitter" in href.get_attribute("href"):
                    print(href.get_attribute("href"))
                elif "discord" in href.get_attribute("href"):
                    print(href.get_attribute("href"))
            
            collection_name = row.text.split(" ")[3]
            print(collection_name)
        # Need to add webiste
        #df.loc[len(df)] = ""
