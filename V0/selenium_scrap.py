from grpc import services
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time

path = "../chromedriver/chromedriver"
browserpath = "/opt/google/chrome/google-chrome"

options = webdriver.ChromeOptions()
options.binary_location = browserpath
options.add_argument('headless')
options.add_argument('window-size=1200x900')

service = Service(path)
driver = webdriver.Chrome(service=service, options=options)
url = "https://dappradar.com/nft/collections"
class_name = "cms-page" #CLASS_NAME
#url = "https://nftgo.io/whale-tracking/whale"
#class_name = "whales_container__mmGF3"
driver.get(url=url)


print("Element identified by class:",driver.find_element(By.ID, "rankCollectionTable").text)

driver.close()
exit()