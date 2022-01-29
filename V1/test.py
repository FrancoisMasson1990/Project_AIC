import requests
import cloudscraper
from bs4 import BeautifulSoup
import json
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import undetected_chromedriver as uc
headers = {"Accept": "application/json"}

def scrap_url(url, key="script"):
    scraper = cloudscraper.create_scraper()
    r = scraper.get(url, headers=headers)
    soup = BeautifulSoup(r.text, 'html.parser')
    a = soup.find("meta", property="og:description")
    print(a.get("content", None).text)
    exit()
    json_data = json.loads(soup.find(key, type='application/json').text)
    return json_data

url = "https://discord.com/invite/Xh9EHpubhv"
scrap_url(url)