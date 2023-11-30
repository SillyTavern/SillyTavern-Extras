from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.firefox.service import Service as FirefoxService
from modules.utils import is_colab
import atexit


def get_driver():
    try:
        print("Initializing Chrome driver...")
        options = ChromeOptions()
        options.add_argument('--disable-infobars')
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument("--lang=en-GB")

        if is_colab():
            return webdriver.Chrome('chromedriver', options=options)
        else:
            chromeService = ChromeService()
            return webdriver.Chrome(service=chromeService, options=options)
    except:
        print("Chrome not found, using Firefox instead.")
        firefoxService = FirefoxService()
        options = FirefoxOptions()
        options.add_argument("--headless")
        options.set_preference("intl.accept_languages", "en,en_US")
        return webdriver.Firefox(service=firefoxService, options=options)


def search_google(query: str) -> str:
    global driver
    print(f"Searching Google for {query}...")
    driver.get("https://google.com/search?hl=en&q=" + query)
    text = ''
    # Answer box
    for el in driver.find_elements(By.CSS_SELECTOR, '.hgKElc'):
        if el and el.text:
            text += el.text + '\n'
    # Knowledge panel
    for el in driver.find_elements(By.CSS_SELECTOR, '.hgKElc'):
        if el and el.text:
            text += el.text + '\n'
    # Page snippets
    for el in driver.find_elements(By.CSS_SELECTOR, '.yDYNvb.lyLwlc'):
        if el and el.text:
            text += el.text + '\n'
    print("Found: " + text)
    return text


def search_duckduckgo(query: str) -> str:
    global driver
    print(f"Searching DuckDuckGo for {query}...")
    driver.get("https://duckduckgo.com/?kp=-2&kl=wt-wt&q=" + query)
    text = ''
    for el in driver.find_elements(By.CSS_SELECTOR, '[data-result="snippet"]'):
        if el and el.text:
            text += el.text + '\n'
    print("Found: " + text)
    return text

driver = get_driver()

def quit_driver():
    driver.quit()

atexit.register(quit_driver)
