from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from modules.utils import is_colab
import atexit
import sys

sys.stdout.reconfigure(encoding='utf-8')

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


def search_google(query: str) -> (str, list[str]):
    global driver
    print(f"Searching Google for {query}...")
    driver.get("https://google.com/search?hl=en&q=" + query)
    wait_for_id('res')
    save_debug()
    text = ''
    # Answer box
    text += get_from_selector('.wDYxhc')
    # Knowledge panel
    text += get_from_selector('.hgKElc')
    # Page snippets
    text += get_from_selector('.r025kc.lVm3ye')
    # Old selectors (for compatibility)
    text += get_from_selector('.yDYNvb.lyLwlc')
    # Links
    links = get_links_from_selector('.yuRUbf a')
    print("Found: " + text, links)
    return (text, links)


def search_duckduckgo(query: str) -> (str, list[str]):
    global driver
    print(f"Searching DuckDuckGo for {query}...")
    driver.get("https://duckduckgo.com/?kp=-2&kl=wt-wt&q=" + query)
    wait_for_id('web_content_wrapper')
    save_debug()
    text = get_from_selector('[data-result="snippet"]')
    links = get_links_from_selector('[data-testid="result-title-a"]')
    print("Found: " + text, links)
    return (text, links)

driver = get_driver()

def quit_driver():
    driver.quit()

def save_debug():
    with open("data/tmp/debug.html", "w", encoding='utf-8') as f:
        f.write(driver.page_source)

def wait_for_id(id: str, delay: int = 5):
    try:
        WebDriverWait(driver, delay).until(EC.presence_of_element_located((By.ID, id)))
    except:
        print(f"Element with id {id} not found, proceeding without.")

def get_from_selector(selector: str):
    result = ''
    for el in driver.find_elements(By.CSS_SELECTOR, selector):
        if el and el.text:
            result += el.text + '\n'
    return result

def get_links_from_selector(selector: str):
    links = []
    for el in driver.find_elements(By.CSS_SELECTOR, selector):
        if el and el.text:
            links.append(el.get_attribute('href'))
    return links

atexit.register(quit_driver)
