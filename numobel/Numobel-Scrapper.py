import json
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

# -------------------------------
# CONFIG
# -------------------------------

BASE_URL = "https://www.numobel.in/shop-online?page={}"
OUTPUT_FILE = "numobel_products.json"
START_PAGE = 1
END_PAGE = 14

# -------------------------------
# DRIVER SETUP
# -------------------------------

chrome_options = Options()
chrome_options.add_argument("--start-maximized")
chrome_options.add_argument("--disable-blink-features=AutomationControlled")

driver = webdriver.Chrome(options=chrome_options)
wait = WebDriverWait(driver, 20)

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------

def get_product_links_from_page(page_no):
    url = BASE_URL.format(page_no)
    print(f"\n🔎 Opening Page {page_no}: {url}")

    driver.get(url)

    # Wait for products to load
    wait.until(
        EC.presence_of_element_located((By.TAG_NAME, "a"))
    )

    time.sleep(2)  # small buffer for JS

    links = set()

    # Adjust this selector if needed
    product_elements = driver.find_elements(By.CSS_SELECTOR, '[data-hook="product-item-container"]')

    for element in product_elements:
        link = element.get_attribute("href")
        if link:
            links.add(link)

    print(f"Found {len(links)} products on page {page_no}")
    return list(links)

def extract_name(driver):
    try:
        return driver.find_element(
            By.CSS_SELECTOR,
            '[data-hook="product-title"]'
        ).text.strip()
    except Exception:
        return None

def extract_price(driver):
    try:
        return driver.find_element(
            By.CSS_SELECTOR,
            '[data-hook="product-price"]'
        ).text.strip()
    except Exception:
        return None


def extract_formatted_price(driver):
    try:
        return driver.find_element(
            By.CSS_SELECTOR,
            '[data-hook="formatted-primary-price"]'
        ).text.strip()
    except Exception:
        return None

def extract_sizes(driver, wait):
    try:
        dropdown = wait.until(
            EC.element_to_be_clickable(
                (By.CSS_SELECTOR, '[data-hook="core-dropdown"]')
            )
        )
        dropdown.click()

        options = wait.until(
            EC.presence_of_all_elements_located(
                (By.CSS_SELECTOR, '[data-hook="dropdown-option"]')
            )
        )

        sizes = [opt.text.strip() for opt in options]

        dropdown.click()  # close dropdown

        return sizes if sizes else None

    except Exception:
        return None

def extract_colors(driver, wait):
    try:
        color_elements = wait.until(
            EC.presence_of_all_elements_located(
                (By.CSS_SELECTOR, 'input[data-hook="hidden-radio"]')
            )
        )

        colors = [
            c.get_attribute("aria-label") 
            for c in color_elements
            if c.get_attribute("aria-label")
        ]

        return colors if colors else None

    except Exception:
        return None

def extract_images(driver, wait):
    try:
        thumbnail_imgs = wait.until(
            EC.presence_of_all_elements_located(
                (By.CSS_SELECTOR, 'button[data-hook="thumbnail-root"] img')
            )
        )

        image_urls = [
            img.get_attribute("src")
            for img in thumbnail_imgs
            if img.get_attribute("src")
        ]

        return image_urls if image_urls else None

    except Exception:
        return None

def extract_description(driver, wait):
    try:
        description_element = wait.until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, '[data-hook="description"]')
            )
        )

        return description_element.text.strip()

    except Exception:
        return None



def scrape_product_page(url):
    print(f"   ➜ Scraping: {url}")
    driver.get(url)

    try:
        wait.until(
            EC.presence_of_element_located((By.TAG_NAME, "h1"))
        )
    except:
        print("   ❌ Failed loading product page")
        return None

    time.sleep(1)

    # ---- Extract Data ----

    return {
        "name": extract_name(driver),
        "price": extract_price(driver),
        "formatted_price": extract_formatted_price(driver),
        "sizes": extract_sizes(driver, wait),
        "colors": extract_colors(driver, wait),
        "images": extract_images(driver, wait),
        "description": extract_description(driver, wait),
        "url": url
    }



# -------------------------------
# MAIN SCRAPING LOOP
# -------------------------------

all_products = []
all_links = set()

for page in range(START_PAGE, END_PAGE + 1):

    page_links = get_product_links_from_page(page)

    for link in page_links:
        if link not in all_links:
            all_links.add(link)

            data = scrape_product_page(link)

            if data:
                all_products.append(data)

    print(f"✅ Completed Page {page}")