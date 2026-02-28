"""Find where the actual transcript content is."""
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time

opts = Options()
opts.add_argument('--disable-gpu')
opts.add_argument('--window-size=1920,1080')
svc = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=svc, options=opts)

url = 'https://news.mongabay.com/podcast/2024/04/jane-goodall-at-90-on-fame-hope-and-empathy/'
driver.get(url)
time.sleep(4)

# Click the transcript link using Selenium click
try:
    expander = driver.find_element(By.CSS_SELECTOR, '.content-expander')
    print(f'Found expander, clicking...')
    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", expander)
    time.sleep(0.5)
    expander.click()
    print('Clicked!')
except Exception as e:
    print(f'Click error: {e}')

time.sleep(3)

# Check what changed
result = driver.execute_script("""
    const transcript = document.querySelector('.transcript');
    return {
        transcriptHTML: transcript?.innerHTML?.substring(0, 500),
        transcriptText: transcript?.innerText?.substring(0, 500),
        bodyLength: document.body.innerText.length,
        articleLength: document.querySelector('article')?.innerText?.length
    };
""")
print('\nAfter click:')
print(f"Body length: {result['bodyLength']}")
print(f"Article length: {result['articleLength']}")
print(f"Transcript HTML: {result['transcriptHTML']}")

# Look for any dynamically added content
new_elements = driver.execute_script("""
    // Look for elements that might be the transcript
    const possibles = document.querySelectorAll('[class*="content"], [class*="text"], [class*="body"]');
    let found = [];
    possibles.forEach(el => {
        if (el.innerText?.length > 5000) {
            found.push({class: el.className, length: el.innerText.length});
        }
    });
    return found;
""")
print(f'\nLarge content elements: {new_elements}')

driver.quit()
