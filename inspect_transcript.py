"""Inspect the transcript HTML structure."""
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
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

# Get the transcript container HTML
html = driver.execute_script("""
    const t = document.querySelector('.transcript');
    return t ? t.outerHTML.substring(0, 3000) : 'not found';
""")
print('Transcript HTML (first 3000 chars):')
print(html)

# Also check the parent container
parent_html = driver.execute_script("""
    const t = document.querySelector('.transcript');
    if (!t) return 'not found';
    const parent = t.parentElement;
    return {
        parentClass: parent?.className,
        childCount: parent?.children?.length,
        siblings: Array.from(parent?.children || []).map(c => ({tag: c.tagName, class: c.className}))
    };
""")
print('\nParent structure:', parent_html)

driver.quit()
