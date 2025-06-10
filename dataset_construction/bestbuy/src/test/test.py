from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions 
from selenium.webdriver.chrome.service import Service as ChromeService      
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.firefox.service import Service as FirefoxService 
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager
from webdriver_manager.microsoft import EdgeChromiumDriverManager 
from selenium.webdriver.chrome.options import Options
def test_chrome_session():
    options = ChromeOptions()
     
    options.add_argument("--headless")  
    driver = webdriver.Chrome(options=options)

    driver.quit()

def test_eight_components():
    options = ChromeOptions()
     
    options.add_argument("--headless")  
    driver = webdriver.Chrome(service=ChromeService(executable_path=ChromeDriverManager().install()),options=options)
    driver.get("https://www.selenium.dev/selenium/web/web-form.html")
    title = driver.title
    assert title == "Web form"
    driver.implicitly_wait(0.5)
    text_box = driver.find_element(by=By.NAME, value="my-text")
    submit_button = driver.find_element(by=By.CSS_SELECTOR, value="button")
    text_box.send_keys("Selenium")
    submit_button.click()
    message = driver.find_element(by=By.ID, value="message")
    value = message.text
    print(value)
    assert value == "Received!"

    driver.quit()
    



def test_driver_manager_chrome():
    service = ChromeService(executable_path=ChromeDriverManager().install())

    driver = webdriver.Chrome(service=service)

    driver.quit()


# def test_edge_session():
#     service = EdgeService(executable_path=EdgeChromiumDriverManager().install())

#     driver = webdriver.Edge(service=service)
#     print(driver)
#     driver.quit()


# def test_firefox_session():
#     service = FirefoxService(executable_path=GeckoDriverManager().install())

#     driver = webdriver.Firefox(service=service)

#     driver.quit()
    


def test2():
     
    
    chrome_options = Options() 
    chrome_options.add_argument("--headless") 
    driver = webdriver.Chrome(options=chrome_options)
    start_url = "https://duckgo.com"
    driver.get(start_url)
    print(driver.page_source.encode("utf-8"))
    # b'<!DOCTYPE html><html xmlns="http://www....
    driver.quit()
    
test_eight_components()