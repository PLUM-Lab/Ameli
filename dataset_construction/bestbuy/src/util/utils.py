def get_product_info(doc):

    # define the name path
    name_path = '//div[starts-with(@id, "shop-product-title")]'

    # get the name ele
    try:
        name_ele = doc.xpath(name_path)[0]
    except:
        # something is wrong with url
        return {}

    # return a dictionary with all information
    try:
        company_name = name_ele.xpath('.//div')[0].xpath('.//a')[0].text
    except:
        company_name = "N/A"


    try:
        product_name = name_ele.find_class('sku-title')[0].text_content()
    except:
        product_name = "N/A"


    return {
            'company': company_name,
            'product_name': product_name
        }
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService    
from webdriver_manager.chrome import ChromeDriverManager
def init_driver_manager_chrome():
    service = ChromeService(executable_path=ChromeDriverManager().install())

    driver = webdriver.Chrome(service=service)
    print(driver)
    driver.quit()     
    
init_driver_manager_chrome()    