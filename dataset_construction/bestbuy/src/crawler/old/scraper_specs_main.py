import requests
import json
from typing import Dict
from pathlib import Path
from time import time
from tqdm import tqdm
from bestbuy.src.crawler.old.specs_scraper import SpecScraper
from lxml.html import fromstring
from bestbuy.src.crawler.scraper_classes import ReviewScraper, ThumbnailScraper
import concurrent.futures

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
# headers
headers = {'User-Agent': 'Mozilla/5.0'}
is_headless=True 

# scraper
def bs4_review_scraper(product_dict: Dict):
    
    url = product_dict['url']
    try:
        lxml_doc = fromstring(
            requests.get(url=url, headers=headers, timeout=30).content
        )
    except TimeoutError:
        print(f'Request timed out for {url}')
        return product_dict
    if 'Spec' not in  product_dict :
        driver=gen_wait(url,is_headless)
    
        reviews = SpecScraper(url=url, driver=driver)
        product_dict['Spec'] = reviews.spec_list_dict
        print(url)
        driver.quit()
    # return the dictionary
    return product_dict

def gen_wait(url,is_headless):
    # service = EdgeService(executable_path=EdgeChromiumDriverManager().install())

    # driver = webdriver.Edge(service=service)
     
    options = ChromeOptions()
    if is_headless:
        options.add_argument("--headless")  
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    driver = webdriver.Chrome(service=ChromeService(executable_path=ChromeDriverManager().install()),options=options)
    driver.get(url) 
    return driver 


import argparse
def get_args():
   


    parser = argparse.ArgumentParser() 
    # parser.add_argument("--lr", default=1e-7, type=float)
    parser.add_argument("--step_size", default=10, type=int)
    parser.add_argument("--start_id", default=0, type=int)
    parser.add_argument("--end_id", default=6000, type=int)
    # parser.add_argument("--use_pre_trained_model", default=True, action="store_false") 
    # parser.add_argument("--desc", type=str  ) 
    args = parser.parse_args()

    print(args)
    return args

# runner
if __name__ == "__main__":  
    args=get_args()
    start_id=args.start_id
    end_id=args.end_id
    incomplete_products_path = Path(
        f'../data/bestbuy_products_40000_0_desc.json'
    )
    complete_products_path = Path(
        f'../data/bestbuy_products_1_desc_spec_from_{args.start_id}_to_{args.end_id}.json'
    )
    with open(incomplete_products_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)

    step_size = args.step_size
    output_list = []
    print_ctr = 0
    result = []

    for i in range(0, len(incomplete_dict_list), step_size):
        if i>=start_id :
            if   i<end_id:
                print(100 * '=')
                print(f'starting at the {print_ctr}th value')
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    try:
                        result = list(
                            tqdm(
                                executor.map(
                                    bs4_review_scraper,
                                    incomplete_dict_list[i: i + step_size]
                                ),
                                total=step_size)
                        )
                    except Exception as e:
                        print(f"ERROR: {e}" )
                        result = []
                end = time()
                if len(result) != 0:
                    output_list.extend(result)
                    with open(complete_products_path, 'w', encoding='utf-8') as fp:
                        json.dump(output_list, fp, indent=4)
                else:
                    print('something is wrong')
                
            else:
                break 
        print_ctr += step_size
