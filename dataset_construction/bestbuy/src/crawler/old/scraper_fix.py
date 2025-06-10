import requests
import json
from typing import Dict
from pathlib import Path
from time import time
from tqdm import tqdm
from lxml.html import fromstring
from bestbuy.src.crawler.util.image_downloader import download_row_images_for_json
 
from bestbuy.src.crawler.util.scraper_classes import ReviewScraperForDebug, SearchReviewScraper, SpecScraper
from bestbuy.src.organize.checker import is_nan, is_nan_or_miss
from bestbuy.src.crawler.util.scraper_classes import OverviewScraper, ReviewScraper, ThumbnailScraper
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

 
import logging  
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
# logging.basicConfig(format=FORMAT,filename="/home/menglong/workspace/code/referred/product_scraper/dataset_construction/bestbuy/output/log.txt", filemode='w',encoding='utf-8', level=logging.DEBUG)
new_crawled_num=0
# scraper

# headers
headers = {'User-Agent': 'Mozilla/5.0'}
is_headless=True 
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
# headers

import copy 
def bs4_review_scraper(product_dict: Dict,crawl_mode):
    url = product_dict['url']
    if crawl_mode=="review":
        needed_properties=[ "reviews"]
    elif crawl_mode=="spec":
        needed_properties=["Spec"]
    else:
        needed_properties=[ "name","path","overview","img_path","img_url"]#"Spec",
 
    is_force_crawl_review=True 
    if "reviews" in needed_properties:
        review=product_dict["reviews"][0]
        review_body=review["body"]
        review_header=review["header"]
        # debug_body="WHY DOES GOOGLE NOT ALLOW THEIR NEST PRODUCT TO BE ADDED TO THE NEST APP???\nDo not purchase these if you have any other Nest products and are trying to include them with the Nest App. These are not supported by the Nest App and you will only be able to view them on the Google Home App. VERY VERY DISAPPONTING!!!\nThey also have very poor video quality. If you are trying to use these to identify a person or vehicle from any distance greater than 20' use a different camera."
        product_name=product_dict["product_name"]
        # review_body="This chest box freezer is great but the delivery guys that delivered it actually dropped it and it’s rattling Service guy came out to fix it what she did an awesome job"
        # review_body="Great find.   I really didn\u2019t want a chest freezer as I normally would purchase an upright freezer.  Couple of times we didn\u2019t close the upright freezer completely and had spoiled or damaged food.  We shouldn\u2019t have this issue with this freezer.  Spacious on the inside.  Freezer also has 2 large baskets to separate some items.  Love it."
        
        # if "has_check_image_url" in review or "corrected_body" in review:
        #     return product_dict
        
        if ((  (product_dict['reviews_urls'] is None or \
                len(product_dict['reviews_urls']) > 0)
                and  is_nan_or_miss(  product_dict,'reviews') ) or is_force_crawl_review):
            # if review["id"]==465:
            #     print("")
            # scrape the reviews
            reviews =  SearchReviewScraper(url=url, review_body=review_body, product_name=product_name,
                                           headers=headers,review_id=review['id'],review_header=review_header)
            if len(reviews.reviews_list)>0:
                image_url=reviews.reviews_list[0]["image_url"]
                review_body=reviews.reviews_list[0]["body"]
                product_dict['reviews'][0]["wrong_image_url"]=copy.deepcopy(product_dict['reviews'][0]["image_url"])
                product_dict['reviews'][0]["corrected_body"] =review_body
                product_dict['reviews'][0]["image_url"] = image_url
            else:
                print(f"not find review for {review['id']}")
                logging.info(f"not find review for {review['id']}")
                product_dict["error"]="Y"
                reviews.reviews_list[0]["error"]="Y"
            # product_dict['reviews'][0]["flag"]="test"
              
         
    return product_dict


import argparse
def get_args():

    parser = argparse.ArgumentParser() 
    parser.add_argument("--file",default='bestbuy/data/temp/bestbuy_products_40000_3.4.7_remove_image_path_prefix.json', type=str  )
    parser.add_argument("--out_file",default='bestbuy/data/temp/bestbuy_products_40000_3.4.7_debug', type=str  ) 
    parser.add_argument("--start_id", default=0, type=int)
    parser.add_argument("--end_id", default=150000, type=int)
    parser.add_argument("--step_size", default=32, type=int)
    # parser.add_argument("--is_review", action='store_true'   ) #
    parser.add_argument("--crawl_mode", default='review', type=str  ) #review, spec
    parser.add_argument("--log_file",default="bestbuy/output/log.txt", type=str  )

    args = parser.parse_args()

    print(args)
    return args
# runner
def get_logger(log_file):
    logging.basicConfig(format=FORMAT,filename=log_file, filemode='w',  level=logging.INFO)

if __name__ == "__main__":
    args=get_args()
    start_id=args.start_id
    end_id=args.end_id
    incomplete_products_path = Path(
        args.file
    )
    complete_products_path = Path(
        f"{args.out_file}_from_{start_id}_to_{end_id}.json"
    )
    crawl_mode=args.crawl_mode
    get_logger(args.log_file)
    with open(incomplete_products_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
    logging.info("start ")
    step_size = args.step_size
    output_list = []
    print_ctr = 0
    result = []
    save_step=1000
    
    crawl_mode_list= [crawl_mode for i in range(step_size)]
    for i in  range(0, len(incomplete_dict_list), step_size) :
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
                                    incomplete_dict_list[i: i + step_size],
                                    crawl_mode_list
                                ),
                                total=step_size)
                        )
                    except Exception as e:
                        print(f"{e}")
                        print(f"__main__ error {e}")
                        result = []
                end = time()
                if len(result) != 0:
                    output_list.extend(result)
                else:
                    print('something is wrong')
                if i%save_step==0:
                    with open(complete_products_path, 'w', encoding='utf-8') as fp:
                        json.dump(output_list, fp, indent=4)
        print_ctr += step_size
    with open(complete_products_path, 'w', encoding='utf-8') as fp:
        json.dump(output_list, fp, indent=4)
    print(f"new crawl {new_crawled_num}")