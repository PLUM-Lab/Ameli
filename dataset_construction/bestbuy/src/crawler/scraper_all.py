import requests
import json
from typing import Dict
from pathlib import Path
from time import time
from tqdm import tqdm
from lxml.html import fromstring
from bestbuy.src.crawler.util.image_downloader import download_row_images_for_json
 
from bestbuy.src.crawler.util.scraper_classes import SpecScraper
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


def bs4_review_scraper(product_dict: Dict,crawl_mode):
    url = product_dict['url']
    if crawl_mode=="review":
        needed_properties=[ "reviews"]
    elif crawl_mode=="spec":
        needed_properties=["Spec"]
    else:
        needed_properties=[ "name","path","overview","img_path","img_url"]#"Spec",
    if "overview" in needed_properties and is_nan_or_miss(  product_dict,'overview_section') :
            
        try:
            lxml_doc = fromstring(
                requests.get(url=url, headers=headers, timeout=30).content
            )

        except TimeoutError:
            logging.warning(f'Request timed out for {url}')
            print(f'Request timed out for {url}')
            return product_dict
        overview = OverviewScraper(url=url, doc=lxml_doc, headers=headers)
        product_dict['overview_section']=overview.overview_dict

    is_force_crawl_review=True 
    if "reviews" in needed_properties:
        
        if ((  (product_dict['reviews_urls'] is None or \
                len(product_dict['reviews_urls']) > 0)
                and  is_nan_or_miss(  product_dict,'reviews') ) or is_force_crawl_review):
            # scrape the reviews
            reviews = ReviewScraper(url=url,   headers=headers)
            product_dict['reviews'] = reviews.reviews_list
         #TODO add image download for review

     # overview class
    
        

    if "img_url" in needed_properties:
        if ( is_nan_or_miss(product_dict, 'thumbnails')  and is_nan_or_miss(  product_dict, 'product_images') ):
            # scrape thumbnail pics
            thumbnails = ThumbnailScraper(url=url,   headers=headers)
            if len(thumbnails.thumbnail_list)==0:
                logging.info(f"empty image: {url}")
            else:
                new_crawled_num+=1
            product_dict['thumbnails'] = thumbnails.thumbnail_list
    if "img_path" in needed_properties:
        if ( is_nan_or_miss(product_dict, 'image_path' ) and is_nan_or_miss(product_dict, 'thumbnail_paths' )):
            
            image_list=download_row_images_for_json(product_dict, headers )
            product_dict["image_path"]=image_list 
            
    if "Spec" in needed_properties:
        if   is_nan_or_miss(product_dict,"Spec") :
            driver=gen_wait(url,is_headless=False)
        
            reviews = SpecScraper(url=url, driver=driver)
            product_dict['Spec'] = reviews.spec_list_dict
            # print(url)
            driver.quit()
    # return the dictionary
    return product_dict


import argparse
def get_args():

    parser = argparse.ArgumentParser() 
    parser.add_argument("--file",default='bestbuy/data/final/v1/bestbuy_products_40000_3.4.3_desc_img_url_from_0_to_150000.json', type=str  )
    parser.add_argument("--out_file",default='bestbuy/data/final/v1/bestbuy_products_40000_debug', type=str  ) 
    parser.add_argument("--start_id", default=0, type=int)
    parser.add_argument("--end_id", default=150000, type=int)
    parser.add_argument("--step_size", default=32, type=int)
    # parser.add_argument("--is_review", action='store_true'   ) #
    parser.add_argument("--crawl_mode", default='product', type=str  ) #review, spec
    parser.add_argument("--log_file",default="bestbuy/output/log.txt", type=str  )

    args = parser.parse_args()

    print(args)
    return args
# runner
def get_logger(log_file):
    logging.basicConfig(format=FORMAT,filename=log_file, encoding='utf-8', level=logging.WARN)

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

    step_size = args.step_size
    output_list = []
    print_ctr = 0
    result = []
    save_step=1000
    crawl_mode_list= [crawl_mode for i in range(step_size)]
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
                                    incomplete_dict_list[i: i + step_size],
                                    crawl_mode_list
                                ),
                                total=step_size)
                        )
                    except Exception as e:
                        logging.warning(f"{e}")
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