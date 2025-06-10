import requests
import json
from typing import Dict
from pathlib import Path
from time import time
from tqdm import tqdm
from lxml.html import fromstring
from bestbuy.src.organize.checker import is_nan_or_miss
from bestbuy.src.crawler.scraper_classes import ReviewScraper, ThumbnailScraper
import concurrent.futures

# headers
headers = {'User-Agent': 'Mozilla/5.0'}


# scraper
def bs4_review_scraper(product_dict: Dict):
    url = product_dict['url']
    # try:
    #     lxml_doc = fromstring(
    #         requests.get(url=url, headers=headers, timeout=30).content
    #     )
    # except TimeoutError:
    #     print(f'Request timed out for {url}')
    #     return product_dict

    if (product_dict['reviews_urls'] is None or \
            len(product_dict['reviews_urls']) > 0) and (is_nan_or_miss(  product_dict,"reviews") ):
        # scrape the reviews
        reviews = ReviewScraper(url=url,  headers=headers)
        product_dict['reviews'] = reviews.reviews_list
    else:
        product_dict['reviews'] = []

 

    # return the dictionary
    return product_dict


import argparse
def get_args():
   


    parser = argparse.ArgumentParser() 
    # parser.add_argument("--lr", default=1e-7, type=float)
    parser.add_argument("--step_size", default=16, type=int)
    parser.add_argument("--start_id", default=25650, type=int)
    parser.add_argument("--end_id", default=50000, type=int)
    # parser.add_argument("--use_pre_trained_model", default=True, action="store_false") 
    # parser.add_argument("--crawl_mode",default="product", type=str  ) 
    args = parser.parse_args()

    print(args)
    return args
# runner
if __name__ == "__main__":
    args=get_args()
    start_id=args.start_id
    end_id=args.end_id
    step_size = args.step_size
    incomplete_products_path = Path(
        f'bestbuy/data/bestbuy_review_2.json'
    )
    complete_products_path = Path(
        f'bestbuy/data/bestbuy_review_2.1_incomplete_from_{start_id}_to_{end_id}.json'
    )
    with open(incomplete_products_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
    
    output_list = []
    print_ctr = 0
    result = []
    save_step=1000
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
                        print(f"error {e}")
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