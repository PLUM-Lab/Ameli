import requests
import json
from typing import Dict
from pathlib import Path
from time import time
from tqdm import tqdm
from lxml.html import fromstring
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

    # if product_dict['reviews_urls'] is None or \
    #         len(product_dict['reviews_urls']) > 0:
    #     # scrape the reviews
    #     reviews = ReviewScraper(url=url, doc=lxml_doc, headers=headers)
    #     product_dict['reviews'] = reviews.reviews_list
    # else:
    #     product_dict['reviews'] = []


    if product_dict['thumbnails'] is None:
        # scrape thumbnail pics
        thumbnails = ThumbnailScraper(url=url, headers=headers)
        product_dict['thumbnails'] = thumbnails.thumbnail_list

    # return the dictionary
    return product_dict


# runner
if __name__ == "__main__":

    incomplete_products_path = Path(
        f'../../data/bestbuy_products_40000_0.1_desc_img_url.json'
    )
    complete_products_path = Path(
        f'../../data/bestbuy_products_40000_0.1_desc_img_url.json'
    )
    with open(incomplete_products_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)

    step_size = 30
    output_list = []
    print_ctr = 0
    result = []

    for i in range(0, len(incomplete_dict_list), step_size):
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
            except Exception as e :
                print(f"error {e}")
                result = []
        end = time()
        if len(result) != 0:
            output_list.extend(result)
            with open(complete_products_path, 'w', encoding='utf-8') as fp:
                json.dump(output_list, fp, indent=4)
        else:
            print('something is wrong')
        print_ctr += step_size
