import requests
import json
from typing import Dict
from pathlib import Path
from time import time
from tqdm import tqdm
from lxml.html import fromstring
from bestbuy.src.crawler.scraper_classes import ReviewScraper, ThumbnailScraper, OverviewScraper
import concurrent.futures

# headers
headers = {'User-Agent': 'Mozilla/5.0'}


# scraper
def bs4_review_scraper(url: str)   :
    try:
        lxml_doc = fromstring(
            requests.get(url=url, headers=headers, timeout=30).content
        )
    except TimeoutError:
        print(f'Request timed out for {url}')
        return

    # scrape thumbnail pics
    thumbnails = ThumbnailScraper(url=url, doc=lxml_doc, headers=headers)

    # overview class
    overview = OverviewScraper(url=url, doc=lxml_doc, headers=headers)

    # scrape the reviews
    reviews = ReviewScraper(url=url, doc=lxml_doc, headers=headers)

    # make a dictionary of what you want to return
    url_dict = {
        'url': url,
        'thumbnails': thumbnails.thumbnail_list,
        'overview_section': overview.overview_dict,
        'reviews': reviews.reviews_list
    }
    return url_dict


# runner
if __name__ == "__main__":

    # offset = 22640
    incomplete_products_path = Path(
        f'bestbuy/data/similar_urls.txt'
    )
    complete_products_path = Path(
        f'bestbuy/data/similar_products.json'
    )
    with open(incomplete_products_path, 'r', encoding='utf-8') as fp:
        similar_urls = [line.strip() for line in fp.readlines()]

    step_size = 10
    output_list = []
    print_ctr = 0
    result = []

    for i in range(0, len(similar_urls), step_size):
        print(100 * '=')
        print(f'starting at the {print_ctr}th value')
        with concurrent.futures.ThreadPoolExecutor() as executor:
            try:
                result = list(
                    tqdm(
                        executor.map(
                            bs4_review_scraper,
                            similar_urls[i: i + step_size]
                        ),
                        total=step_size)
                )
            except:
                result = []
        end = time()
        if len(result) != 0:
            output_list.extend(result)
            with open(complete_products_path, 'w', encoding='utf-8') as fp:
                json.dump(output_list, fp, indent=4)
        else:
            print('something is wrong')
        print_ctr += step_size
