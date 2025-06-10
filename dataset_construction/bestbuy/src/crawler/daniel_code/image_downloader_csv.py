import pandas as pd
import requests
import os
from pathlib import Path
from typing import Dict
from tqdm import tqdm
from ast import literal_eval

import requests
import json
from typing import Dict
from pathlib import Path
from time import time
from tqdm import tqdm
from lxml.html import fromstring
from bestbuy.src.organize.checker import is_nan, is_nan_or_miss
from bestbuy.src.crawler.scraper_classes import ReviewScraper, ThumbnailScraper
import concurrent.futures

# headers
headers = {'User-Agent': 'Mozilla/5.0'}
IMAGE_PATH=r"bestbuy/data/product_images"
# scraper
def bs4_review_scraper(product_dict: Dict):
    headers = {"User-Agent": "Mozilla/5.0"}
    
    if 'thumbnail_paths' not in product_dict or is_nan(product_dict['thumbnail_paths'] ):
        image_list=download_row_images_for_json(product_dict, headers)
        product_dict["thumbnail_paths"]=image_list 
    
  

    # return the dictionary
    return product_dict

    """
    "https://pisces.bbystatic.com/image2/BestBuy_US/ugc/photos/thumbnail/432fbe121bc689a7b55924d338b395f7.jpg;maxHeight=140;maxWidth=140
    https://pisces.bbystatic.com/image2/BestBuy_US/images/products/6076/6076906_sd.jpg;maxHeight=54;maxWidth=54
    """
def clean_img_url(url):
    original_url=url.split(";maxHeight")[0]
    return original_url

def download_image( url, headers,output_dir= r'bestbuy/data/product_images'):
    """
    :param url:
    :param headers:
    :return:
    """
    
    resp = requests.get(url, headers=headers)
    if resp.status_code == 200:
        fname = os.path.join(output_dir, url.split('.jpg')[0].split('/')[-1] + '.jpg')
        with open(fname, 'wb') as fp:
            fp.write(resp.content)
    return 'images' + fname.split('images')[-1]


def download_row_images_for_csv(db_row: pd.core.series.Series, headers: Dict) \
        -> pd.core.series.Series:
    """
    :param db_row:
    :return:
    """
    thumbnails = literal_eval(db_row['thumbnails'])
    reviews = literal_eval(db_row['reviews'])

    thumbnail_fnames = []
    if len(thumbnails) != 0:
        for thumbnail in tqdm(thumbnails):
            thumbnail_cleaned = thumbnail.split(';')[0]
            local_fname = download_image(url=thumbnail_cleaned, headers=headers)
            thumbnail_fnames.append(local_fname)
    db_row['thumbnail_paths'] = thumbnail_fnames

    review_list = []
    if len(reviews) != 0:
        for review in tqdm(reviews):
            review_fnames = []
            review_images = review['product_images']
            for review_image in review_images:
                review_cleaned = review_image.split(';')[0]
                local_fname = download_image(url=review_cleaned, headers=headers)
                review_fnames.append(local_fname)
            review['product_paths'] = review_fnames
            review_list.append(review)
    db_row['reviews'] = review_list

    return db_row


def download_row_images_for_json(product_dict , headers: Dict,image_path=IMAGE_PATH)  :
    """
    :param db_row:
    :return:
    """
    thumbnail_fnames = []
    if is_nan_or_miss(product_dict,'thumbnails') :

        if not is_nan_or_miss(product_dict,'product_images') :
            thumbnails_list  =  product_dict['product_images'] 
        else:
            thumbnails_list = []
    else:
        thumbnails_list  =  product_dict['thumbnails'] 

        
    if len(thumbnails_list) != 0:
        for thumbnail in  thumbnails_list :
            thumbnail_cleaned = thumbnail.split(';')[0]
            thumbnail_cleaned = clean_img_url(thumbnail_cleaned)
            local_fname = download_image(url=thumbnail_cleaned, headers=headers,output_dir=image_path)
            thumbnail_fnames.append(local_fname)
    

     
    return thumbnail_fnames

def download_images_by_csv(df: pd.DataFrame, headers_dict: Dict) -> None:
    """
    :param headers_dict:
    :param df:
    :return:
    """
    image_row_list = []
    for idx, row in tqdm(df.iterrows()):
        image_row = download_row_images_for_csv(row, headers_dict)
        image_row_list.append(image_row)
    image_df = pd.DataFrame(image_row_list)
    image_df.to_csv('../data/bestbuy_data_with_ids_specs_and_paths.csv', index=False)

# def download_images_by_json(incomplete_dict_list,complete_products_path, headers_dict: Dict) -> None:
 
#     output_list = []
#     total_len=len(incomplete_dict_list)
#     for idx,product_dict  in  tqdm(enumerate(incomplete_dict_list )):
#         image_list=download_row_images_for_json(product_dict, headers_dict)
#         product_dict["thumbnail_paths"]=image_list 
#         output_list.extend(product_dict)
#         if idx%50==0:
#             print(f"{idx}/{total_len}")
#             with open(complete_products_path, 'w', encoding='utf-8') as fp:
#                 json.dump(output_list, fp, indent=4)
            
def download_by_csv():
    headers = {"User-Agent": "Mozilla/5.0"}
    db_path = Path('bestbuy/data/bestbuy_data_with_ids_and_specs.csv')
    db_df = pd.read_csv(db_path)
    download_images_by_csv(db_df, headers)
import json 
# def download_by_json():
#     headers = {"User-Agent": "Mozilla/5.0"}
#     db_path = Path('../../data/bestbuy_products_40000_0.05_desc_img_url.json')
#     complete_products_path = Path('../../data/bestbuy_products_40000_0.15_desc_img_url_img_path.json')
#     with open(db_path, 'r', encoding='utf-8') as fp:
#         incomplete_dict_list = json.load(fp)
#         download_images_by_json(incomplete_dict_list,complete_products_path, headers)    
    
# if __name__ == "__main__":
#     download_by_json()


import argparse
def get_args():
   


    parser = argparse.ArgumentParser() 
    # parser.add_argument("--lr", default=1e-7, type=float)
    parser.add_argument("--step_size", default=15, type=int)
    parser.add_argument("--start_id", default=0, type=int)
    parser.add_argument("--end_id", default=6000, type=int)
    # parser.add_argument("--use_pre_trained_model", default=True, action="store_false") 
    # parser.add_argument("--desc", type=str  ) 
    args = parser.parse_args()

    print(args)
    return args


def main(start_id,end_id,step_size,image_path):
    incomplete_products_path = Path(
        f'bestbuy/data/bestbuy_products_40000_2.5_desc_img_url.json'
    )
    complete_products_path = Path(
        f'bestbuy/data/bestbuy_products_40000_3_desc_img_url_img_path_from_{start_id}_to_{end_id}.json'
    )
     
    with open(incomplete_products_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)

 
    output_list = []
    print_ctr = 0
    result = []

    for i in tqdm(range(0, len(incomplete_dict_list), step_size)):
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
# runner
 
    
   
# runner
if __name__ == "__main__":  
    args=get_args()
    start_id=args.start_id
    end_id=args.end_id
    main(start_id,end_id,args.step_size)