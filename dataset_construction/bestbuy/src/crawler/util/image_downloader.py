from numpy import product
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
import concurrent.futures
import logging  
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"


# headers
headers = {'User-Agent': 'Mozilla/5.0'}
IMAGE_PATH=r"bestbuy/data/product_images"
# scraper
def bs4_review_scraper(product_dict: Dict,crawl_mode):
    headers = {"User-Agent": "Mozilla/5.0"}
    if crawl_mode=="product":
        if 'thumbnail_paths' not in product_dict or is_nan(product_dict['thumbnail_paths'] ):
            image_list=download_row_images_for_json(product_dict, headers)
            product_dict["thumbnail_paths"]=image_list 
    else:
        
        if not is_nan_or_miss(product_dict,"reviews"):
            review_with_image_num=0
            review_with_path_list = []
            for review in product_dict["reviews"] :
                if is_nan_or_miss(review,'image_path'):
                    # if review_with_image_num<=100: 
                        
                    image_list,review_with_image_num=download_row_images_for_one_review(review, headers,review_with_image_num)
                    if len(image_list)>0:
                        review["image_path"]=image_list 
                review_with_path_list.append(review)
            product_dict["reviews"]=review_with_path_list
  

    # return the dictionary
    return product_dict

    """
    "https://pisces.bbystatic.com/image2/BestBuy_US/ugc/photos/thumbnail/432fbe121bc689a7b55924d338b395f7.jpg;maxHeight=140;maxWidth=140
    https://pisces.bbystatic.com/image2/BestBuy_US/images/products/6076/6076906_sd.jpg;maxHeight=54;maxWidth=54
    """
def download_row_images_for_one_review(review, headers,review_with_image_num):
     
         
    review_fnames = []
    if not is_nan_or_miss(review,'image_url') :
        review_with_image_num+=1
        review_images = review['image_url']
        review_id=review["id"]
        image_id=0
        for review_image in review_images:
            review_cleaned = review_image.split(';')[0]
            local_fname = download_image(review_id=review_id, url=review_cleaned, headers=headers,output_dir=r'bestbuy/data/final/v2/review_images')
            # image_id+=1
            if local_fname is not None:
                review_fnames.append(local_fname)
            else:
                logging.info(f"error: image {review_cleaned}")
    return review_fnames,review_with_image_num
           

def clean_img_url(url):
    original_url=url.split(";maxHeight")[0]
    return original_url

def download_image( review_id,url, headers,output_dir= r'bestbuy/data/product_images'):
    """
    :param url:
    :param headers:
    :return:
    """
    
    resp = requests.get(url, headers=headers)
    if resp.status_code == 200:
        original_image_name=str(review_id).zfill(7)+"-"+url.split('.jpg')[0].split('/')[-1] 
        fname = os.path.join(output_dir, original_image_name+ '.jpg')
        with open(fname, 'wb') as fp:
            fp.write(resp.content)
        return  fname.split('images/')[-1]
    return None 

 

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
    parser.add_argument("--step_size", default=32, type=int)
    parser.add_argument("--start_id", default=0, type=int)
    parser.add_argument("--end_id", default=150000, type=int)
    # parser.add_argument("--use_pre_trained_model", default=True, action="store_false") 
    parser.add_argument("--crawl_mode",default="review", type=str  ) 
    args = parser.parse_args()

    print(args)
    return args


def main(start_id,end_id,step_size,args):
    logging.basicConfig(format=FORMAT,filename="bestbuy/output/review_image.txt",  level=logging.WARN)
    # incomplete_products_path = Path(
    #     f'bestbuy/data/bestbuy_products_40000_3.2_desc_img_url.json'
    # )
    # complete_products_path = Path(
    #     f'bestbuy/data/bestbuy_products_40000_3.5_desc_img_url_img_path_from_{start_id}_to_{end_id}.json'
    # )
    incomplete_products_path = Path(
        f'bestbuy/data/final/v2/bestbuy_review_2.3.16.6_change_image_path_nan.json'
    )
    complete_products_path = Path(
        f'bestbuy/data/final/v2/bestbuy_review_2.3.16.7_download_image_from_{start_id}_to_{end_id}.json'
    )
     
    with open(incomplete_products_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)

 
    output_list = []
    print_ctr = 0
    result = []
    save_step=100
    is_parallel=False 
    crawl_mode_list= [args.crawl_mode for i in range(step_size)]
    for i in tqdm(range(0, len(incomplete_dict_list), step_size)):
        if i>=start_id :
            if   i<end_id:
                if is_parallel:
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
                        except Exception as e :
                            logging.warning(f"{e}")
                            print(f"error {e}")
                            result = []
                else:
                    result=[]
                    for product_json in incomplete_dict_list[i: i + step_size]:
                        one_result=bs4_review_scraper(product_json,args.crawl_mode )
                        result.append(one_result)
                end = time()
                if len(result) != 0:
                    output_list.extend(result)
                    
                else:
                    print('something is wrong')
                if   i%save_step==0:
                    with open(complete_products_path, 'w', encoding='utf-8') as fp:
                        json.dump(output_list, fp, indent=4)
        print_ctr += step_size
    with open(complete_products_path, 'w', encoding='utf-8') as fp:
        json.dump(output_list, fp, indent=4)
# runner
 
    
   
# runner
if __name__ == "__main__":  
    args=get_args()
    start_id=args.start_id
    end_id=args.end_id
    main(start_id,end_id,args.step_size,args)