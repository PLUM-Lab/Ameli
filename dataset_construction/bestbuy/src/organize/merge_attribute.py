from pathlib import Path
import json

import os
import pandas as pd
import shutil  
from random import sample
from tqdm import tqdm 
import random

from bestbuy.src.organize.checker import is_nan_or_miss
def merge_img_path_20000():
    incomplete_products_path = Path(
        f'../../data/bestbuy_products_40000_0_desc.json'
    )
    to_merge_products_path = Path(
        f'../../data/bestbuy_products_0.5_w_img_url.json'
    )
    output_products_path = Path(
        f'../../data/bestbuy_products_40000_0.05_desc_img_url.json'
    )
    output_list=[]
    product_id=0
    i=0
    with open(incomplete_products_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        with open(to_merge_products_path, 'r', encoding='utf-8') as fp:
            to_merge_img_url_dict_list = json.load(fp)
            to_merge_img_url_dict_dict=json_to_dict(to_merge_img_url_dict_list)
            for product_json in incomplete_dict_list   :
                
                i+=1
                if i%500==0:
                    print(i)
                product_id=product_json["url"]
                if product_id in to_merge_img_url_dict_dict and product_json["thumbnails"] is None :
                   to_merge_img_url_dict=to_merge_img_url_dict_dict [product_id]
                   product_json["thumbnails"]=to_merge_img_url_dict["thumbnails"]
                output_list.append(product_json)
            
         
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(output_list, fp, indent=4)
def review_json_to_product_dict(review_dataset_json_array):
    output_review_dict={}
    for review_dataset_json in review_dataset_json_array:
        
        review=review_dataset_json 
        review_id=review["review_id"]
        output_review_dict[review_id]=review_dataset_json
           
                
         
    return output_review_dict 

def review_json_to_product_dict_for_old_format(review_dataset_json_array):
    output_review_dict={}
    for review_dataset_json in review_dataset_json_array:
        
        review=review_dataset_json["reviews"][0]
        review_id=review["id"]
        output_review_dict[review_id]=review_dataset_json
           
                
         
    return output_review_dict 


def new_review_json_to_product_dict(review_dataset_json_array):
    output_review_dict={}
    for review in review_dataset_json_array:
       
        review_id=review["review_id"]
        output_review_dict[review_id]=review
           
                
         
    return output_review_dict 


def json_to_review_dict(review_dataset_json_array):
    output_review_dict={}
    for review_dataset_json in review_dataset_json_array:
        if not is_nan_or_miss(review_dataset_json,"reviews"):
            reviews=review_dataset_json["reviews"]
            for review in reviews:
                review_id=review["id"]
                output_review_dict[review_id]=review
         
    return output_review_dict 
def json_to_dict(to_merge_img_url_dict_list):
    output_dict={}
    for to_merge_img_url_dict in to_merge_img_url_dict_list:
        output_dict[to_merge_img_url_dict["url"]]=to_merge_img_url_dict
    return output_dict

def json_to_product_id_dict(to_merge_img_url_dict_list):
    output_dict={}
    for to_merge_img_url_dict in to_merge_img_url_dict_list:
        output_dict[to_merge_img_url_dict["id"]]=to_merge_img_url_dict
    return output_dict
# merge_img_path_20000()                

def split_image(image_corpus,splited_data_path,filepath,is_copy):
    source_path=os.path.join(image_corpus,filepath)
    target_path=os.path.join(splited_data_path ,filepath)
    
    # 
    if is_copy:
        shutil.copy(source_path, target_path)
    else:
        os.rename(source_path,target_path )   
def merge_image(source_path,target_path,is_copy=True):
 
    image_corpus=source_path
    img_names=os.listdir(image_corpus)

    move_num=0   
    for filepath in  tqdm(img_names):
        split_image(image_corpus,target_path,filepath,is_copy)
        move_num+=1
     
    print(f"finish split_image_for_one_split {move_num}")    
    
if __name__ == "__main__":      
    # merge_image("bestbuy/data/product_images","bestbuy/data/final/v1/product_images")    
    merge_image("bestbuy/data/review_images","bestbuy/data/final/v1/review_images")    