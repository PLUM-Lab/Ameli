import json
from typing import Dict
from pathlib import Path
import json
from pathlib import Path
from time import time
from pathlib import Path
import json
import pandas as pd
import requests
import os
from tqdm import tqdm
from ast import literal_eval
from sklearn.model_selection import train_test_split
from bestbuy.src.organize.checker import is_nan_or_miss
def split_url_txt(start_idx):
    incomplete_products_path = Path(
        f'bestbuy/data/similar_urls.txt'
    )
    with open(incomplete_products_path, 'r', encoding='utf-8') as fp:
        lines=fp.readlines()
        lines_from_start_idx=lines[start_idx:]


    with open(f'bestbuy/data/similar_urls_starting_from_{start_idx}.txt', 'w') as filehandle:
        for listitem in lines_from_start_idx:
            filehandle.write('%s' % listitem)


def split_json(start_idx,end_idx):
    incomplete_products_path = Path(
        f'bestbuy/data/bestbuy_products_incomplete.json'
    )
    incomplete_products_from_xx_path = Path(
        f'bestbuy/data/bestbuy_products_incomplete_from_{start_idx}.json'
    )
    with open(incomplete_products_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        incomplete_dict_list_from_start_idx=incomplete_dict_list[start_idx: end_idx]
        with open(incomplete_products_from_xx_path, 'w', encoding='utf-8') as fp:
                json.dump(incomplete_dict_list_from_start_idx, fp, indent=4)
# split_url_txt(400)            
def split_review_product():
    review_path="/home/menglong/workspace/code/referred/product_scraper/dataset_construction/bestbuy/data/intermediate/bestbuy_review_1.json"
    incomplete_products_path = Path(
        review_path
    )
    output_products_path = Path(
        "/home/menglong/workspace/code/referred/product_scraper/dataset_construction/bestbuy/data/bestbuy_review_2.json"
    )
    review_err=0
    out_list=[]
    with open(incomplete_products_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        for product_json in tqdm(incomplete_dict_list):
            product_json["overview_section"]=None 
            product_json["thumbnails"]=None 
            product_json["Spec"]=None 
            product_json["thumbnail_paths"]=None 
            out_list.append(product_json)

    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4)    
    
def separate_missed_review():
    review_path="/home/menglong/workspace/code/referred/product_scraper/dataset_construction/bestbuy/data/bestbuy_review_2.2.json"
    incomplete_products_path = Path(
        review_path
    )
    output_products_path = Path(
        "/home/menglong/workspace/code/referred/product_scraper/dataset_construction/bestbuy/data/bestbuy_review_2.3.1_missed_review_after_25650.json"
    )
    review_err=0
    out_list=[]
    
    with open(incomplete_products_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        for i,product_json in tqdm(enumerate(incomplete_dict_list) ):
            if i<25650:
                continue 
            if (is_nan_or_miss(product_json,"reviews")): 
  
                out_list.append(product_json)

    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4)    
    

def separate_product_with_image_review():
    threshold=6020
    review_path="/home/menglong/workspace/code/referred/product_scraper/dataset_construction/bestbuy/data/bestbuy_review_2.3_incomplete.json"
    incomplete_products_path = Path(
        review_path
    )
    output_products_path = Path(
        "/home/menglong/workspace/code/referred/product_scraper/dataset_construction/bestbuy/data/bestbuy_review_2.3.1_product_with_image_review_after_6020.json"
    )
    review_err=0
    out_list=[]
    
    with open(incomplete_products_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        for i,product_json in tqdm(enumerate(incomplete_dict_list) ):
            if i<threshold:
                continue 
            
            if (not is_nan_or_miss(product_json,"reviews")): 
                has_image_review=False
                for review in product_json["reviews"]:
                    if not is_nan_or_miss(review, "product_images"):
                        has_image_review=True 
                        break
                if has_image_review:
                    out_list.append(product_json)

    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4)    
import copy

def separate_reviews_for_each_product():
    
    review_path="bestbuy/data/final/v0/bestbuy_review_2.3.6_w_image_not_long.json"
    incomplete_products_path = Path(
        review_path
    )
    output_products_path = Path(
        "bestbuy/data/final/v0/bestbuy_review_2.3.7_separate_review.json"
    )
    
    out_list=[]
    
    with open(incomplete_products_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        for i,product_json in tqdm(enumerate(incomplete_dict_list) ):
             
            
            if (not is_nan_or_miss(product_json,"reviews")): 
                reviews=product_json["reviews"]
                product_json["reviews"]=None 
                for review in reviews:
                    product_json_copy = copy.deepcopy(product_json)
                    product_json_copy["reviews"]=[review]
                    out_list.append(product_json_copy)
            
                
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4)       
         
def entity_train_test_split():    
    review_path="bestbuy/data/final/v2_course/bestbuy_review_2.3.16.10_remove_error_image.json"     
    incomplete_products_path = Path(
        review_path
    )
    output_train_products_path = Path(
        "bestbuy/data/final/v2_course/train/bestbuy_review_2.3.16.10.1_split.json"
    )
    output_dev_products_path = Path(
        "bestbuy/data/final/v2_course/val/bestbuy_review_2.3.16.10.1_split.json"
    )
    output_test_products_path = Path(
        "bestbuy/data/final/v2_course/test/bestbuy_review_2.3.16.10.1_split.json"
    )
    out_list=[]
    
    with open(incomplete_products_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        
        
        
        train_set,evaluation_set=train_test_split(incomplete_dict_list,test_size=0.2 )
        dev_set,test_set=train_test_split(evaluation_set,test_size=0.5)
        
            
        
    with open(output_train_products_path, 'w', encoding='utf-8') as fp:
        json.dump(train_set, fp, indent=4)   
    with open(output_dev_products_path, 'w', encoding='utf-8') as fp:
        json.dump(dev_set, fp, indent=4) 
    with open(output_test_products_path, 'w', encoding='utf-8') as fp:
        json.dump(test_set, fp, indent=4)       
         
         
def get_all_reviews_in_same_product( new_crawled_products_url_json_dict,product_id):
    review_product_json_list=[]
    num=0
    for review_dataset_json   in  new_crawled_products_url_json_dict  :
        if product_id==review_dataset_json["id"]:
            review_product_json_list.append(review_dataset_json)
   
    return review_product_json_list         
         
def   split():
    has_test=False 
    review_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v3/bestbuy_review_train_val_2.3.16.25_wrong_image.json"     
    incomplete_products_path = Path(
        review_path
    )
    output_train_products_path = Path(
        "bestbuy/data/final/v2_course/train/bestbuy_review_2.3.16.25.1_split.json"
    )
    output_dev_products_path = Path(
        "bestbuy/data/final/v2_course/val/bestbuy_review_2.3.16.25.1_split.json"
    )
    if has_test:
        output_test_products_path = Path(
            "bestbuy/data/final/v2_course/test/bestbuy_review_2.3.16.20.1_split.json"
        )
    out_list=[]
    num_dict={}
    test_set=[]
    dev_set=[]
    train_set=[]
    cur_product_id_list=[]
    with open(incomplete_products_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        for review_product_dataset_json   in tqdm(incomplete_dict_list):
            review_json=review_product_dataset_json["reviews"][0]
            attribute_num=len(review_json["attribute"])
            if attribute_num in num_dict:
                num_dict[attribute_num]+=1
            else:
                num_dict[attribute_num]=1
            if review_product_dataset_json["id"] not in cur_product_id_list:
                cur_product_id_list.append(review_product_dataset_json["id"])
                cur_review_list=get_all_reviews_in_same_product(incomplete_dict_list,review_product_dataset_json["id"])
                if attribute_num>=3:
                    
                    if has_test and len(test_set)<4000:
                        
                        test_set.extend(cur_review_list)
                    elif len(dev_set)<2000:
                        dev_set.extend(cur_review_list)
                    else:
                        train_set.extend(cur_review_list)
                else:
                    if len(dev_set)<2000:
                        dev_set.extend(cur_review_list)
                    else:
                        train_set.extend(cur_review_list)
                    
        print(num_dict)
        print(len(test_set),len(dev_set),len(train_set))
        
   
    
        # train_set,more_dev_set=train_test_split(train_set,test_size=0.1 )
        # dev_set.extend(more_dev_set)
    #     dev_set,test_set=train_test_split(evaluation_set,test_size=0.5)
        
            
        
    with open(output_train_products_path, 'w', encoding='utf-8') as fp:
        json.dump(train_set, fp, indent=4)   
    with open(output_dev_products_path, 'w', encoding='utf-8') as fp:
        json.dump(dev_set, fp, indent=4) 
    if has_test:
        with open(output_test_products_path, 'w', encoding='utf-8') as fp:
            json.dump(test_set, fp, indent=4)       
                
         
def move_dev_data_to_test(data_num,max_ratio):
    dev_products_path="bestbuy/data/final/v2_course/val/bestbuy_review_2.3.16.20.1_split.json" 
    incomplete_products_path = Path(
        dev_products_path
    )
    output_dev_products_path = Path(
        "bestbuy/data/final/v2_course/val/bestbuy_review_2.3.16.20.2_split.json"
    )
    output_test_products_path = Path(
        "bestbuy/data/final/v2_course/test/bestbuy_review_2.3.16.20.3_split.json"
    )       
   
    test_set=[]
    dev_set=[]
    num_dict={}
    cur_product_id_list=[]
    with open(incomplete_products_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        for review_product_dataset_json   in tqdm(incomplete_dict_list):
        
             
            if review_product_dataset_json["id"] not in cur_product_id_list:
                cur_product_id_list.append(review_product_dataset_json["id"])
                cur_review_list=get_all_reviews_in_same_product(incomplete_dict_list,review_product_dataset_json["id"])
                if len(cur_review_list) in num_dict:
                        num_dict[len(cur_review_list)]+=1
                else:
                    num_dict[len(cur_review_list)]=1
                if len(test_set)<data_num  :
                    
                    test_set.extend(cur_review_list)
                     
                else:
                    dev_set.extend(cur_review_list)
                
       
        print(len(test_set),len(dev_set) ,num_dict)
    with open(output_dev_products_path, 'w', encoding='utf-8') as fp:
        json.dump(dev_set, fp, indent=4) 
    with open(output_test_products_path, 'w', encoding='utf-8') as fp:
        json.dump(test_set, fp, indent=4)   
         
if __name__ == "__main__":  
    # move_dev_data_to_test(116,1.3)
    split()
    # separate_reviews_for_each_product()
# split_json(15000,20000)
# separate_product_with_image_review()