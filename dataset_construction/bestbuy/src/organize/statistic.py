from pathlib import Path
import json
from unicodedata import category
import pandas as pd
import requests
import os
from tqdm import tqdm
from ast import literal_eval 



import argparse

from bestbuy.src.organize.checker import is_nan_or_miss

def count_review_num(review_path):
    incomplete_products_path = Path(
        review_path
    )
    review_num=0
    with open(incomplete_products_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        for product_json in incomplete_dict_list:
            
            if not is_nan_or_miss(product_json,"reviews"):
                reviews=product_json["reviews"]
                review_num+=len(reviews)
    return review_num

def spec_to_json_attribute(spec_object,is_key_attribute):
    merged_attribute_json ={}
    if is_key_attribute:
        important_attribute_json_list=spec_object[:2]
    else:
        important_attribute_json_list=spec_object
    for attribute_subsection in important_attribute_json_list:
        attribute_list_in_one_section=attribute_subsection["text"]
        for attribute_json  in attribute_list_in_one_section:
            attribute_key=attribute_json["specification"]
            attribute_value=attribute_json["value"]
            merged_attribute_json[attribute_key]=attribute_value
    return merged_attribute_json

def product_statistic(product_path):
    incomplete_products_path = Path(
        product_path
    ) 
    key_attribute_num=0
    all_attribute_num=0
    category_set=set()
    product_image_num=0
    category_num_dict={}
    with open(incomplete_products_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        product_num=len(incomplete_dict_list)
        for product_json in incomplete_dict_list:
            category=product_json["product_category"]
            category_set.add(category)
            if category not in category_num_dict:
                category_num_dict[category]=0
            category_num_dict[category]+=1
            if "image_path" in product_json:
                if len(product_json["image_path"])==0:
                    print(f"ERROR: {product_json['id']}")
                product_image_num+=len(product_json["image_path"])
            if "Spec" in product_json:
                spec_json=product_json["Spec"]
                key_attribute_dict=spec_to_json_attribute(spec_json,is_key_attribute=True)
                key_attribute_num+=len(key_attribute_dict)
                all_attribute_dict=spec_to_json_attribute(spec_json,is_key_attribute=False)
                all_attribute_num+=len(all_attribute_dict)
    sorted_category_num_dict=sorted(category_num_dict.items(), key=lambda x:x[1],reverse=True)
    
    print(f"product_num:{product_num}, product_image_num:{product_image_num}, category:{len(category_set)}, key_attribute_num:{key_attribute_num/product_num}, all_attribute_num:{all_attribute_num/product_num}")
    for idx,(category,num) in enumerate(sorted_category_num_dict ):
        if idx>=10:
            break
        print(f"frequent category {idx}: {num},{round(100*num/len(incomplete_dict_list),2)} {category}")

def review_statistic(review_path):
    if review_path is not None:
        incomplete_products_path = Path(
            review_path
        )
        review_id_set=set()
        review_num=0
        review_image_num=0
        with open(incomplete_products_path, 'r', encoding='utf-8') as fp:
            incomplete_dict_list = json.load(fp)
            for review in incomplete_dict_list:
                review_id=review["review_id"]
                if review_id in review_id_set:
                    exit("error")
                else:
                    review_id_set.add(review_id)
              
              
                if "review_image_path" in review:
                    review_image_num+=len(review["review_image_path"])
        review_num=len(incomplete_dict_list)  
        print(f"review_image_num:{review_image_num},review_num:{review_num}, ratio:{review_image_num/review_num}")    
        return review_num


def statistic(file_path,review_path_name):
    
    product_statistic(file_path)
    for sub in ["val","test","train"]:
        review_path=os.path.join("/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6",sub,review_path_name)
        review_statistic(review_path)
    
    
    
    # incomplete_products_path = Path(
    #     file_path
    # )
    # review_num=count_review_num(review_path)
     
    
    
    # with open(incomplete_products_path, 'r', encoding='utf-8') as fp:
    #     incomplete_dict_list = json.load(fp)
    #     product_num=len(incomplete_dict_list)

    # print(f"product_num:{product_num},review_num:{review_num}")    
# runner


def gen_category_statistic(  review_path):
    category_review_num_dict={}
    incomplete_products_path = Path(
        review_path
    )
    review_num=0
    with open(incomplete_products_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        for product_json in incomplete_dict_list:
            
            if not is_nan_or_miss(product_json,"reviews"):
                category=product_json["product_category"]
                category=category.split(" -> ")[1]
                if category in category_review_num_dict:
                    review_num=category_review_num_dict[category]
                    review_num+=len(product_json["reviews"])
                    category_review_num_dict[category]=review_num
                else:
                    review_num+=len(product_json["reviews"])
                    category_review_num_dict[category]=review_num
    print(category_review_num_dict)

def statistic_similar_product_num(review_path):
 
    incomplete_products_path = Path(
        review_path
    )
    review_num=0
    review_image_num=0
    with open(incomplete_products_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        for product_json in incomplete_dict_list:
            
            if not is_nan_or_miss(product_json,"reviews"):
                similar_product_id_list=product_json["similar_product_id"]
                product_id_with_similar_image_list=product_json["product_id_with_similar_image"]
                similar_product_id_list.extend(product_id_with_similar_image_list)
                similar_product_id_list=list(set(similar_product_id_list))
                review_num+=len(similar_product_id_list)
              
                
    print(f" similar_product_num:{review_num}")    
    return review_num


def get_args():
   


    parser = argparse.ArgumentParser() 
    # parser.add_argument("--lr", default=1e-7, type=float)
    # parser.add_argument("--step_size", default=10, type=int)
    # parser.add_argument("--start_id", default=0, type=int)
    # parser.add_argument("--end_id", default=6000, type=int)
    # parser.add_argument("--use_pre_trained_model", default=True, action="store_false") 
    parser.add_argument("--file",default='/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/bestbuy_products_40000_3.4.19_final_format.json', type=str  ) 
    parser.add_argument("--review_file",default='bestbuy_review_2.3.17.0.1_format_wo_candidates_review_id.json', type=str  ) 
    parser.add_argument("--review_parent_dir",default='/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6', type=str  ) 
    args = parser.parse_args()

    print(args)
    return args


def split_statistic(product_path,review_parent_dir,review_file_name):
    
    product_statistic(product_path)
    review_statistic(os.path.join(review_parent_dir,"train",review_file_name))
    review_statistic(os.path.join(review_parent_dir,"val",review_file_name))
    review_statistic(os.path.join(review_parent_dir,"test",review_file_name))

if __name__ == "__main__":  
    args=get_args()
    # gen_category_statistic(args.review_file)
    statistic(args.file,args.review_file)
    # statistic_similar_product_num(args.review_file)
    # split_statistic(args.file,args.review_parent_dir,review_file_name="bestbuy_review_2.3.16.27.1_similar_split.json")
     
        