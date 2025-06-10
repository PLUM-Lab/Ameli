from pathlib import Path
import json
import pandas as pd
import requests
import os
from tqdm import tqdm
from ast import literal_eval
def change_path_without_parent_dir_for_product():
    incomplete_products_path = Path(
        "/home/menglong/workspace/code/referred/product_scraper/dataset_construction/bestbuy/data/final/v1/bestbuy_products_40000_3.4.6_remove_missed_product.json"
    )
    out_list=[]
    with open(incomplete_products_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        for product_json in incomplete_dict_list:
            thumbnail_paths=product_json["thumbnail_paths"]
            real_image_name_list=change_img_path_list(thumbnail_paths)
            product_json["image_path"]=real_image_name_list
            product_json["image_url"]=product_json["thumbnails"]
            product_json["thumbnails"]=None 
            product_json["thumbnail_paths"]=None 
            out_list.append(product_json)
    with open("bestbuy/data/final/v1/bestbuy_products_40000_3.4.7_remove_image_path_prefix.json", 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4) 
     
     
def change_img_path_list(thumbnail_paths):
    real_image_name_list=[]
    for filepath in  thumbnail_paths:
        real_image_name=filepath
        if "images/" in filepath:
            real_image_name=filepath.split("images/")[-1]
        elif "images\\" in filepath: 
            real_image_name=filepath.split("images\\")[-1]
        elif "images" in filepath: 
            print(f"ERROR! {filepath}")
            continue 
        # else:
        #     print("ERROR! in extract_images_for_one_list")
        #     continue 
        real_image_name_list.append(real_image_name)       
    return real_image_name_list
            
def change_path_without_parent_dir_for_review():
    incomplete_products_path = Path(
        "bestbuy/data/final/v1/bestbuy_review_2.3.13_fix_image_url_bug.json"
    )
    out_list=[]
    with open(incomplete_products_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        
        for product_json in incomplete_dict_list:
            review=product_json["reviews"][0]
            thumbnail_paths=review["image_path"]
            if thumbnail_paths is not None:
                real_image_name_list=change_img_path_list(thumbnail_paths)
                review["image_path"]=real_image_name_list
            # review["image_url"]=review["product_images"]
            # review["thumbnail_paths"]=None 
            # review["product_images"]=None 
            product_json["reviews"]=[review]
            out_list.append(product_json)
    with open("bestbuy/data/final/v1/bestbuy_review_2.3.13.1_fix_image_url.json", 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4) 
            
if __name__ == "__main__":                 
    change_path_without_parent_dir_for_review()
# change_path_without_parent_dir_for_product()