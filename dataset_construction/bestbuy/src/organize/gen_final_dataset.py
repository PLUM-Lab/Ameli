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
from bestbuy.src.organize.checker import is_nan, is_nan_or_miss

from bestbuy.src.organize.merge_attribute import json_to_dict,json_to_review_dict, new_review_json_to_product_dict, review_json_to_product_dict, review_json_to_product_dict_for_old_format

def merge_attribute():
     
     
    output_products_path = Path(
        f'/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/train/attribute/bestbuy_review_2.3.17.9.4_merge_chatgpt.json'
    ) 
    product_json_dict={}
    output_list=[]
    dir_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/train/attribute/gpt"
    spec_file_list=os.listdir(dir_path)
    for path in spec_file_list:
        spec_file=os.path.join(dir_path, path)
        new_crawled_img_url_json_path = Path(
            spec_file
            ) 
        with open(new_crawled_img_url_json_path, 'r', encoding='utf-8') as fp:
            product_dataset_json_array = json.load(fp)
            for product_dataset_json   in tqdm(product_dataset_json_array):
                review_id=product_dataset_json["review_id"] #["reviews"][0]
                product_json_dict[review_id]=product_dataset_json

    for i in sorted(product_json_dict.keys()):
        output_list.append(product_json_dict[i])
    # output_list=list(product_json_dict.values())
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(output_list, fp, indent=4)                          
# merge_all()        
if __name__ == "__main__":  
    # merge_attribute()
    merge_all_for_review_for_attribute("test")
    # merge_attribute()