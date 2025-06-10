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
import concurrent.futures
import logging
import spacy 
import numpy as np

def gen_right_case_product_id(input_products_path):
    with open(input_products_path, 'r', encoding='utf-8') as fp:
        product_dataset_json_array = json.load(fp)
        correct_id_list=[]
        k_val=10
        for product_dataset_json   in product_dataset_json_array :
            category=product_dataset_json["product_category"]
            product_id=product_dataset_json["reviews"][0]["id"]
            
            
            mention_to_category_list=product_dataset_json["reviews"][0]["mention_to_category_list"]
       
            if category in mention_to_category_list[0:k_val]:
                correct_id_list.append(product_id)
                
    return correct_id_list


def check_error(input_products_path,correct_product_id_list):
    with open(input_products_path, 'r', encoding='utf-8') as fp:
        product_dataset_json_array = json.load(fp)
        cur_correct_id_list=[]
        for product_dataset_json   in product_dataset_json_array :
            
            category=product_dataset_json["product_category"]
            product_id=product_dataset_json["reviews"][0]["id"]
            k_val=10
  
            mention_to_category_list=product_dataset_json["reviews"][0]["mention_to_category_list"]
            
            if category not in mention_to_category_list[0:k_val] :
                if   product_id   in correct_product_id_list:
                    print(product_id)
            else:
                cur_correct_id_list.append(product_id)
    print("pass")
            
                
                

correct_product_id_list=gen_right_case_product_id("/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v4/val/bestbuy_review_2.3.16.29.11.3_10000_standard_rerank.json")
check_error("/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v4/val/bestbuy_review_2.3.16.29.11.3_1_standard_rerank.json",correct_product_id_list)