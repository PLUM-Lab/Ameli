# from attribute.gen_review_attribute import match_attribute_by
 
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
import numpy as np 
from util.env_config import *

from pathlib import Path
import json

import os
import pandas as pd
import shutil  
from random import sample
from tqdm import tqdm 
import random
from nltk.tokenize import word_tokenize
from disambiguation.data_util.inner_util import gen_gold_attribute, gen_gold_attribute_without_key_section, gen_review_text, json_to_dict
import re 

import copy 
#find attributes which are correct for gpt2 but not for gpt
def find_difference(review_file_str,output_path):
    new_crawled_img_url_json_path = Path(
        review_file_str
        ) 
    filter_gold_num=0
    extraction_num_dict={}
    extraction_right_num_dict={}
    gold_num_dict={}
    out_list=[]
    total_right_extraction_num,total_extracted_attributes_num,total_gold_product_attributes_num =0,0,0
    with open(new_crawled_img_url_json_path, 'r', encoding='utf-8') as fp:
        product_dataset_json_array = json.load(fp)
        for review_product_json   in product_dataset_json_array :#tqdm(
            review_text=gen_review_text(review_product_json)
            gold_product_id=review_product_json["gold_entity_info"]["id"]
            review_json=review_product_json
            similar_product_id_list=review_product_json["fused_candidate_list"]
            annotated_attributes=review_product_json["gold_attribute_for_predicted_category"]
            gpt2_attributes=review_product_json["predicted_attribute_gpt2"]
            gpt2_few_attributes=review_product_json["predicted_attribute_gpt2_few"]
            out_json={}
            out_json["predicted_attribute_gpt2"]={}
            out_json["gold_attribute_for_predicted_category"]={}
            out_json["predicted_attribute_gpt2_few"]={}
            out_json["predicted_attribute_context_gpt2"]={}
            out_json["predicted_attribute_context_gpt2_few"]={}
            out_json["is_attribute_correct_gpt2"]={}
            out_json["is_attribute_correct_gpt2_few"]={}
            for attribute_key, attribute_value in annotated_attributes.items():
                if attribute_key in gpt2_attributes and attribute_value in gpt2_attributes[attribute_key]:
                    if attribute_key not in gpt2_few_attributes or  attribute_value not in gpt2_few_attributes[attribute_key]:
                        # predicted_attribute_gpt2_json={attribute_key:gpt2_attributes[attribute_key]}
                        # predicted_attribute_gpt2_few_json={attribute_key:gpt2_few_attributes[attribute_key]}
                        # out_json[""]=review_product_json[""]
                        out_json["gold_attribute_for_predicted_category"][attribute_key]=review_product_json["gold_attribute_for_predicted_category"][attribute_key]
                        out_json["predicted_attribute_gpt2"][attribute_key]=gpt2_attributes[attribute_key]
                        if attribute_key in gpt2_few_attributes:
                            out_json["predicted_attribute_gpt2_few"][attribute_key]=gpt2_few_attributes[attribute_key]
                        out_json["is_attribute_correct_gpt2"][attribute_key]=gpt2_few_attributes["is_attribute_correct_gpt2"][attribute_key]
                        if attribute_key in gpt2_few_attributes["is_attribute_correct_gpt2_few"]:
                            out_json["is_attribute_correct_gpt2_few"][attribute_key]=gpt2_few_attributes["is_attribute_correct_gpt2_few"][attribute_key]
                        out_json["predicted_attribute_context_gpt2"][attribute_key]=review_product_json["predicted_attribute_context_gpt2"][attribute_key]
                        if attribute_key in review_product_json["predicted_attribute_context_gpt2_few"]:
                            out_json["predicted_attribute_context_gpt2_few"][attribute_key]=review_product_json["predicted_attribute_context_gpt2_few"][attribute_key]
                        out_json["review_id"]=review_product_json["review_id"]
                        # out_json[""]=review_product_json[""]
            if len(out_json["predicted_attribute_context_gpt2_few"])>0:            
                out_list.append(copy.deepcopy(out_json))
                        # break
    with open(output_path , 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4) 
            
             
             
find_difference("/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/disambiguation/bestbuy_review_2.3.17.11.21_add_gpt2_vicuna.json",
                "/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/temp/bestbuy_review_2.3.17.11.21_diff_gpt2_v2.json")             