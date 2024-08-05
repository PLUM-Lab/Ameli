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

def remove_to_10_candidate(new_crawled_img_url_json_path,output_products_path):
    with open(new_crawled_img_url_json_path, 'r', encoding='utf-8') as fp:
        product_dataset_json_array = json.load(fp)
    out_list=[]
    for product_dataset_json   in  product_dataset_json_array :
        new_product_json={}
         
        gold_product_id=product_dataset_json["gold_entity_info"]["id"]
        fused_candidate_list=product_dataset_json["fused_candidate_list"]
        fused_score_dict=product_dataset_json["fused_score_dict"]
        out_score_dict={}
        score_dict_list=list(fused_score_dict.values()) 
        sorted_score_dict_list= sorted(score_dict_list, key=lambda x: x["fused_score"], reverse=True)
        for idx,( score_json) in enumerate(sorted_score_dict_list):
            product_id=score_json["corpus_id"]
            if idx>=10:
                break
            out_score_dict[product_id]=score_json
        # new_product_json=copy.deepcopy(product_dataset_json)
        product_dataset_json["fused_candidate_list"]=fused_candidate_list[:10]
        product_dataset_json["fused_score_dict"]=out_score_dict
        
        out_list.append(product_dataset_json)
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4)   
        
        
in_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/val/disambiguation/bestbuy_review_2.3.17.11.19.13_add_token_id.json"
out_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/val/disambiguation/bestbuy_review_2.3.17.11.19.15.1_remove_to_10_candidate.json"        
remove_to_10_candidate(in_path,out_path)        