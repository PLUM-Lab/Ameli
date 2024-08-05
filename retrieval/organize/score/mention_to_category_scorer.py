
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

 
import argparse


import json
from pathlib import Path
from time import time
from pathlib import Path
import json
import pandas as pd
def metric(review_file_str):
    data_name=f"{review_file_str.split('/')[-2]}"
    new_crawled_img_url_json_path = Path(
        review_file_str
        ) 
    
    category_right_num=0
    category_wrong_num=0
    category_right_top1_num=0
    product_id_right_num=0
    prediction_num=0
    precision_recall_at_k=[100000]#1,2,3,4,5,6,7,8,9,10 
    recall_at_k_result = {k: [] for k in precision_recall_at_k}
    
    with open(new_crawled_img_url_json_path, 'r', encoding='utf-8') as fp:
        product_dataset_json_array = json.load(fp)
        for product_dataset_json   in product_dataset_json_array :#tqdm(
            gold_product_id=product_dataset_json["gold_entity_info"]["id"]
            category=product_dataset_json["product_category"]
            mention_to_category_list=product_dataset_json["mention_to_category_list"]
            mention_to_product_id_list=product_dataset_json["mention_to_product_id_list"]
            if mention_to_product_id_list is not None and gold_product_id in mention_to_product_id_list:
                product_id_right_num+=1
            if mention_to_category_list is not None and  category in mention_to_category_list:
                category_right_num+=1
                for k in precision_recall_at_k:
                    for k_val in  precision_recall_at_k:
                        if category in mention_to_category_list[0:k_val]:
                            recall_at_k_result[k_val].append(1)
                        else:
                            recall_at_k_result[k_val].append(0)
                      
            else:
                category_wrong_num+=1
                print(product_dataset_json["review_id"])
            prediction_num+=len(set(mention_to_category_list))
            
    for k in recall_at_k_result:
        recall_at_k_result[k] = np.mean(recall_at_k_result[k])
    print(f"recall_category:{category_right_num/len(product_dataset_json_array)},precision_category:{category_right_num/prediction_num},len(product_dataset_json):{len(product_dataset_json_array)},category_right_num:{category_right_num},category_wrong_num:{category_wrong_num}, recall_category_right_topk:{recall_at_k_result}, recall_product_id:{product_id_right_num/len(product_dataset_json_array)}")
    
    
    


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',type=str,help=" ",default="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v4/val/bestbuy_review_2.3.16.29.9.3_standard_add_4.1_mention_to_category_product_id_remove_score_dict.json") #politifact_v3,mode3_latest_v5
    parser.add_argument('--out_path',type=str,help=" ",default="bestbuy/output/html")
    # parser.add_argument('--example_type',type=str,help=" ",default="verification")
    parser.add_argument('--mode',type=str,help=" ",default="val")
    parser.add_argument('--attribute_source',type=str,help=" ",default="similar")#gold
    parser.add_argument('--attribute_field',type=str,help=" ",default="all")
    
    parser.add_argument('--attribute_logic',type=str,help=" ",default="numeral")#exact
    parser.add_argument('--filter_by_attribute_field',type=str,help=" ",default="Brand,Color")#all
    # parser.add_argument('--generate_claim_level_report',type=str,help=" ",default="y")
    args = parser.parse_args()
    return args


  
if __name__ == '__main__':
    args = parser_args()
    data_path=args.data_path   
    mode=args.mode
    # "/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v4/val/bestbuy_review_2.3.16.29.9.3_standard_add_4.1_mention_to_category_product_id_remove_score_dict.json"     
    metric(args.data_path)