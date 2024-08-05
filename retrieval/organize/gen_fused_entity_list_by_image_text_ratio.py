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

from retrieval.organize.score.scorer_helper import gen_score_dict, gen_sorted_candidate_list, gen_sorted_candidate_list_for_three 




def gen_new_retrieved_dataset(review_file_str,output_products_path_str,text_score_key,text_ratio,is_save=True,candidate_num=1000):
    new_crawled_img_url_json_path = Path(
        review_file_str
        ) 
 
    with open(new_crawled_img_url_json_path, 'r', encoding='utf-8') as fp:
        product_dataset_json_array = json.load(fp)
        gen_new_retrieved_dataset_for_one_json_list(product_dataset_json_array,output_products_path_str,text_score_key,text_ratio)


import copy
def gen_new_retrieved_dataset_for_one_json_list(product_dataset_json_array,output_products_path_str,
                                                text_score_key,text_ratio,is_save=True,candidate_num=None,is_cross=False,obtain_score_dict_num=None,
                                                is_remove_prior_probability=False,second_score_key="image_score",out_score_key="fused_score",
                                                use_original_fused_score_dict=False,image_ratio=None,
                                                                                           third_score_key=None):
    
    output_products_path = Path(
        output_products_path_str
    )
    
    out_list=[]
    new_product_json_list=[]
     
    for product_dataset_json   in  product_dataset_json_array :
        new_product_json={}
        new_product_dataset_json=copy.deepcopy(product_dataset_json)
        gold_product_id=product_dataset_json["gold_entity_info"]["id"]
        score_dict=gen_score_dict(new_product_dataset_json,is_cross,use_original_fused_score_dict=use_original_fused_score_dict)
        if third_score_key is not None:
            candidate_id_list,fused_score_dict=gen_sorted_candidate_list_for_three(score_dict,text_ratio,text_score_key,obtain_score_dict_num=obtain_score_dict_num,
                                                                     gold_product_id=gold_product_id,second_score_key= second_score_key,  
                                                                     out_score_key=out_score_key,image_ratio=image_ratio,third_score_key=third_score_key)
        else:
            candidate_id_list,fused_score_dict=gen_sorted_candidate_list(score_dict,text_ratio,text_score_key,obtain_score_dict_num=obtain_score_dict_num,
                                                                        gold_product_id=gold_product_id,second_score_key= second_score_key,
                                out_score_key=out_score_key)
        # new_product_json=copy.deepcopy(product_dataset_json)
        new_product_dataset_json["fused_candidate_list"]=candidate_id_list[:candidate_num] if candidate_num is not None else candidate_id_list
        new_product_dataset_json["fused_score_dict"]=fused_score_dict
        new_product_dataset_json["image_score_dict"]={}
        new_product_dataset_json["text_score_dict"]={}
        new_product_dataset_json["desc_score_dict"]={}
        new_product_dataset_json["product_id_with_similar_image_by_review"]={}
        new_product_dataset_json["product_id_with_similar_text_by_review"]={}
        if "product_id_with_similar_desc_by_review" in product_dataset_json:
            new_product_dataset_json["product_id_with_similar_desc_by_review"]={}
        if is_remove_prior_probability and "mention_to_category_list" in new_product_dataset_json:
            del new_product_dataset_json["mention_to_category_list"]
            del new_product_dataset_json["mention_to_product_id_list"]
        out_list.append(new_product_dataset_json)
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4)      
    return out_list
       

import argparse
def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',type=str,help=" ",default="mocheg2/test") #politifact_v3,mode3_latest_v5
    parser.add_argument('--out_path',type=str,help=" ",default="bestbuy/output/html")
    # parser.add_argument('--example_type',type=str,help=" ",default="verification")
    parser.add_argument('--mode',type=str,help=" ",default="val")
    # parser.add_argument('--generate_claim_level_report',type=str,help=" ",default="y")
    args = parser.parse_args()
    return args


  
  
  
  
if __name__ == '__main__':
    args = parser_args()
    # data_path=args.data_path 
    # out_path=args.out_path
    mode=args.mode
# path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v4/test/bestbuy_review_2.3.16.28.4_similar_text_desc_split.json"
# out_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v4/test/bestbuy_review_2.3.16.28.5_fuse_score.json"
 
# path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v4/val/bestbuy_review_2.3.16.28.4_similar_text_desc_split.json"
# out_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v4/val/bestbuy_review_2.3.16.28.5_fuse_score.json"
 
    data_path=f"/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v4/{mode}/bestbuy_review_2.3.16.29.4.1_desc_text_similar_image_max_mode_split.json"
    out_path=f"/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v4/{mode}/bestbuy_review_2.3.16.29.5.1_fuse_score_title_0.6_max_image.json"
    text_score_key,text_ratio='bi_score', 0.6
    gen_new_retrieved_dataset(data_path,out_path ,text_score_key,text_ratio)