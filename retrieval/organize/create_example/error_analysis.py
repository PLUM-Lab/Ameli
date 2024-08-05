import os 
import pandas as pd 
 
from random import sample
import json
from pathlib import Path
from time import time
from pathlib import Path
import json
import pandas as pd
import requests
import os
import copy
from util.common_util import json_to_product_id_dict, review_json_to_product_dict     
from util.env_config import * 
from tqdm import tqdm
from ast import literal_eval

from retrieval.retrieval_organize_main import merge_all_for_review
def is_nan_or_miss(json,key):
    if key  not in json:
        return True
    elif json[key] is None or len( json[key]) == 0:          
        return True 
    else:
        return False   


def gen_review_product_json_dict():
    product_dataset_path = Path(test_dir)
    with open(product_dataset_path, 'r', encoding='utf-8') as fp:
        product_dataset_json_array = json.load(fp)
        review_product_json_dict=review_json_to_product_dict(product_dataset_json_array)
        
    with open(products_path_str, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        product_json_dict=json_to_product_id_dict(new_crawled_products_url_json_array)
    return review_product_json_dict,product_json_dict

def gen_score_for_product_id(score_dict,product_id_list,product_id):
    new_score_dict={}
    if str(product_id) in score_dict:
        new_score_dict[product_id]=score_dict[str(product_id)]
    for idx, (item_key, item_value) in enumerate(score_dict.items()):
        item_key=int(item_key)
        if item_key in product_id_list:
            
            new_score_dict[item_key]=item_value
    return new_score_dict

def gen_first_k(score_dict,k):
    new_score_dict={}
    for idx, (item_key, item_value) in enumerate(score_dict.items()):
        if idx < k:
            item_key=int(item_key)
            new_score_dict[item_key]=item_value
    return new_score_dict

def gen_example(ordered_candidate_id_list,product_json_dict,review_dataset_json,k,gold_product_index
            ):
    product_id=review_dataset_json["gold_entity_info"]["id"]
    review_id=review_dataset_json["review_id"]
    product_json=copy.deepcopy(product_json_dict[product_id])
    product_json["reviews"]=review_dataset_json 
    product_json["fused_candidate_list"]=review_dataset_json["fused_candidate_list"]
    # ordered_candidate_id_list.append(product_id)
    product_json["similar_product_id"]=ordered_candidate_id_list
    product_json["product_id_with_similar_image"]=[]
    product_json["fused_score_dict"]=review_dataset_json["fused_score_dict"]
    # product_json["image_score_dict"]=gen_score_for_product_id(review_product_json_array_with_score_dict[review_id]["image_score_dict"],ordered_candidate_id_list,product_id)
    # product_json["text_score_dict"]=gen_score_for_product_id(review_product_json_array_with_score_dict[review_id]["text_score_dict"],ordered_candidate_id_list,product_id)
    # product_json["desc_score_dict"]=gen_score_for_product_id(review_product_json_array_with_score_dict[review_id]["desc_score_dict"],ordered_candidate_id_list,product_id)
    # product_json["fused_score_dict"]=gen_score_for_product_id(review_product_json_array_with_fused_score_dict[review_id]["fused_score_dict"],ordered_candidate_id_list,product_id)
    # product_json["product_id_with_similar_image_by_review"]=review_product_json_array_with_score_dict[review_id]["product_id_with_similar_image_by_review"][:k]
    # product_json["product_id_with_similar_text_by_review"]=review_product_json_array_with_score_dict[review_id]["product_id_with_similar_text_by_review"][:k]
    # if "product_id_with_similar_desc_by_review" in review_product_json_array_with_score_dict[review_id]:
    #     product_json["product_id_with_similar_desc_by_review"]=review_product_json_array_with_score_dict[review_id]["product_id_with_similar_desc_by_review"][:k]
    product_json["gold_product_index"]=gold_product_index
    return product_json  

def gen_error_candidate_list_with_filter_issue(retrieval_output_file_path):
    with open(products_path_str, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        product_json_dict=json_to_product_id_dict(new_crawled_products_url_json_array)
    # original_retrieval_path=f"/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/retrieval/bestbuy_review_2.3.17.5.1.2_finetuned_26_text_title_no_reranker.json"
    # update_score_retrieval_path=f"/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/retrieval/bestbuy_review_2.3.17.6.4.2_update_image_score_fine_tune.json"
    # new_retrieval_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/retrieval/bestbuy_review_2.3.17.3.2_27_finetuned.json"
    # review_product_json_array_with_score=merge_all_for_review(new_retrieval_path,original_retrieval_path,update_score_retrieval_path,is_save=False)
    # with open(original_retrieval_path, 'r', encoding='utf-8') as fp:
    #         review_product_json_array_with_score = json.load(fp)    
    # review_product_json_array_with_score_dict=review_json_to_product_dict(review_product_json_array_with_score)
    # review_product_json_array_with_fused_score_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/retrieval/bestbuy_review_2.3.17.7.1_top10_fused_score_fine_tune_27_26.json"
    # with open(review_product_json_array_with_fused_score_path, 'r', encoding='utf-8') as fp:
    #         review_product_json_array_with_fused_score = json.load(fp)    
    #         review_product_json_array_with_fused_score_dict=review_json_to_product_dict(review_product_json_array_with_fused_score)
            
    output_error_path = Path(
        f'output/error/retrieval/bestbuy_50_error_high_retrieval_filter_issue.json'
    )  
    error_review_product_json_list=[]
    k=1 
    with open(retrieval_output_file_path, 'r', encoding='utf-8') as fp:
        review_product_json_array = json.load(fp)
        for review_product_json in review_product_json_array:
            gold_product_id=review_product_json["gold_entity_info"]["id"]
            similar_product_id_list=review_product_json["fused_candidate_list"]
            fused_score_dict=review_product_json["fused_score_dict"]
            if gold_product_id not in similar_product_id_list[:k]:#  k
                if str(gold_product_id) in fused_score_dict:
                    gold_product_index=fused_score_dict[str(gold_product_id)]["gold_product_position_in_retrieval"]
                    if gold_product_index==0:
                        similar_product_id_list.insert(1,gold_product_id)
                        review_product_json["fused_candidate_list"]=similar_product_id_list
                        gold_product_index=1
                        new_score_dict={}#TODO
                        for product_id,score_dict in review_product_json["fused_score_dict"].items():
                            del score_dict["disambiguation_image_score"]
                            new_score_dict[product_id]=score_dict
                        review_product_json["fused_score_dict"]=new_score_dict
                        error_review_dataset_json=gen_example(similar_product_id_list[:k],product_json_dict,review_product_json,
                                                            k ,gold_product_index)
                        error_review_product_json_list.append(error_review_dataset_json)
    print(len(error_review_product_json_list))
    # sampled_error_list=sample(error_review_product_json_list,100)
 
    with open(output_error_path, 'w', encoding='utf-8') as fp:
        json.dump(error_review_product_json_list, fp, indent=4)
    

def gen_error_candidate_list_with_high_retrieval(retrieval_output_file_path):
    with open(products_path_str, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        product_json_dict=json_to_product_id_dict(new_crawled_products_url_json_array)
    # original_retrieval_path=f"/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/retrieval/bestbuy_review_2.3.17.5.1.2_finetuned_26_text_title_no_reranker.json"
    # update_score_retrieval_path=f"/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/retrieval/bestbuy_review_2.3.17.6.4.2_update_image_score_fine_tune.json"
    # new_retrieval_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/retrieval/bestbuy_review_2.3.17.3.2_27_finetuned.json"
    # review_product_json_array_with_score=merge_all_for_review(new_retrieval_path,original_retrieval_path,update_score_retrieval_path,is_save=False)
    # with open(original_retrieval_path, 'r', encoding='utf-8') as fp:
    #         review_product_json_array_with_score = json.load(fp)    
    # review_product_json_array_with_score_dict=review_json_to_product_dict(review_product_json_array_with_score)
    # review_product_json_array_with_fused_score_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/retrieval/bestbuy_review_2.3.17.7.1_top10_fused_score_fine_tune_27_26.json"
    # with open(review_product_json_array_with_fused_score_path, 'r', encoding='utf-8') as fp:
    #         review_product_json_array_with_fused_score = json.load(fp)    
    #         review_product_json_array_with_fused_score_dict=review_json_to_product_dict(review_product_json_array_with_fused_score)
            
    output_error_path = Path(
        f'output/error/retrieval/bestbuy_50_error_high_retrieval_low_disambiguation.json'
    )  
    error_review_product_json_list=[]
    k=1 
    with open(retrieval_output_file_path, 'r', encoding='utf-8') as fp:
        review_product_json_array = json.load(fp)
        for review_product_json in review_product_json_array:
            gold_product_id=review_product_json["gold_entity_info"]["id"]
            similar_product_id_list=review_product_json["fused_candidate_list"]
            fused_score_dict=review_product_json["fused_score_dict"]
            if gold_product_id not in similar_product_id_list[:k]:#  k
                if str(gold_product_id) in fused_score_dict:
                    gold_product_index=fused_score_dict[str(gold_product_id)]["gold_product_position_in_retrieval"]
                    if gold_product_index==0:
                        similar_product_id_list.insert(1,gold_product_id)
                        review_product_json["fused_candidate_list"]=similar_product_id_list
                        gold_product_index=1
                        new_score_dict={}#TODO
                        for product_id,score_dict in review_product_json["fused_score_dict"].items():
                            del score_dict["disambiguation_image_score"]
                            new_score_dict[product_id]=score_dict
                        review_product_json["fused_score_dict"]=new_score_dict
                        error_review_dataset_json=gen_example(similar_product_id_list[:k],product_json_dict,review_product_json,
                                                            k ,gold_product_index)
                        error_review_product_json_list.append(error_review_dataset_json)
    print(len(error_review_product_json_list))
    # sampled_error_list=sample(error_review_product_json_list,100)
 
    with open(output_error_path, 'w', encoding='utf-8') as fp:
        json.dump(error_review_product_json_list, fp, indent=4)


def gen_error_candidate_list(retrieval_output_file_path):
    with open(products_path_str, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        product_json_dict=json_to_product_id_dict(new_crawled_products_url_json_array)
    # original_retrieval_path=f"/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/retrieval/bestbuy_review_2.3.17.5.1.2_finetuned_26_text_title_no_reranker.json"
    # update_score_retrieval_path=f"/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/retrieval/bestbuy_review_2.3.17.6.4.2_update_image_score_fine_tune.json"
    # new_retrieval_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/retrieval/bestbuy_review_2.3.17.3.2_27_finetuned.json"
    # review_product_json_array_with_score=merge_all_for_review(new_retrieval_path,original_retrieval_path,update_score_retrieval_path,is_save=False)
    # with open(original_retrieval_path, 'r', encoding='utf-8') as fp:
    #         review_product_json_array_with_score = json.load(fp)    
    # review_product_json_array_with_score_dict=review_json_to_product_dict(review_product_json_array_with_score)
    # review_product_json_array_with_fused_score_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/retrieval/bestbuy_review_2.3.17.7.1_top10_fused_score_fine_tune_27_26.json"
    # with open(review_product_json_array_with_fused_score_path, 'r', encoding='utf-8') as fp:
    #         review_product_json_array_with_fused_score = json.load(fp)    
    #         review_product_json_array_with_fused_score_dict=review_json_to_product_dict(review_product_json_array_with_fused_score)
            
    output_error_path = Path(
        f'output/error/retrieval/bestbuy_50_error.json'
    )  
    error_review_product_json_list=[]
    k=1 
    with open(retrieval_output_file_path, 'r', encoding='utf-8') as fp:
        review_product_json_array = json.load(fp)
        for review_product_json in review_product_json_array:
            gold_product_id=review_product_json["gold_entity_info"]["id"]
            similar_product_id_list=review_product_json["fused_candidate_list"]
            fused_score_dict=review_product_json["fused_score_dict"]
            if gold_product_id not in similar_product_id_list[:k]:#  k
                if gold_product_id in similar_product_id_list:
                    gold_product_index=similar_product_id_list.index(gold_product_id)
                elif str(gold_product_id) in fused_score_dict:
                    gold_product_index=fused_score_dict[str(gold_product_id)]["gold_product_position_in_retrieval"]
                
                else:
                    
                    gold_product_index=5001
                error_review_dataset_json=gen_example(similar_product_id_list[:k],product_json_dict,review_product_json,
                                                      k ,gold_product_index)
                error_review_product_json_list.append(error_review_dataset_json)
    sampled_error_list=sample(error_review_product_json_list,100)
 
    with open(output_error_path, 'w', encoding='utf-8') as fp:
        json.dump(sampled_error_list, fp, indent=4)
   


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
    # gen_error_candidate_list(args.data_path)
    gen_error_candidate_list_with_high_retrieval(args.data_path)