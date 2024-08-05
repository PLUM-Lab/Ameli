
from pathlib import Path
import json

import os
import pandas as pd
import shutil  
from random import sample
from tqdm import tqdm 
import random
from nltk.tokenize import word_tokenize
from attribute.extractor.amazon_review_gen_review_attribute_by_rule import match_attribute_main
from attribute.scorer import evaluate
from disambiguation.data_util.inner_util import gen_review_text, json_to_dict, review_json_to_product_dict
import re

from retrieval.organize.filter import filter_by_category, filter_non_attribute_main
 
 
 
 
import argparse


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
from disambiguation.data_util.inner_util import review_json_to_product_dict
from retrieval.organize.gen_fused_entity_list_by_image_text_ratio import gen_new_retrieved_dataset_for_one_json_list

from retrieval.organize.score.scorer_helper import compute_for_one_json_list, gen_score_dict, gen_sorted_candidate_list, search_one_json_list 
 
 
def gen_all_result():
    data_path=f'/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/retrieval/bestbuy_review_2.3.17.2_clean_missed_product.json'
    evaluate(data_path,False,"fused_candidate_list")  
    data_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/retrieval/bestbuy_review_2.3.17.5.0.1_pre_trained_text_title_no_reranker.json"
    evaluate(data_path,False,"product_id_with_similar_text_by_review")   #product_id_with_similar_desc_by_review 
    data_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/retrieval/bestbuy_review_2.3.17.3_pre_trained_image.json"
    evaluate(data_path,False,"product_id_with_similar_image_by_review")    
    data_path=f'/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/retrieval/bestbuy_review_2.3.17.7_fused_score_pre_trained.json'
    evaluate(data_path,False,"fused_candidate_list")    
 
    # data_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/retrieval/bestbuy_review_2.3.17.5.1.2_finetuned_26_text_title_no_reranker.json"
    data_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/retrieval/bestbuy_review_2.3.17.5.1.2_finetuned_145.json"
    evaluate(data_path,False,"product_id_with_similar_text_by_review")  
    data_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/retrieval/bestbuy_review_2.3.17.3.2_27_finetuned.json"
    evaluate(data_path,False,"product_id_with_similar_image_by_review" ) 
    data_path=f'/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/retrieval/bestbuy_review_2.3.17.7.1_top100_fused_score_fine_tune_27_145.json'
    # data_path=f'/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/retrieval/bestbuy_review_2.3.17.7.1_top100_fused_score_fine_tune_27_26.json'
    evaluate(data_path,False,"fused_candidate_list" )    
    
def product_recall_separately(data_path,key,precision_recall_at_k_type):
    # data_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v5/test/retrieval/bestbuy_review_2.3.16.29.4.1_desc_text_similar_image_max_mode_split.json"
    # evaluate(data_path,False,"product_id_with_similar_image_by_review")    
    # evaluate(data_path,False,"product_id_with_similar_text_by_review")    
    # evaluate(data_path,False,"product_id_with_similar_desc_by_review")     
    # data_path=f'/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v4/val/bestbuy_review_2.3.16.29.5.1.1_remove_score_dict_fuse_score_title_0.6_max_image.json'
    # evaluate(data_path,False,"fused_candidate_list")    
    # data_path=f'/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v4/val/bestbuy_review_2.3.16.29.9.3_standard_add_4.1_mention_to_category_product_id_remove_score_dict.json'
    # evaluate(data_path,False,"mention_to_product_id_list")    
    # data_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v4/val/bestbuy_review_2.3.16.29.11.3_10000_standard_rerank.json"
    # evaluate(data_path,False,"fused_candidate_list")  
    # data_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v5/test/retrieval/bestbuy_review_2.3.16.29.11.3_standard_rerank_fine_tune.json"
    # evaluate(data_path,False,"fused_candidate_list") 
    # data_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v5/test/retrieval/bestbuy_review_2.3.16.29.11.3_standard_rerank_20_55.json"
    if precision_recall_at_k_type!="1to10":
        evaluate(data_path,False,key) 
    else:
        evaluate(data_path,False,key,precision_recall_at_k=[1,2,3,4,5,6,7,8,9,10]) 
    # data_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v5/test/retrieval/bestbuy_review_2.3.16.29.11.3_standard_rerank_20_54.json"
    # evaluate(data_path,False,"fused_candidate_list") 
    
  

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',type=str,help=" ",default="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v5/test/retrieval/bestbuy_review_2.3.16.29.11.3_standard_rerank_20_54.json")  
    parser.add_argument('--out_path',type=str,help=" ",default="bestbuy/output/html")
    parser.add_argument('--key',type=str,help=" ",default="fused_candidate_list")
    # parser.add_argument('--example_type',type=str,help=" ",default="verification")
    parser.add_argument('--precision_recall_at_k_type',type=str,help=" ",default="standard")
    parser.add_argument('--attribute_source',type=str,help=" ",default="similar")#gold
    parser.add_argument('--attribute_field',type=str,help=" ",default="all")
    parser.add_argument('--version',type=str,help=" ",default="0 ")
    parser.add_argument('--is_cross',type=str,help=" ",default="n")
    parser.add_argument('--attribute_logic',type=str,help=" ",default="numeral")#exact
    parser.add_argument('--filter_by_attribute_field',type=str,help=" ",default="Brand,Color")#all
    parser.add_argument('--original_retrieval_file_name',type=str,help=" ",default="bestbuy_review_2.3.16.29.4.1_desc_text_similar_image_max_mode_split.json")  
    # parser.add_argument('--generate_claim_level_report',type=str,help=" ",default="y")
    args = parser.parse_args()
    return args

  
if __name__ == '__main__':
    args = parser_args()
    data_path=args.data_path       
    product_recall_separately(args.data_path,args.key,args.precision_recall_at_k_type)
    # gen_all_result()