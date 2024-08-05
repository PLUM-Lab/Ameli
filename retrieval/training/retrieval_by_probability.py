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
from pathlib import Path
import json
import os
import pandas as pd
import shutil  
from random import sample
from tqdm import tqdm 
import random
from nltk.tokenize import word_tokenize
from util.common_util import json_to_product_id_dict 
from util.env_config import * 
from util.data_util.entity_linking_data_util import gen_review_text



import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize 
import nltk 


def find_category_by_mention(category,mode,review_mention,alias_category_product_id_json,alias_base_category_product_id_json,review_id):
    import spacy 
    nlp = spacy.load("en_core_web_lg")
    review_mention_doc=nlp(review_mention)
    review_mention_lower=review_mention.lower()
    review_mention_base=review_mention_doc[:].lemma_.lower()
    total_category_list,total_product_id_list=[],[]
    if mode =="standard":
        is_exist=False
        
        if review_mention_lower in alias_category_product_id_json:
            is_exist=True
            category_list=alias_category_product_id_json[review_mention_lower]["category"]
            product_id_list=alias_category_product_id_json[review_mention_lower]["product_id"] 
            total_category_list.extend(category_list)
            total_product_id_list.extend(product_id_list)
        if review_mention_base in alias_base_category_product_id_json:
            is_exist=True
            category_list=alias_base_category_product_id_json[review_mention_base]["category"]
            product_id_list=alias_base_category_product_id_json[review_mention_base]["product_id"] 
            total_category_list.extend(category_list)
            total_product_id_list.extend(product_id_list)
        # elif review_mention_base=="gaming laptop":
        #     category_list=alias_category_product_id_json[review_mention]["category"]
        #     product_id_list=alias_category_product_id_json[review_mention]["product_id"] 
        #     return category_list,product_id_list
        if not is_exist:
            print(f"can not find statistics by mention: {review_mention_base}, {review_id}")
        # elif  category_list is None or len(category_list)<=0:
        #         print(f"no category by mention : {review_id}")
        elif   category not in total_category_list:
            print(f" category out of found category: {category}, {review_mention}, {review_id}" )
            
            
    else:
        if review_mention_base in alias_base_category_product_id_json:
            category_list=alias_base_category_product_id_json[review_mention_base]["category"]
            product_id_list=alias_base_category_product_id_json[review_mention_base]["product_id"] 
            total_category_list,total_product_id_list= category_list,product_id_list
        elif review_mention.lower() in alias_base_category_product_id_json:
            category_list=alias_base_category_product_id_json[review_mention.lower()]["category"]
            product_id_list=alias_base_category_product_id_json[review_mention.lower()]["product_id"] 
            total_category_list,total_product_id_list= category_list,product_id_list
        else:
            print(f"error: miss product_alias_base {review_mention_base}")
             
    
    
    return total_category_list, total_product_id_list


def gen_top_k(dict_object,k):
    new_dict={}
    for i,(key,value) in enumerate(dict_object.items()):
        if i>=k:
            break 
        new_dict[key]=value
    return new_dict

def gen_fused_candidate_id(category_list,product_id_list,product_json_dict):
    selected_product_id_list=[]
    if len(product_id_list)<=10  :
        selected_product_id_list= product_id_list
    else:
        category=category_list[0]
        for product_id in product_id_list:
            cur_category=product_json_dict[product_id]["product_category"]
            if len(selected_product_id_list)>=100:
                break
            if category==cur_category   :
                selected_product_id_list.append(product_id)
    if len(selected_product_id_list)==0:
        if len(product_id_list)>0:
            selected_product_id_list=product_id_list[:100]
        else:
            selected_product_id_list=sample(list(product_json_dict.keys()),100)
    return selected_product_id_list                
        
def gen_candidate_list_by_mention_category_product_id_probability(mode,review_file_str,product_alias_base_category_product_id_path,
                                                                  product_alias_category_product_id_path,
                                                                  output_products_path,is_gen_fused_candidate_id=False):
    with open(product_alias_base_category_product_id_path, 'r', encoding='utf-8') as fp:
        alias_base_category_product_id_json = json.load(fp)
    with open(product_alias_category_product_id_path, 'r', encoding='utf-8') as fp:
        alias_category_product_id_json = json.load(fp)    
    new_crawled_img_url_json_path = Path(
        review_file_str
        ) 
    with open(products_path_str, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        product_json_dict=json_to_product_id_dict(new_crawled_products_url_json_array)
    out_list=[]   
    with open(new_crawled_img_url_json_path, 'r', encoding='utf-8') as fp:
        product_dataset_json_array = json.load(fp)
        for review_product_json   in tqdm(product_dataset_json_array) :#tqdm(
            # review_text=gen_review_text(review_product_json)
            category=review_product_json["gold_entity_info"]["product_category"]
            review_mention=review_product_json["mention"]
            category_list,product_id_list=find_category_by_mention(category,mode,review_mention,alias_category_product_id_json,
                                                                   alias_base_category_product_id_json,review_product_json["review_id"])
            
         
            if is_gen_fused_candidate_id:
                fused_candidate_list=gen_fused_candidate_id(category_list,product_id_list,product_json_dict)
            review_product_json["mention_to_category_list"]=category_list
            review_product_json["mention_to_product_id_list"]=product_id_list 
            # review_product_json["desc_score_dict"]=""
            # review_product_json["image_score_dict"]=""
            # review_product_json["text_score_dict"]=""
            review_product_json["fused_candidate_list"]=fused_candidate_list
            # review_product_json["fused_score_dict"]=""#gen_top_k(review_product_json["fused_score_dict"],10)
            # review_product_json["product_id_with_similar_text_by_review"]=""
            # review_product_json["product_id_with_similar_image_by_review"]=""
            # review_product_json["product_id_with_similar_desc_by_review"]=""#fused_score_dict, fused_candidate_list
            # review_product_json["image_similarity_score_list"]=""
            out_list.append(review_product_json)
    
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4) 
        
        
        
        
if __name__ == "__main__":    
    mode="standard"
    review_file_str=data_dir+"test/bestbuy_review_2.3.17_format_wo_candidates.json"
    product_alias_base_category_product_id_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/mention_to_entity_statistics/bestbuy_products_40000_3.4.19.1_alias_base_to_category_product_id.json"
    product_alias_category_product_id_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/mention_to_entity_statistics/bestbuy_products_40000_3.4.19.2_alias_to_category_product_id.json"
    output_products_path=data_dir+f"test/retrieval/bestbuy_review_2.3.17.1_{mode}_add_mention_to_category_product_id.json"
    
    gen_candidate_list_by_mention_category_product_id_probability(mode,review_file_str,product_alias_base_category_product_id_path
                                                                  ,product_alias_category_product_id_path,output_products_path,is_gen_fused_candidate_id=True)