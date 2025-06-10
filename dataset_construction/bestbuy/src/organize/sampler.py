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
from transformers import BertTokenizer
from bestbuy.src.organize.merge_attribute import json_to_dict, json_to_product_id_dict
import logging 
logging.basicConfig( filename="bestbuy/output/review_image.txt", filemode="w", level=logging.INFO)
def gen_100_review_set_to_generate_mention():
    product_json_path = Path(
        f'bestbuy/data/final/v0/bestbuy_products_40000_3.4.2_desc_img_url.json'
    )        
    review_dataset_path = Path('bestbuy/data/final/v0/bestbuy_review_2.3.4_w_image.json')
    output_products_path = Path(
        f'bestbuy/data/example/bestbuy_review_50_200.json'
    )  
    number=0
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    start_id=50
    # fields=["reviews","overview_section","product_images_fnames","product_images","Spec"]
    
    # fields=["thumbnails", "thumbnail_paths","reviews", "Spec"]
    output_list=[]
    with open(product_json_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        product_json_dict=json_to_dict(new_crawled_products_url_json_array)
        
        with open(review_dataset_path, 'r', encoding='utf-8') as fp:
            product_dataset_json_array = json.load(fp)
            for review_dataset_json   in tqdm(product_dataset_json_array):
                
                url=review_dataset_json["url"]
                if url in product_json_dict :
                    product_json=product_json_dict[url]
                    review_dataset_json["overview_section"]= product_json["overview_section"]
                    reviews=review_dataset_json["reviews"]
                    if len(reviews)>0:
                        output_review_list=[]
                        
                        for review in reviews:
                            review_text=review["body"]
                            review_id=review["id"]
                            # if review_id <start_id:
                            #     continue 
                            if not is_too_long(review_text,tokenizer):   
                                review["gt_mention"]=[""]
                                output_review_list.append(review)
                                break 
                    if len(output_review_list)>0:
                        review_dataset_json["reviews"]=output_review_list
                        if number>start_id:
                            
                            if number>200:
                                break 
                            else:
                                output_list.append(review_dataset_json)
                         
                        number+=1
                        
                        
                            
                            
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(output_list, fp, indent=4)   
        
def is_too_long(text,tokenizer):
    text_tokens=tokenizer.tokenize(text)        
    if len(text_tokens)>500:
        return True 
    else:
        return False 
    
def check_id(review_id_list):
    logging.info(review_id_list)
    logging.info(len(review_id_list))
    review_id_set=set()
    for review_id in review_id_list:  
        if review_id in review_id_set:
            print("ERROR!")
        else:
            review_id_set.add(review_id)    
    logging.info(len(review_id_set))
    
def gen_current_review():
    review_dataset_path = Path('bestbuy/data/example/annotated_example.json')
    review_id_list=[]
    with open(review_dataset_path, 'r', encoding='utf-8') as fp:
        review_dataset_json_array = json.load(fp)
        for review_dataset_json in review_dataset_json_array:
            if not is_nan_or_miss(review_dataset_json,"reviews"):
                review=review_dataset_json["reviews"][0]
                review_id=review["id"]
                review_id_list.append(review_id)
                
    return review_id_list 



def gen_brand_from_spec(review_dataset_json):
    sepc_list=review_dataset_json["Spec"]
    brand=None 
    for attribute_subsection in sepc_list:

        attribute_list_in_one_section=attribute_subsection["text"]
   
        for attribute_json  in attribute_list_in_one_section:
            attribute_key=attribute_json["specification"]
            attribute_value=attribute_json["value"]
            if attribute_key=="Brand":
                brand=attribute_value
    if brand is None :
        product_name=review_dataset_json["product_name"]
        brand=product_name.split("-")[0].strip()
    return brand 

def filter_similar_product_id_list(similar_product_list,brand,product_json_dict,filter_num):
    new_list=[]
    for similar_product_id in similar_product_list:
        product_json=product_json_dict[similar_product_id]
        
        brand_in_product= gen_brand_from_spec(product_json)
        if brand_in_product is   None or brand_in_product ==brand:
            new_list.append(similar_product_id)
        else:
            filter_num+=1
    return new_list,filter_num
 
 
def get_brand_from_review(product_json):
    review_json=product_json["reviews"][0]
    attribute_dict=review_json["attribute"]
    if "Brand" in attribute_dict:
        brand=attribute_dict["Brand"]
    else:
        brand =None 
    return brand
 
def filter_similar_products(product_json, attribute_dict,product_json_dict,filter_num,brand_wrong_num):
    similar_product_list=product_json["similar_product_id"]
    product_id_with_similar_image_list=product_json["product_id_with_similar_image"]
    if "Brand" in attribute_dict:
        brand=attribute_dict["Brand"]
        brand_in_product= gen_brand_from_spec(product_json)
        
        if   brand_in_product ==brand:
            product_json["similar_product_id"],filter_num=filter_similar_product_id_list(similar_product_list,brand,product_json_dict,filter_num)
            product_json["product_id_with_similar_image"],filter_num=filter_similar_product_id_list(product_id_with_similar_image_list,brand,product_json_dict,filter_num) 
        else:
            brand_wrong_num+=1
            print(f"ERROR {brand_in_product}, {brand},{product_json['id']}")
    return product_json ,filter_num,brand_wrong_num

import copy     
from random import sample    
def sample_to_evaluate_testset():
    product_json_path = Path(
        f'bestbuy/data/final/v2_course/bestbuy_products_40000_3.4.13_sensitive.json'
    )        
    review_dataset_path = Path('bestbuy/data/final/v2_course/test/bestbuy_review_2.3.16.20.3_added_split.json')
    output_products_path = Path(
        f'bestbuy/data/example/bestbuy_100_human_performance_added.json'
    )  
    number=0
    
    # fields=["reviews","overview_section","product_images_fnames","product_images","Spec"]
    current_review_id_list=gen_current_review()
    # current_review_id_list=[]
    # fields=["thumbnails", "thumbnail_paths","reviews", "Spec"]
    output_list=[]
    with open(product_json_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        product_json_dict=json_to_product_id_dict(new_crawled_products_url_json_array)
        
        with open(review_dataset_path, 'r', encoding='utf-8') as fp:
            review_dataset_json_array = json.load(fp)
            idx_list=[]
            for i in range(len(review_dataset_json_array)):
                if review_dataset_json_array[i]["reviews"][0]["id"] not in current_review_id_list:
                    idx_list.append(i)
            
            print(len(idx_list))
            sampled_idx_list=idx_list
            # sampled_idx_list=sample(idx_list,116)
            filter_num=0
            brand_wrong_num=0
            for idx in tqdm(sampled_idx_list):
             
                review_dataset_json=review_dataset_json_array[idx]
                url=review_dataset_json["id"]
                if url in product_json_dict :
                    product_json=copy.deepcopy(product_json_dict[url])
                    product_json["reviews"]=review_dataset_json["reviews"]
                    product_json["text_similarity_score"]=review_dataset_json["text_similarity_score"]
                    product_json["image_similarity_score_list"]=review_dataset_json["image_similarity_score_list"]
                    product_json["product_title_similarity_score"]=review_dataset_json["product_title_similarity_score"]
                    product_json["predicted_is_low_quality_review"]=review_dataset_json["predicted_is_low_quality_review"]
                    product_json,filter_num,brand_wrong_num=filter_similar_products(product_json,review_dataset_json["reviews"][0]["attribute"],product_json_dict,filter_num,brand_wrong_num)
                    product_json=add_reshuffle_position(product_json)
                    output_list.append(product_json)
                                  
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(output_list, fp, indent=4)    
    print(filter_num, brand_wrong_num)
    
import random
def add_reshuffle_position(product_json):
    similar_product_list=product_json["similar_product_id"]
    product_id_with_similar_image_list=product_json["product_id_with_similar_image"]
    new_list=[]
    new_list.extend(product_id_with_similar_image_list)
    new_list.extend(similar_product_list)
    merge_similar_product_list_num=len(set(new_list))
    product_json["reviews"][0]["reshuffled_target_product_id_position"]=random.randint(0,merge_similar_product_list_num)
    return product_json 
    
def sample_100_examples_to_check_quality():
    product_json_path = Path(
        f'bestbuy/data/final/v2_course/bestbuy_products_40000_3.4.10_remove_wrong_image.json'
    )        
    review_dataset_path = Path('bestbuy/data/final/v2_course/bestbuy_review_2.3.16.16_filter_low_information.json')
    output_products_path = Path(
        f'bestbuy/data/example/bestbuy_100_human_performance.json'
    )  
    number=0
    
    # fields=["reviews","overview_section","product_images_fnames","product_images","Spec"]
    # current_review_id_list=gen_current_review()
    current_review_id_list=[]
    # fields=["thumbnails", "thumbnail_paths","reviews", "Spec"]
    output_list=[]
    with open(product_json_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        product_json_dict=json_to_dict(new_crawled_products_url_json_array)
        
        with open(review_dataset_path, 'r', encoding='utf-8') as fp:
            review_dataset_json_array = json.load(fp)
            idx_list=[]
            for i in range(len(review_dataset_json_array)):
                if review_dataset_json_array[i]["reviews"][0]["id"] not in current_review_id_list:
                    idx_list.append(i)
            
            sampled_idx_list=sample(idx_list,250)
            
            # review_id_list=[review_dataset_json_array[sample_id]["reviews"][0]["id"] for sample_id in sampled_idx_list]
            # check_id(review_id_list)
            # if len(set(review_id_list))<100:
            #     print("issue < 100")
                
            #     sampled_idx_list=sample(idx_list,110)
            #     return 
            for idx in sampled_idx_list:
             
                review_dataset_json=review_dataset_json_array[idx]
                url=review_dataset_json["url"]
                if url in product_json_dict :
                    product_json=copy.deepcopy(product_json_dict[url])
                    product_json["reviews"]=review_dataset_json["reviews"]
                    product_json["text_similarity_score"]=review_dataset_json["text_similarity_score"]
                    product_json["image_similarity_score_list"]=review_dataset_json["image_similarity_score_list"]
                    product_json["product_title_similarity_score"]=review_dataset_json["product_title_similarity_score"]
                    product_json["predicted_is_low_quality_review"]=review_dataset_json["predicted_is_low_quality_review"]
                    
                    output_list.append(product_json)
                                  
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(output_list, fp, indent=4)        

def check_review_id_number(review_products_json_list):
    id_list=[]
    for review_product_json in review_products_json_list:
        review_json=review_product_json["reviews"][0]
        id=review_json["id"] 
        id_list.append(id )
        
    print(f"{len(set(id_list))}")
    
    
if __name__ == "__main__":  
    # sample_100_examples_to_check_quality()
    sample_to_evaluate_testset()        