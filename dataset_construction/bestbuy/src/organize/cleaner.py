import json
from typing import Dict
from pathlib import Path
import json
from pathlib import Path
from time import time
from pathlib import Path
import json
import pandas as pd
import requests
import os
from transformers import AutoProcessor
from tqdm import tqdm
from ast import literal_eval
from bestbuy.src.extract_mention.name_entity_recognition.util.data_util import longest_match_base_form
from bestbuy.src.extract_mention.name_entity_recognition.util.data_util import prepare_produt
from transformers import CLIPTokenizer
from bestbuy.src.organize.checker import is_miss_desc, is_nan_or_miss
from transformers import BertTokenizer
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import shutil
from transformers import CLIPFeatureExtractor
from bestbuy.src.organize.merge_attribute import json_to_product_id_dict, review_json_to_product_dict
from bestbuy.src.organize.score import gen_reivew_gold_product_dict
 
import logging  
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"

def remove_review_wo_image_and_add_review_id():
     
    review_path="/home/menglong/workspace/code/referred/product_scraper/dataset_construction/bestbuy/data/milestone/bestbuy_6507768_review_2.3.1_incomplete_before_remove_reviews_wo_image.json"
    incomplete_products_path = Path(
        review_path
    )
    output_products_path = Path(
        "/home/menglong/workspace/code/referred/product_scraper/dataset_construction/bestbuy/data/bestbuy_review_2.3.2_w_image_incomplete.json"
    )
    review_err=0
    out_list=[]
    review_id=0
    with open(incomplete_products_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        for i,product_json in tqdm(enumerate(incomplete_dict_list) ):
            
            
            if (not is_nan_or_miss(product_json,"reviews")): 
                has_image_review=False 
                review_list=[]
                for review in product_json["reviews"]:
                    if not is_nan_or_miss(review, "product_images"):
                        review["id"]=review_id
                        review_id+=1
                        review_list.append(review)
                        has_image_review=True 
                        
                if has_image_review:
                    product_json["reviews"]=review_list
                    out_list.append(product_json)

    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4) 
        
def clean_example():
    review_json_path = Path(
        f'bestbuy/data/example/bestbuy_review_50.json'
    )        
     
    output_products_path = Path(
        f'bestbuy/data/example/bestbuy_review_50_cleaned_annotation.json'
    )  
    output_list=[]
     
     
        
    with open(review_json_path, 'r', encoding='utf-8') as fp:
        review_json_array = json.load(fp)
            
        for idx,review_json   in tqdm(enumerate(review_json_array)):
            review_json["url"]=None 
            review_json["Spec"]=None 
            if "overview_section" in review_json:
                review_json["overview_section"]["features"]=None 
                # review_json["description"]=review_json["overview_section"]["description"]  
                # review_json["overview_section"]["description"]  =None 
            reviews=review_json["reviews"]
            review_json["thumbnail_paths"]=None 
            review_json["thumbnails"]=None 
            
            for review in reviews:
                review["user"]=None 
                review["feedback"]=None 
                review["product_images"]=None 
                review["thumbnail_paths"]=None 
                review["rating"]=None 
                review["recommendation"]=None 
                review["is_predict_right"]=None 
            review_json["reviews"]=[review]
            output_list.append(review_json)
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(output_list, fp, indent=4)
                
                

def clean_example2():
    review_json_path = Path(
        f'bestbuy/data/example/bestbuy_review_50_200.json'
    )        
     
    output_products_path = Path(
        f'bestbuy/data/example/bestbuy_review_50_200_cleaned_annotation.json'
    )  
    output_list=[]
     
     
        
    with open(review_json_path, 'r', encoding='utf-8') as fp:
        review_json_array = json.load(fp)
            
        for idx,review_json   in tqdm(enumerate(review_json_array)):
            # review_json["url"]=None 
            review_json["Spec"]=None 
            if "overview_section" in review_json:
                review_json["overview_section"]["features"]=None 
                # review_json["description"]=review_json["overview_section"]["description"]  
                # review_json["overview_section"]["description"]  =None 
            reviews=review_json["reviews"]
            review_json["thumbnail_paths"]=None 
            review_json["thumbnails"]=None 
            
            for review in reviews:
                review["user"]=None 
                review["feedback"]=None 
                review["product_images"]=None 
                review["thumbnail_paths"]=None 
                review["rating"]=None 
                review["recommendation"]=None 
                # review["is_predict_right"]=None 
            review_json["reviews"]=[review]
            output_list.append(review_json)
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(output_list, fp, indent=4)
def has_overlap(product_category,noisy_product_name,product_desc,gt_mention_list):
    for gt_mention in gt_mention_list:
        for context in [product_category,noisy_product_name,product_desc]:
            is_found=context.lower().find(gt_mention.lower()) 
            if  is_found !=-1:
                return True 
            else:
                _,is_found_flag=longest_match_base_form(context,gt_mention) 
                if is_found_flag:
                    return True 
    return False 
    
    
def filter_non_overlap_reviews():
    review_json_path = Path(
        f'bestbuy/data/example/bestbuy_review_50_200_cleaned_annotation.json'
    )        
     
    output_products_path = Path(
        f'bestbuy/data/example/bestbuy_review_50_200_cleaned_annotation_v2.json'
    )  
    output_list=[]
     
    filter_num=0
        
    with open(review_json_path, 'r', encoding='utf-8') as fp:
        review_json_array = json.load(fp)
            
        for idx,review_json   in tqdm(enumerate(review_json_array)):
            product_category,noisy_product_name,product_desc=prepare_produt(review_json,False)
            review_sub_json=review_json["reviews"]
            gt_mention_list=review_sub_json[0]["gt_mention"]
             
            if has_overlap(product_category,noisy_product_name,product_desc,gt_mention_list):
                output_list.append(review_json)
            else:
                filter_num+=1
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(output_list, fp, indent=4)
    print(f"filter {filter_num}")
                          
def filter_too_long():
    review_path="/home/menglong/workspace/code/referred/product_scraper/dataset_construction/bestbuy/data/final/v0/bestbuy_review_2.3.4_w_image.json"
    incomplete_products_path = Path(
        review_path
    )
    output_products_path = Path(
        "/home/menglong/workspace/code/referred/product_scraper/dataset_construction/bestbuy/data/final/v0/bestbuy_review_2.3.5_w_image_not_long.json"
    )
 
    out_list=[]
    filter_num=0
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    with open(incomplete_products_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        for i,product_json in tqdm(enumerate(incomplete_dict_list) ):
            
            
            if (not is_nan_or_miss(product_json,"reviews")): 
                
                review_list=[]
                for review in product_json["reviews"]:
                    text=review["header"]+"."+review["body"]
                    if not is_too_long(text,tokenizer):
                        review_list.append(review)
                    else:
                        
                        filter_num+=1
                    
                        
                if len(review_list)>0:
                    product_json["reviews"]=review_list
                    out_list.append(product_json)

    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4)                               
    print(f"filter {filter_num}")
                     
                     
def filter_unhelpful():
    review_path="/home/menglong/workspace/code/referred/product_scraper/dataset_construction/bestbuy/data/final/v0/bestbuy_review_2.3.5_w_image_not_long.json"
    incomplete_products_path = Path(
        review_path
    )
    output_products_path = Path(
        "/home/menglong/workspace/code/referred/product_scraper/dataset_construction/bestbuy/data/final/v0/bestbuy_review_2.3.6_w_image_not_long.json"
    )
 
    out_list=[]
    filter_num=0
    # tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    with open(incomplete_products_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        for i,product_json in tqdm(enumerate(incomplete_dict_list) ):
            
            
            if (not is_nan_or_miss(product_json,"reviews")): 
                
                review_list=[]
                for review in product_json["reviews"]:
                    if "feedback" in review and review["feedback"] is not None :
                        feedback=review["feedback"]
                        if "number_helpful" in feedback:
                            number_helpful=feedback["number_helpful"]
                            number_unhelpful=feedback["number_unhelpful"]
                            if not (number_unhelpful>0 and number_helpful<=0):
                                review_list.append(review)
                            else:
                                # print(f"filter {number_unhelpful}, {review}")
                                filter_num+=1
                if len(review_list)>0:
                    product_json["reviews"]=review_list
                    out_list.append(product_json)

    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4)                               
    print(f"filter {filter_num}")                     
                     
                      
def is_too_long(text,tokenizer):
    text_tokens=tokenizer.tokenize(text)        
    if len(text_tokens)>512:
        return True 
    else:
        return False                           
        
def filter_review_wo_mention_and_change_mention_to_review_sub_json(review_path,output_products_path,is_mention_in_review_sub_json):
    # review_path="/home/menglong/workspace/code/referred/product_scraper/dataset_construction/bestbuy/data/final/v1/bestbuy_review_2.3.8_w_mention.json"
    incomplete_products_path = Path(
        review_path
    )
    # output_products_path = Path(
    #     "/home/menglong/workspace/code/referred/product_scraper/dataset_construction/bestbuy/data/final/v1/bestbuy_review_2.3.9_w_mention.json"
    # )
 
    out_list=[]
    filter_num=0
    # tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    with open(incomplete_products_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        for i,product_json in tqdm(enumerate(incomplete_dict_list) ):
            if not is_mention_in_review_sub_json:
                mention=product_json["mention"]
            else:
                mention=product_json["reviews"][0]["mention"]
            product_json["mention"]=None 
            if mention is not None and mention !="":
                if (not is_nan_or_miss(product_json,"reviews")): 
                    
                    review_list=[]
                    for review in product_json["reviews"]:
                        review["mention"]=mention
                        
                        review_list.append(review)
                                
                    if len(review_list)>0:
                        product_json["reviews"]=review_list
                        out_list.append(product_json)
            else:
                filter_num+=1

    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4)                               
    print(f"filter {filter_num}")     


def clean_product_missing_fields():
    review_path="bestbuy/data/final/v1/bestbuy_products_40000_3.4.5_fix_spec_format.json"
    incomplete_products_path = Path(
        review_path
    )
    output_products_path = Path(
        "bestbuy/data/final/v1/bestbuy_products_40000_3.4.6_remove_missed_product.json"
    )
 
    out_list=[]
    filter_num=0
    # tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    with open(incomplete_products_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        for i,product_json in tqdm(enumerate(incomplete_dict_list) ):
            if  (is_nan_or_miss(product_json,"Spec")) or is_nan_or_miss(product_json,"thumbnail_paths") or is_nan_or_miss(product_json,"thumbnails") or is_miss_desc(product_json):
                continue 
            else:
                out_list.append(product_json)
    
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4) 
        
        
        


def clean_review_missing_fields():
    products_path = Path(
        "bestbuy/data/final/v1/bestbuy_products_40000_3.4.6_remove_missed_product.json"
    )
    review_path="bestbuy/data/final/v1/bestbuy_review_2.3.13.8_merge_fix.json"
    incomplete_review_path = Path(
        review_path
    )
    output_products_path = Path(
        "bestbuy/data/final/v1/bestbuy_review_2.3.14_clean_duplicate_review.json"
    )
    product_id_list=[]
    with open(products_path, 'r', encoding='utf-8') as fp:
        product_dict_list = json.load(fp)
        for i,product_json in  enumerate(product_dict_list)  :
            product_id_list.append(product_json["id"])
            
    out_list=[]
    filter_num=0
    # tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    with open(incomplete_review_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        for i,review_json in tqdm(enumerate(incomplete_dict_list) ):
            if  review_json["id"] not in product_id_list:
                continue 
            else:
                out_list.append(review_json)
    
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4)        
    
    
    

def remove_review_wo_image():
     
   
    review_path="bestbuy/data/final/v1/bestbuy_review_2.3.16.7_fix_image_path_nan_from_0_to_150000.json"
    incomplete_review_path = Path(
        review_path
    )
    output_products_path = Path(
        "bestbuy/data/final/v1/bestbuy_review_2.3.16.4_remove_image_url_null.json"
    )
    review_err=0
    out_list=[]
    review_id=0
    with open(incomplete_review_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        for i,product_json in tqdm(enumerate(incomplete_dict_list) ):
            
            
            if (not is_nan_or_miss(product_json,"reviews")): 
                has_image_review=False 
                review_list=[]
                for review in product_json["reviews"]:
                    if not is_nan_or_miss(review, "image_url"):
                       
                      
                        review_list.append(review)
                        has_image_review=True 
                        
                if has_image_review:
                    product_json["reviews"]=review_list
                    out_list.append(product_json)

    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4)     
    
def clean_duplicate_review_same_product():
    products_path = Path(
        "bestbuy/data/final/v1/bestbuy_products_40000_3.4.7_remove_image_path_prefix.json"
    )
    review_path="bestbuy/data/final/v1/bestbuy_review_2.3.13.2_remove_nan_image_url.json"
    incomplete_review_path = Path(
        review_path
    )
    output_products_path = Path(
        "bestbuy/data/final/v1/bestbuy_review_2.3.13.3_remove_duplicate_review_same_product.json"
    )
    product_id_list=[]
    # with open(products_path, 'r', encoding='utf-8') as fp:
    #     product_dict_list = json.load(fp)
    #     for i,product_json in  enumerate(product_dict_list)  :
    #         product_id_list.append(product_json["id"])
            
    out_list=[]
    filter_num=0
    review_body_product_json_dict={}
    # tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    with open(incomplete_review_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        for i,cur_product_json in tqdm(enumerate(incomplete_dict_list) ):
            review=cur_product_json["reviews"][0]
            review_body=review["body"]
            if review["id"]==2186:
                print("")
            is_found=False 
            if  review_body   in review_body_product_json_dict:
                product_json_list=review_body_product_json_dict[review_body]
                
                for product_json in product_json_list:
                    if equal_json(product_json,cur_product_json):
                        is_found=True  
                if not is_found:
                    review_body_product_json_dict[review_body].append(cur_product_json)
            else:
                review_body_product_json_dict[review_body]=[cur_product_json]
            if not is_found:        
                out_list.append(cur_product_json)
    
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4)       
        
    
def equal_json(product_json,cur_product_json):
    if product_json["product_name"] !=cur_product_json["product_name"]:
        return False 
    elif product_json["url"] !=cur_product_json["url"]:
        return False 
    else:
        review=product_json["reviews"][0]
        cur_review=cur_product_json["reviews"][0]
        for field in ["user","header","body"]:#,"image_path","image_url"
            if review[field]!=cur_review[field]:
                return False 
        return True 
    
def clean_error_Y_review():
    review_path="bestbuy/data/final/v1/bestbuy_review_2.3.13.5_fix_duplicate_review_incomplete_from_0_to_150000.json"
    incomplete_review_path = Path(
        review_path
    )
    output_products_path = Path(
        "bestbuy/data/final/v1/bestbuy_review_2.3.13.6_clean_error_Y.json"
    )
    
    out_list=[]
    filter_num=0
 
    # tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    with open(incomplete_review_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        for i,cur_product_json in tqdm(enumerate(incomplete_dict_list) ):
            if "error"   in cur_product_json and cur_product_json["error"]=="Y":
                cur_product_json["reviews"][0]["body"]=""
                cur_product_json["reviews"][0]["image_url"]=[]
                filter_num+=1
            out_list.append(cur_product_json)
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4)      
    print(filter_num)
         
def choose_one_from_duplicate_review():
  
    review_path="bestbuy/data/final/v1/bestbuy_review_2.3.13.6_clean_error_Y.json"
    incomplete_review_path = Path(
        review_path
    )
    output_products_path = Path(
        "bestbuy/data/final/v1/bestbuy_review_2.3.13.7_choose_one_from_duplicate.json"
    )
 
    out_list=[]
    filter_num=0
    review_body_product_json_dict={}
    review_body_set=set()
    # tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    with open(incomplete_review_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        for i,cur_product_json in tqdm(enumerate(incomplete_dict_list) ):
            review=cur_product_json["reviews"][0]
            review_body=review["body"]
            if review_body!="" and review_body in review_body_set:
                cur_product_json["reviews"][0]["body"]=""
                cur_product_json["reviews"][0]["image_url"]=[]
                filter_num+=1
            review_body_set.add(review_body)
            out_list.append(cur_product_json)
    
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4)         
    print(filter_num)
    
    
def is_image_valid(real_review_image_path,processor):
    is_valid=True 
    try:               
        image=Image.open(real_review_image_path)
        processor(images=image, return_tensors="pt")
        # model.encode([image], convert_to_tensor=True, show_progress_bar=False )
    except Exception as e:
        is_valid=False 
        print(f"{real_review_image_path}, {e}")
    return is_valid

 

is_clip=False
def remove_review_with_error_image(review_path,out_path ):
    if is_clip:
        processor  =CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32") 
    else:
        processor=AutoProcessor.from_pretrained("facebook/flava-full")
    
    incomplete_review_path = Path(
        review_path
    )
    output_products_path = Path(
        out_path
    )
    wrong_image_dir="bestbuy/data/final/v6/wrong_image"
    review_err=0
    out_list=[]
    review_id=0
    filter_num=0
    review_id=0
    filter_image_num=0
    review_image_dir="bestbuy/data/final/v6/cleaned_review_images"
    with open(incomplete_review_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        for i,review in tqdm(enumerate(incomplete_dict_list) ):
            
            review_image_path_list=review["review_image_path"]
            valid_review_image_path_list=[]
            has_valid_image=False  
            for review_image_path  in review_image_path_list:
                real_review_image_path=os.path.join(review_image_dir,review_image_path)
                is_valid=is_image_valid(real_review_image_path,processor)
                if is_valid:
                    valid_review_image_path_list.append(review_image_path)
                    
                    has_valid_image=True  
                else:
                    filter_image_num+=1
                    print(f"{review['review_id']}. Error! product without image  ")
                # else:
                #     src_path = real_review_image_path
                #     dst_path =os.path.join(wrong_image_dir,review_image_path)
                #     shutil.move(src_path, dst_path)
            review["review_image_path"]=valid_review_image_path_list
             
            if has_valid_image:
                out_list.append(review)
            else:
                filter_num+=1
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4)     
    
      
    print(filter_num,filter_image_num)

def remove_product_with_error_image(  ):
    
    if is_clip:
        processor  =CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32") 
    else: 
        processor=AutoProcessor.from_pretrained("facebook/flava-full")
    review_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/bestbuy_products_40000_3.4.19_final_format.json"
    incomplete_review_path = Path(
        review_path
    )
    output_products_path = Path(
        "/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/bestbuy_products_40000_3.4.20_remove_error_image.json"
    )
    wrong_image_dir="bestbuy/data/final/v6/wrong_image"
    review_err=0
    out_list=[]
    filter_num=0
    review_id=0
    filter_image_num=0
    review_image_dir="bestbuy/data/final/v6/product_images"
    with open(incomplete_review_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        for i,product_json in tqdm(enumerate(incomplete_dict_list) ):
            
            product_image_path_list=product_json["image_path"]
            valid_review_image_path_list=[]
            has_valid_image=False  
            for review_image_path  in product_image_path_list:
                real_review_image_path=os.path.join(review_image_dir,review_image_path)
                is_valid=is_image_valid(real_review_image_path,processor)
                if is_valid:
                    valid_review_image_path_list.append(review_image_path)
                    
                    has_valid_image=True  
                else:
                    filter_image_num+=1
                # else:
                #     src_path = real_review_image_path
                #     dst_path =os.path.join(wrong_image_dir,review_image_path)
                #     shutil.move(src_path, dst_path)
            product_json["image_path"]=valid_review_image_path_list
            if has_valid_image:
                
                
                out_list.append(product_json)
            else:
                print(f"{product_json['id']}. Error! product without image {product_json['id']}")
                filter_num+=1
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4)     
    print(filter_num,filter_image_num)

def filter_low_information():
    product_json_path_str=f'bestbuy/data/example/bestbuy_100_human_performance_w_similar_score.json'
    output_products_path_str=f'bestbuy/data/example/bestbuy_100_human_performance_w_system_predict.json'
        # mode="text"
    image_model = SentenceTransformer('clip-ViT-L-14')#all-mpnet-base-v2
    product_json_path = Path(
        product_json_path_str
    )    
    output_products_path = Path(
        output_products_path_str
    ) 
    output_list=[]    
    with open(product_json_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        # check_review_id_number(new_crawled_products_url_json_array)
        for review_dataset_json   in tqdm(new_crawled_products_url_json_array):
            product_text=review_dataset_json["overview_section"]["description"]
            product_images=review_dataset_json["image_path"]
            product_id=review_dataset_json["id"]
            review_json=review_dataset_json["reviews"][0]
            review_text=review_json["header"]+". "+review_json["body"]
            review_images=review_json["image_path"]
            # product_text_similarity_score=gen_text_similarity_score(review_text,product_text)
            product_image_similarity_score=gen_image_similarity_score(review_images,product_images,image_model,product_image_dir,review_image_dir)  
            review_dataset_json["reviews"][0]["image_similarity_score"]=product_image_similarity_score
            output_list.append(review_dataset_json)
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(output_list, fp, indent=4) 

import unidecode
def clean_unidecode():
    
    
    product_json_path_str=f'bestbuy/data/final/v2_course/bestbuy_review_2.3.16.16_filter_low_information.json'
    output_products_path_str=f'bestbuy/data/final/v2_course/bestbuy_review_2.3.16.17_unidecode.json'
     
   
    product_json_path = Path(
        product_json_path_str
    )    
    output_products_path = Path(
        output_products_path_str
    ) 
    output_list=[]    
    with open(product_json_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        # check_review_id_number(new_crawled_products_url_json_array)
        for review_dataset_json   in tqdm(new_crawled_products_url_json_array):
            
    
            
            review_json=review_dataset_json["reviews"][0]
            review_header=unidecode.unidecode(review_json["header"])
            review_body=unidecode.unidecode(review_json["body"])
            review_dataset_json["product_name"]=unidecode.unidecode(review_dataset_json["product_name"])
            
            
            review_json["body"]=review_body
            review_json["header"]=review_header
            review_json["mention"]=unidecode.unidecode(review_json["mention"])
            attribute_dict=review_json["attribute"]
            new_attribute_dict={}
            for key,value in attribute_dict.items():
                
                value=unidecode.unidecode(value)
                new_attribute_dict[key]=value 
            review_json["attribute"]=new_attribute_dict
            if "corrected_body" in review_json:
                review_json["corrected_body"]=unidecode.unidecode(review_json["corrected_body"])
            review_dataset_json["reviews"][0]=review_json 
            output_list.append(review_dataset_json)
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(output_list, fp, indent=4) 

def clean_unidecode_for_product():
       
    
    product_json_path_str=f'bestbuy/data/final/v2_course/bestbuy_products_40000_3.4.10_remove_wrong_image.json'
    output_products_path_str=f'bestbuy/data/final/v2_course/bestbuy_products_40000_3.4.11_unidecode.json'
     
   
    product_json_path = Path(
        product_json_path_str
    )    
    output_products_path = Path(
        output_products_path_str
    ) 
    output_list=[]    
    with open(product_json_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        # check_review_id_number(new_crawled_products_url_json_array)
        for review_dataset_json   in tqdm(new_crawled_products_url_json_array):
            
            review_dataset_json["product_name"]=unidecode.unidecode(review_dataset_json["product_name"])
            review_dataset_json["overview_section"]["description"]=unidecode.unidecode(review_dataset_json["overview_section"]["description"])
            if "features" in review_dataset_json["overview_section"]:
                attribute_list=review_dataset_json["overview_section"]["features"]
                new_attribute_list=[]
                for attribute_dict in attribute_list:
                    attribute_dict["header"]=unidecode.unidecode(attribute_dict["header"])
                    if attribute_dict["description"] is not None:
                        attribute_dict["description"]=unidecode.unidecode(attribute_dict["description"])
                        new_attribute_list.append(attribute_dict)
                
                review_dataset_json["overview_section"]["features"]=new_attribute_list 
         
            sepc_list=review_dataset_json["Spec"]
            new_spec_list=[]
            for attribute_subsection in sepc_list:
        
                attribute_list_in_one_section=attribute_subsection["text"]
                new_attribute_list_in_one_section=[]
                for attribute_json  in attribute_list_in_one_section:
                    attribute_key=attribute_json["specification"]
                    attribute_value=attribute_json["value"]
                    attribute_value=unidecode.unidecode(attribute_value)
                    attribute_json["value"]=attribute_value
                    new_attribute_list_in_one_section.append(attribute_json )
                attribute_subsection["text"]=new_attribute_list_in_one_section    
                new_spec_list.append(attribute_subsection)
            review_dataset_json["Spec"]=new_spec_list
            output_list.append(review_dataset_json)
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(output_list, fp, indent=4) 



def find_review_num(new_crawled_products_url_json_dict,product_id):
    num=0
    for review_dataset_json   in new_crawled_products_url_json_dict.values() :
        if product_id==review_dataset_json["id"]:
            num+=1
    return num 

        
def search_model_number(sepc_list,key):
    for attribute_subsection in sepc_list:
        
        attribute_list_in_one_section=attribute_subsection["text"]
 
        for attribute_json  in attribute_list_in_one_section:
            attribute_key=attribute_json["specification"]
            attribute_value=attribute_json["value"]
            if attribute_key==key:
                return attribute_value 
    return None 
def gen_review_id_list(new_crawled_products_url_json_dict,product_id):
    review_id_list=[]
    num=0
    for review_dataset_json   in  new_crawled_products_url_json_dict.values() :
        if product_id==review_dataset_json["id"]:
            review_id_list.append(review_dataset_json["reviews"][0]["id"])
            num+=1
    return review_id_list 
        
        
def update_similar_products(product_json,remove_product_id_set ):
    similar_product_id_set=set(product_json["similar_product_id"])
    new_similar_product_id_set=similar_product_id_set-remove_product_id_set
    product_json["similar_product_id"]=list(new_similar_product_id_set)
    
    similar_product_id_image_set=set(product_json["product_id_with_similar_image"])
    new_similar_product_image_id_set=similar_product_id_image_set-remove_product_id_set
    product_json["product_id_with_similar_image"]=list(new_similar_product_image_id_set)
    return product_json
        
def filter_product_variant():
    product_json_path_str=f'bestbuy/data/final/v2_course/bestbuy_products_40000_3.4.11_unidecode.json'
    output_products_path_str=f'bestbuy/data/final/v2_course/bestbuy_products_40000_3.4.12_filter_variant.json'
     
    
    review_json_str = Path(
    f'bestbuy/data/final/v2_course/bestbuy_review_2.3.16.18_attribute.json'
    )  
    output_review_json_str = Path(
    f'bestbuy/data/final/v2_course/bestbuy_review_2.3.16.19_filter_variant.json'
    )     
    
 
    
    
    with open(review_json_str, 'r', encoding='utf-8') as fp:
        new_crawled_review_url_json_array = json.load(fp)
        new_crawled_products_url_json_dict=review_json_to_product_dict(new_crawled_review_url_json_array)
        
    product_json_path = Path(
        product_json_path_str
    )    
    output_products_path = Path(
        output_products_path_str
    ) 
 
    product_title_dict={}
    remove_product_id_list=[]
    remove_review_id_list=[]
    with open(product_json_path, 'r', encoding='utf-8') as fp:
        products_url_json_array = json.load(fp)
        # check_review_id_number(new_crawled_products_url_json_array)
        for review_dataset_json   in tqdm(products_url_json_array):
            product_name=review_dataset_json["product_name"]
            if product_name  not in product_title_dict:
                product_title_dict[product_name]=review_dataset_json
            else:
                previous_model_number=search_model_number(product_title_dict[product_name]["Spec"],"Model Number")
                model_number=search_model_number(review_dataset_json["Spec"],"Model Number")
                if model_number!=previous_model_number:
                    # print(f"{model_number},{previous_model_number}")
                    previous_product_id=product_title_dict[product_name]["id"]
                    current_product_id=review_dataset_json["id"]
                    previous_review_num=find_review_num(new_crawled_products_url_json_dict,previous_product_id)
                    current_review_num=find_review_num(new_crawled_products_url_json_dict,current_product_id)
                    if previous_review_num>current_review_num:
                        remove_product_id_list.append(current_product_id)
                        cur_removed_review_id_list=gen_review_id_list(new_crawled_products_url_json_dict,current_product_id)
                        
                    else:
                        remove_product_id_list.append(previous_product_id)
                        cur_removed_review_id_list=gen_review_id_list(new_crawled_products_url_json_dict,previous_product_id)
                    if len(cur_removed_review_id_list)>100:
                        print(f"{len(cur_removed_review_id_list)},{remove_product_id_list[-1]}")
                    remove_review_id_list.extend(cur_removed_review_id_list)
            
      
        print(f"{len(remove_product_id_list)},{len(remove_review_id_list)}")


        remove_product_id_set=set(remove_product_id_list)
        output_product_list=[]
        for product_json   in tqdm(products_url_json_array):
            if product_json["id"] not in remove_product_id_list:
                product_json=update_similar_products(product_json,remove_product_id_set)
                output_product_list.append(product_json)
            
        with open(output_products_path, 'w', encoding='utf-8') as fp:
            json.dump(output_product_list, fp, indent=4) 
                
                
        remove_review_id_set=set(remove_review_id_list)
        output_review_list=[]
        with open(review_json_str, 'r', encoding='utf-8') as fp:
            new_crawled_review_url_json_array = json.load(fp)  
            for review_product_json   in tqdm(new_crawled_review_url_json_array):
                if review_product_json["reviews"][0]["id"] not in remove_review_id_list:
                    output_review_list.append(review_product_json)
        with open(output_review_json_str, 'w', encoding='utf-8') as fp:
            json.dump(output_review_list, fp, indent=4)     
         
    
    
def filter_sensitive():
    product_json_path_str=f'bestbuy/data/final/v2_course/bestbuy_products_40000_3.4.12_filter_variant.json'
    output_products_path_str=f'bestbuy/data/final/v2_course/bestbuy_products_40000_3.4.13_sensitive.json'
     
   
    product_json_path = Path(
        product_json_path_str
    )    
    output_products_path = Path(
        output_products_path_str
    ) 
    output_list=[]    
    remove_product_id_list=[]
    with open(product_json_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        # check_review_id_number(new_crawled_products_url_json_array)
        for review_dataset_json   in tqdm(new_crawled_products_url_json_array):
            
            category=review_dataset_json["product_category"] 
            if "Sexual Wellness"   in category:
            
                remove_product_id_list.append(review_dataset_json["id"])
                 
        for review_dataset_json   in tqdm(new_crawled_products_url_json_array):
            if review_dataset_json["id"] not in remove_product_id_list:
                review_dataset_json=update_similar_products(review_dataset_json,set(remove_product_id_list))
                output_list.append(review_dataset_json)
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(output_list, fp, indent=4) 

    
    
    
    
def clean_offsive():
    with open("/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/temp/Facebook Bad Words List - May 1, 2022.txt") as f:
        lines = f.readlines()
        block_word_list=lines[0].split(",")
    product_json_path_str=f'bestbuy/data/final/v2_course/bestbuy_review_2.3.16.19_filter_variant.json'
    output_products_path_str=f'bestbuy/data/final/v2_course/bestbuy_review_2.3.16.20_sensitive.json'
     
   
    product_json_path = Path(
        product_json_path_str
    )    
    output_products_path = Path(
        output_products_path_str
    ) 
    output_list=[]    
    remove_num=0
    with open(product_json_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        # check_review_id_number(new_crawled_products_url_json_array)
        for review_dataset_json   in tqdm(new_crawled_products_url_json_array):
            
    
            
            review_json=review_dataset_json["reviews"][0]
            review_header= review_json["header"] 
            review_body= review_json["body"] 
            review=review_header+". "+review_body 
            is_remove=False 
            cur_block_word=""
            for block_word in block_word_list:
                if " "+block_word+" " in review:
                    is_remove=True   
                    cur_block_word=block_word
            if not is_remove:
                output_list.append(review_dataset_json)
            else:
                review_id=review_dataset_json["reviews"][0]["id"]
                remove_num+=1
                print(f"{review_id},{cur_block_word} ")
    print(remove_num)
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(output_list, fp, indent=4) 
    
 
 

def get_error_examples_after_scraper_fix():
    # review_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v2_course/test/bestbuy_review_2.3.16.20.2_no_search_crawl_from_0_to_150000.json"
    # out_path_str="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v2_course/test/bestbuy_review_2.3.16.20.2.1_error_example.json"
    review_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/example/bestbuy_100_human_performance_added_wrong_review_crawl_from_0_to_150000.json"
    out_path_str="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/example/bestbuy_100_human_performance_added_v2_merged_wrong_review.json"
    incomplete_review_path = Path(
        review_path
    )
    output_products_path = Path(
       out_path_str
    )
    
    out_list=[]
    filter_num=0
 
    # tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    with open(incomplete_review_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        for i,cur_product_json in tqdm(enumerate(incomplete_dict_list) ):
            if "error"   in cur_product_json and cur_product_json["error"]=="Y":
                
                out_list.append(cur_product_json)
                filter_num+=1
            elif cur_product_json['reviews'][0]["corrected_body"]=="":
                out_list.append(cur_product_json)
                filter_num+=1
            
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4)      
    print(filter_num)
    
def get_correct_review_id_list():
    gold_product_dict=gen_reivew_gold_product_dict()
    
    report_dir="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/example/annotation"
 
    review_id_list=[]
    report_path_list=os.listdir(report_dir)
    for report_path in  report_path_list :
        report_df=pd.read_excel(os.path.join(report_dir,report_path), index_col=0,skiprows=0)
        annotator_id=1+int(report_path.split(".xlsx")[0].split("report")[1])
     
        for idx,(_, row) in enumerate(report_df.iterrows()):
         
            review_id=row["Review Id"] 
            human_prediction=row["Target Product Position (1-21)"]
      
                
            if not pd.isna(human_prediction):
                gold_product=gold_product_dict[review_id]
          
                if gold_product==human_prediction:    
                    review_id_list.append(review_id)
    print(len(review_id_list))
    return review_id_list

def gen_mingchen_list():
    report_dir="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/example/annotation"
 
    review_id_list=[]
 
 
    report_df=pd.read_excel(os.path.join(report_dir,"report8.xlsx"), index_col=0,skiprows=0)
    
    
    for idx,(_, row) in enumerate(report_df.iterrows()):
        
        review_id=row["Review Id"] 
        human_prediction=row["Target Product Position (1-21)"]
        if not pd.isna(human_prediction):
                
            review_id_list.append(review_id)
    print(len(review_id_list))
    return review_id_list

def gen_jeet_unannotated_list():
    report_dir="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/example/annotation"
 
    review_id_list=[]
    report_df=pd.read_excel(os.path.join(report_dir,"report6.xlsx"), index_col=0,skiprows=0)
    for idx,(_, row) in enumerate(report_df.iterrows()):  
        review_id=row["Review Id"] 
        human_prediction=row["Target Product Position (1-21)"]
        if not pd.isna(review_id) and   pd.isna(human_prediction):  
            review_id_list.append(review_id)
    print(len(review_id_list))
    return review_id_list

to_be_added_review_id_list=[
    533, 538, 925, 173, 175, 177, 179, 184, 825, 185, 827, 826, 829, 984, 988, 989, 991, 992, 995, 996, 998, 491, 1003, 1004, 894
]

def obtain_human_correct_example_as_testset():
    correct_review_id_list=get_correct_review_id_list()
    # mingchen_list=gen_mingchen_list()
    # jeet_unannotated_list=gen_jeet_unannotated_list()
    # correct_review_id_list.extend(mingchen_list)
    # correct_review_id_list.extend(jeet_unannotated_list)
    # correct_review_id_list.extend(to_be_added_review_id_list)
    print(len(correct_review_id_list))
    
    review_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v3/test/bestbuy_100_human_performance_v4.json"
    out_path_str="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v3/test/bestbuy_review_2.3.16.21.1_right_human_prediction.json"
    incomplete_review_path = Path(
        review_path
    )
    output_products_path = Path(
       out_path_str
    )
    
    out_list=[]
    with open(incomplete_review_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        for i,cur_product_json in tqdm(enumerate(incomplete_dict_list) ):
            review_id=cur_product_json["reviews"][0]["id"]
            if review_id in correct_review_id_list:
                
                out_list.append(cur_product_json)
                  
            
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4)  
    #Mingchen's set all in 
    #Jeet's set, unannotated in 
    #some examples to be added (20)
    
def copy_review_image():
    incomplete_products_path = Path(
    "/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v3/bestbuy_review_2.3.16.26_train_dev_test.json"
    )
    review_err=0
    image_dir="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v2_course/review_images"
    target_image_dir="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v3/review_images"
    out_list=[]
    review_id=0
    image_num=0
    with open(incomplete_products_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        for i,product_json in tqdm(enumerate(incomplete_dict_list) ):
            if (not is_nan_or_miss(product_json,"reviews")): 
                has_image_review=False 
                review_list=[]
                for review in product_json["reviews"]:
                    if not is_nan_or_miss(review, "image_path"):
                        image_path_list=review["image_path"]
                        for image_path in image_path_list:
                            image_num+=1
                            split_image(image_dir,target_image_dir,image_path,True)
    print(image_num)
                            

def copy_product_image():
    incomplete_products_path = Path(
    "/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v3/bestbuy_products_40000_3.4.16_all_text_image_similar.json"
    )
    review_err=0
    image_dir="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v2_course/product_images"
    target_image_dir="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v3/product_images"
    out_list=[]
    review_id=0
    image_num=0
    with open(incomplete_products_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        for i,product_json in tqdm(enumerate(incomplete_dict_list) ):
            image_path_list=product_json["image_path"]
            for image_path in image_path_list:
                image_num+=1
                split_image(image_dir,target_image_dir,image_path,True)
            
                        
    print(image_num)
                            
def extract_images_for_one_list(img_list,source_img_path,target_image_corpus):
    is_copy=True 
    for filepath in  img_list:
        
        real_image_name=filepath
            # print("ERROR! in extract_images_for_one_list")
            # continue 
            
        split_image(source_img_path,target_image_corpus,real_image_name,is_copy)
        

def split_image(image_corpus,splited_data_path,filepath,is_copy):
    source_path=os.path.join(image_corpus,filepath)
    target_path=os.path.join(splited_data_path ,filepath)
    
    # 
    if not os.path.exists(target_path):
        if os.path.exists(source_path):
            if is_copy :
                shutil.copy(source_path, target_path)
            else:
                os.rename(source_path,target_path )   
                


def clean_review_with_empty_body():
 
    review_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/train/bestbuy_review_2.3.16.30_merge_merged_review_from_29.4.1.json"
    incomplete_review_path = Path(
        review_path
    )
    output_products_path = Path(
        "/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/train/bestbuy_review_2.3.16.31_remove_merged_review.json"
    )
    
            
    out_list=[]
    filter_num=0
    # tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    with open(incomplete_review_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        for i,review_product_json in tqdm(enumerate(incomplete_dict_list) ):
            review_json=review_product_json["reviews"][0]
            if  "has_check_image_url"  in review_json or ("corrected_body"   in review_json and review_json["corrected_body"]!="" ) :
             
                out_list.append(review_product_json)
            else:
                filter_num+=1
    
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4)        
    print(filter_num)
    
    
def keep_top_100():
    pass     
import copy 

def remove_empty_review_and_add_review_id():
     
    review_path="/home/menglong/workspace/code/referred/product_scraper/dataset_construction/bestbuy/data/milestone/bestbuy_6507768_review_2.3.1_incomplete_before_remove_reviews_wo_image_part1.json"
    incomplete_products_path = Path(
        review_path
    )
    output_products_path = Path(
        "/home/menglong/workspace/code/referred/product_scraper/dataset_construction/bestbuy/data/mention_to_entity_statistics/bestbuy_review_S2.3.2_statistic_remove_empty_review.json"
    )
    review_err=0
    out_list=[]
    review_id=0
    with open(incomplete_products_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        for i,product_json in tqdm(enumerate(incomplete_dict_list) ):
            if (not is_nan_or_miss(product_json,"reviews")): 
                has_image_review=False 
                review_list=product_json["reviews"]
                product_json["reviews"]=[]
                for review in review_list:
                    new_product_json={}
                    new_product_json["id"]=product_json["id"]
                    new_product_json["url"]=product_json["url"]
                    new_product_json["product_name"]=product_json["product_name"]
                    new_product_json["product_category"]=product_json["product_category"]
                    review["id"]=review_id
                    review_id+=1
                    new_product_json["reviews"]=[review]
                    out_list.append(new_product_json)

    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4) 
    
 


def bs4_review_scraper(product_dict: Dict):
    out_list=[]
    
    if (not is_nan_or_miss(product_dict,"reviews")): 
        has_image_review=False 
        review_list=product_dict["reviews"]
        product_dict["reviews"]=[]
        for review in review_list:
            new_product_json={}
            new_product_json["id"]=product_dict["id"]
            new_product_json["url"]=product_dict["url"]
            new_product_json["product_name"]=product_dict["product_name"]
            new_product_json["product_category"]=product_dict["product_category"]
            new_product_json["reviews"]=[review]
            out_list.append(new_product_json)
    return out_list


def mp_clean():
    args=get_args()
    start_id=args.start_id
    end_id=args.end_id
    incomplete_products_path = Path(
        args.file
    )
    complete_products_path = Path(
        f"{args.out_file}_from_{start_id}_to_{end_id}.json"
    )
    crawl_mode=args.crawl_mode
    get_logger(args.log_file)
    with open(incomplete_products_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)

    step_size = args.step_size
    output_list = []
    print_ctr = 0
    result = []
    save_step=1000
    # crawl_mode_list= [crawl_mode for i in range(step_size)]
    for i in range(0, len(incomplete_dict_list), step_size):
        if i>=start_id :
            if   i<end_id:
                print(100 * '=')
                print(f'starting at the {print_ctr}th value')
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    try:
                        result = list(
                            tqdm(
                                executor.map(
                                    bs4_review_scraper,
                                    incomplete_dict_list[i: i + step_size] 
                                ),
                                total=step_size)
                        )
                    except Exception as e:
                        logging.warning(f"{e}")
                        print(f"__main__ error {e}")
                        result = []
                end = time()
                if len(result) != 0:
                    for  review_list_of_one_product in result:
                        output_list.extend(review_list_of_one_product)
                else:
                    print('something is wrong')
                if i%save_step==0:
                    with open(complete_products_path, 'w', encoding='utf-8') as fp:
                        json.dump(output_list, fp, indent=4)
        print_ctr += step_size
    with open(complete_products_path, 'w', encoding='utf-8') as fp:
        json.dump(output_list, fp, indent=4)
        
         
def set_empty_if_not_none(json_dict,key):
    if key in json_dict:
        json_dict[key]=""
    return json_dict
            
def remove_score_dict(file_path,output_products_path):
    new_crawled_img_url_json_path = Path(
        file_path
        ) 
    out_list=[]   
    with open(new_crawled_img_url_json_path, 'r', encoding='utf-8') as fp:
        product_dataset_json_array = json.load(fp)
        for review_product_json   in product_dataset_json_array :#tqdm(
            # review_text=gen_review_text(review_product_json)
            review_product_json=set_empty_if_not_none(review_product_json,"desc_score_dict")
            review_product_json=set_empty_if_not_none(review_product_json,"image_score_dict")
            review_product_json=set_empty_if_not_none(review_product_json,"text_score_dict")
            review_product_json=set_empty_if_not_none(review_product_json,"fused_score_dict")
            review_product_json=set_empty_if_not_none(review_product_json,"product_id_with_similar_desc_by_review")
            review_product_json=set_empty_if_not_none(review_product_json,"product_id_with_similar_image_by_review")
            review_product_json=set_empty_if_not_none(review_product_json,"product_id_with_similar_text_by_review")
            review_product_json=set_empty_if_not_none(review_product_json,"image_similarity_score_list")
            out_list.append(review_product_json)
    
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4) 
        
def gen_real_path(image_dir,relative_image_path_list):
    return [os.path.join(image_dir,relative_image_path) for relative_image_path in relative_image_path_list]        
def clean_by_prediction(review_path,cleaned_path,review_image_path):
    with open(review_path, 'r', encoding='utf-8') as fp:
        review_product_json_array = json.load(fp)
    output_list=[]
    filter_image_num=0
    filter_num=0
    for review_product_json  in  tqdm(review_product_json_array):
     
        cleaned_image_path_list=[]
        for relative_image_path in review_product_json["reviews"][0]["image_path"]:
            real_image_path=os.path.join(review_image_path,relative_image_path)
            if os.path.exists(real_image_path):
                cleaned_image_path_list.append(relative_image_path)
            else:
                filter_image_num+=1
        if len(cleaned_image_path_list)>0:
            review_product_json["reviews"][0]["image_path"]=cleaned_image_path_list
            output_list.append(review_product_json)
        else:
            filter_num+=1
    with open(cleaned_path, 'w', encoding='utf-8') as fp:
        json.dump(output_list, fp, indent=4)
    print(filter_image_num,filter_num)
            
def is_low_quality(image_path):
    file_stats=os.stat(image_path)
    if file_stats.st_size>20000:
        return False
    else:
        return True            
            
def clean_by_low_quality_image(review_path,cleaned_path,review_image_path,wrong_image_dir,is_product=False):
    with open(review_path, 'r', encoding='utf-8') as fp:
        review_product_json_array = json.load(fp)
    output_list=[]
    filter_image_num=0
    filter_num=0
    for review_product_json  in  tqdm(review_product_json_array):
     
        cleaned_image_path_list=[]
        if not is_product:
            image_list=review_product_json["reviews"][0]["image_path"]
        else:
            image_list=review_product_json["image_path"]
        for relative_image_path in image_list:
            real_image_path=os.path.join(review_image_path,relative_image_path)
            if not is_low_quality(real_image_path):
                cleaned_image_path_list.append(relative_image_path)
            else:
                 
                # dst_path =os.path.join(wrong_image_dir,relative_image_path)
                # shutil.move(real_image_path, dst_path)
                filter_image_num+=1
        if len(cleaned_image_path_list)>0:
            if not is_product:
                review_product_json["reviews"][0]["image_path"]=cleaned_image_path_list
            else:
                review_product_json["image_path"]=cleaned_image_path_list
            output_list.append(review_product_json)
        else:
            filter_num+=1
    with open(cleaned_path, 'w', encoding='utf-8') as fp:
        json.dump(output_list, fp, indent=4)
    print(filter_image_num,filter_num)
            
def remove_review_wo_gold_product(product_json_path,review_dataset_path,cleaned_path):
    output_list=[]
    filter_num=0
    with open(product_json_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        product_json_dict=json_to_product_id_dict(new_crawled_products_url_json_array)
        
    with open(review_dataset_path, 'r', encoding='utf-8') as fp:
        review_dataset_json_array = json.load(fp) 
        for review_dataset_json in review_dataset_json_array:
            gold_product_id=review_dataset_json["id"]
            if gold_product_id not in product_json_dict:
                filter_num+=1
            else:
                output_list.append(review_dataset_json)
    print(filter_num)
    with open(cleaned_path, 'w', encoding='utf-8') as fp:
        json.dump(output_list, fp, indent=4)
         
def clean_object_detect_error_by_merge(review_path,out_path,cleaned_review_reference_path):
    with open(cleaned_review_reference_path, 'r', encoding='utf-8') as fp:
        cleaned_review_reference_array = json.load(fp) 
        cleaned_review_reference_dict=review_json_to_product_dict(cleaned_review_reference_array)
    out_list=[]
    with open(review_path, 'r', encoding='utf-8') as fp:
        review_dataset_json_array = json.load(fp) 
        for review_dataset_json  in review_dataset_json_array:
            review_id=review_dataset_json["reviews"][0]["id"]
            if review_id in cleaned_review_reference_dict:
                cleaned_review_reference=cleaned_review_reference_dict[review_id]
                cleaned_image_path_list=cleaned_review_reference["reviews"][0]["image_path"]
                review_dataset_json["reviews"][0]["image_path"]=cleaned_image_path_list
                out_list.append(review_dataset_json)
    with open(out_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4)         
    print(len(out_list))
         

def remove_images_not_in_json_file(json_path,output_image_father_dir, image_father_dir,out_path,is_product):
    image_dir=os.path.join(image_father_dir,"review_images")
    cleaned_image_dir=os.path.join(image_father_dir,"cleaned_review_images")
    output_image_dir=os.path.join(output_image_father_dir,"review_images")
    output_cleaned_image_dir=os.path.join(output_image_father_dir,"cleaned_review_images")
    
    incomplete_review_path = Path(
        json_path
    )  
    filter_num=0
    filter_image_num=0
    image_num=0
    out_list=[]
    with open(incomplete_review_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        for i,review_product_json in tqdm(enumerate(incomplete_dict_list) ):
            if not is_product:
                image_list=review_product_json["reviews"][0]["image_path"]
            else:
                image_list=review_product_json["image_path"]
            new_image_list=[]
            for relative_image_path in image_list:
                src_path = os.path.join(image_dir,relative_image_path)
                dst_path =os.path.join(output_image_dir,relative_image_path)
                cleaned_src_path = os.path.join(cleaned_image_dir,relative_image_path)
                cleaned_dst_path =os.path.join(output_cleaned_image_dir,relative_image_path)
                if    ( os.path.exists(src_path) or  os.path.exists(dst_path)) and (  os.path.exists(cleaned_src_path) or os.path.exists(cleaned_dst_path)  ) :
                   
                    new_image_list.append(relative_image_path)
                    image_num+=1
                    if   os.path.exists(src_path): 
                        shutil.move(src_path, dst_path)
                    if os.path.exists(cleaned_src_path): 
                    
                        shutil.move(cleaned_src_path, cleaned_dst_path)
                else:
                    filter_image_num+=1
                
                
            if len(new_image_list)>0:
                if not is_product:
                    review_product_json["reviews"][0]["image_path"]=new_image_list
                else:
                    review_product_json["image_path"]=new_image_list
                out_list.append(review_product_json)
            else:
                filter_num+=1
    with open(out_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4)         
                    
                    
    print(image_num,filter_image_num,filter_num)
               
def clean_format_product(json_path,out_path):
    incomplete_review_path = Path(
        json_path
    )  
    out_list=[]
    with open(incomplete_review_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        for i,product_json in tqdm(enumerate(incomplete_dict_list) ):
            del product_json["thumbnails"]
            del product_json["reviews_urls"]
            del product_json["reviews"]
            product_json["product_category"]=product_json["product_category"]
            del product_json["product_category"]
            del product_json["thumbnail_paths"]
            del product_json["similar_product_id"]
            del product_json["product_id_with_similar_image"]
            out_list.append(product_json)
    with open(out_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4)   

    
def clean_format_review(json_path,out_path):
    incomplete_review_path = Path(
        json_path
    )  
    out_list=[]
    with open(incomplete_review_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        for i,review_product_json in tqdm(enumerate(incomplete_dict_list) ):
            review_json=review_product_json["reviews"][0]
            del review_json["product_images"]
            del review_json["thumbnail_paths"]
            if "wrong_image_url" in review_json:
                del review_json["wrong_image_url"]
            if "corrected_body" in review_json:
                del review_json["corrected_body"]
            if "image_similarity_score" in review_json:
                del review_json["image_similarity_score"]
            if "reshuffled_target_product_id_position" in review_json:
                del review_json["reshuffled_target_product_id_position"]
            if "has_check_image_url" in review_json:
                del review_json["has_check_image_url"]
            review_json["attribute_by_match"]=review_json["attribute"]
            del review_json["attribute"]
            review_json["gold_entity_info"]={}
            review_json["gold_entity_info"]["product_name"]=review_product_json["product_name"]
            review_json["gold_entity_info"]["id"]=review_product_json["id"]
            review_json["gold_entity_info"]["product_category"]=review_product_json["product_category"]
            for key in review_json.keys():
                if key not in ["user","header","rating","recommendation","feedback","body","id","mention","image_path","image_url","attribute_by_match","gold_entity_info"]:
                    del review_json[key]
            out_list.append(review_json)
    with open(out_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4)   
        
        

    
def clean_format_review_from_id_to_review_id(json_path,out_path):
    incomplete_review_path = Path(
        json_path
    )  
    out_list=[]
    with open(incomplete_review_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        for i,review_product_json in tqdm(enumerate(incomplete_dict_list) ):
            review_json=review_product_json 
            review_product_json["review_id"]=review_product_json["id"]
            del review_product_json["id"]
            review_product_json["review_image_url"]=review_product_json["image_url"]
            del review_product_json["image_url"]
            review_product_json["review_image_path"]=review_product_json["image_path"]
            del review_product_json["image_path"]
            out_list.append(review_json)
    with open(out_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4)           



    
def clean_format_review_from_id_to_review_id_for_one_dir(json_dir ):
    for file_name in os.listdir(json_dir):
        if file_name.endswith(".json") and not file_name.startswith("max_setting_dict"):
            one_path=os.path.join(json_dir,file_name)
            clean_format_review_from_id_to_review_id(one_path,one_path)
         

def clean_missed_product_in_csv(product_json_path):
    with open(product_json_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        product_json_dict=json_to_product_id_dict(new_crawled_products_url_json_array)
        
    df=pd.read_csv('/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/mention_to_entity_statistics/product_alias_base_to_product_id_statistic_1.csv')  
    print(len(df))
    df=df[df["product_id"].isin(list(product_json_dict.keys()))]
    print(len(df))
    df.to_csv("/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/mention_to_entity_statistics/product_alias_base_to_product_id_statistic_2.csv",index=False)  
     
def clean_missed_product_in_prior(product_json_path,review_json_path,out_path):
    with open(product_json_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        product_json_dict=json_to_product_id_dict(new_crawled_products_url_json_array)
    incomplete_review_path = Path(
        review_json_path
    )  
    filter_num=0
    with open(incomplete_review_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        new_incomplete_dict_list={}
        for alias, statistic_dict in incomplete_dict_list.items():
            new_product_id_list=[]
            product_id_list=statistic_dict["product_id"]
            for product_id in product_id_list:
                if product_id in product_json_dict:
                    new_product_id_list.append(product_id)
                else:
                    filter_num+=1
            statistic_dict["product_id"]=new_product_id_list
            new_incomplete_dict_list[alias]=statistic_dict
             
    with open(out_path, 'w', encoding='utf-8') as fp:
        json.dump(new_incomplete_dict_list, fp, indent=4)   
    print(filter_num)
            

def clean_missed_product_in_2_3_17_1(product_json_path,review_json_path,out_path):
    with open(product_json_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        product_json_dict=json_to_product_id_dict(new_crawled_products_url_json_array)
    incomplete_review_path = Path(
        review_json_path
    )  
    filter_num=0
    out_list=[]
    with open(incomplete_review_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
         
        for review_json in incomplete_dict_list:
            new_candidate_id=[]
            candidate_list=review_json["fused_candidate_list"]
            for candidate_id in candidate_list:
                if candidate_id in product_json_dict:
                    new_candidate_id.append(candidate_id)
                else:
                    filter_num+=1
            review_json["fused_candidate_list"]=new_candidate_id
            out_list.append(review_json)
    with open(out_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4)   
    print(filter_num)

def    clean_err1(json_dir):
    for file_name in os.listdir(json_dir):
        if file_name.endswith(".json") and not file_name.startswith("max_setting_dict"):
            one_path=os.path.join(json_dir,file_name)
            incomplete_review_path = Path(
                one_path
            )  
            out_list=[]
            review_id_set=set()
            with open(incomplete_review_path, 'r', encoding='utf-8') as fp:
                incomplete_dict_list = json.load(fp)
                for i,review_product_json in tqdm(enumerate(incomplete_dict_list) ):
                    review_json=review_product_json 
                    review_id=review_product_json["review_id"] 
                    if review_id not in review_id_set:
                        out_list.append(review_json)
             
                    review_id_set.add(review_id)
            with open(one_path, 'w', encoding='utf-8') as fp:
                json.dump(out_list, fp, indent=4)         
            print(len(out_list))
                        
def clean_format_review2(json_path,out_path):
    incomplete_review_path = Path(
        json_path
    )  
    out_list=[]
    with open(incomplete_review_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        for i,review_product_json in tqdm(enumerate(incomplete_dict_list) ):
            review_json=review_product_json 
            if "total_predicted_attribute_exact" in review_product_json:
                del review_product_json["total_predicted_attribute_exact"]
             
                del review_product_json["confidence_score_exact"]
             
                del review_product_json["predicted_attribute_context_exact"]
            del review_product_json["attribute_by_match"]
            out_list.append(review_json)
    with open(out_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4)           


            

# def clean_format_review3(json_path,out_path):
#     incomplete_review_path = Path(
#         json_path
#     )  
#     out_list=[]
#     with open(incomplete_review_path, 'r', encoding='utf-8') as fp:
#         incomplete_dict_list = json.load(fp)
#         for i,review_product_json in tqdm(enumerate(incomplete_dict_list) ):
#             review_json=review_product_json 
#             if "total_predicted_attribute_exact" in review_product_json:
#                 del review_product_json["total_predicted_attribute_exact"]
             
#                 del review_product_json["confidence_score_exact"]
             
#                 del review_product_json["predicted_attribute_context_exact"]
#             del review_product_json["attribute_by_match"]
#             out_list.append(review_json)
#     with open(out_path, 'w', encoding='utf-8') as fp:
#         json.dump(out_list, fp, indent=4)  
                    
import argparse
def get_args():

    parser = argparse.ArgumentParser() 
    parser.add_argument("--file",default='bestbuy/data/final/v6/val/retrieval', type=str  )
    parser.add_argument("--product_path",default='bestbuy/data/final/v6/bestbuy_products_40000_3.4.19_final_format.json', type=str  )
    parser.add_argument("--out_file",default='bestbuy/data/final/v6/val/bestbuy_review_2.3.17.0.1_format_wo_candidates_review_id.json', type=str  ) 
    parser.add_argument("--image_dir",default='/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v5/cleaned_review_images', type=str  )
    parser.add_argument("--error_image_dir",default="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v5/error_review_images_low_quality", type=str  )
    parser.add_argument("--cleaned_image_dir",default="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v5/error_review_images_low_quality", type=str  )
    parser.add_argument("--start_id", default=0, type=int)
    parser.add_argument("--end_id", default=150000, type=int)
    parser.add_argument("--step_size", default=8, type=int)
    # parser.add_argument("--is_review", action='store_true'   ) #
    parser.add_argument("--crawl_mode", default='product', type=str  ) #review, spec
    parser.add_argument("--log_file",default="bestbuy/output/log.txt", type=str  )
    parser.add_argument("--mode", default='review', type=str  ) #review, spec
    args = parser.parse_args()

    print(args)
    return args
# runner
def get_logger(log_file):
    logging.basicConfig(format=FORMAT,filename=log_file, encoding='utf-8', level=logging.WARN)
    
import concurrent.futures
if __name__ == "__main__":
    
     
 
 
  
    args=get_args()  
    # clean_by_low_quality_image(args.file,args.out_file,args.image_dir ,args.error_image_dir,is_product=True)
    # remove_review_wo_gold_product(args.product_path,args.file,args.out_file)
    # clean_object_detect_error_by_merge(args.file,args.out_file,args.product_path )
    # remove_images_not_in_json_file(args.file,args.cleaned_image_dir, args.image_dir,args.out_file,is_product=False)
    # clean_format_review(args.file,args.out_file)
    # clean_missed_product_in_csv(args.file)
    remove_review_with_error_image(args.file,args.out_file )
    # remove_product_with_error_image()
    # clean_format_review2( args.file,args.out_file )
    # clean_format_review2( args.file,args.out_file )
    # clean_by_prediction(args.file,args.out_file,args.image_dir )
    # clean_review_with_empty_body()
    # mp_clean()
    # remove_empty_review_and_add_review_id()
    # filter_product_variant()
    # filter_sensitive()
    # clean_offsive()
    # get_error_examples_after_scraper_fix()
    # choose_one_from_duplicate_review()
    # obtain_human_correct_example_as_testset()
    # clean_review_with_empty_body()
    # copy_review_image()
    # 
    # copy_product_image()
    # if args.mode=="review":
    #     remove_review_with_error_image()
    # else:
    #     remove_product_with_error_image() 
    # choose_one_from_duplicate_review()
    # remove_score_dict(args.file,args.out_file)
    # filter_review_wo_mention_and_change_mention_to_review_sub_json(args.file,args.out_file,is_mention_in_review_sub_json=True)
# filter_review_wo_mention_and_change_mention_to_review_sub_json()
# filter_unhelpful()                          
# filter_non_overlap_reviews()   
# remove_review_wo_image_and_add_review_id()