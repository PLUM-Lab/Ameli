import os 
import re
import json

import json
from pathlib import Path
from time import time
from pathlib import Path
import json
import pandas as pd
import requests
import os
import copy     
from util.env_config import * 
from tqdm import tqdm
from ast import literal_eval 
def setup_with_args(args,outdir,run_desc=""):
    if  run_desc !=None and args.desc!=None:
        run_desc+="-"+args.desc
    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    args.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{run_desc}')
    args.cur_run_id=cur_run_id
    assert not os.path.exists(args.run_dir)
    # Create output directory.
    print(f'Creating output directory...{args.run_dir}')
    os.makedirs(args.run_dir)
    
    
    
    # Print options.
    # print()
    # print(f'Training options:  ')
    # print(json.dumps(args, indent=2))
    # print()
    # print(f'Output directory:   {args.run_dir}')
    # with open(os.path.join(args.run_dir, 'training_options.json'), 'wt') as f:
    #     json.dump(args, f, indent=2)
 
    return args.run_dir,args 

def is_nan_or_miss(json,key):
    if key  not in json:
        return True
    elif json[key] is None or len( json[key]) == 0:          
        return True 
    else:
        return False   

def json_to_product_id_dict(to_merge_img_url_dict_list):
    output_dict={}
    for to_merge_img_url_dict in to_merge_img_url_dict_list:
        output_dict[to_merge_img_url_dict["id"]]=to_merge_img_url_dict
    return output_dict
 
 

def product_json_to_product_id_dict(to_merge_img_url_dict_list):
    output_dict={}
    for to_merge_img_url_dict in to_merge_img_url_dict_list:
        output_dict[to_merge_img_url_dict["id"]]=to_merge_img_url_dict
    return output_dict
      
def json_to_dict(to_merge_img_url_dict_list):
    output_dict={}
    for to_merge_img_url_dict in to_merge_img_url_dict_list:
        output_dict[to_merge_img_url_dict["id"]]=to_merge_img_url_dict
    return output_dict

def json_to_review_dict(review_dataset_json_array):
    output_review_dict={}
    for review_dataset_json in review_dataset_json_array:
       
    
        review_id=review_dataset_json["review_id"]
        output_review_dict[review_id]=review_dataset_json
    return output_review_dict 
 
 
 
def review_json_to_product_dict(review_dataset_json_array):
    output_review_dict={}
    for review_dataset_json in review_dataset_json_array:
        
        review=review_dataset_json
        review_id=review["review_id"]
        output_review_dict[review_id]=review_dataset_json
           
                
         
    return output_review_dict 
 

 

def gen_extracted_attributes_except_ocr_str(review_product_json,dataset):
    exact_attribute_str=append_one_kind_of_attribute("predicted_attribute_exact",review_product_json,"EXACT")
    gpt2_attribute_str=append_one_kind_of_attribute("predicted_attribute_gpt2",review_product_json,"GPT2")
    model_version_attribute_str=append_one_kind_of_attribute("predicted_attribute_model_version",review_product_json,"VERSION",is_allow_multiple=True)
    model_version_to_product_title_attribute_str=append_one_kind_of_attribute("predicted_attribute_model_version_to_product_title",review_product_json,"PRODUCT")
    if dataset=="test" and "predicted_attribute_chatgpt" in review_product_json:
        chatgpt_attribute_str=append_one_kind_of_attribute("predicted_attribute_chatgpt",review_product_json,"CHATGPT")
    else:
        chatgpt_attribute_str=""
    return exact_attribute_str+model_version_attribute_str+model_version_to_product_title_attribute_str+gpt2_attribute_str+chatgpt_attribute_str
    
    

def append_one_kind_of_attribute(attribute_field_name,review_product_json,special_token,is_allow_multiple=False):
    predicted_attribute_ocr_dict=review_product_json[attribute_field_name]
    if len(predicted_attribute_ocr_dict)>0:
        one_kind_attributes=f" {special_token}: " 
        for attribute_key,predicted_attribute_ocr_list in predicted_attribute_ocr_dict.items():
            if not is_allow_multiple:
                one_kind_attributes+=predicted_attribute_ocr_list[0]+". "
            else:
                for predict_attribute in predicted_attribute_ocr_list:
                    one_kind_attributes+=predict_attribute+". "
        one_kind_attributes=one_kind_attributes[:-1]
    else:
        one_kind_attributes=""
    return one_kind_attributes
    
def gen_ocr_info(review_product_json):
    raw_ocr=" RAW: "+". ".join(review_product_json["raw_ocr"])+"."
    ocr_attributes=append_one_kind_of_attribute("predicted_attribute_ocr",review_product_json,"OCR")
 
    return raw_ocr,ocr_attributes    

def gen_review_text_rich(review_product_json):
    # predicted_attribute_dict=review_product_json["predicted_attribute"]
     
    review_text=review_product_json["header"]+". "+review_product_json["body"]
    ocr_raw_text,ocr_attributes=gen_ocr_info(review_product_json)
    sentence1="\n Review: "+review_text+". \n"+ ocr_attributes+ocr_raw_text
  
    extracted_attributes_except_ocr_str=gen_extracted_attributes_except_ocr_str(review_product_json,"test")
    sentence1+=extracted_attributes_except_ocr_str
    return sentence1