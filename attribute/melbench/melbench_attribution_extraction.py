#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
"""
from pathlib import Path
import json
from tqdm import tqdm
import os
import pandas as pd
from nltk.tokenize import word_tokenize
import argparse
import logging


from bs4 import BeautifulSoup
import numpy as np
import torch

from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
)
from attribute.extractor.chatgpt_extractor import ChatGPTExtractor, ChatGPTParser
from attribute.extractor.amazon_review_gen_review_attribute_by_rule import AmazonMatchExtractor

from attribute.extractor.gpt2_extractor import MODEL_CLASSES, GPT2Extractor, GPT2ResultParser, adjust_length_to_model, gen_model_gpt2, set_seed
from attribute.extractor.gpt3 import GPT3Extractor
from attribute.extractor.melbench_vicuna_complete_extractor import MELBenchVicunaCompleteExtractor
# from attribute.extractor.llava_extractor import LLaVAExtractor  #TODO
from attribute.extractor.model_version_extractor import ModelVersionExtractor, ModelVersionToProductTitleExtractor
from attribute.extractor.ocr_extractor import OCRExtractor, OCRResultParser
from attribute.extractor.vicuna_complete_extractor import VicunaCompleteExtractor
from attribute.extractor.vicuna_extractor import VicunaExtractor
from attribute.melbench.melbench_util import load_attribute_key_value 
from disambiguation.data_util.inner_util import   json_to_dict, review_json_to_product_dict, spec_to_json_attribute, spec_to_json_attribute_w_special_attribute_success_flag
from util.env_config import * 
import concurrent.futures
import pandas as pd
import gzip
import json 
from tqdm import tqdm 

# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.chrome.options import Options as ChromeOptions
# from selenium.webdriver.edge.options import Options as EdgeOptions
# from selenium.webdriver.firefox.options import Options as FirefoxOptions 
# from selenium.webdriver.chrome.service import Service as ChromeService      
# from selenium.webdriver.edge.service import Service as EdgeService
# from selenium.webdriver.firefox.service import Service as FirefoxService 
# from webdriver_manager.chrome import ChromeDriverManager
# from webdriver_manager.firefox import GeckoDriverManager
# from webdriver_manager.microsoft import EdgeChromiumDriverManager 
import re

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
 

def _init(args ):
    # args=gen_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    logger.warning(f"device: {args.device}, n_gpu: {args.n_gpu}, 16-bits training: {args.fp16}")
    set_seed(args)
    return args
def match_for_multiple_extraction( attribute_value_candidate_list,attribute_value_gold_list,acc_dict,predict_num_dict,attribute_key):
    is_attribute_correct="n"
    for attribute_value_candidate in attribute_value_candidate_list:
        predict_num_dict[attribute_key]+=1
        if attribute_value_candidate.lower() in attribute_value_gold_list :
            acc_dict[attribute_key]+=1
            is_attribute_correct="y"
        elif attribute_value_candidate in attribute_value_gold_list :
            acc_dict[attribute_key]+=1
            is_attribute_correct="y"
        else:
            print("")
        # else:
        #     for  attribute_value_gold in attribute_value_gold_list:
        #         if  attribute_value_candidate.lower()  in attribute_value_gold:
        #             return True 
    return acc_dict,predict_num_dict ,is_attribute_correct


def  match(attribute_value_candidate_list,attribute_value_gold_list):
    # print(attribute_value_candidate_list,attribute_value_gold_list)
    for attribute_value_candidate in attribute_value_candidate_list:
        if attribute_value_candidate.lower() in attribute_value_gold_list :
            return True 
        elif attribute_value_candidate in attribute_value_gold_list :
            return True 
        # else:
        #     for  attribute_value_gold in attribute_value_gold_list:
        #         if  attribute_value_candidate.lower()  in attribute_value_gold:
        #             return True 
    return False 
  



def generate_per_review_attribute(args, review,attribute_key,candidate_attribute_list,is_constrained_beam_search,extractor,review_dataset_json,
                                  review_image_dir,total_attribute_value_candidate_list,total_confidence_score_list,mention,
                                  attribute_value,ocr_raw,gpt_context,chatgpt_context):
    
    
    return extractor.generate_per_review_attribute(args, args.prefix,review,attribute_key,candidate_attribute_list,is_constrained_beam_search,
                                                   review_dataset_json,review_image_dir,total_attribute_value_candidate_list,
                                                   total_confidence_score_list,mention,attribute_value,ocr_raw,gpt_context,chatgpt_context)
    # else:
    #     return generate_by_gpt3(args,model,tokenizer,input_text,postfix,candidate_attribute_list,is_constrained_beam_search)



def gen_candidate_attribute(review_id,product_dict,review_dataset_json,candidate_num):
    total_attribute_json_in_review={"Product Title":[]}
     
    product_id_list=review_dataset_json["fused_candidate_list"][:candidate_num]#100 
    for product_id in product_id_list:
        product_json=product_dict[product_id]
        total_attribute_json_in_review=gen_gold_attribute_without_key_section( product_json["Spec"],product_json["product_name"],total_attribute_json_in_review)
        # product_title=product_json["product_name"]
        
    return  total_attribute_json_in_review 
    
def gen_key_name(args):
    key_name=args.attribute_logic
    if args.attribute_logic=="gpt2_result_parser":
        key_name="gpt2"
    elif args.attribute_logic=="ocr_result_parser":
        key_name="ocr"
    elif args.attribute_logic=="chatgpt_parser":
        key_name="chatgpt"
    return key_name
def init_json(review_dataset_json,key_name,is_allow_override):
    if f"predicted_attribute_{key_name}" not in review_dataset_json or is_allow_override:
        review_dataset_json[f"predicted_attribute_{key_name}"]={ }#{args.attribute_logic}
    if "gold_attribute" not in review_dataset_json  :
        review_dataset_json[f"gold_attribute"]={ }
    if args.attribute_logic in ["gpt2","gpt2_few","chatgpt","vicuna","llava","vicuna_complete"] and (f"predicted_attribute_context_{args.attribute_logic}" not in review_dataset_json):
        review_dataset_json[f"predicted_attribute_context_{args.attribute_logic}"]={ }
    if f"is_attribute_correct_{key_name}" not in review_dataset_json or is_allow_override:
        review_dataset_json[f"is_attribute_correct_{key_name}"]={ }
    # if f"predicted_attribute_out_of_top10_{key_name}" not in review_dataset_json:
    #     review_dataset_json[f"predicted_attribute_out_of_top10_{key_name}"]={}
    #     review_dataset_json[f"is_attribute_correct_out_of_top10_{key_name}"]={}
    return review_dataset_json



def load_attribute(tech1):
    soup = BeautifulSoup(tech1, 'html.parser')

    # Extract attribute keys and values
    attributes = {}
    for th, td in zip(soup.find_all('th', class_='a-color-secondary a-size-base prodDetSectionEntry'),
                    soup.find_all('td', class_='a-size-base')):
        key = th.get_text(strip=True)
        value = td.get_text(strip=True)
        attributes[key] = [value]
    return attributes

def gen_gold_attribute_without_key_section(tech1,tech2,section_list=None,dataset_class="v6",is_allow_multiple_attribute=True):
    # Parse the HTML using BeautifulSoup
    attribute={}
    if tech1!="" and tech1 is not None:
        attribute_tech1=load_attribute(tech1)
        attribute.update(attribute_tech1)
    if tech2!="" and tech2 is not None:
        attribute_tech2=load_attribute(tech2)
        attribute.update(attribute_tech2)
    
    
    
     
    return attribute  



def handle_one_review_dataset_json(review_dataset_json,args,idx,start_idx,end_idx,review_id_to_debug,is_allow_override,product_dict,
                                   valid_product_id_list,
                                   key_name,extractor,is_constrained_beam_search,output_list,output_products_path,save_step,acc_dict,predict_num_dict):
    should_skip=False
    is_first_non_selected_attribute=False
    if idx>=start_idx or args.is_slice_start_to_end=="y" :
        if   idx<end_idx or args.is_slice_start_to_end=="y":
            if review_id_to_debug is not None and review_dataset_json["review_id"]!=review_id_to_debug:
                should_skip=True
                return  should_skip,output_list,acc_dict,predict_num_dict
            if f"predicted_attribute_{args.attribute_logic}" not in review_dataset_json or is_allow_override:
                product_id=review_dataset_json["answer"]
                review_json=review_dataset_json
                review_id=review_json["id"]
                # mention=review_json["mention"]
                mention=""
                review=review_json["sentence"] 
                 
                gold_attribute_json=product_dict[product_id]
                all_ocr_raw,all_gpt_context,all_chatgpt_context={},{},{}
           
                if args.attribute_source=="similar":
                    candidate_attribute_json=gen_candidate_attribute(review_id,product_dict,review_dataset_json,candidate_num=10)
                    attribute_key_list=list(candidate_attribute_json.keys())
                elif args.attribute_source=="gold_attribute_category_similar_attribute_value":
                    candidate_attribute_json=gen_candidate_attribute(review_id,product_dict,review_dataset_json,candidate_num=10)
                    gold_attribute_for_predicted_category=review_json["gold_attribute_for_predicted_category"]
                    attribute_key_list=list(gold_attribute_for_predicted_category.keys())
                else:
                    candidate_attribute_json=gold_attribute_json
                    attribute_key_list=list(gold_attribute_json.keys())
                is_match_all_attribute=True
                review_dataset_json=init_json(review_dataset_json,key_name,is_allow_override)
                _,_,total_attribute_value_candidate_list,_,total_confidence_score_list=extractor.generate_per_review(args, args.prefix,review,"",[],is_constrained_beam_search,review_dataset_json,review_image_dir)
                for attribute_key in  attribute_key_list :
                    if (f"predicted_attribute_{key_name}" in review_dataset_json and attribute_key in review_dataset_json[f"predicted_attribute_{key_name}"]) and not is_allow_override:
                        continue 
                    if args.attribute_field !="all" and attribute_key not in args.attribute_field:
                        continue 
            
                    if attribute_key not in acc_dict:
                        acc_dict[attribute_key]=0
                    if attribute_key not in predict_num_dict:
                        predict_num_dict[attribute_key]=0
                    if attribute_key in gold_attribute_json:
                        gold_attribute_value=gold_attribute_json[attribute_key]
                    else:
                        gold_attribute_value=""
                    # if attribute_key  in review_json["attribute"]:
                    #     attribute_value_predicted_by_stringmatch=review_json["attribute"][attribute_key ]
                    # else:
                    #     attribute_value_predicted_by_stringmatch=""
                    if attribute_key in candidate_attribute_json:
                        candidate_attribute_list=candidate_attribute_json[attribute_key]
                    else:
                        candidate_attribute_list=[]
                    if attribute_key in all_gpt_context:
                        gpt_context=all_gpt_context[attribute_key]
                    else:
                        gpt_context=[]
                    if attribute_key in all_chatgpt_context:
                        chatgpt_context=all_chatgpt_context[attribute_key]
                    else:
                        chatgpt_context=[]
                            
                    _,_,attribute_value_candidate_list,generated_sequences,confidence_score_list,attribute_value_candidate_list_out_of_top10=generate_per_review_attribute(args, review,attribute_key,  candidate_attribute_list,
                                                                                    is_constrained_beam_search,extractor,review_dataset_json,
                                                                                    review_image_dir,total_attribute_value_candidate_list,
                                                                                    total_confidence_score_list,mention,gold_attribute_value,all_ocr_raw, gpt_context, chatgpt_context)
                    
                    if len(gold_attribute_value)>1 :
                        acc_dict,predict_num_dict,is_attribute_correct=match_for_multiple_extraction( attribute_value_candidate_list,gold_attribute_value,acc_dict,predict_num_dict,attribute_key)
                    else:    
                        if len(attribute_value_candidate_list)>0   :
                            predict_num_dict[attribute_key]+=1
                        if match( attribute_value_candidate_list,gold_attribute_value):
                            acc_dict[attribute_key]+=1
                            is_attribute_correct="y"
                            # print(f"Right: predicted:{generated_sequence}, gold:{attribute_value}, predicted_by_stringmatch:{attribute_value_predicted_by_stringmatch}")
                        else:
                            is_match_all_attribute=False 
                            is_attribute_correct="n"
        
                    if len(gold_attribute_value)>0 and gold_attribute_value !=" " and len(attribute_value_candidate_list)>0:
                        review_dataset_json["gold_attribute"][ attribute_key ]=gold_attribute_value
                    # review_dataset_json["predicted_attribute_by_stringmatch"][ attribute_key ]=attribute_value_predicted_by_stringmatch
                    if args.attribute_logic in ["gpt2","gpt2_few","chatgpt" ,"vicuna","llava","vicuna_complete"]:
                        review_dataset_json[f"predicted_attribute_context_{args.attribute_logic}"][ attribute_key ]=generated_sequences
                    # if len(attribute_value_candidate_list)>0   :
                    if len(attribute_value_candidate_list)>0 :
                        review_dataset_json[f"predicted_attribute_{key_name}"][ attribute_key ]= attribute_value_candidate_list
                        review_dataset_json[f"is_attribute_correct_{key_name}"][ attribute_key ]=is_attribute_correct
                    elif    attribute_key in review_dataset_json[f"predicted_attribute_{key_name}"]:
                        del review_dataset_json[f"predicted_attribute_{key_name}"][ attribute_key ]
                    # if len(attribute_value_candidate_list_out_of_top10)>0:
                    #     review_dataset_json[f"predicted_attribute_out_of_top10_{key_name}"][ attribute_key ]= attribute_value_candidate_list_out_of_top10
                    #     review_dataset_json[f"is_attribute_correct_out_of_top10_{key_name}"][ attribute_key ]=is_attribute_correct_out_of_top10
                        
                if args.attribute_logic in ["ocr"]:
                    # review_dataset_json[f"total_predicted_attribute_{args.attribute_logic}"]=total_attribute_value_candidate_list
                    review_dataset_json[f"confidence_score_{args.attribute_logic}"]=total_confidence_score_list
                if is_match_all_attribute:
                    acc_dict["all"]+=1 
            output_list.append(review_dataset_json)
            if idx%save_step==0:
                with open(output_products_path, 'w', encoding='utf-8') as fp:
                    json.dump(output_list, fp, indent=4)
  
    return  should_skip,output_list,acc_dict,predict_num_dict


def generate_for_gz_dataset(data_path,args,  output_products_path,
                          product_dict,is_constrained_beam_search,extractor,review_id_to_debug,valid_product_id_list,is_allow_override=False):
    new_crawled_data_json_path = data_path
    # test_val_gpt_attribute_set=load_val_test_attribute_key_set( )
    output_list=[]
    start_idx=args.start_idx
    end_idx=args.end_idx
    output_products_path = Path(
        f"{output_products_path}_from_{start_idx}_to_{end_idx}.json"
    )
    save_step=30
    acc_dict={}
    acc_dict["all" ]=0
    
    with gzip.open(data_path, "rt") as input_file:
       
        valid_num=0
        predict_num_dict={}
        key_name=gen_key_name(args)
        # review_dataset_json_array_to_use=review_dataset_json_array[start_idx:end_idx]
        print(f"start_idx:{start_idx},end_idx:{end_idx} ")
        for idx,line in tqdm(enumerate(input_file )):
        # review_dataset_json_array = json.load(fp)
            review_dataset_json = json.loads(line.strip())
            should_skip,output_list=handle_one_review_dataset_json(idx,start_idx,end_idx,review_id_to_debug,is_allow_override,product_dict,valid_product_id_list,
                                   key_name,extractor,is_constrained_beam_search,output_list,output_products_path,save_step)
    out_acc_dict={}
    out_predict_num_dict={}
    for key,value in acc_dict.items():
        if key!="all" and key in predict_num_dict  and predict_num_dict[key]>0:
            acc=value/predict_num_dict[key]
            if acc>0:
                out_acc_dict[key]=acc
            out_predict_num_dict[key]=predict_num_dict[key]
     
    print(out_acc_dict,len(output_list))
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(output_list, fp, indent=4)  
    print(out_predict_num_dict)
    
    
    
def generate_for_dataset(data_path,args,  output_products_path,
                          product_dict,is_constrained_beam_search,extractor,review_id_to_debug,valid_product_id_list,is_allow_override=False):
    new_crawled_data_json_path = data_path
    # test_val_gpt_attribute_set=load_val_test_attribute_key_set( )
    output_list=[]
    start_idx=args.start_idx
    end_idx=args.end_idx
    output_products_path = Path(
        f"{output_products_path}_from_{start_idx}_to_{end_idx}.json"
    )
    save_step=100
    acc_dict={}
    acc_dict["all" ]=0
    
    with open(data_path, 'r', encoding='utf-8') as fp:
        review_dataset_json_array = json.load(fp)
        total_num=min(end_idx,len(review_dataset_json_array))-start_idx
        predict_num_dict={}
        key_name=gen_key_name(args)
        # review_dataset_json_array_to_use=review_dataset_json_array[start_idx:end_idx]
        # print(f"start_idx:{start_idx},end_idx:{end_idx},len:{len(review_dataset_json_array_to_use)}")
        for idx,(review_id,review_dataset_json) in enumerate(tqdm(review_dataset_json_array.items())):
             
            should_skip,output_list,acc_dict,predict_num_dict=handle_one_review_dataset_json(review_dataset_json,args,idx,start_idx,end_idx,review_id_to_debug,is_allow_override,product_dict,valid_product_id_list,
                                   key_name,extractor,is_constrained_beam_search,output_list,output_products_path,save_step,
                                   acc_dict,predict_num_dict)
    out_acc_dict={}
    out_predict_num_dict={}
    for key,value in acc_dict.items():
        if key!="all" and key in predict_num_dict  and predict_num_dict[key]>0:
            acc=value/predict_num_dict[key]
            if acc>0:
                out_acc_dict[key]=acc
            out_predict_num_dict[key]=predict_num_dict[key]
     
    print(out_acc_dict,len(output_list))
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(output_list, fp, indent=4)  
    print(out_predict_num_dict)    

# def gen_attribute_key_list(attribute_field,attribute_source):
#     attribute_key_list=["Brand"] 
#     return attribute_key_list



def gen_model(args,device):
    if args.attribute_logic in ["gpt2","gpt2_few"]:
        extractor=GPT2Extractor(device,args)
    # if args.attribute_logic in ["llava"]:
    #     extractor=LLaVAExtractor( args) #TODO 
    elif args.attribute_logic=="gpt3":
        extractor=GPT3Extractor( )
    # elif args.attribute_logic in ["exact","numeral"]:
    #     extractor=MatchExtractor( )
    elif args.attribute_logic=="ocr":
        extractor=OCRExtractor(device)
    elif args.attribute_logic=="gpt2_result_parser":
        extractor=GPT2ResultParser(device,args)
    elif args.attribute_logic=="ocr_result_parser":
        extractor=OCRResultParser(device)
    elif args.attribute_logic=="chatgpt":
        extractor=ChatGPTExtractor()
    elif args.attribute_logic=="chatgpt_parser":
        extractor=ChatGPTParser()
    elif args.attribute_logic=="vicuna":
        extractor=VicunaExtractor()
    elif args.attribute_logic=="vicuna_complete":
        extractor=VicunaCompleteExtractor()
    elif args.attribute_logic=="melbench_vicuna_complete":
        extractor=MELBenchVicunaCompleteExtractor()
            
    elif args.attribute_logic=="model_version_to_product_title":
        extractor=ModelVersionToProductTitleExtractor()
    elif args.attribute_logic=="model_version":
        extractor=ModelVersionExtractor()
    elif args.attribute_logic in ["amazon_exact"]:
        extractor=AmazonMatchExtractor( )
    return extractor

def init(fused_score_path,product_path,args):
    logger.info(args)
    with open(fused_score_path, 'r', encoding='utf-8') as fp:
        review_product_json_array_with_fused_score = json.load(fp)    
        review_product_json_array_with_fused_score_dict=review_json_to_product_dict(review_product_json_array_with_fused_score)

    # prompt_text = args.prompt if args.prompt else input("Model prompt >>> ")
    # attribute_key_list=gen_attribute_key_list(args.attribute_field,args.attribute_source,review_product_json_array_with_fused_score_dict)
    # attribute_key_list=["Brand" ,"Color","Product Name","Storage Capacity","Capacity"]
    
    with open(product_path, 'r', encoding='utf-8') as fp:
        products_url_json_array = json.load(fp)
        product_dict=json_to_dict(products_url_json_array)
    return review_product_json_array_with_fused_score_dict,product_dict

def gen_valid_product_id_list(product_path):
    valid_product_id_list=[]
    with gzip.open(product_path, "rt") as input_file :
       
        for i,line in tqdm(enumerate(input_file )) :
            try:
                # Assuming each line is a JSON object
                item = json.loads(line.strip())
                valid_product_id_list.append(item["asin"])
            except json.JSONDecodeError:
                print("Error decoding JSON from line:", line)
            
    return valid_product_id_list



def gen_melbench_product_dict(product_path):
    return load_attribute_key_value(product_path),[]
    
    
    # with gzip.open(product_path, "rt") as input_file :
       
    #     for i,line in tqdm(enumerate(input_file )) :
    #         try:
    #             # Assuming each line is a JSON object
    #             item = json.loads(line.strip())
    #             product_dict[item["asin"]]=item 
    #             valid_product_id_list.append(item["asin"])
    #         except json.JSONDecodeError:
    #             print("Error decoding JSON from line:", line)
            
    # return product_dict,valid_product_id_list

 


def main(args,data_path,output_products_path, products_path_str,review_id_to_debug):
    extractor=gen_model(args,args.device)
    
    # review_product_json_array_with_fused_score_dict,product_dict=init(fused_score_path,product_path,args)
    product_dict,valid_product_id_list=gen_melbench_product_dict( products_path_str)
    # Initialize the model and tokenizer
    # prompt_text = args.prompt if args.prompt else input("Model prompt >>> ")
    # attribute_key_list=gen_attribute_key_list(args.attribute_field,args.attribute_source,review_product_json_array_with_fused_score_dict)
    # attribute_key_list=["Brand" ,"Color","Product Name","Storage Capacity","Capacity"]
    is_constrained_beam_search=False 

    # generated_sequence=generate(args,model,tokenizer,prompt_text,postfix)
    
    generate_for_dataset(data_path,args, 
                            output_products_path,
                            product_dict,is_constrained_beam_search,extractor,review_id_to_debug ,valid_product_id_list,is_allow_override=True)
 
 
 

def bs4_review_scraper(review_dataset_json,extractor,args, product_dict,test_val_gpt_attribute_set):
    is_constrained_beam_search=False  
    product_id=review_dataset_json["answer"]
    review_json=review_dataset_json
    review_id=review_json["id"]
    # mention=review_json["mention"]
    mention=""
    review=review_json["sentence"] 
        
    attribute_json=product_dict[product_id]
                 
     
    
    
    
    if args.attribute_source=="similar":
        candidate_attribute_json=gen_candidate_attribute(review_id,product_dict,review_dataset_json)
        attribute_key_list=list(candidate_attribute_json.keys())
    elif args.attribute_source=="gold_attribute_category_similar_attribute_value":
        candidate_attribute_json=gen_candidate_attribute(review_id,product_dict,review_dataset_json,candidate_num=10)
        gold_attribute_for_predicted_category=review_json["gold_attribute_for_predicted_category"]
        attribute_key_list=list(gold_attribute_for_predicted_category.keys())
    else:
        candidate_attribute_json=attribute_json
        attribute_key_list=list(attribute_json.keys())
    is_match_all_attribute=True
    review_dataset_json[f"predicted_attribute_{args.attribute_logic}"]={ }
    if "gold_attribute" not in review_dataset_json:
        review_dataset_json[f"gold_attribute"]={ }
    if args.attribute_logic in ["gpt2","gpt2_few","chatgpt","llava","vicuna","vicuna_complete"] and f"predicted_attribute_context_{args.attribute_logic}" not in review_dataset_json:
        review_dataset_json[f"predicted_attribute_context_{args.attribute_logic}"]={ }
    review_dataset_json[f"predicted_attribute_context_{args.attribute_logic}"]={ }
    review_dataset_json[f"is_attribute_correct_{args.attribute_logic}"]={ }
    _,_,total_attribute_value_candidate_list,_,total_confidence_score_list=extractor.generate_per_review(args, args.prefix,review,"",[],is_constrained_beam_search,review_dataset_json,review_image_dir)
    for attribute_key in tqdm(attribute_key_list):
       
        # if attribute_key not in test_val_gpt_attribute_set:
        #     continue 
        # if attribute_key not in acc_dict:
        #     acc_dict[attribute_key]=0
        
        if attribute_key in attribute_json:
            attribute_value=attribute_json[attribute_key]
        else:
            attribute_value=""
        # if attribute_key  in review_json["attribute"]:
        #     attribute_value_predicted_by_stringmatch=review_json["attribute"][attribute_key ]
        # else:
        #     attribute_value_predicted_by_stringmatch=""
        if attribute_key in candidate_attribute_json:
            candidate_attribute_list=candidate_attribute_json[attribute_key]
        else:
            candidate_attribute_list=[] 
        _,_,attribute_value_candidate_list,generated_sequences,[],confidence_score_list=generate_per_review_attribute(args, review,attribute_key,  candidate_attribute_list,
                                                                        is_constrained_beam_search,extractor,review_dataset_json,review_image_dir,total_attribute_value_candidate_list,total_confidence_score_list,
                                                                        mention,
                                  attribute_value,[],[],[])
        
        if match( attribute_value_candidate_list,attribute_value):
            # acc_dict[attribute_key]+=1
            is_attribute_correct="y"
            # print(f"Right: predicted:{generated_sequence}, gold:{attribute_value}, predicted_by_stringmatch:{attribute_value_predicted_by_stringmatch}")
        else:
            is_match_all_attribute=False 
            is_attribute_correct="n"
            # print(f"Wrong: predicted:{generated_sequence}, gold:{attribute_value}, predicted_by_stringmatch:{attribute_value_predicted_by_stringmatch}")
        
        if len(attribute_value)>0 and attribute_value !=" " and len(attribute_value_candidate_list)>0 :
            review_dataset_json["gold_attribute"][ attribute_key ]=attribute_value
        # review_dataset_json["predicted_attribute_by_stringmatch"][ attribute_key ]=attribute_value_predicted_by_stringmatch
        if len(attribute_value_candidate_list)>0   :
            review_dataset_json[f"predicted_attribute_{args.attribute_logic}"][ attribute_key ]= attribute_value_candidate_list
            review_dataset_json[f"predicted_attribute_context_{args.attribute_logic}"][ attribute_key ]=generated_sequences
            review_dataset_json[f"is_attribute_correct_{args.attribute_logic}"][ attribute_key ]=is_attribute_correct
    review_dataset_json[f"total_predicted_attribute_{args.attribute_logic}"]=total_attribute_value_candidate_list
    review_dataset_json[f"confidence_score_{args.attribute_logic}"]=total_confidence_score_list
    return review_dataset_json


def load_val_test_attribute_key_set( ):
    import pickle
    
        # # Open the file in binary mode
        # with open('/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/train/attribute/test_val_gpt_attribute_key.pkl', 'rb') as file:
            
        #     # Call load method to deserialze
        #     test_val_gpt_attribute_set = pickle.load(file)
        # return test_val_gpt_attribute_set
    attribute_dict={'Brand': 0.8831908831908832, 'Product Title':1,'Model Number': 0.8823529411764706,  'Color': 1,  
                    'Processor Brand': 0.7669902912621359, 'GPU Brand': 0.7906976744186046,
                    "Processor Model":1,"Screen Size":1,"Solid State Drive Capacity":1,"System Memory (RAM)":1,"Graphics":1 }    
    attribute_key_set=set(attribute_dict.keys())
     
        
    return attribute_key_set


def mp_main(args):
    start_id=args.start_idx
    end_id=args.end_idx
    incomplete_products_path = Path(
        args.data_path
    )
    complete_products_path = Path(
        f"{args.out_path}_from_{start_id}_to_{end_id}.json"
    )
 
    test_val_gpt_attribute_set=load_val_test_attribute_key_set()
    with open(incomplete_products_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)

    step_size =  args.step_size
    output_list = []
    print_ctr = 0
    result = []
    save_step=5
    world_size=torch.cuda.device_count()
    gpu_list=[i for i in range(world_size)]
     
    args_list= [args for i in range(step_size)]
    test_val_gpt_attribute_set_list=[test_val_gpt_attribute_set for i in range(step_size)]
    extractor_list=[ gen_model(args,torch.device('cuda',gpu_list[i%world_size])) for i in range(step_size)]
    product_dict,_=gen_melbench_product_dict( args.product_path)
    product_dict_list=[product_dict for i in range(step_size)] 
    incomplete_dict_list=incomplete_dict_list[start_id:end_id]
    is_check_start=False              
    for i in tqdm( range(0, len(incomplete_dict_list), step_size)):
        if i>=start_id or not is_check_start:
            if   i<end_id or not is_check_start:
                print(100 * '=')
                print(f'starting at the {print_ctr}th value')
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    try:
                        result = list(
                            tqdm( 
                                executor.map(
                                    bs4_review_scraper,
                                    incomplete_dict_list[i: i + step_size],
                                    extractor_list,
                                    args_list,
                                
                                    product_dict_list ,
                                    test_val_gpt_attribute_set_list
                                ),
                                total=step_size)
                        )
                    except Exception as e:
                        logging.warning(f"{e}")
                        print(f"__main__ error {e}")
                        result = []
             
                if len(result) != 0:
                    output_list.extend(result)
                else:
                    print('something is wrong')
                if i%save_step==0:
                    with open(complete_products_path, 'w', encoding='utf-8') as fp:
                        json.dump(output_list, fp, indent=4)
        print_ctr += step_size
    with open(complete_products_path, 'w', encoding='utf-8') as fp:
        json.dump(output_list, fp, indent=4)
    # print(f"new crawl {new_crawled_num}") 

import argparse
def parser_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--data_path',type=str,help=" ",default="data/melbench/Richpedia-MEL_w_attribute_from_0_to_10000000.json")
    parser.add_argument('--out_path',type=str,help=" ",default="data/melbench/Richpedia-MEL_w_attribute_vicuna")
    parser.add_argument('--attribute_logic',type=str,help=" ",default="melbench_vicuna_complete")#exact numeral amazon_exact vicuna_complete
    parser.add_argument('--product_path',type=str,help=" ", default="data/melbench/qid2abs_long.json")
    # parser.add_argument('--example_type',type=str,help=" ",default="verification")
    # parser.add_argument('--mode',type=str,help=" ",default="test")
    parser.add_argument('--attribute_source',type=str,help=" ",default="gold")#gold
    parser.add_argument('--attribute_field',type=str,help=" ",default="all")#all Brand
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=10000000)
    parser.add_argument("--step_size", type=int, default=5)
    parser.add_argument("--is_mp", type=str, default="y", help="Text added prior to input.")
    parser.add_argument("--is_slice_start_to_end", type=str, default="n", help="Text added prior to input.")
    
    # parser.add_argument('--filter_by_attribute_field',type=str,help=" ",default="Brand|Color")#all
    # parser.add_argument('--generate_claim_level_report',type=str,help=" ",default="y")
    parser.add_argument(
        "--model_type",
        default="gpt2",
        type=str,
         
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default="gpt2-xl",
        type=str,
        
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument("-prompt", type=str, default="")#Perfect companion to the Surface Pro 8. This is the keyboard to get with the new Surface Pro 8. The Surface Pro Keyboard with Slim Pen 2 is comfortable to type on, although it feels somewhat delicate due to its thinness, so if you need to write an angry email, don't take it out on this keyboard. The built-in trackpad is responsive and works well for pointing, scrolling, and tapping. The keyboard also includes a space to store the Slim Pen 2 in the section that normally faces up against the tablet, so you can be sure the pen doesn't fall out. The Slim Pen 2 works well for doodling or marking up a PDF, and there is an adjustable tactile engine for the pen that can make it feel like you're writing on paper (although, I had to adjust the tactile feedback to the max of 100 to really notice it). When not in use, this keyboard cover folds up against the screen to protect it with a light magnetic closure that helps keep it closed.
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--stop_token", type=str, default=None, help="Token at which text generation is stopped")

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)

    parser.add_argument("--prefix", type=str, default="Attribute Value Extraction: (If the output is a date, please convert the data format to yyyy-MM-dd, like 1705-02-24.)\n ", help="Text added prior to input.")
    parser.add_argument("--padding_text", type=str, default="", help="Deprecated, the use of `--prefix` is preferred.")
    parser.add_argument("--xlm_language", type=str, default="", help="Optional language when used with the XLM model.")
 
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument("--review_id_to_debug", type=int,default=None )
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v0")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--data_format", type=str, default="json")
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    args = parser.parse_args()
    return args


   

if __name__ == '__main__':
    args = parser_args()
    args=_init( args)
    # data_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/multimodal_entity_linking/output/error/retrieval/bestbuy_50_error.json"
    # output_products_path=f"/home/menglong/workspace/code/multimodal/multimodal_entity_linking/multimodal_entity_linking/output/error/retrieval/bestbuy_50_error_with_attribute_{args.attribute_logic}.json"
    # data_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v5/test/retrieval/bestbuy_review_2.3.16.29.5_fused_score_20_54.json"
    # output_products_path=f"/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v5/test/retrieval/bestbuy_review_2.3.16.29.6_fused_score_20_54_attribute_{args.attribute_logic}.json"
    # out_path=f"/home/menglong/workspace/code/multimodal/multimodal_entity_linking/multimodal_entity_linking/output/error/disambiguation/50/bestbuy_50_error_with_attribute_{args.attribute_logic}.json"
    # fused_score_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v5/test/retrieval/bestbuy_review_2.3.16.29.5_fused_score_20_54.json"
    
    if args.is_mp=="y":
        mp_main(args)
    else:
        main(args,args.data_path, args.out_path, args.product_path,args.review_id_to_debug)