# from attribute.gen_review_attribute import match_attribute_by
from retrieval.utils.retrieval_util import old_spec_to_json_attribute
from retrieval.organize.score.scorer_helper import retrieval_recall_metric
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
from util.env_config import *

from pathlib import Path
import json

import os
import pandas as pd
import shutil  
from random import sample
from tqdm import tqdm 
import random
from nltk.tokenize import word_tokenize
from disambiguation.data_util.inner_util import gen_gold_attribute, gen_gold_attribute_without_key_section, gen_review_text, json_to_dict
import re 

# def extract_attribute(gold_product_id,similar_product_id_list,product_dict,review_text):
#     product_json=product_dict[gold_product_id]
#     total_attribute={} 
#     attribute_json_in_review=match_attribute_by(review_text,product_json["Spec"])
 
#     total_attribute.update(attribute_json_in_review)
#     return total_attribute
 
def compare_with_gold(extracted_attributes,gold_product_attributes,attribute_key_list,extraction_num_dict,
                                                                                                          extraction_right_num_dict,
                                                                                                        gold_num_dict,annotated_attributes=None):
    extracted_num=0
    right_extraction_num=0
    is_valid_example=True 
    if len(extracted_attributes)>0:
        for extracted_attribute_key,extracted_attribute_value in extracted_attributes.items():
            if len(extracted_attribute_value)>0:
                if attribute_key_list is  None or  extracted_attribute_key in attribute_key_list:
                    if extracted_attribute_key not in extraction_num_dict:
                        extraction_num_dict[extracted_attribute_key]=0
                    if (annotated_attributes is not None):
                        attributes_dict_to_be_check=annotated_attributes
                    else:
                        attributes_dict_to_be_check=gold_product_attributes
                    if  extracted_attribute_key in attributes_dict_to_be_check:
                        
                        if isinstance(attributes_dict_to_be_check[extracted_attribute_key],str):
                            one_gold_product_attribute_value=attributes_dict_to_be_check[extracted_attribute_key]
                        elif attributes_dict_to_be_check[extracted_attribute_key] is not None and len(attributes_dict_to_be_check[extracted_attribute_key])>0:
                            one_gold_product_attribute_value=attributes_dict_to_be_check[extracted_attribute_key][0]
                        else:
                            continue
                        extraction_num_dict[extracted_attribute_key]+=1
                        extracted_num+=1
                        # for one_gold_product_attribute_value in attributes_dict_to_be_check[extracted_attribute_key] :
                        if   one_gold_product_attribute_value in extracted_attribute_value or (len(extracted_attribute_value)>0 and extracted_attribute_value[0] in one_gold_product_attribute_value)  :
                            right_extraction_num+=1 
                            if extracted_attribute_key not in extraction_right_num_dict:
                                extraction_right_num_dict[extracted_attribute_key]=0
                            extraction_right_num_dict[extracted_attribute_key]+=1
                            # break
                            # else:
                                # print(f"filter gold by {extracted_attribute_key}, gold: {gold_product_attributes[extracted_attribute_key]}, review: {extracted_attribute_value}")
                                # pass 
                    # else:
                    #     print(f"missing gold:{gold_product_attributes}")
                # else:
                #     print(f"not expected attribute")
            # else:
            #     print(f"len=0,{extracted_attribute_value}")
    else:
        # print(f"len(extracted_attributes)=0")
        is_valid_example=False 
    
    gold_num=0
    # for  attribute_key in extracted_attributes:
    #     if  attribute_key_list is not None and  attribute_key not in attribute_key_list:
    #         continue 
    #     extracted_num+=1
    if  annotated_attributes is not None :  
        product_attribute_dict_to_be_check=annotated_attributes
    else:
        product_attribute_dict_to_be_check=gold_product_attributes
    for  attribute_key in product_attribute_dict_to_be_check:
        if  attribute_key_list is not None and  attribute_key not in attribute_key_list:
            continue 
        gold_num+=1
        if attribute_key not in gold_num_dict:
            gold_num_dict[attribute_key]=0
        gold_num_dict[attribute_key]+=1
    return right_extraction_num,extracted_num,gold_num,extraction_num_dict,extraction_right_num_dict,gold_num_dict
    
    
    

#1. only compute based on annotated attribute categories
def attribute_metric(review_file_str,attribute_key="predicted_attribute_ocr",attribute_key_list=None,product_dict=None,is_print=True,
                     extract_num_threshold=50,is_retrieval_correct="n",is_gold_attribute="n"):
    if product_dict is None:
        product_path=Path(products_path_str)
        with open(product_path, 'r', encoding='utf-8') as fp:
            products_url_json_array = json.load(fp)
            product_dict=json_to_dict(products_url_json_array)
    new_crawled_img_url_json_path = Path(
        review_file_str
        ) 
    filter_gold_num=0
    extraction_num_dict={}
    extraction_right_num_dict={}
    gold_num_dict={}
    total_right_extraction_num,total_extracted_attributes_num,total_gold_product_attributes_num =0,0,0
    with open(new_crawled_img_url_json_path, 'r', encoding='utf-8') as fp:
        product_dataset_json_array = json.load(fp)
        for review_product_json   in product_dataset_json_array :#tqdm(
            review_text=gen_review_text(review_product_json)
            gold_product_id=review_product_json["gold_entity_info"]["id"]
            gold_product_json=product_dict[gold_product_id]
            review_json=review_product_json
            similar_product_id_list=review_product_json["fused_candidate_list"]
            if attribute_key   in review_json:
                extracted_attributes=review_json[attribute_key]#attribute
            else:
                extracted_attributes=[]
            if gold_product_id not in similar_product_id_list[:10]:
                filter_gold_num+=1
                if is_retrieval_correct=="y":
                    continue
            
            # extracted_attributes=extract_attribute(gold_product_id,similar_product_id_list,product_dict,review_text)
            gold_product_attributes=gen_gold_attribute_without_key_section(gold_product_json["Spec"],gold_product_json["product_name"],{},dataset_class="v6",section_list=None,is_allow_multiple_attribute=True)
            if is_gold_attribute=="y":
                annotated_attributes=review_product_json["gold_attribute_for_predicted_category"]
            else:
                annotated_attributes=None
            # gold_product_attributes=spec_to_json_attribute(gold_product_json["Spec"],is_key_attribute=False)
            right_extraction_num, extracted_attributes_num,gold_product_attributes_num,extraction_num_dict,extraction_right_num_dict,gold_num_dict =compare_with_gold(extracted_attributes,
                                                                                                          gold_product_attributes,
                                                                                                          attribute_key_list,extraction_num_dict,
                                                                                                          extraction_right_num_dict,gold_num_dict,
                                                                                                          annotated_attributes)
            total_extracted_attributes_num+=extracted_attributes_num
            total_gold_product_attributes_num+=gold_product_attributes_num
            total_right_extraction_num+= right_extraction_num
    total_precision=total_right_extraction_num/total_extracted_attributes_num
    total_recall=total_right_extraction_num/total_gold_product_attributes_num
    total_f1=2*total_precision*total_recall/(total_precision+total_recall)
    total_metric={"f1":total_f1,"precision":total_precision,"recall":total_recall,"extract_num":total_extracted_attributes_num,"gold_num":total_gold_product_attributes_num,"correct_num":total_right_extraction_num}
    if is_print:
        print(f"filter_gold_num:{filter_gold_num}, precision:{total_right_extraction_num/total_extracted_attributes_num} in {total_extracted_attributes_num},recall {total_right_extraction_num/total_gold_product_attributes_num} in {total_gold_product_attributes_num}")
    # print(f"extraction_num_dict:{extraction_num_dict},extraction_right_num_dict:{extraction_right_num_dict},gold_num_dict:{gold_num_dict}")
    attribute_metric_dict,cleaned_out_dict=print_pre_recall(extraction_num_dict,extraction_right_num_dict,gold_num_dict,is_print,extract_num_threshold)
    return attribute_metric_dict,cleaned_out_dict,total_metric
    
def print_pre_recall(extraction_num_dict,extraction_right_num_dict,gold_num_dict,is_print,extract_num_threshold)    :
    out_dict={}
    for key,value in extraction_right_num_dict.items():
        if key in extraction_num_dict:
            extract_num=extraction_num_dict[key]
            precision_ignore_no_extraction=value/extract_num
        else:
            extract_num=0
            precision_ignore_no_extraction=-1
        
        if key in gold_num_dict:
            gold_num=gold_num_dict[key]
            recall=value/gold_num
            
        else:
            gold_num=0
            recall=-1
        f1=recall
        if key not in out_dict:
            out_dict[key]={}
        # out_dict[key]["precision_ignore_no_extraction"]=precision_ignore_no_extraction
        out_dict[key]["precision"]=precision_ignore_no_extraction
        out_dict[key]["recall"]=recall
        out_dict[key]["f1"]=f1
        out_dict[key]["extract_num"]=extract_num
        out_dict[key]["gold_num"]=gold_num
        out_dict[key]["correct_num"]=value
    # print(out_dict)
    cleaned_out_dict={}
 
    for key,value in out_dict.items():
        if meet_threshold(value,extract_num_threshold):#
            cleaned_out_dict[key]=out_dict[key]
    if is_print:
        print(cleaned_out_dict)
    return out_dict,cleaned_out_dict
    
def evaluate(data_path,is_check_attribute=True,metric_key="fused_candidate_list",precision_recall_at_k=[1,10,20,50, 100,1000]):
   
    if is_check_attribute:
        attribute_metric(data_path)
    
     
    retrieval_recall_metric(data_path,precision_recall_at_k,metric_key)
    
import copy     
def overall_metric(data_path,mode,is_print=True,extract_num_threshold=50,is_gold_attribute="n"):
    if mode=="test":
        extracted_attribute_field_list=["predicted_attribute_chatgpt","predicted_attribute_vicuna","predicted_attribute_gpt2_few"] #predicted_attribute_out_of_top10_chatgpt
    else:
        extracted_attribute_field_list=[]
    extracted_attribute_field_list.extend(["predicted_attribute_exact","predicted_attribute_ocr","predicted_attribute_gpt2","predicted_attribute_model_version","predicted_attribute_model_version_to_product_title","predicted_attribute","gold_attribute_for_predicted_category"])#
    
    metric_dict={}
    product_path=Path(products_path_str)
    with open(product_path, 'r', encoding='utf-8') as fp:
        products_url_json_array = json.load(fp)
        product_dict=json_to_dict(products_url_json_array)
    total_metric_dict={}
    for extracted_attribute_field in extracted_attribute_field_list:
        print(extracted_attribute_field)
        attribute_level_metric_dict,cleaned_attribute_level_metric_dict,total_metric=attribute_metric( data_path, extracted_attribute_field,None,
                                                                                         product_dict,is_print,extract_num_threshold,is_gold_attribute=is_gold_attribute)
        metric_dict[extracted_attribute_field]=cleaned_attribute_level_metric_dict
        total_metric_dict[extracted_attribute_field]=copy.deepcopy(total_metric)
    # print(total_metric_dict)
    for key,value in total_metric_dict.items():
        print(f"{key}: {value}")
    # print(f"gpt2 {total_metric_dict['predicted_attribute_gpt2']}")
    # print(f"gpt2_few  {total_metric_dict['predicted_attribute_gpt2_few']}" )
    # print(f"vicuna  {total_metric_dict['predicted_attribute_vicuna']}")
    return metric_dict 

def meet_threshold(value,extract_num_threshold):
    if value["extract_num"]>extract_num_threshold: #value["precision"]>0.6    :# and  
        return True
    else:
        return False
    
def count_number(data_path,attribute_key="gold_attribute_for_predicted_category"):
    number=0
    with open(data_path, 'r', encoding='utf-8') as fp:
        product_dataset_json_array = json.load(fp)
        for review_product_json   in product_dataset_json_array :#tqdm(
            
            gold_attribute_for_predicted_category=review_product_json[attribute_key] 
            cur_number=len(gold_attribute_for_predicted_category)
            number+=cur_number
        print(f"{number},{len(product_dataset_json_array)},{number/len(product_dataset_json_array)}")
        
import argparse
def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',type=str,help=" ",default="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/disambiguation/bestbuy_review_2.3.17.11.21.1_no_run_clean_fuse1.json") 
    parser.add_argument('--mode',type=str,help=" ",default="test")
    parser.add_argument('--output_path',type=str,help=" ",default="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v5/test/attribute/bestbuy_review_2.3.16.29.12_114_gold_attribute.json")
    parser.add_argument('--file_with_ocr_attribute',type=str,help=" ",default="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v5/test/retrieval/bestbuy_review_2.3.16.29.5_fused_score_20_54.json")
    parser.add_argument('--file_with_gpt_attribute',type=str,help=" ",default="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v5/test/retrieval/bestbuy_review_2.3.16.29.5_fused_score_20_54.json")
    parser.add_argument('--product_file',type=str,help=" ",default=products_path_str)
    parser.add_argument('--attribute_key',type=str,help=" ",default="predicted_attribute")#exact numeral
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=50000)
    parser.add_argument("--is_mp", type=str, default="y", help="Text added prior to input.")
    parser.add_argument("--is_retrieval_correct", type=str, default="n", help="Text added prior to input.")
 
    args = parser.parse_args()
    return args


   

if __name__ == '__main__':
    args = parser_args()
    extracted_attribute_field_list=["Color","Brand","Color Category"]#"Product Name",,"Model Number"
    # data_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/disambiguation/bestbuy_review_2.3.17.11.19_gold_attribute_for_predict_category.json" 
    # attribute_key="attribute"#"predicted_attribute_ocr"
    # evaluate(data_path)
    # attribute_metric(args.data_path,args.attribute_key,is_retrieval_correct=args.is_retrieval_correct)#attribute_key_list=["Product Title"],
    
    overall_metric(args.data_path, args.mode,is_print=False,is_gold_attribute="y")
    # count_number(args.data_path,attribute_key=args.attribute_key)