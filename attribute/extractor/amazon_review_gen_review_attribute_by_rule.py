from pathlib import Path
import json

import os
import pandas as pd
import shutil  
from random import sample
from tqdm import tqdm 
import random
from nltk.tokenize import word_tokenize
from attribute.extractor.base import Extractor
from attribute.util.util import is_numeral_match, is_sublist_in_list
 
import re
import nltk
from nltk.corpus import stopwords

from util.common_util import json_to_dict

 


def gen_review_text(review_product_json):
    review_header=review_product_json["header"]
    review_body=review_product_json["body"]
    review_text=review_header+". "+review_body
    
    return review_text
def is_attribute_in_review(review,attribute_value,review_tokens,numeral_in_review,attribute_logic):
    attribute_value_tokens=word_tokenize(attribute_value.lower())
    if is_sublist_in_list(attribute_value_tokens,review_tokens):
        return True  
    elif attribute_logic=="numeral" and is_numeral_match(attribute_value,numeral_in_review):
        return True
    else:
        return False 
     
def match_attribute_by(review,product_spec_json ,attribute_field,attribute_logic,total_attribute_json_in_review):
    attribute_json_in_review={}
    review_tokens=word_tokenize(review.lower())
    numeral_in_review = re.findall('[0-9]+\.[0-9]+|[0-9]+', review)
    important_attribute_json_list=product_spec_json   #[:2]
    for attribute_subsection in important_attribute_json_list:
        attribute_list_in_one_section=attribute_subsection["text"]
        for attribute_json  in attribute_list_in_one_section:
            attribute_key=attribute_json["specification"]
            attribute_value=attribute_json["value"]
            attribute_value_to_check=attribute_value
            if attribute_value.lower()=="yes":
                attribute_value_to_check=attribute_key
            elif attribute_value.lower()=="no"  :
                continue 
            elif attribute_value.lower()=="other":
                continue 
            # elif len(attribute_value)==1:
            #     continue  
            if is_attribute_in_review(review,attribute_value_to_check,review_tokens,numeral_in_review ,
                                      attribute_logic) :
                if attribute_value not in stopwords.words('english'):
                    if attribute_key in total_attribute_json_in_review:
                        if attribute_value not in total_attribute_json_in_review[attribute_key]:
                            total_attribute_json_in_review[attribute_key].append(attribute_value)
                    else:
                        total_attribute_json_in_review[attribute_key]=[attribute_value]
    
    # for 
    return total_attribute_json_in_review

def extract_attribute_method(review_text,product_dict,gold_product_json,review_product_json,attribute_source 
                             ,attribute_field,attribute_logic ):
    if attribute_source=="gold":
        return extract_attribute_method_1(review_text,product_dict,gold_product_json,review_product_json,attribute_source ,attribute_field,attribute_logic)
    else:
        return extract_attribute_method_2(review_text,product_dict,gold_product_json,review_product_json,attribute_source ,attribute_field,attribute_logic)
        
def extract_attribute_method_1(review_text,product_dict,gold_product_json,review_product_json,attribute_source ,attribute_field,attribute_logic):
    attribute_json_in_review=match_attribute_by(review_text,gold_product_json["Spec"] ,attribute_field,attribute_logic)
    return attribute_json_in_review

def extract_attribute_method_2(review_text,product_dict,gold_product_json,review_product_json,attribute_source ,attribute_field,attribute_logic ):
    attribute_json_in_review_from_text_similar_product=match_attribute_by_similar_products(review_product_json["fused_candidate_list"][:1000],product_dict,review_text,attribute_field,attribute_logic)
    return attribute_json_in_review_from_text_similar_product

def match_attribute_main( review_path,output_products_path,attribute_source ,attribute_field,attribute_logic):
    product_path=Path("/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v4/bestbuy_products_40000_3.4.16_all_text_image_similar.json")
    with open(product_path, 'r', encoding='utf-8') as fp:
        products_url_json_array = json.load(fp)
        product_dict=json_to_dict(products_url_json_array)
    
    
    # review_path = Path(
    #     f'/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/example/bestbuy_100_human_performance_w_similar_score.json'
    # )
     
    # output_products_path = Path(
    #     f'/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/example/bestbuy_100_human_performance_w_similar_score_v2.json'
    # )
    output_list=[]
  
    with open(review_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        for review_product_json in tqdm(incomplete_dict_list )  :
            review_text=gen_review_text(review_product_json)
            product_id=review_product_json["gold_entity_info"]["id"]
            product_json=product_dict[product_id]
            total_attribute=extract_attribute_method(review_text,product_dict, product_json,
                                                     review_product_json,attribute_source ,attribute_field,attribute_logic)
            # 
            review_product_json["attribute"]=total_attribute
            output_list.append(review_product_json)
            
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(output_list, fp, indent=4)  
        
  
  
def match_attribute_by_similar_products(product_id_list,product_dict,review_text,attribute_field,attribute_logic )  :
    total_attribute_json_in_review={}
    for product_id in product_id_list:
        product_json=product_dict[product_id]
        total_attribute_json_in_review=match_attribute_by(review_text,product_json["Spec"],attribute_field,attribute_logic,total_attribute_json_in_review)
        # total_attribute_json_in_review.update(attribute_json_in_review)
    
    return  total_attribute_json_in_review 



class AmazonMatchExtractor(Extractor):
    def generate_per_review_attribute(self,args, prefix,prompt_text,attribute_key,candidate_attribute_list,is_constrained_beam_search,
                                      review_dataset_json,review_image_dir,total_attribute_value_candidate_list=None,
                                      total_confidence_score_list=None,mention=None,attribute_value=None,
                                      ocr_raw=None,gpt_context=None,chatgpt_context=None):
        
        review_tokens=word_tokenize(prompt_text.lower())
        pattern='[a-zA-Z]+[0-9]+\w*|[0-9]+\.[0-9]+'
        if args.attribute_logic=="numeral":
            numeral_in_review = re.findall(pattern, prompt_text)
        extracted_attribute_value_list=[]
        for attribute_value in candidate_attribute_list:
            if len(attribute_value)<=2 or   attribute_value   in stopwords.words('english')  :
                continue 
            attribute_value_tokens=word_tokenize(attribute_value.lower())
            if args.attribute_logic=="numeral":
                if is_numeral_match(attribute_value,numeral_in_review,pattern):
                    if not attribute_value.isdigit():
                        extracted_attribute_value_list.append(attribute_value)  
            elif is_sublist_in_list(attribute_value_tokens,review_tokens):
           
                extracted_attribute_value_list.append(attribute_value)  
           
        return None,None,extracted_attribute_value_list,[],[],[]
         

    
    


import argparse
def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',type=str,help=" ",default="mocheg2/test") #politifact_v3,mode3_latest_v5
    parser.add_argument('--out_path',type=str,help=" ",default="bestbuy/output/html")
    # parser.add_argument('--example_type',type=str,help=" ",default="verification")
    parser.add_argument('--mode',type=str,help=" ",default="val")
    parser.add_argument('--attribute_source',type=str,help=" ",default="similar")#gold
    parser.add_argument('--attribute_field',type=str,help=" ",default="all")
    
    parser.add_argument('--attribute_logic',type=str,help=" ",default="numeral")#exact
    parser.add_argument('--filter_by_attribute_field',type=str,help=" ",default="Brand|Color")#all
    # parser.add_argument('--generate_claim_level_report',type=str,help=" ",default="y")
    args = parser.parse_args()
    return args


   

if __name__ == '__main__':
    args = parser_args()
    data_path=args.data_path   
    mode=args.mode 
    # review_path = Path(
    #     f'/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v4/{mode}/bestbuy_review_2.3.16.29.5.1_fuse_score_title_0.6_max_image.json'
    # )
     
    # output_attribute_path = Path(
    #     f'/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v4/{mode}/bestbuy_review_2.3.16.29.6.1_attribute_by_{args.attribute_source}_{args.attribute_field}_{args.attribute_logic}_multiple_attribute_value_image_max.json'
    # )  
    # output_filtered_path =  f'/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v4/{mode}/bestbuy_review_2.3.16.29.7_filtered_{args.attribute_source}_{args.attribute_field}_{args.attribute_logic}_{args.filter_by_attribute_field}.json'
          
    match_attribute_main( args.data_path,args.out_path ,args.attribute_source ,args.attribute_field, args.attribute_logic)
    # filter_non_attribute_main(output_attribute_path,output_filtered_path,args.filter_by_attribute_field)
    # evaluate(output_filtered_path)
    # check_all()