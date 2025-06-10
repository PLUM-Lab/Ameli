from numpy import product
import pandas as pd
import requests
import os
from pathlib import Path
from typing import Dict
from tqdm import tqdm
from ast import literal_eval

import requests
import json
from typing import Dict
from pathlib import Path
from time import time
from tqdm import tqdm
from lxml.html import fromstring 
import concurrent.futures
import logging
import spacy 
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize 
import nltk 
nlp = spacy.load("en_core_web_lg")



def get_statistic(input_products_path,output_products_path,category_csv_path,product_id_csv_path,output_alias_to_product_id_category_list_path):
    product_alias_base_dict={}
    alias_category_statistic={}
    alias_product_id_statistic={}
    with open(input_products_path, 'r', encoding='utf-8') as fp:
        product_dataset_json_array = json.load(fp)
        for product_dataset_json   in tqdm(product_dataset_json_array):
            product_id=product_dataset_json["id"]
            product_category=product_dataset_json["product_category"]
            product_alias_dict=product_dataset_json["displayed_review"]["mention_candidate_list_before_review_match"]
            for product_alias_root,product_alias_list in product_alias_dict.items():
                for product_alias in product_alias_list:
                    product_context_doc = nlp(product_alias)
                    product_alias_base=product_context_doc[:].lemma_.lower()
                    if product_alias_base in product_alias_base_dict:
                        product_alias_base_dict[product_alias_base].append({"product_category":product_category,"product_id":product_id,"product_alias":product_alias})
                    else:
                        product_alias_base_dict[product_alias_base]=[{"product_category":product_category,"product_id":product_id,"product_alias":product_alias}]
                    if product_alias_base not in alias_category_statistic:
                        alias_category_statistic[product_alias_base]={}
                    if product_category not in alias_category_statistic[product_alias_base]:
                        alias_category_statistic[product_alias_base][product_category]=0
                    alias_category_statistic[product_alias_base][product_category]+=1
                    if product_alias_base not in alias_product_id_statistic:
                        alias_product_id_statistic[product_alias_base]={}
                    if product_id not in alias_product_id_statistic[product_alias_base]:
                        alias_product_id_statistic[product_alias_base][product_id]=0
                    alias_product_id_statistic[product_alias_base][product_id]+=1
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(product_alias_base_dict, fp, indent=4)  
        
    save_statistic(alias_category_statistic,alias_product_id_statistic,category_csv_path,product_id_csv_path,output_alias_to_product_id_category_list_path)


def gen_sorted_key_by_value(category_statistic):
    sorted_dict=sorted(category_statistic.items(), key=lambda kv:  (kv[1], kv[0]),reverse=True)
    sorted_keys=[sorted_dict_item[0] for sorted_dict_item in sorted_dict]
    return sorted_keys

def save_statistic(alias_base_category_statistic,alias_base_product_id_statistic,alias_category_statistic,alias_product_id_statistic
                   ,category_csv_path,product_id_csv_path,output_alias_base_path,output_alias_path):
    report_list=[]
    alias_category_product_id_json ={}
    for product_alias_base, category_statistic in alias_base_category_statistic.items():
        for category, number in category_statistic.items():
            report_list.append([product_alias_base,category,number])
        sorted_keys=gen_sorted_key_by_value(category_statistic)
        alias_category_product_id_json[product_alias_base]={"category":sorted_keys}
        
    df = pd.DataFrame(report_list, columns =['product_alias_base', 'category','number' ])
    df.to_csv(category_csv_path,index=False)  
            
    report_list=[]
    for product_alias_base, product_id_statistic in alias_base_product_id_statistic.items():
        for product_id, number in product_id_statistic.items():
            report_list.append([product_alias_base,product_id,number])
        sorted_keys=gen_sorted_key_by_value(product_id_statistic)
        if product_alias_base not in alias_category_product_id_json:
            alias_category_product_id_json[product_alias_base]={}
            
        alias_category_product_id_json[product_alias_base]["product_id"]=sorted_keys 
         
            
    df = pd.DataFrame(report_list, columns =['product_alias_base', 'product_id','number' ])
    df.to_csv(product_id_csv_path,index=False)  
    
    with open(output_alias_base_path, 'w', encoding='utf-8') as fp:
        json.dump(alias_category_product_id_json, fp, indent=4) 
    
    report_list=[]
    alias_category_product_id_json ={}
    for product_alias_base, category_statistic in alias_category_statistic.items():
         
        sorted_keys=gen_sorted_key_by_value(category_statistic)
        alias_category_product_id_json[product_alias_base]={"category":sorted_keys}
         
 
    for product_alias_base, product_id_statistic in alias_product_id_statistic.items():
      
        sorted_keys=gen_sorted_key_by_value(product_id_statistic)
        if product_alias_base not in alias_category_product_id_json:
            alias_category_product_id_json[product_alias_base]={}
            
        alias_category_product_id_json[product_alias_base]["product_id"]=sorted_keys 
         
  
    with open(output_alias_path, 'w', encoding='utf-8') as fp:
        json.dump(alias_category_product_id_json, fp, indent=4) 
    
    

    
def get_statistic_from_product_alias_dict(input_product_with_only_alias_path,category_csv_path,
                                          product_id_csv_path,output_products_path,output_alias_path):
    #1. get #product_context_doc_base:{"product_category":"","product_id":"","product_alias":""}
 
    alias_base_category_statistic={}
    alias_base_product_id_statistic={}
    alias_category_statistic={}
    alias_product_id_statistic={}
 
    with open(input_product_with_only_alias_path, 'r', encoding='utf-8') as fp:
        product_alias_base_dict = json.load(fp)
        for product_alias_base,product_dict_list   in tqdm(product_alias_base_dict.items()):
            for product_dict in product_dict_list :
                product_id=product_dict["product_id"]
                product_category=product_dict["product_category"]
                product_alias =product_dict["product_alias"].lower()
                if product_alias_base not in alias_base_category_statistic:
                    alias_base_category_statistic[product_alias_base]={}
                if product_category not in alias_base_category_statistic[product_alias_base]:
                    alias_base_category_statistic[product_alias_base][product_category]=0
                alias_base_category_statistic[product_alias_base][product_category]+=1
                if product_alias_base not in alias_base_product_id_statistic:
                    alias_base_product_id_statistic[product_alias_base]={}
                if product_id not in alias_base_product_id_statistic[product_alias_base]:
                    alias_base_product_id_statistic[product_alias_base][product_id]=0
                alias_base_product_id_statistic[product_alias_base][product_id]+=1
                
                if product_alias  not in alias_category_statistic:
                    alias_category_statistic[product_alias]={}
                if product_category not in alias_category_statistic[product_alias]:
                    alias_category_statistic[product_alias][product_category]=0
                alias_category_statistic[product_alias][product_category]+=1
                if product_alias not in alias_product_id_statistic:
                    alias_product_id_statistic[product_alias]={}
                if product_id not in alias_product_id_statistic[product_alias]:
                    alias_product_id_statistic[product_alias][product_id]=0
                alias_product_id_statistic[product_alias][product_id]+=1
    save_statistic(alias_base_category_statistic,alias_base_product_id_statistic,alias_category_statistic,alias_product_id_statistic
                   ,category_csv_path,product_id_csv_path,output_products_path,output_alias_path)
  
    
    #2. get #csv product_context_doc_base product_category num product_id num
    #3. use review mention to match the csv file  
    # review_mention[:].lemma_.lower()
    
    
def find_category_by_mention(review_mention,alias_category_product_id_json):
    review_mention_base=review_mention[:].lemma_.lower()
    if review_mention_base in alias_category_product_id_json:
        category_list=alias_category_product_id_json[review_mention_base]["category"]
        product_id_list=alias_category_product_id_json[review_mention_base]["product_id"] 
    else:
        print(f"error: miss product_alias_base {review_mention_base}")
    return category_list,product_id_list
    
    
def main():
    input_products_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/mention_to_entity_statistics/bestbuy_products_40000_3.4.16.2.1_fix_product_alias_v38_from_0_to_5000000000.json"
    output_products_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/mention_to_entity_statistics/bestbuy_products_40000_3.4.16.3.1_only_product_alias.json"
    category_csv_path= "/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/mention_to_entity_statistics/product_alias_base_to_product_category_statistic_1.csv"
    product_id_csv_path ="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/mention_to_entity_statistics/product_alias_base_to_product_id_statistic_1.csv"
    output_alias_base_to_product_id_category_list_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/mention_to_entity_statistics/bestbuy_products_40000_3.4.16.4.1_alias_base_to_category_product_id.json"
    output_alias_to_product_id_category_list_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/mention_to_entity_statistics/bestbuy_products_40000_3.4.16.4.2_alias_to_category_product_id.json"

    # get_statistic(input_products_path,output_products_path,category_csv_path,product_id_csv_path,output_alias_to_product_id_category_list_path)
    get_statistic_from_product_alias_dict( output_products_path,category_csv_path,product_id_csv_path,output_alias_base_to_product_id_category_list_path,output_alias_to_product_id_category_list_path)
     

    
main()