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
import random
from disambiguation.data_util.inner_util import gen_gold_attribute, spec_to_json_attribute
import numpy as np 
from util.common_util import json_to_product_id_dict
from util.visualize.create_example.generate_html import generate 
import torch
def sample_100_product():
    product_dataset_path = Path('/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v5/bestbuy_products_40000_3.4.17_remove_empty_image.json')
    cluster_product_id_dict={}
    sampled_product_id_list=[]
    with open(product_dataset_path, 'r', encoding='utf-8') as fp:
        product_dataset_json_array = json.load(fp)
        sampled_products=random.sample(product_dataset_json_array,300)
        for product_json in sampled_products:
            temp_attributes=gen_gold_attribute(product_json["Spec"],product_json["product_name"],is_list=False,section_list=None)
            attributes=copy.deepcopy(temp_attributes)
            product_id=product_json["id"]
            if "Brand" in attributes:
                
                brand=attributes["Brand"] 
                if brand not in cluster_product_id_dict: 
                    cluster_product_id_dict[brand]=[]
                cluster_product_id_dict[brand].append(product_id)
                if len(cluster_product_id_dict[brand])>1:
                    print(f"{brand}")
            sampled_product_id_list.append(product_id)
    new_y={}
    for key, value in cluster_product_id_dict.items():
        if len(value)>1:
            new_y[key]=value
    print(len(new_y) )
    torch.save({"product_id":sampled_product_id_list,"cluster_by_brand":new_y},"/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/example/cluster_data.npy")
    
    
def create_example_for_human_annotation():
    
    a = torch.load("/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/example/cluster_data.npy")    
    cluster=a["cluster_by_brand"] 
    new_cluster={}
    for key, value in cluster.items():
        if len(value)>1:
            new_cluster[key]=value
    
    print(new_cluster)
    test_dir="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v5/test/bestbuy_review_2.3.16.27.1_similar_split.json"
    output_products_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/example/cluster_review.json"
    output_list=[]
    with open(test_dir, 'r', encoding='utf-8') as fp:
        product_dataset_json_array = json.load(fp)
        sampled_review_product_dataset_json_array=random.sample(product_dataset_json_array,len(new_cluster))
        for review_product_json ,cluster_product_id  in  zip(sampled_review_product_dataset_json_array,new_cluster.values())  :#tqdm(
            review_product_json["fused_candidate_list"]=cluster_product_id
            output_list.append(review_product_json)
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(output_list, fp, indent=4) 
        
    similar_products_path = Path(
        f'/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v5/bestbuy_products_40000_3.4.16_all_text_image_similar.json'
    )
    review_image_dir="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v5/cleaned_review_images"
    product_image_dir="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v5/product_images"
    out_path="cluster_analysis"
    generate(  out_path,output_products_path , similar_products_path,product_image_dir,review_image_dir,
             generate_claim_level_report=True ,is_reshuffle=False,is_need_image=False ,is_tar=False,is_need_corpus=True,
             is_generate_report_for_human_evaluation=False,is_add_gold=False, is_add_gold_at_end=True ) 



# sample_100_product()
create_example_for_human_annotation()        