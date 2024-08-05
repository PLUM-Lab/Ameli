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
from disambiguation.data_util.inner_util import   review_json_to_product_dict
import re
from disambiguation.utils.post_process import post_process_for_json

from retrieval.organize.filter import filter_by_category 
 
 
 
 
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

def gen_new_score_dict(old_score_dict,candidate_id_list):
    new_score_dict={}
    for product_id, one_score_dict in old_score_dict.items():
        if int(product_id) in candidate_id_list:
            new_score_dict[product_id]=one_score_dict
    return new_score_dict
    
def merge_score_into_json(review_product_json_array,old_review_product_json_array_with_score_dict):
    out_list=[]
    for review_product_json   in tqdm(review_product_json_array):
     
        review_json=review_product_json
        review_id=review_json["review_id"]# 
        #merge desc_score_dict text_score_dict image_score_dict into the current json_list
        ## only keep score for current candidates
        old_review_product_json=old_review_product_json_array_with_score_dict[review_id]
        if "desc_score_dict" in old_review_product_json:
            review_product_json["desc_score_dict"]=gen_new_score_dict(old_review_product_json["desc_score_dict"],review_product_json["fused_candidate_list"])
        review_product_json["text_score_dict"]=gen_new_score_dict(old_review_product_json["text_score_dict"],review_product_json["fused_candidate_list"])
        review_product_json["image_score_dict"]=gen_new_score_dict(old_review_product_json["image_score_dict"],review_product_json["fused_candidate_list"])
        # compute fused id list 
        out_list.append(review_product_json)
    return out_list

def retrieval_re_search_recall_metric(review_file_str,review_file_str_with_score,
                                      output_products_path ,mode,old_product_dataset_json_array_with_score_dict=None,max_setting_dict_path=None):
    data_name=mode
    precision_recall_at_k_space=[1,10,20,30,40,50,60,70,80,90,100,500,1000]#100
    text_ratio_list=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    text_score_key_list=["bi_score"]#"cross_score",,"desc_cross_score","desc_bi_score"
    if old_product_dataset_json_array_with_score_dict is None:
        with open(review_file_str_with_score, 'r', encoding='utf-8') as fp:
            old_product_dataset_json_array_with_score = json.load(fp)
            old_product_dataset_json_array_with_score_dict=review_json_to_product_dict(old_product_dataset_json_array_with_score)
    new_crawled_img_url_json_path = Path(
        review_file_str
    ) 
    with open(new_crawled_img_url_json_path, 'r', encoding='utf-8') as fp:
        product_dataset_json_array = json.load(fp)
    review_product_json_array_with_score=merge_score_into_json(product_dataset_json_array,old_product_dataset_json_array_with_score_dict)
    if mode=="val":
        searched_all_recall_at_k_result,max_setting_dict=search_one_json_list(review_product_json_array_with_score,precision_recall_at_k_space,text_score_key_list,text_ratio_list)
        save_val_max_setting_dict(max_setting_dict_path,max_setting_dict,"text_image_category")
        # searched_all_recall_at_k_result,max_setting_dict=search_one_json_list(None,precision_recall_at_k_space,text_score_key_list,text_ratio_list,update_score_retrieval_path,is_cross)
    else:
        max_setting_dict=read_max_setting_dict( max_setting_dict_path,"text_image_category")
    text_score_key,text_ratio,k =get_value(max_setting_dict,10)
    review_product_json_array_after_fuse_score=gen_new_retrieved_dataset_for_one_json_list(review_product_json_array_with_score,output_products_path,
                                                                                           text_score_key,text_ratio,obtain_score_dict_num=10,
                                                                                           candidate_num=100,is_remove_prior_probability=True)
    compute_for_one_json_list(review_product_json_array_after_fuse_score, data_name,precision_recall_at_k_space)
    # review_product_json_array_after_fuse_score=gen_new_retrieved_dataset_for_one_json_list(review_product_json_array_with_score,output_products_path,
    #                                                                                        text_score_key,text_ratio,obtain_score_dict_num=10,
    #                                                                                        candidate_num=10,is_remove_prior_probability=True)
    

  
def check_all( ):
    mode="val"
    attribute_field="all"
    for attribute_source in ["similar"]:#,"gold"
        for attribute_logic in ["exact","numeral"]:
            for filter_by_attribute_field in ["Brand,Color","all"]:
                print(f"{attribute_source}, {attribute_logic},{filter_by_attribute_field} ---------------------------------------------")
                output_attribute_path = Path(
                    f'/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v4/{mode}/bestbuy_review_2.3.16.29.6.1_attribute_by_{attribute_source}_{attribute_field}_{attribute_logic}_multiple_attribute_value_image_max.json'
                ) 
                output_filtered_path=f"/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v4/{mode}/bestbuy_review_2.3.16.29.7.1_filtered_{attribute_source}_{attribute_field}_{attribute_logic}_{filter_by_attribute_field}_multiple_attribute_value_image_max.json"
                
                # match_attribute_main( review_path,output_attribute_path ,args.attribute_source ,args.attribute_field, args.attribute_logic)
                # filter_non_attribute_main(output_attribute_path,output_filtered_path, filter_by_attribute_field)
                evaluate(output_filtered_path)
                review_file_str_with_score_path=f"/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v4/{mode}/bestbuy_review_2.3.16.29.4.1_desc_text_similar_image_max_mode_split.json"
                final_output_review_path=f"/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v4/{mode}/bestbuy_review_2.3.16.29.8.1_rerank_{attribute_source}_{attribute_field}_{attribute_logic}_{filter_by_attribute_field}_multiple_attribute_value_image_max.json"
                retrieval_re_search_recall_metric(output_filtered_path,review_file_str_with_score_path,final_output_review_path )
  

def check_category():
    input_path = Path(
        f'/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v4/val/bestbuy_review_2.3.16.29.9.3_standard_add_4.1_mention_to_category_product_id_remove_score_dict.json'
    ) 
    review_file_str_with_score_path=f"/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v4/{mode}/bestbuy_review_2.3.16.29.4.1_desc_text_similar_image_max_mode_split.json"
    with open(review_file_str_with_score_path, 'r', encoding='utf-8') as fp:
        old_product_dataset_json_array_with_score = json.load(fp)
        old_product_dataset_json_array_with_score_dict=review_json_to_product_dict(old_product_dataset_json_array_with_score)
    precision_recall_at_k=[10000]#1,2,3,4,5,6,7,8,9,10 
    for k in precision_recall_at_k:
        print(f"{k}-----------------------------")
        output_filtered_path=f"/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v4/val/bestbuy_review_2.3.16.29.10.3_{k}_standard_filter_category.json"
        filter_by_category(input_path,output_filtered_path,k)
        evaluate(output_filtered_path,False)
        
        final_output_review_path=f"/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v4/{mode}/bestbuy_review_2.3.16.29.11.3_{k}_standard_rerank.json"
        
        retrieval_re_search_recall_metric(output_filtered_path,review_file_str_with_score_path,final_output_review_path ,old_product_dataset_json_array_with_score_dict)





def merge_all_for_review(new_retrieval_path,
                         original_retrieval_path,update_score_retrieval_path,
                         fields=["product_id_with_similar_image_by_review","image_score_dict"],level="product",is_save=True):
     
    # mode="train"
    new_crawled_img_url_json_path = Path(
        new_retrieval_path
    )        
    review_dataset_path = Path(original_retrieval_path)
    output_products_path = Path(
        update_score_retrieval_path
    )
 
    # fields=[ "overview_section","product_images_fnames","product_images","Spec"]
    
    #,"thumbnails", "thumbnail_paths",  "Spec"
    
    output_list=[]
    with open(new_crawled_img_url_json_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        new_crawled_products_url_json_dict=review_json_to_product_dict(new_crawled_products_url_json_array)
        
        with open(review_dataset_path, 'r', encoding='utf-8') as fp:
            product_dataset_json_array = json.load(fp)
            for product_dataset_json   in tqdm(product_dataset_json_array):
                review=product_dataset_json
                review_id=review["review_id"]
                 
                if review_id in new_crawled_products_url_json_dict:
                    new_crawled_products_url_json=new_crawled_products_url_json_dict[review_id]
                    for field in fields:
                        # if field not in product_dataset_json  or is_nan(product_dataset_json[field]) :
                            # if not is_nan_or_miss(new_crawled_products_url_json, field ):
                            if level=="review":
                                if field in new_crawled_products_url_json:
                                    product_dataset_json[field]=new_crawled_products_url_json[field]
                            else:
                                
                                product_dataset_json[field]=new_crawled_products_url_json[field]
                            
                    
                output_list.append(product_dataset_json)
    if is_save:
        with open(output_products_path, 'w', encoding='utf-8') as fp:
            json.dump(output_list, fp, indent=4)  
    return output_list

def save_val_max_setting_dict(max_setting_dict_path,one_max_setting_dict,key="text_image"):
    if os.path.exists(max_setting_dict_path):
        with open(max_setting_dict_path, 'r', encoding='utf-8') as fp:
            multiple_max_setting_dict = json.load(fp)
    else:
        multiple_max_setting_dict={}
    multiple_max_setting_dict[key]=one_max_setting_dict
    with open(max_setting_dict_path, 'w', encoding='utf-8') as fp:
        json.dump(multiple_max_setting_dict, fp, indent=4)  
     

def read_max_setting_dict( max_setting_dict_path,key="text_image"):
    with open(max_setting_dict_path, 'r', encoding='utf-8') as fp:
        multiple_max_setting_dict = json.load(fp)
        return multiple_max_setting_dict[key]    
    
    
def get_value(one_dict,int_key):
    if isinstance(list(one_dict.keys())[0],int):
        return one_dict[int_key]
    else:
        return one_dict[str(int_key)]

def save_top_10_fused_file(review_product_json_array_with_score ,real_top_10_fused_score_path,is_cross , mode,precision_recall_at_k_space,
                           max_setting_dict,real_top_100_fused_score_path):
    text_score_key,text_ratio,k =get_value(max_setting_dict,1)
    review_product_json_array_after_fuse_score=gen_new_retrieved_dataset_for_one_json_list(review_product_json_array_with_score
                                                                                           ,real_top_100_fused_score_path,text_score_key,
                                                                                           text_ratio,is_cross=is_cross,is_save=True,
                                                                                           candidate_num=100,obtain_score_dict_num=100,is_remove_prior_probability=True)
    
    compute_for_one_json_list(review_product_json_array_after_fuse_score, mode,precision_recall_at_k_space)
    gen_new_retrieved_dataset_for_one_json_list(review_product_json_array_with_score
                                                                        ,real_top_10_fused_score_path,text_score_key,
                                                                        text_ratio,is_cross=is_cross,is_save=True,
                                                                        candidate_num=10,obtain_score_dict_num=10,is_remove_prior_probability=True)
    
def try_new_retrieval(new_retrieval_path,version=0,is_cross=False,mode="test",original_retrieval_file_name="bestbuy_review_2.3.16.29.4_desc_text_similar_image_split.json",media="image"):
    original_retrieval_path=f"/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/{mode}/retrieval/{original_retrieval_file_name}"
    update_score_retrieval_path=f"/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/{mode}/retrieval/bestbuy_review_2.3.17.6_merge_image_score_{version}.json"
    real_top_10_fused_score_path=f"/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/{mode}/retrieval/bestbuy_review_2.3.17.7.1_top10_fused_score_{version}.json"
    real_top_100_fused_score_path=f"/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/{mode}/retrieval/bestbuy_review_2.3.17.7.1_top100_fused_score_{version}.json"
    max_setting_dict_path=f"/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/val/retrieval/max_setting_dict_{version}.json"
    

    precision_recall_at_k_space=[1,10,20,50,100,1000]#100ss
    text_ratio_list=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    text_score_key_list=["bi_score"]#"cross_score","desc_cross_score","desc_bi_score"
    if media=="image":
        fields=["product_id_with_similar_image_by_review","image_score_dict"]
    else:
        fields=["product_id_with_similar_text_by_review","text_score_dict"]
    review_product_json_array_with_score=merge_all_for_review(new_retrieval_path,original_retrieval_path,update_score_retrieval_path,is_save=False,fields=fields)
    if mode=="val":#
        print("begin to search")
        searched_all_recall_at_k_result,max_setting_dict=search_one_json_list(review_product_json_array_with_score,
                                                                            precision_recall_at_k_space,text_score_key_list,
                                                                            text_ratio_list,is_cross=is_cross )
        save_val_max_setting_dict(max_setting_dict_path,max_setting_dict)
    else:
        max_setting_dict=read_max_setting_dict( max_setting_dict_path)
    save_top_10_fused_file(review_product_json_array_with_score ,real_top_10_fused_score_path,is_cross , mode,
                           precision_recall_at_k_space,max_setting_dict,real_top_100_fused_score_path)
    
    
    

def try_new_image_retrieval(new_retrieval_path,version=0,is_cross=False,mode="test",original_retrieval_file_name="bestbuy_review_2.3.16.29.4_desc_text_similar_image_split.json",media="image"):
    original_retrieval_path=f"/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/{mode}/retrieval/{original_retrieval_file_name}"
    update_score_retrieval_path=f"/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/{mode}/retrieval/bestbuy_review_2.3.17.6_merge_image_score_{version}.json"
    fused_score_path=f"/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/{mode}/retrieval/bestbuy_review_2.3.17.7_fused_score_{version}.json"
    real_top_10_fused_score_path=f"/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/{mode}/retrieval/bestbuy_review_2.3.17.7.1_top10_fused_score_{version}.json"
    real_top_100_fused_score_path=f"/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/{mode}/retrieval/bestbuy_review_2.3.17.7.1_top100_fused_score_{version}.json"
    # mention_to_category_info_path=f"/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/{mode}/retrieval/bestbuy_review_2.3.16.29.9.3_standard_add_4.1_mention_to_category_product_id_remove_score_dict.json" 
    # fused_score_with_category_path=f"/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/{mode}/retrieval/bestbuy_review_2.3.16.29.9.1_10000_with_category_score_{version}.json"
    output_filtered_path=f"/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/{mode}/retrieval/bestbuy_review_2.3.17.8_10000_standard_filter_category_{version}.json"
    final_output_review_path=f"/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/{mode}/retrieval/bestbuy_review_2.3.17.8.1_standard_rerank_{version}.json"
    max_setting_dict_path=f"/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/val/retrieval/max_setting_dict_{version}.json"
    product_alias_base_category_product_id_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/mention_to_entity_statistics/bestbuy_products_40000_3.4.19.1_alias_base_to_category_product_id.json"
    product_alias_category_product_id_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/mention_to_entity_statistics/bestbuy_products_40000_3.4.19.2_alias_to_category_product_id.json"

    precision_recall_at_k_space=[1,10,20,50,100,1000]#100ss
    text_ratio_list=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    text_score_key_list=["bi_score"]#"cross_score","desc_cross_score","desc_bi_score"
    if media=="image":
        fields=["product_id_with_similar_image_by_review","image_score_dict"]
    else:
        fields=["product_id_with_similar_text_by_review","text_score_dict"]
    review_product_json_array_with_score=merge_all_for_review(new_retrieval_path,original_retrieval_path,update_score_retrieval_path,is_save=False,fields=fields)
    if mode=="val":#
        print("begin to search")
        searched_all_recall_at_k_result,max_setting_dict=search_one_json_list(review_product_json_array_with_score,
                                                                            precision_recall_at_k_space,text_score_key_list,
                                                                            text_ratio_list,is_cross=is_cross )
        save_val_max_setting_dict(max_setting_dict_path,max_setting_dict)
    else:
        max_setting_dict=read_max_setting_dict( max_setting_dict_path)
    
    
    save_top_10_fused_file(review_product_json_array_with_score ,real_top_10_fused_score_path,is_cross , mode,
                           precision_recall_at_k_space,max_setting_dict,real_top_100_fused_score_path)
    
    
    print("category")
    # if not os.path.exists(mention_to_category_info_path):
    #     gen_data_with_category(fused_score_path,product_alias_base_category_product_id_path,product_alias_category_product_id_path,mention_to_category_info_path)
    # merge_all_for_review(mention_to_category_info_path,fused_score_path,fused_score_with_category_path,fields=["mention_to_product_id_list","mention_to_category_list"],level="review")
    
    text_score_key,text_ratio,k =get_value(max_setting_dict,1000)
    review_product_json_array_after_fuse_score=gen_new_retrieved_dataset_for_one_json_list(review_product_json_array_with_score
                                                                                           ,fused_score_path,text_score_key,
                                                                                           text_ratio,is_cross=is_cross,is_save=True)
    # compute_for_one_json_list(review_product_json_array_after_fuse_score, mode,precision_recall_at_k_space)
    # 
    filter_by_category(fused_score_path,output_filtered_path)
    # evaluate(output_filtered_path,False,precision_recall_at_k=precision_recall_at_k_space)
    
    retrieval_re_search_recall_metric(output_filtered_path,update_score_retrieval_path,final_output_review_path,mode,
                                      max_setting_dict_path=max_setting_dict_path,
                                      old_product_dataset_json_array_with_score_dict=review_json_to_product_dict(review_product_json_array_with_score))


def gen_data_with_category(review_file_str,product_alias_base_category_product_id_path,product_alias_category_product_id_path,output_products_path):
    mode="standard"
    from retrieval.training.retrieval_by_probability import gen_candidate_list_by_mention_category_product_id_probability 
 
    gen_candidate_list_by_mention_category_product_id_probability(mode,review_file_str,product_alias_base_category_product_id_path,product_alias_category_product_id_path,output_products_path)

def new_image_retrieval(args):
     
    # mode=args.mode 
    # review_path=f'/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v4/{mode}/bestbuy_review_2.3.16.29.5.1_fuse_score_title_0.6_max_image.json'
    
     
    # output_attribute_path = Path(
    #     f'/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v4/{mode}/bestbuy_review_2.3.16.29.6.1_attribute_by_{args.attribute_source}_{args.attribute_field}_{args.attribute_logic}_multiple_attribute_value_image_max.json'
    # )  
    # output_filtered_path =  f'/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v4/{mode}/bestbuy_review_2.3.16.29.7_filtered_{args.attribute_source}_{args.attribute_field}_{args.attribute_logic}_{args.filter_by_attribute_field}.json'
          
    # match_attribute_main( review_path,output_attribute_path ,args.attribute_source ,args.attribute_field, args.attribute_logic)
    # filter_non_attribute_main(output_attribute_path,output_filtered_path,args.filter_by_attribute_field)
    # evaluate(output_filtered_path)
    # check_all()
    # check_category()
    if args.is_cross=="y":
        is_cross=True
    else:
        is_cross=False
    try_new_image_retrieval(args.data_path,args.version,is_cross,args.mode,args.original_retrieval_file_name,args.media)#"/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/val/bestbuy_review_2.3.16.28.1.1_fine_tune_image_encoder.json"
    
    
    
    
    
def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',type=str,help=" ",default="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/val/retrieval/bestbuy_review_2.3.17.3.2_27_finetuned.json")  
    parser.add_argument('--out_path',type=str,help=" ",default="bestbuy/output/html")
    parser.add_argument('--second_out_path',type=str,help=" ",default="bestbuy/output/html")
    # parser.add_argument('--example_type',type=str,help=" ",default="verification")
    parser.add_argument('--mode',type=str,help=" ",default="train")
    parser.add_argument('--media',type=str,help=" ",default="image")
    parser.add_argument('--attribute_source',type=str,help=" ",default="similar")#gold
    parser.add_argument('--attribute_field',type=str,help=" ",default="all")
    parser.add_argument('--version',type=str,help="000",default="temp")
    parser.add_argument('--is_cross',type=str,help=" ",default="n")
    parser.add_argument('--attribute_logic',type=str,help=" ",default="numeral")#exact
    parser.add_argument('--filter_by_attribute_field',type=str,help=" ",default="Brand,Color")#all
    parser.add_argument('--original_retrieval_file_name',type=str,help=" ",default="bestbuy_review_2.3.17.5.1.2_finetuned_26_text_title_no_reranker.json")  #bestbuy_review_2.3.17.5.1.1_finetuned_text_title_no_reranker.json
    # parser.add_argument('--generate_claim_level_report',type=str,help=" ",default="y")
    args = parser.parse_args()
    return args


  
if __name__ == '__main__':
    args = parser_args()
    if args.is_cross=="y":
        is_cross=True
    else:
        is_cross=False
    try_new_retrieval(args.data_path,args.version,is_cross,args.mode,args.original_retrieval_file_name,args.media )
    # merge_disambiguation_score(args)
    
    # post_process_for_json( args.out_path,10, args.second_out_path)