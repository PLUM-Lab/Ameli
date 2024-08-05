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
from retrieval.organize.score.scorer_helper import compute_for_one_json_list, gen_score_dict, gen_sorted_candidate_list, search_one_json_list, search_one_json_list_for_three
from retrieval.organize.gen_fused_entity_list_by_image_text_ratio import gen_new_retrieved_dataset_for_one_json_list
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
from disambiguation.utils.post_process import post_process
from util.common_util import product_json_to_product_id_dict, review_json_to_product_dict     
from util.env_config import * 
from tqdm import tqdm
from ast import literal_eval

from util.visualize.create_example.generate_html import generate
 

def gen_review_product_json_dict(test_dir):
    product_dataset_path = Path(test_dir)
    with open(product_dataset_path, 'r', encoding='utf-8') as fp:
        product_dataset_json_array = json.load(fp)
        review_product_json_dict=review_json_to_product_dict(product_dataset_json_array)
        
    with open(products_path_str, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        product_json_dict=product_json_to_product_id_dict(new_crawled_products_url_json_array)
    return review_product_json_dict,product_json_dict
  
def get_value(one_dict,int_key):
    if isinstance(list(one_dict.keys())[0],int):
        return one_dict[int_key]
    else:
        return one_dict[str(int_key)]


def search_disambiguation_score(fused_score_data_path,search_retrieval_score_field,search_disambiguation_score_field,third_score_key):
    precision_recall_at_k_space=[1]#100ss
    # text_ratio_list=[ 1]
    # search_one_json_list(None,precision_recall_at_k_space,text_score_key_list=["fused_score"],
    #                      text_ratio_list=text_ratio_list,review_product_json_path_with_score=args.data_path,is_cross=False
    #                      ,second_score_key="disambiguation_text_score",out_score_key="temp_fused_score",use_original_fused_score_dict=True )
    second_score_key=search_disambiguation_score_field#"disambiguation_text_score"
     
    text_ratio_list=[ 0.1,0.2,0.3, 0.4, 0.5, 0.6, 0.7,0.8] #
    if third_score_key is not None:
        _,max_setting_dict=search_one_json_list_for_three(None,precision_recall_at_k_space,text_score_key_list=[search_retrieval_score_field],
                         text_ratio_list=text_ratio_list,review_product_json_path_with_score= fused_score_data_path,is_cross=False
                         ,second_score_key=second_score_key,out_score_key="temp_fused_score",
                         use_original_fused_score_dict=True ,third_score_key=third_score_key)
        text_score_key,text_ratio,image_ratio,k =get_value(max_setting_dict,1)
    else:
        _,max_setting_dict=search_one_json_list(None,precision_recall_at_k_space,text_score_key_list=[search_retrieval_score_field],
                         text_ratio_list=text_ratio_list,review_product_json_path_with_score= fused_score_data_path,is_cross=False
                         ,second_score_key=second_score_key,out_score_key="temp_fused_score",use_original_fused_score_dict=True )
        image_ratio=0
        text_score_key,text_ratio,k =get_value(max_setting_dict,1)
    
    return text_ratio,image_ratio
    
        
def merge_disambiguation_score(args):
    precision_recall_at_k_space=[1]#100ss
    # text_ratio_list=[ 1]
    # search_one_json_list(None,precision_recall_at_k_space,text_score_key_list=["fused_score"],
    #                      text_ratio_list=text_ratio_list,review_product_json_path_with_score=args.data_path,is_cross=False
    #                      ,second_score_key="disambiguation_text_score",out_score_key="temp_fused_score",use_original_fused_score_dict=True )
    second_score_key="disambiguation_text_score"
    out_score_key="disambiguation_fused_score"
    text_ratio_list=[ 0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75 ]#[0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]
    _,max_setting_dict=search_one_json_list(None,precision_recall_at_k_space,text_score_key_list=["fused_score"],
                         text_ratio_list=text_ratio_list,review_product_json_path_with_score=args.fused_score_data_path,is_cross=False
                         ,second_score_key="disambiguation_text_score",out_score_key="temp_fused_score",use_original_fused_score_dict=True )
    text_score_key,text_ratio,k =get_value(max_setting_dict,1)
    
    with open(args.fused_score_data_path, 'r', encoding='utf-8') as fp:
        review_product_json_array_with_score = json.load(fp)
    review_product_json_array_after_fuse_score=gen_new_retrieved_dataset_for_one_json_list(review_product_json_array_with_score,args.out_path,
                                                                                           text_score_key,text_ratio,obtain_score_dict_num=100,
                                                                                           candidate_num=100,is_remove_prior_probability=True,
                                                                                           second_score_key="disambiguation_text_score",
                                                                                           out_score_key=out_score_key,use_original_fused_score_dict=True)
    compute_for_one_json_list(review_product_json_array_after_fuse_score, "test",precision_recall_at_k_space )
    post_process_for_json( args.out_path,10, args.second_out_path,filter_method="m1")
    
import torch 
from torch import nn
sigmoid_method = nn.Sigmoid()
def cross_score_to_sigmoid(score):
     
    input=torch.tensor(score)
    score_sigmoid=sigmoid_method(input)
    return score_sigmoid.item()

def gen_example(ordered_candidate_id_list,product_json_dict,review_dataset_json,fused_score_dict,entity_id_list):
    # product_id=review_dataset_json["gold_entity_info"]["id"]
    # product_json=copy.deepcopy(product_json_dict[product_id])
 
    # product_json["similar_product_id"]=ordered_candidate_id_list
    # product_json["product_id_with_similar_image"]=[]
    # product_json["fused_candidate_list"]=ordered_candidate_id_list
    # product_json["fused_score_dict"]=fused_score_dict
    product_json =copy.deepcopy(review_dataset_json) 
    # product_json["similar_product_id"]=ordered_candidate_id_list
    # product_json["product_id_with_similar_image"]=[]
    product_json["fused_candidate_list"]=ordered_candidate_id_list
    product_json["fused_score_dict"]=fused_score_dict
    product_json["entity_id_list"]=entity_id_list
    return product_json

def gen_error_candidate_list(corpus,test_dir,output_products_path='output/disambiguation/bestbuy_100_human_performance_added.json',is_sample=True,
                             is_error=True ,keep_top_J=None ,is_need_sigmoid=False):
 
    corpus_df = pd.read_csv(corpus ,encoding="utf8")  
    review_product_json_dict,product_json_dict=gen_review_product_json_dict(test_dir)
    new_corpus_df=corpus_df
    if is_error:
        new_corpus_df=corpus_df[corpus_df.apply(lambda x: x['predict_entity_position'] != x['gold_entity_position'], axis = 1)]
        # new_corpus_df=new_corpus_df[new_corpus_df.apply(lambda x: x['gold_entity_position'] != 10, axis = 1)]
    if is_sample:
        new_corpus_df=new_corpus_df.sample(n=100, random_state=1)
    output_list=[]
    for index, row in new_corpus_df.iterrows():
        query_id =row["query_id"]
        y_pred=row["predict_entity_position"]
        y_true=row["gold_entity_position"]
        # if is_error:
        #     if y_true==10:
        #         continue
        entity_id_list_str=row["entity_id_list"]
        order_list_str=row["order_list"]
        text_score_list_str=row["text_score"]
        image_score_list_str=row["image_score"]
        if "image_score_before_softmax" in row:
            image_score_before_softmax_list_str=row["image_score_before_softmax"]
            text_score_before_softmax_list_str=row["text_score_before_softmax"]
            
        else:
            image_score_before_softmax_list_str=row["image_score"]
            text_score_before_softmax_list_str=row["text_score"]
        
        entity_id_list=entity_id_list_str.split("|")
        order_list=order_list_str.split("|")
        text_score_list=text_score_list_str.split("|")
        image_score_list=image_score_list_str.split("|")
        image_score_before_softmax_list =image_score_before_softmax_list_str.split("|")
        text_score_before_softmax_list=text_score_before_softmax_list_str.split("|")
        # if query_id== 499:
        #     print("") 
        review_dataset_json=review_product_json_dict[query_id]
        fused_score_dict=review_dataset_json["fused_score_dict"]
        ordered_candidate_id_list=[]
        for  order in order_list:
            entity_id= entity_id_list[int(order)]
            if entity_id !="-1.0":
                ordered_candidate_id_list.append(int(float(entity_id)))
                if str(int(float(entity_id))) in fused_score_dict and len(text_score_list)>0 :
                    text_score=float(text_score_list[int(order)])
                    image_score=float(image_score_list[int(order)])
                    text_score_before_softmax=float(text_score_before_softmax_list[int(order)])
                    image_score_before_softmax=float(image_score_before_softmax_list[int(order)])
                    
                    if is_need_sigmoid:
                        text_score_before_softmax=cross_score_to_sigmoid(text_score)
          
                    fused_score_dict[str(int(float(entity_id)))]["disambiguation_text_score"]=text_score
                    fused_score_dict[str(int(float(entity_id)))]["disambiguation_image_score"]=image_score
                    fused_score_dict[str(int(float(entity_id)))]["disambiguation_text_score_before_softmax"]=text_score_before_softmax
                    fused_score_dict[str(int(float(entity_id)))]["disambiguation_image_score_before_softmax"]=image_score_before_softmax
        if keep_top_J is not None:
            ordered_candidate_id_list=ordered_candidate_id_list[:keep_top_J]
        # predict_candidate_id=int(float(entity_id_list[y_pred]))
        # gold_candidate_id=int(float(entity_id_list[y_true]))
        
        
        example_json=gen_example(ordered_candidate_id_list,product_json_dict,review_dataset_json,fused_score_dict,entity_id_list )
        output_list.append(example_json)
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(output_list, fp, indent=4)
    return ordered_candidate_id_list    

def search_best_retrieval_weight_by_loading_file(prediction_path, data_path , fused_score_data_path,search_retrieval_score_field
                                                 ,search_disambiguation_score_field,third_score_key ):
    gen_error_candidate_list(prediction_path, data_path ,is_sample=False,is_error=False,is_need_sigmoid=True,output_products_path= fused_score_data_path )
    best_retrieval_ratio,image_ratio=search_disambiguation_score(fused_score_data_path,search_retrieval_score_field,
                                                                 search_disambiguation_score_field,third_score_key)
    return best_retrieval_ratio,image_ratio



def gen_retrieval_field( use_image,use_text,search_retrieval_score_field_in_args):
    third_score_key=None
    if   use_image and  use_text:
        search_retrieval_score_field="fused_score"
        search_disambiguation_score_field="disambiguation_text_score_before_softmax"
        third_score_key="disambiguation_image_score"
    elif  use_image:
        search_retrieval_score_field="image_score"
        search_disambiguation_score_field="disambiguation_image_score"
    else:
        search_retrieval_score_field="bi_score"
        search_disambiguation_score_field="disambiguation_text_score_before_softmax"
    if  search_retrieval_score_field_in_args is not None:
        search_retrieval_score_field=search_retrieval_score_field_in_args
    return search_retrieval_score_field,search_disambiguation_score_field,third_score_key


def apply_best_retrieval_weight_without_filter(prediction_path,data_path,fused_score_data_path,best_retrieval_ratio,weighted_score_data_path,
                                filtered_data_path,use_image,use_text,search_retrieval_score_field_in_args,image_ratio ):
    search_retrieval_score_field,search_disambiguation_score_field,third_score_key=gen_retrieval_field( use_image,use_text,search_retrieval_score_field_in_args)
    gen_error_candidate_list(prediction_path, data_path ,is_sample=False,is_error=False,is_need_sigmoid=True,output_products_path= fused_score_data_path )
    precision_recall_at_k_space=[1]#100ss
    out_score_key="disambiguation_fused_score"
    text_score_key,text_ratio,k =search_retrieval_score_field,best_retrieval_ratio,1
    with open( fused_score_data_path, 'r', encoding='utf-8') as fp:
        review_product_json_array_with_score = json.load(fp)
    review_product_json_array_after_fuse_score=gen_new_retrieved_dataset_for_one_json_list(review_product_json_array_with_score,weighted_score_data_path,
                                                                                           text_score_key,text_ratio,obtain_score_dict_num=100,
                                                                                           candidate_num=100,is_remove_prior_probability=True,
                                                                                           second_score_key=search_disambiguation_score_field,
                                                                                           out_score_key=out_score_key,
                                                                                           use_original_fused_score_dict=True
                                                                                           ,image_ratio=image_ratio,
                                                                                           third_score_key=third_score_key)
    f1_score,pre, recall=compute_for_one_json_list(review_product_json_array_after_fuse_score, "test",precision_recall_at_k_space )
    return f1_score,pre, recall

def apply_best_retrieval_weight(prediction_path,data_path,fused_score_data_path,best_retrieval_ratio,weighted_score_data_path,
                                filtered_data_path,use_image,use_text,search_retrieval_score_field_in_args,image_ratio,
                                predicted_attribute_field="predicted_attribute",filter_method="m1"):
    apply_best_retrieval_weight_without_filter(prediction_path,data_path,fused_score_data_path,best_retrieval_ratio,weighted_score_data_path,
                                filtered_data_path,use_image,use_text,search_retrieval_score_field_in_args,image_ratio) 
    f1=post_process_for_json( weighted_score_data_path,10, filtered_data_path,filter_method=filter_method,predicted_attribute_field=predicted_attribute_field)
    return f1

def main1(args):
    # gen_error_candidate_list(prediction_path,args.data_path ,is_sample=False,is_error=False,is_need_sigmoid=True,output_products_path=args.fused_score_data_path )
    # merge_disambiguation_score(args)  
    use_image=True
    use_text=True 
    search_retrieval_score_field_in_args=None
    
    search_retrieval_score_field,search_disambiguation_score_field,third_score_key=gen_retrieval_field( use_image,use_text,search_retrieval_score_field_in_args)
    
    if args.compute_method=="test":
        search_best_retrieval_weight_by_loading_file(args.prediction_path, args.data_path,args.fused_score_data_path,search_retrieval_score_field
                                                    ,search_disambiguation_score_field,third_score_key)
    else:
        best_retrieval_ratio,image_ratio=search_best_retrieval_weight_by_loading_file(args.val_prediction_path, args.val_data_path,args.fused_score_data_path,search_retrieval_score_field
                                                    ,search_disambiguation_score_field,third_score_key)
        apply_best_retrieval_weight(args.prediction_path,args.data_path,args.fused_score_data_path,best_retrieval_ratio, 
                                             args.out_path,args.second_out_path, use_image,
                                              use_text, search_retrieval_score_field_in_args,image_ratio,predicted_attribute_field=args.predicted_attribute_field,
                                              filter_method=args.filter_method )
    # apply_best_retrieval_weight(prediction_path,args.data_path,args.fused_score_data_path,0.75,args.out_path,args.second_out_path,True,True,None,0)

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',type=str,help=" ",default="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/disambiguation/bestbuy_review_2.3.17.11.20.1_update_annotated_gold_attribute_update_candidate.json") 
    parser.add_argument('--prediction_path',type=str,help=" ",default="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/multimodal_entity_linking/output/runs/00711-/entity_link_reproduce_test0630_v3.csv")    
    parser.add_argument('--val_data_path',type=str,help=" ",default="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/val/disambiguation/bestbuy_review_2.3.17.11.19.16_update_gold_attribute.json") #politifact_v3,mode3_latest_v5
    parser.add_argument('--val_prediction_path',type=str,help=" ",default="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/multimodal_entity_linking/output/runs/00711-/val_entity_link_reproduce_test0630_v3.csv")
    parser.add_argument('--fused_score_data_path',type=str,help=" ",default="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/disambiguation/postprocess/bestbuy_review_2.3.17.11.20_add_disambiguation_score.json")
    parser.add_argument('--out_path',type=str,help=" ",default="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/disambiguation/postprocess/bestbuy_review_2.3.17.11.21_fuse_disambiguation_score.json")
    parser.add_argument('--second_out_path',type=str,help=" ",default="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/disambiguation/postprocess/bestbuy_review_2.3.17.11.22_filter_by_attribute.json")
    # parser.add_argument('--example_type',type=str,help=" ",default="verification")
    parser.add_argument('--filter_method',type=str,help=" ",default="m1")#exact
    parser.add_argument('--predicted_attribute_field',type=str,help=" ",default="predicted_attribute")#exact
    parser.add_argument('--mode',type=str,help=" ",default="train")
    parser.add_argument('--process_type',type=str,help=" ",default="v2vt")
    parser.add_argument('--compute_method',type=str,help=" ",default="val_test")#test
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
    
    # main1(args)
    if args.process_type=="v2vt":
        gen_error_candidate_list(args.prediction_path, args.data_path ,is_sample=False,is_error=False,is_need_sigmoid=True,
                                output_products_path= args.fused_score_data_path,keep_top_J=5 )