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
def is_nan_or_miss(json,key):
    if key  not in json:
        return True
    elif json[key] is None or len( json[key]) == 0:          
        return True 
    else:
        return False   


def gen_review_product_json_dict(test_dir):
    product_dataset_path = Path(test_dir)
    with open(product_dataset_path, 'r', encoding='utf-8') as fp:
        product_dataset_json_array = json.load(fp)
        review_product_json_dict=review_json_to_product_dict(product_dataset_json_array)
        
    with open(products_path_str, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        product_json_dict=product_json_to_product_id_dict(new_crawled_products_url_json_array)
    return review_product_json_dict,product_json_dict


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



import torch 
from torch import nn
sigmoid_method = nn.Sigmoid()
def cross_score_to_sigmoid(score):
     
    input=torch.tensor(score)
    score_sigmoid=sigmoid_method(input)
    return score_sigmoid.item()

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
        entity_id_list=entity_id_list_str.split("|")
        order_list=order_list_str.split("|")
        text_score_list=text_score_list_str.split("|")
        image_score_list=image_score_list_str.split("|")
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
                    if is_need_sigmoid:
                        text_score=cross_score_to_sigmoid(text_score)
                        image_score=cross_score_to_sigmoid(image_score)
                    fused_score_dict[str(int(float(entity_id)))]["disambiguation_text_score"]=text_score
                    fused_score_dict[str(int(float(entity_id)))]["disambiguation_image_score"]=image_score
        if keep_top_J is not None:
            ordered_candidate_id_list=ordered_candidate_id_list[:keep_top_J]
        # predict_candidate_id=int(float(entity_id_list[y_pred]))
        # gold_candidate_id=int(float(entity_id_list[y_true]))
        
        
        example_json=gen_example(ordered_candidate_id_list,product_json_dict,review_dataset_json,fused_score_dict,entity_id_list )
        output_list.append(example_json)
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(output_list, fp, indent=4)
    return ordered_candidate_id_list

def sample_50_from_testdataset():
    testdata_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v5/test/attribute/temp_bestbuy_review_2.3.16.29.12.2_114_gold_attribute_filter.json"
    output_products_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/multimodal_entity_linking/output/disambiguation/temp_bestbuy_review_2.3.16.29.12.2_114_gold_attribute_filter_sample_50.json"
    with open(testdata_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        sampled_test=random.sample(new_crawled_products_url_json_array, 50)
    example_list=[]
    for one_example in sampled_test:
        product_id=one_example["gold_entity_info"]["id"]
        one_example["original_fused_candidate_list"]=one_example["fused_candidate_list"][:100]
        candidate_list=one_example["fused_candidate_list"][:10]
        if product_id not in candidate_list:
            if len(candidate_list)>0:
                candidate_list.insert(1,product_id)
                reshuffed_id=1
            else:
                candidate_list.insert(0,product_id)
                reshuffed_id=0
        else:
            reshuffed_id=candidate_list.index(product_id)
        
        one_example["fused_candidate_list"]=candidate_list
        one_example["reshuffled_target_product_id_position"]=reshuffed_id
        example_list.append(one_example)
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(example_list, fp, indent=4)
        

def generate_train_example_by_review_id(review_id):
    with open(products_path_str, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        product_json_dict=product_json_to_product_id_dict(new_crawled_products_url_json_array)
        
    testdata_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v5/test/attribute/bestbuy_review_2.3.16.29.14_114_select_one_image_from_0_to_1000000.json"
    output_products_path=f"/home/menglong/workspace/code/multimodal/multimodal_entity_linking/multimodal_entity_linking/output/disambiguation/one_example_{review_id}.json"
    with open(testdata_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        
    example_list=[]
    for one_example in new_crawled_products_url_json_array:
        product_id=one_example["gold_entity_info"]["id"]
        cur_review_id=one_example["review_id"]
        if cur_review_id==review_id:
            
            
            candidate_list=one_example["fused_candidate_list"][:10]
            if product_id not in candidate_list:
                if len(candidate_list)>0:
                    candidate_list.insert(1,product_id)
                    reshuffed_id=1
                else:
                    candidate_list.insert(0,product_id)
                    reshuffed_id=0
            else:
                reshuffed_id=candidate_list.index(product_id)
             
            product_json=copy.deepcopy(product_json_dict[product_id])
            product_json["reviews"]=one_example
            product_json["fused_candidate_list"]=candidate_list
            product_json["reshuffled_target_product_id_position"]=reshuffed_id
            example_list.append(product_json)
            break
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(example_list, fp, indent=4)        
    similar_products_path = Path(
        f'/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v5/bestbuy_products_40000_3.4.16_all_text_image_similar.json'
    )
    review_image_dir="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v5/cleaned_review_images"
    product_image_dir="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v5/product_images"
    out_path="one_example"
    generate(  out_path,output_products_path , similar_products_path,product_image_dir,review_image_dir,
             generate_claim_level_report=True ,is_reshuffle=False,is_need_image=False ,is_tar=False,is_need_corpus=True,
             is_generate_report_for_human_evaluation=False,is_add_gold=False, is_add_gold_at_end=True ) 
    
    
    

def sample_50_error_from_testdataset():
    testdata_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/disambiguation/postprocess/bestbuy_review_2.3.17.11.21_fuse_disambiguation_score.json"
    output_products_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/multimodal_entity_linking/output/disambiguation/temp_bestbuy_review_2.3.17.11.21.1_50_error.json"
    error_list=[]
    with open(testdata_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        for review_dataset_json in new_crawled_products_url_json_array:
            gold_product_id=review_dataset_json["gold_entity_info"]["id"]
            predicted_product_id=review_dataset_json["fused_candidate_list"][0]
            if gold_product_id!=predicted_product_id:
                error_list.append(review_dataset_json)
        print(len(error_list),len(new_crawled_products_url_json_array),len(error_list)/len(new_crawled_products_url_json_array))    
        sampled_test=random.sample(error_list, 50)
    example_list=[]
    for one_example in sampled_test:
        gold_product_id=one_example["gold_entity_info"]["id"]
        one_example["original_fused_candidate_list"]=one_example["fused_candidate_list"][:100]
        candidate_list=one_example["fused_candidate_list"][:10]
        if gold_product_id not in candidate_list:
            if len(candidate_list)>0:
                candidate_list.insert(1,gold_product_id)
                reshuffed_id=1
            else:
                candidate_list.insert(0,gold_product_id)
                reshuffed_id=0
        else:
            reshuffed_id=candidate_list.index(gold_product_id)
        
        one_example["fused_candidate_list"]=candidate_list
        one_example["reshuffled_target_product_id_position"]=reshuffed_id
        example_list.append(one_example)
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(example_list, fp, indent=4)
    
    
def count_retrieval_error(output_products_path):
    # output_products_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/disambiguation/example/temp_bestbuy_review_2.3.17.11.20.1.1_test_sample_50_end_to_end.json"
    with open(output_products_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
    retrieval_error_num=0
    for one_example in new_crawled_products_url_json_array:
        product_id=one_example["gold_entity_info"]["id"]
        one_example["original_fused_candidate_list"]=one_example["fused_candidate_list"][:100]
        candidate_list=one_example["fused_candidate_list"][:10]
        if product_id not in candidate_list:
            is_retrieval_error="no_top_10"
            gold_label=10
            retrieval_error_num+=1
        else:
            is_retrieval_error="no_error"
            gold_label=candidate_list.index(product_id)
    print(retrieval_error_num)
            

def count_same_review_id(output_products_path):
    # output_products_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/disambiguation/example/temp_bestbuy_review_2.3.17.11.20.1.1_test_sample_50_end_to_end.json"
    with open(output_products_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
    retrieval_error_num=0
    review_id_set=set()
    for one_example in new_crawled_products_url_json_array:
        review_id=one_example["review_id"]
        if review_id in review_id_set:
            print(f"duplicated id:{review_id}")
        else:
            review_id_set.add(review_id)
         
    print(len(review_id_set))            
            
def sample_50_from_testdataset_for_human(testdata_path,output_products_path,number=50):
    
    
    with open(testdata_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        sampled_test=random.sample(new_crawled_products_url_json_array, number)
    example_list=[]
    for one_example in sampled_test:
        product_id=one_example["gold_entity_info"]["id"]
        one_example["original_fused_candidate_list"]=one_example["fused_candidate_list"][:100]
        candidate_list=one_example["fused_candidate_list"][:10]
        if product_id not in candidate_list:
            is_retrieval_error="no_top_10"
            gold_label=10
        else:
            is_retrieval_error="no_error"
            gold_label=candidate_list.index(product_id)
        
        one_example["fused_candidate_list"]=candidate_list
        one_example["target_product_id_position"]=gold_label
        one_example["is_retrieval_error"]=is_retrieval_error
        example_list.append(one_example)
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(example_list, fp, indent=4)    
    
import argparse
def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',type=str,help=" ",default="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/disambiguation/correct_retrieval_subset/bestbuy_review_2.3.17.11.19.20.2_correct_retrieval_subset_10_update_gold_attribute.json") #politifact_v3,mode3_latest_v5
    parser.add_argument('--prediction_path',type=str,help=" ",default="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/multimodal_entity_linking/output/runs/00711-/entity_link_reproduce_test0630_v3_subset.csv")
    parser.add_argument('--out_path',type=str,help=" ",default="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/disambiguation/bestbuy_review_2.3.17.12.1_add_disambiguation_score.json")
    # parser.add_argument('--example_type',type=str,help=" ",default="verification")
    parser.add_argument('--mode',type=str,help=" ",default="val")
    parser.add_argument('--review_id',type=int,help=" ",default=19022)
    # parser.add_argument('--generate_claim_level_report',type=str,help=" ",default="y")
    args = parser.parse_args()
    return args


  
  
  
  
if __name__ == '__main__':
    args = parser_args()
    is_post=False
     
    prediction_path=args.prediction_path
    # if is_post:
    #     post_process(prediction_path, args.data_path,10,True)
    #     disambiguation_result_postfix=""
    #     prediction_path=f"/home/menglong/workspace/code/multimodal/multimodal_entity_linking/multimodal_entity_linking/output/runs/00711-/filter_entity_link_reproduce_{disambiguation_result_postfix}.csv"
    
    # sample_50_error_from_testdataset()
    
    # testdata_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/disambiguation/correct_retrieval_subset/bestbuy_review_2.3.17.11.19.20.3.1_correct_retrieval_subset_10_update_gold_attribute_update_candidate.json"
    # output_products_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/disambiguation/example/temp_bestbuy_review_2.3.17.11.20.1.1_test_sample_20_disambiguation_v1.json"
    # sample_50_from_testdataset_for_human(testdata_path,output_products_path,number=20)
    # count_retrieval_error(output_products_path)
    output_products_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/disambiguation/example/temp_bestbuy_review_2.3.17.11.20.1.1_test_sample_50_v1_5.json"
    count_same_review_id(output_products_path)
    count_retrieval_error(output_products_path)
    
    
    
    
    # gen_error_candidate_list(prediction_path,args.data_path ,is_sample=False,is_error=False,is_need_sigmoid=True,output_products_path=args.out_path )
    # keep_top_J=5
    # gen_error_candidate_list(prediction_path,args.data_path , output_products_path=f'/home/menglong/workspace/code/multimodal/multimodal_entity_linking/multimodal_entity_linking/output/runs/01012-/bestbuy_review_2.3.16.29.15_114_ameli_{keep_top_J}.json',is_sample=False,is_error=False ,keep_top_J= keep_top_J )
    
    
    # generate_train_example_by_review_id(args.review_id)