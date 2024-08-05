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
from util.env_config import *
from retrieval.utils.retrieval_util import cross_score_to_sigmoid
def update_one_dict(precision_recall_at_k,similar_product_id_list_text,gold_product_id,recall_at_k_result):
    for k_val in  precision_recall_at_k:
        num_correct = 0
        if gold_product_id in similar_product_id_list_text[0:k_val]:
            num_correct += 1
        recall_at_k_result[k_val].append(num_correct  )
    return recall_at_k_result

def update_combined_dict(precision_recall_at_k,similar_product_id_list_text,similar_product_id_list_image,gold_product_id,text_recall_at_k):
    for k_val in  precision_recall_at_k:
        num_correct = 0
        if gold_product_id in similar_product_id_list_text[0:k_val] or gold_product_id in similar_product_id_list_image[0:k_val]:
            num_correct += 1
        text_recall_at_k[2*k_val].append(num_correct  )
    return text_recall_at_k
    
def compute_recall_for_image_or_text_separately(review_file_str,precision_recall_at_k):
    
    print(review_file_str.split("/")[-2])
    new_crawled_img_url_json_path = Path(
        review_file_str
        ) 
    
    combined_recall_at_k = {2*k: [] for k in precision_recall_at_k}
    image_recall_at_k = {k: [] for k in precision_recall_at_k}
    text_recall_at_k = {k: [] for k in precision_recall_at_k}
    with open(new_crawled_img_url_json_path, 'r', encoding='utf-8') as fp:
        product_dataset_json_array = json.load(fp)
        for product_dataset_json   in tqdm(product_dataset_json_array):
            gold_product_id=product_dataset_json["gold_entity_info"]["id"]
            review_json=product_dataset_json
            similar_product_id_list_text=product_dataset_json["product_id_with_similar_text_by_review"]
            similar_product_id_list_image=product_dataset_json["product_id_with_similar_image_by_review"]
            text_recall_at_k=update_one_dict(precision_recall_at_k,similar_product_id_list_text,gold_product_id,text_recall_at_k)
            image_recall_at_k=update_one_dict(precision_recall_at_k,similar_product_id_list_image,gold_product_id,image_recall_at_k)
            combined_recall_at_k=update_combined_dict(precision_recall_at_k,similar_product_id_list_text,similar_product_id_list_image,gold_product_id,combined_recall_at_k)
            
    for k in text_recall_at_k:
        text_recall_at_k[k] = np.mean(text_recall_at_k[k])
    for k in image_recall_at_k:
        image_recall_at_k[k] = np.mean(image_recall_at_k[k])
    for k in combined_recall_at_k:
        combined_recall_at_k[k] = np.mean(combined_recall_at_k[k])
    print(combined_recall_at_k)
    print(image_recall_at_k)
    print(text_recall_at_k)
            
def compute_for_one_json_list(product_dataset_json_array,data_name,precision_recall_at_k,key="fused_candidate_list"):
    combined_recall_at_k = {2*k: [] for k in precision_recall_at_k}
    image_recall_at_k = {k: [] for k in precision_recall_at_k}
    text_recall_at_k = {k: [] for k in precision_recall_at_k}
    for review_json   in product_dataset_json_array :#tqdm(
        gold_product_id=review_json["gold_entity_info"]["id"]
        # review_json=product_dataset_json
        if key in review_json :
            if key!="mention_to_product_id_list":
                similar_product_id_list_text=review_json[key]
            else:
                similar_product_id_list_text=review_json["mention_to_product_id_list"]
                if similar_product_id_list_text is None:
                    similar_product_id_list_text=[]
        else:
            return
        
        # similar_product_id_list_image=product_dataset_json["product_id_with_similar_image_by_review"]
        text_recall_at_k=update_one_dict(precision_recall_at_k,similar_product_id_list_text,gold_product_id,text_recall_at_k)
        # image_recall_at_k=update_one_dict(precision_recall_at_k,similar_product_id_list_image,gold_product_id,image_recall_at_k)
        # combined_recall_at_k=update_combined_dict(precision_recall_at_k,similar_product_id_list_text,similar_product_id_list_image,gold_product_id,combined_recall_at_k)
        
    for k in text_recall_at_k:
        text_recall_at_k[k] = np.mean(text_recall_at_k[k])*100
    # for k in image_recall_at_k:
    #     image_recall_at_k[k] = np.mean(image_recall_at_k[k])
    # for k in combined_recall_at_k:
    #     combined_recall_at_k[k] = np.mean(combined_recall_at_k[k])
    # print(combined_recall_at_k)
    # print(image_recall_at_k)
    print(f"{data_name}:{key}, {text_recall_at_k}")
    return text_recall_at_k[1],text_recall_at_k[1],text_recall_at_k[1]

def retrieval_recall_metric(review_file_str,precision_recall_at_k,key="fused_candidate_list"):
    
    data_name=f"{review_file_str.split('/')[-2]}"
    new_crawled_img_url_json_path = Path(
        review_file_str  
        ) 
    
    
    with open(new_crawled_img_url_json_path, 'r', encoding='utf-8') as fp:
        product_dataset_json_array = json.load(fp)
        compute_for_one_json_list(product_dataset_json_array,data_name,precision_recall_at_k,key)


 

def update_combined_dict_unbalanced(precision_recall_at_k,similar_product_id_list_text,similar_product_id_list_image,gold_product_id,text_recall_at_k,text_candidate_num_list):
    for k_val in  precision_recall_at_k:
        for text_candidate_num in text_candidate_num_list:
            num_correct = 0
            if gold_product_id in similar_product_id_list_text[0:int(text_candidate_num*k_val)] or gold_product_id in similar_product_id_list_image[0:int((1-text_candidate_num)*k_val)]:
                num_correct += 1
            text_recall_at_k[k_val][text_candidate_num].append(num_correct  )
    return text_recall_at_k


def search_recall(review_file_str,precision_recall_at_k,text_candidate_num_list):
    
    print(review_file_str.split("/")[-2])
    new_crawled_img_url_json_path = Path(
        review_file_str
        ) 
    combined_recall_at_k={k: [] for k in precision_recall_at_k}
    for k in precision_recall_at_k:
        combined_recall_at_k[k]={k: [] for k in text_candidate_num_list}
       
    with open(new_crawled_img_url_json_path, 'r', encoding='utf-8') as fp:
        product_dataset_json_array = json.load(fp)
        for product_dataset_json   in tqdm(product_dataset_json_array):
            gold_product_id=product_dataset_json["gold_entity_info"]["id"]
            review_json=product_dataset_json
            similar_product_id_list_text=product_dataset_json["product_id_with_similar_text_by_review"]
            similar_product_id_list_image=product_dataset_json["product_id_with_similar_image_by_review"]
            combined_recall_at_k=update_combined_dict_unbalanced(precision_recall_at_k,similar_product_id_list_text,similar_product_id_list_image,gold_product_id,combined_recall_at_k,text_candidate_num_list)
            
    max_score=-1
    max_setting=[]
    for k in combined_recall_at_k:
        for text_candidate_num in text_candidate_num_list:
            score=np.mean(combined_recall_at_k[k][text_candidate_num])
            if score>max_score:
                max_score= score
                max_setting=[k,text_candidate_num]
            combined_recall_at_k[k][text_candidate_num] = score
    print(combined_recall_at_k)
    print(f"max:{max_score},set:{max_setting}")
    return max_score,max_setting,combined_recall_at_k 
            
def check_balanced():
    precision_recall_at_k=[5,10,20,50,100,500,1000]
    train_dir=os.path.join(data_dir,'bestbuy/data/final/v4/train/bestbuy_review_2.3.16.27.1_similar_split.json')
    val_dir=os.path.join(data_dir,'bestbuy/data/final/v4/val/bestbuy_review_2.3.16.27.1_similar_split.json')
    test_dir=os.path.join(data_dir,'bestbuy/data/final/v4/test/bestbuy_review_2.3.16.27.1_similar_split.json') 
    compute_recall_for_image_or_text_separately(test_dir,precision_recall_at_k)
    compute_recall_for_image_or_text_separately(train_dir,precision_recall_at_k)
    compute_recall_for_image_or_text_separately(val_dir,precision_recall_at_k)
    
def hyper_search_test():
    train_dir=os.path.join(data_dir,'bestbuy/data/final/v4/train/bestbuy_review_2.3.16.27.1_similar_split.json')
    val_dir=os.path.join(data_dir,'bestbuy/data/final/v4/val/bestbuy_review_2.3.16.27.1_similar_split.json')
    test_dir=os.path.join(data_dir,'bestbuy/data/final/v4/test/bestbuy_review_2.3.16.27.1_similar_split.json') 
    precision_recall_at_k=[10,20,50,100]
    text_candidate_num_list=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1 ]
    search_recall(test_dir,precision_recall_at_k,text_candidate_num_list)
    
    
def hyper_search_val():
    train_dir=os.path.join(data_dir,'bestbuy/data/final/v4/train/bestbuy_review_2.3.16.27.1_similar_split.json')
    val_dir=os.path.join(data_dir,'bestbuy/data/final/v4/val/bestbuy_review_2.3.16.27.1_similar_split.json')
    test_dir=os.path.join(data_dir,'bestbuy/data/final/v4/test/bestbuy_review_2.3.16.27.1_similar_split.json') 
    precision_recall_at_k=[10,20,50]#100
    text_candidate_num_list=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1 ]
    for k in [10,20,50,100]:#
        print(k)
        print("-----------------------------")
        max_score,max_setting,combined_recall_at_k=search_recall(train_dir,[k],text_candidate_num_list)   
        k,text_ratio=max_setting
        max_score,max_setting,combined_recall_at_k=result_dict=search_recall(test_dir,[k],[text_ratio])     

def merge_image_dict(score_dict,image_score_dict,is_cross=False):
    for corpus_id, one_score_dict in image_score_dict.items():
        score=one_score_dict["score"]
        if is_cross:
            score=cross_score_to_sigmoid(score)
        if str(corpus_id) in  score_dict    :
            score_dict[str(corpus_id)]["image_score"]=score 
        elif corpus_id  in  score_dict    :
            score_dict[corpus_id]["image_score"]=score 
        else:
            score_dict[corpus_id]={"corpus_id":corpus_id}
            score_dict[corpus_id]["image_score"]=score 
    return score_dict 


 

def merge_text_dict(score_dict,title_score_dict,bi_key="bi_score",cross_key="cross_score"):
    #score_dict[corpus_id]={"idx":idx,"cross_score":cross_score,"bi_score":bi_score}
    for corpus_id, one_score_dict in title_score_dict.items():
        if "corss_score_sigmoid" in one_score_dict:
            cross_score=one_score_dict["cross_score_sigmoid"]
        else:
            cross_score=one_score_dict["cross_score"]
            if cross_score is not None:
                cross_score=cross_score_to_sigmoid(cross_score)
        bi_score=one_score_dict["bi_score"]
        if corpus_id in  score_dict:
            score_dict[corpus_id][cross_key]=cross_score 
            score_dict[corpus_id][bi_key]=bi_score 
        else:
            score_dict[corpus_id]={"corpus_id":corpus_id}
            score_dict[corpus_id][cross_key]=cross_score 
            score_dict[corpus_id][bi_key]=bi_score
    return score_dict 


def gen_sorted_candidate_list_for_three(score_dict,text_ratio,text_score_key,obtain_score_dict_num=None,gold_product_id=None,
                                        second_score_key ="image_score",
                              out_score_key="fused_score",third_score_key=None,image_ratio=None):
    third_ratio=1-text_ratio -image_ratio
    if third_ratio<0:
        return [],{}
    for corpus_id, one_score_dict in  score_dict.items():
        if text_score_key in one_score_dict:
            text_score=one_score_dict[text_score_key]
        else:
            text_score=-1
        if second_score_key in one_score_dict:
            image_score=one_score_dict[second_score_key]
        else:
            image_score=-1
        if third_score_key in one_score_dict:
            third_score=one_score_dict[third_score_key]
        else:
            third_score=-1
   
        fused_score=text_score*text_ratio+image_score*image_ratio +third_score*third_ratio
        one_score_dict[out_score_key]=fused_score
    score_dict_list=list(score_dict.values()) 
    sorted_score_dict_list= sorted(score_dict_list, key=lambda x: x[out_score_key], reverse=True)
    candidate_id_list=[]
 
    output_score_dict={}
    for idx,sorted_score_dict in enumerate(sorted_score_dict_list):
        corpus_id=int(sorted_score_dict["corpus_id"])
        candidate_id_list.append(corpus_id )
        if obtain_score_dict_num is not None and idx<obtain_score_dict_num:
            output_score_dict[corpus_id]=sorted_score_dict
    if gold_product_id is not None and str(gold_product_id) in score_dict:
        gold_score_dict=score_dict[str(gold_product_id)]
        if gold_product_id in candidate_id_list:
            gold_product_position_in_retrieval=candidate_id_list.index(gold_product_id)
        else:
            gold_product_position_in_retrieval=-1
        gold_score_dict["gold_product_position_in_retrieval"]=gold_product_position_in_retrieval
        output_score_dict[gold_product_id]=gold_score_dict
    return candidate_id_list[:5000]  ,output_score_dict  

def gen_sorted_candidate_list(score_dict,text_ratio,text_score_key,obtain_score_dict_num=None,gold_product_id=None,second_score_key ="image_score",
                              out_score_key="fused_score"):
    image_ratio=1-text_ratio 
    for corpus_id, one_score_dict in  score_dict.items():
        if text_score_key in one_score_dict:
            text_score=one_score_dict[text_score_key]
        else:
            text_score=-1
        if second_score_key in one_score_dict:
            image_score=one_score_dict[second_score_key]
        else:
            image_score=-1
   
        fused_score=text_score*text_ratio+image_score*image_ratio 
        one_score_dict[out_score_key]=fused_score
    score_dict_list=list(score_dict.values()) 
    sorted_score_dict_list= sorted(score_dict_list, key=lambda x: x[out_score_key], reverse=True)
    candidate_id_list=[]
 
    output_score_dict={}
    for idx,sorted_score_dict in enumerate(sorted_score_dict_list):
        corpus_id=int(sorted_score_dict["corpus_id"])
        candidate_id_list.append(corpus_id )
        if obtain_score_dict_num is not None and idx<obtain_score_dict_num:
            output_score_dict[corpus_id]=sorted_score_dict
    if gold_product_id is not None and str(gold_product_id) in score_dict:
        gold_score_dict=score_dict[str(gold_product_id)]
        if gold_product_id in candidate_id_list:
            gold_product_position_in_retrieval=candidate_id_list.index(gold_product_id)
        else:
            gold_product_position_in_retrieval=-1
        gold_score_dict["gold_product_position_in_retrieval"]=gold_product_position_in_retrieval
        output_score_dict[gold_product_id]=gold_score_dict
    return candidate_id_list[:5000]  ,output_score_dict 
        
def gen_score_dict(product_dataset_json,is_cross=False,use_original_fused_score_dict=False):
    score_dict = {}
    image_score_dict=product_dataset_json["image_score_dict"]
    title_score_dict=product_dataset_json["text_score_dict"]
    if "desc_score_dict" in product_dataset_json:
        desc_score_dict=product_dataset_json["desc_score_dict"]
    else:
        desc_score_dict={}
    if use_original_fused_score_dict:
        if "fused_score_dict" in product_dataset_json:
            score_dict=product_dataset_json["fused_score_dict"]
    score_dict=merge_image_dict(score_dict,image_score_dict,is_cross)
    score_dict=merge_text_dict(score_dict,title_score_dict)
    score_dict=merge_text_dict(score_dict,desc_score_dict,bi_key="desc_bi_score",cross_key="desc_cross_score")   
    return score_dict  
    
def search_one_file(review_file_str,precision_recall_at_k_space,text_score_key_list=["cross_score"],text_ratio_list=[0.5]):
    #product_id_with_similar_image_by_review, product_id_with_similar_image_by_review, product_id_with_similar_image_by_review
    #image_score_dict text_score_dict desc_score_dict
    print(review_file_str.split("/")[-2])
    new_crawled_img_url_json_path = Path(
        review_file_str
        ) 
    with open(new_crawled_img_url_json_path, 'r', encoding='utf-8') as fp:
        product_dataset_json_array = json.load(fp)
     
    return  search_one_json_list(product_dataset_json_array,precision_recall_at_k_space,text_score_key_list,text_ratio_list)

def search_one_json_list_for_three(product_dataset_json_array,precision_recall_at_k_space,text_score_key_list=["cross_score"],
                         text_ratio_list=[0.5],review_product_json_path_with_score=None,is_cross=False,second_score_key="image_score",
                         out_score_key="fused_score" ,use_original_fused_score_dict=False,third_score_key=None):
    #product_id_with_similar_image_by_review, product_id_with_similar_image_by_review, product_id_with_similar_image_by_review
    #image_score_dict text_score_dict desc_score_dict
 
    if review_product_json_path_with_score is not None:
        with open(review_product_json_path_with_score, 'r', encoding='utf-8') as fp:
            product_dataset_json_array = json.load(fp)
         
    searched_all_recall_at_k_result={}
    for text_score_key in text_score_key_list :
        searched_all_recall_at_k_result[text_score_key]={}
        for text_ratio in text_ratio_list:
            searched_all_recall_at_k_result[text_score_key][text_ratio]={}
            for image_ratio in text_ratio_list:
                searched_all_recall_at_k_result[text_score_key][text_ratio][image_ratio]={}
                for k in precision_recall_at_k_space:
                    searched_all_recall_at_k_result[text_score_key][text_ratio][image_ratio][k]=[]
            
  
    for product_dataset_json   in tqdm(product_dataset_json_array):
        gold_product_id=product_dataset_json["gold_entity_info"]["id"]
        score_dict=gen_score_dict(product_dataset_json,is_cross,use_original_fused_score_dict)
        for text_ratio in text_ratio_list:
            for image_ratio in text_ratio_list:
                for text_score_key in text_score_key_list: 
                    candidate_id_list,_=gen_sorted_candidate_list_for_three(score_dict,text_ratio,text_score_key,second_score_key=second_score_key,
                                                                out_score_key=out_score_key,third_score_key=third_score_key,image_ratio=image_ratio)
                    searched_all_recall_at_k_result[text_score_key][text_ratio][image_ratio]=update_one_dict(precision_recall_at_k_space,candidate_id_list,gold_product_id,searched_all_recall_at_k_result[text_score_key][text_ratio][image_ratio])
            
    max_setting_dict={}
    for k in precision_recall_at_k_space:
        max_score=-1
        max_setting=[]
        for text_score_key in text_score_key_list :
            for text_ratio in text_ratio_list:
                for image_ratio in text_ratio_list:
                    score=np.mean(searched_all_recall_at_k_result[text_score_key][text_ratio][image_ratio][k])
                    searched_all_recall_at_k_result[text_score_key][text_ratio][image_ratio][k]=score 
                    print(f"k:{k}, cur:{score},text_score_key:{text_score_key},text_ratio:{text_ratio},image_ratio:{image_ratio }")
                    if score>max_score:
                        max_score= score
                        max_setting=[text_score_key,text_ratio,image_ratio,k ]
                        max_setting_dict[k]=max_setting
        print(f"k:{k}, max:{max_score},set:{max_setting}")
    # print(searched_all_recall_at_k_result)
    return  searched_all_recall_at_k_result,max_setting_dict

def search_one_json_list(product_dataset_json_array,precision_recall_at_k_space,text_score_key_list=["cross_score"],
                         text_ratio_list=[0.5],review_product_json_path_with_score=None,is_cross=False,second_score_key="image_score",
                         out_score_key="fused_score" ,use_original_fused_score_dict=False):
    #product_id_with_similar_image_by_review, product_id_with_similar_image_by_review, product_id_with_similar_image_by_review
    #image_score_dict text_score_dict desc_score_dict
 
    if review_product_json_path_with_score is not None:
        with open(review_product_json_path_with_score, 'r', encoding='utf-8') as fp:
            product_dataset_json_array = json.load(fp)
         
    searched_all_recall_at_k_result={}
    for text_score_key in text_score_key_list :
        searched_all_recall_at_k_result[text_score_key]={}
        for text_ratio in text_ratio_list:
            searched_all_recall_at_k_result[text_score_key][text_ratio]={}
            for k in precision_recall_at_k_space:
                searched_all_recall_at_k_result[text_score_key][text_ratio][k]=[]
            
  
    for product_dataset_json   in tqdm(product_dataset_json_array):
        gold_product_id=product_dataset_json["gold_entity_info"]["id"]
        score_dict=gen_score_dict(product_dataset_json,is_cross,use_original_fused_score_dict)
        for text_ratio in text_ratio_list:
            for text_score_key in text_score_key_list: 
                candidate_id_list,_=gen_sorted_candidate_list(score_dict,text_ratio,text_score_key,second_score_key=second_score_key,out_score_key=out_score_key)
                searched_all_recall_at_k_result[text_score_key][text_ratio]=update_one_dict(precision_recall_at_k_space,candidate_id_list,gold_product_id,searched_all_recall_at_k_result[text_score_key][text_ratio])
            
    max_setting_dict={}
    for k in precision_recall_at_k_space:
        max_score=-1
        max_setting=[]
        for text_score_key in text_score_key_list :
            for text_ratio in text_ratio_list:
                score=np.mean(searched_all_recall_at_k_result[text_score_key][text_ratio][k])
                searched_all_recall_at_k_result[text_score_key][text_ratio][k]=score 
                print(f"k:{k}, cur:{score},set:{[text_score_key,text_ratio,k ]}")
                if score>max_score:
                    max_score= score
                    max_setting=[text_score_key,text_ratio,k ]
                    max_setting_dict[k]=max_setting
        print(f"k:{k}, max:{max_score},set:{max_setting}")
    # print(searched_all_recall_at_k_result)
    return  searched_all_recall_at_k_result,max_setting_dict


def search_one_json_list_for_four_scores(product_dataset_json_array,precision_recall_at_k_space,text_score_key_list=["cross_score"],
                         text_ratio_list=[0.5],review_product_json_path_with_score=None,is_cross=False ):
    #product_id_with_similar_image_by_review, product_id_with_similar_image_by_review, product_id_with_similar_image_by_review
    #image_score_dict text_score_dict desc_score_dict
 
    if review_product_json_path_with_score is not None:
        with open(review_product_json_path_with_score, 'r', encoding='utf-8') as fp:
            product_dataset_json_array = json.load(fp)
         
    searched_all_recall_at_k_result={}
    for text_score_key in text_score_key_list :
        searched_all_recall_at_k_result[text_score_key]={}
        for text_ratio in text_ratio_list:
            searched_all_recall_at_k_result[text_score_key][text_ratio]={}
            for k in precision_recall_at_k_space:
                searched_all_recall_at_k_result[text_score_key][text_ratio][k]=[]
            
  
    for product_dataset_json   in tqdm(product_dataset_json_array):
        gold_product_id=product_dataset_json["gold_entity_info"]["id"]
        score_dict=gen_score_dict(product_dataset_json,is_cross)
        for text_ratio in text_ratio_list:
            for text_score_key in text_score_key_list: 
                candidate_id_list,_=gen_sorted_candidate_list(score_dict,text_ratio,text_score_key)
                searched_all_recall_at_k_result[text_score_key][text_ratio]=update_one_dict(precision_recall_at_k_space,candidate_id_list,gold_product_id,searched_all_recall_at_k_result[text_score_key][text_ratio])
            
    max_setting_dict={}
    for k in precision_recall_at_k_space:
        max_score=-1
        max_setting=[]
        for text_score_key in text_score_key_list :
            for text_ratio in text_ratio_list:
                score=np.mean(searched_all_recall_at_k_result[text_score_key][text_ratio][k])
                searched_all_recall_at_k_result[text_score_key][text_ratio][k]=score 
                if score>max_score:
                    max_score= score
                    max_setting=[text_score_key,text_ratio,k ]
                    max_setting_dict[k]=max_setting
        print(f"k:{k}, max:{max_score},set:{max_setting}")
    # print(searched_all_recall_at_k_result)
    return  searched_all_recall_at_k_result,max_setting_dict



def compute_recall_w_fused_score(review_file_str, max_setting_dict,precision_recall_at_k_space,text_ratio_list ,text_score_key_list ):
    #product_id_with_similar_image_by_review, product_id_with_similar_image_by_review, product_id_with_similar_image_by_review
    #image_score_dict text_score_dict desc_score_dict
    print(review_file_str.split("/")[-2])
    new_crawled_img_url_json_path = Path(
        review_file_str
        ) 
    
    searched_all_recall_at_k_result={}
    for text_score_key in text_score_key_list :
        searched_all_recall_at_k_result[text_score_key]={}
        for text_ratio in text_ratio_list:
            searched_all_recall_at_k_result[text_score_key][text_ratio]={}
            for k in precision_recall_at_k_space:
                searched_all_recall_at_k_result[text_score_key][text_ratio][k]=[]
            
    with open(new_crawled_img_url_json_path, 'r', encoding='utf-8') as fp:
        product_dataset_json_array = json.load(fp)
        for product_dataset_json   in tqdm(product_dataset_json_array):
            gold_product_id=product_dataset_json["gold_entity_info"]["id"]
            score_dict=gen_score_dict(product_dataset_json)
            for k,max_setting in max_setting_dict.items():
                text_score_key,text_ratio,_ =max_setting
                candidate_id_list,_=gen_sorted_candidate_list(score_dict,text_ratio,text_score_key)
                searched_all_recall_at_k_result[text_score_key][text_ratio]=update_one_dict([k],candidate_id_list,gold_product_id,searched_all_recall_at_k_result[text_score_key][text_ratio])
            

    for k in precision_recall_at_k_space:
        for text_score_key in text_score_key_list :
            for text_ratio in text_ratio_list:
                if len(searched_all_recall_at_k_result[text_score_key][text_ratio][k])>0:
                    score=np.mean(searched_all_recall_at_k_result[text_score_key][text_ratio][k])
                    searched_all_recall_at_k_result[text_score_key][text_ratio][k]=score 
                    # if score>max_score:
                        
                    max_setting=[text_score_key,text_ratio,k ]
                        # max_setting_dict[k]=max_setting
                    print(f"k:{k}, max:{score},set:{max_setting}")
    # print(searched_all_recall_at_k_result)
    return  searched_all_recall_at_k_result,max_setting_dict
    
     
    
# check_balanced()
# hyper_search_val()

def hyper_search_val_score(val_dir,is_check_test=True):
    
    test_dir=os.path.join(data_dir,'bestbuy/data/final/v4/test/bestbuy_review_2.3.16.29.4_desc_text_similar_image_split.json') 
    precision_recall_at_k=[10,20,30,40,50,60,70,80,90,100,500,1000]#100
    text_ratio_list=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    text_score_key_list=["cross_score","bi_score","desc_cross_score","desc_bi_score"]
    searched_all_recall_at_k_result,max_setting_dict=search_one_file(val_dir,precision_recall_at_k,text_ratio_list=text_ratio_list,text_score_key_list=text_score_key_list)
    if is_check_test:
        compute_recall_w_fused_score(test_dir, max_setting_dict,precision_recall_at_k,text_ratio_list=text_ratio_list,text_score_key_list=text_score_key_list)
    


import argparse
def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',type=str,help=" ",default="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v4/val/bestbuy_review_2.3.16.29.4.1_desc_text_similar_image_max_mode_split.json") #politifact_v3,mode3_latest_v5
    parser.add_argument('--out_path',type=str,help=" ",default="bestbuy/output/html")
    # parser.add_argument('--example_type',type=str,help=" ",default="verification")
    parser.add_argument('--mode',type=str,help=" ",default="val")
    # parser.add_argument('--generate_claim_level_report',type=str,help=" ",default="y")
    args = parser.parse_args()
    return args


  
  
  
  
if __name__ == '__main__':
    args = parser_args()
    data_path=args.data_path    
    val_dir=os.path.join(data_dir,'bestbuy/data/final/v4/val/bestbuy_review_2.3.16.29.4_desc_text_similar_image_split.json')
    # hyper_search_val_score(data_path,False)
    # review_file_str="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v4/val/bestbuy_review_2.3.16.28.9.1_filter_attribute.json"#6,8,9
    precision_recall_at_k=[10,20,30,40,50,60,70,80,90,100,500,1000]#100
    text_ratio_list=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    text_score_key_list=["cross_score","bi_score","desc_cross_score","desc_bi_score"]
    # retrieval_recall_metric(data_path,precision_recall_at_k)
    # hyper_search_val
    text_ratio_list ,text_score_key_list
    text_score_key,text_ratio,k="desc_bi_score",0.3,10
    max_setting_dict={}
    max_setting_dict[k]=text_score_key,text_ratio,k
    max_setting_dict[1000]=text_score_key,text_ratio,1000
    # check_test(data_path, max_setting_dict,precision_recall_at_k,text_ratio_list ,text_score_key_list )
    
 
    retrieval_recall_metric(data_path,precision_recall_at_k,key="mention_to_product_id_list")
     
    # compute_recall_for_image_or_text_separately(data_path,precision_recall_at_k)