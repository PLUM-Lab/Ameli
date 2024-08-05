import torch
from torch.utils.data import Dataset  
from torch.utils.data import DataLoader 
from pathlib import Path
import json
from tqdm import tqdm

def gen_top_10_id_and_gold_score(fused_score_dict,gold_id,score_field):
    gold_score =-1 
    score_dict={}
    for product_id_str,fused_score_json in fused_score_dict.items():
        if score_field in fused_score_json:
            
            score=fused_score_json[score_field]
        else:
            score=-1
        score_dict[product_id_str]= score
        if str(gold_id)==product_id_str:
            gold_score=score
    sorted_list=sorted(score_dict.items(), key=lambda x:x[1] ,reverse=True)
    return gold_score,sorted_list[9][0] 
def compute(query_path,score_field="bi_score"):
    review_json_path = Path(
        query_path
    )      
    score_diff_sum=0
    score_diff_list=[]
    with open(review_json_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        for review_dataset_json   in tqdm(new_crawled_products_url_json_array):
            review_id=review_dataset_json["review_id"]
            fused_score_dict=review_dataset_json["fused_score_dict"]
            gold_product_id=review_dataset_json["gold_entity_info"]["id"]
            gold_score,sorted_top_10_id=gen_top_10_id_and_gold_score(fused_score_dict,gold_product_id,score_field)
            top_10_score_json=fused_score_dict[sorted_top_10_id]
            if score_field in top_10_score_json:
                top_10_score=top_10_score_json[score_field]
            else:
                top_10_score=-1
            # if "bi_score" in list(fused_score_dict.values())[0]:
            #     sorted_fused_score_dicts = sorted(fused_score_dict.items(), key=lambda x:x[1]["bi_score"],reverse=True)
            #     gold_score,top_10_score=-1,-1
            #     for idx,(product_id_str,sorted_fused_score_json) in enumerate(sorted_fused_score_dicts):
            #         score=sorted_fused_score_json["bi_score"]
            #         if str(gold_product_id)==product_id_str:
            #             gold_score=score
            #         if idx==10:
            #             top_10_score=score
            if gold_score>top_10_score and top_10_score>-1:
                score_diff=gold_score-top_10_score 
            else:
                score_diff=0
            score_diff_list.append(score_diff)
            score_diff_sum+=score_diff
    print(score_diff_sum/len(score_diff_list),len(score_diff_list))
                
            
def compute_remaining(query_path,score_field="bi_score",score_threshold=0):
    review_json_path = Path(
        query_path
    )      
    score_diff_sum=0
    score_diff_list=[]
    remaining_num=0
    less_then_20_num=0
    with open(review_json_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        for review_dataset_json   in tqdm(new_crawled_products_url_json_array):
            cur_remain_num=0
            review_id=review_dataset_json["review_id"]
            fused_score_dict=review_dataset_json["fused_score_dict"]
            gold_product_id=review_dataset_json["gold_entity_info"]["id"]
            if str(gold_product_id) in fused_score_dict and score_field in fused_score_dict[str(gold_product_id)]:
                gold_score=fused_score_dict[str(gold_product_id)][score_field]
            else:
                gold_score=-1
            for product_id_str,fused_score_json in fused_score_dict.items():
                # if gold_score==-1:
                #     remaining_num+=1
                # else:
                if product_id_str in fused_score_dict and score_field in fused_score_dict[product_id_str]:
                    score=fused_score_dict[product_id_str][score_field]
                else:
                    score=-1
                if gold_score-score>score_threshold:
                    remaining_num+=1
                    cur_remain_num+=1
            if cur_remain_num<20:
                 
                less_then_20_num+=1
                 
        
            score_diff_list.append(remaining_num)    
            # score_diff_sum+=remaining_num
    print(remaining_num/len(score_diff_list),len(score_diff_list))            
    print(f"less than 20:{less_then_20_num} ")   
    
import random
def sample_50(product_json_dict,gold_product_id):
    gold_product_category=product_json_dict[gold_product_id]["product_category"]
    product_id_list=list(product_json_dict.keys())
    max_negative_num=50
    cur_negative_num=0
    candidate_id_list=[]
    sampled_product_id_list=random.sample(product_id_list, 5000)
    for idx,random_product_id in enumerate(sampled_product_id_list):
     
           
        product_json=product_json_dict[random_product_id] 
        cur_product_category=product_json["product_category"]
        if gold_product_category != cur_product_category:
            
            candidate_id_list.append(product_json["id"])
            cur_negative_num+=1
        if cur_negative_num>=max_negative_num:
            break 
    return candidate_id_list

def load_candidates_over_threshold(review_dataset_json, product_json_dict,score_field="bi_score",score_threshold=0):
    
    score_diff_sum=0
    score_diff_list=[]
    remaining_num=0
    remaining_list=[]
    less_then_20_num=0
    fused_score_dict=review_dataset_json["fused_score_dict"]
    gold_product_id=review_dataset_json["gold_entity_info"]["id"]
    if str(gold_product_id) in fused_score_dict and score_field in fused_score_dict[str(gold_product_id)]:
        gold_score=fused_score_dict[str(gold_product_id)][score_field]
    else:
        gold_score=-1
    for product_id_str,fused_score_json in fused_score_dict.items():
        # if gold_score==-1:
        #     remaining_num+=1
        # else:
        if product_id_str in fused_score_dict and score_field in fused_score_dict[product_id_str]:
            score=fused_score_dict[product_id_str][score_field]
        else:
            score=-1
        if gold_score-score>score_threshold:
            remaining_num+=1
            remaining_list.append(int(product_id_str))
    if len(remaining_list)<20:
        remaining_list=sample_50(product_json_dict,gold_product_id)
        less_then_20_num+=1
        remaining_num=len(remaining_list)
        # print(f"less than 20:{less_then_20_num} ")   
            
        
    score_diff_list.append(remaining_num)    
            # score_diff_sum+=remaining_num
             
    return remaining_list
 
# compute("/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/disambiguation/bestbuy_review_2.3.17.11.16_merge_top_100.json")            
# compute("/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/train/disambiguation/bestbuy_review_2.3.17.11.16_merge_top_100.json")            
# compute("/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/val/disambiguation/bestbuy_review_2.3.17.11.16_merge_top_100.json")            
       
# compute("/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/disambiguation/bestbuy_review_2.3.17.11.16_merge_top_100.json","image_score")            
# compute("/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/train/disambiguation/bestbuy_review_2.3.17.11.16_merge_top_100.json","image_score")            
# compute("/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/val/disambiguation/bestbuy_review_2.3.17.11.16_merge_top_100.json","image_score")            

# compute("/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/disambiguation/bestbuy_review_2.3.17.11.16_merge_top_100.json","fused_score")            
# compute("/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/train/disambiguation/bestbuy_review_2.3.17.11.16_merge_top_100.json","fused_score")            
# compute("/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/val/disambiguation/bestbuy_review_2.3.17.11.16_merge_top_100.json","fused_score")            

      
# compute_remaining("/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/disambiguation/bestbuy_review_2.3.17.11.16_merge_top_100.json",score_threshold=0.08)            
# compute_remaining("/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/train/disambiguation/bestbuy_review_2.3.17.11.16_merge_top_100.json",score_threshold=0.08)            
# compute_remaining("/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/val/disambiguation/bestbuy_review_2.3.17.11.16_merge_top_100.json",score_threshold=0.08)            
       
# compute_remaining("/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/disambiguation/bestbuy_review_2.3.17.11.16_merge_top_100.json","image_score",0.09)            
# compute_remaining("/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/train/disambiguation/bestbuy_review_2.3.17.11.16_merge_top_100.json","image_score",0.09)            
# compute_remaining("/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/val/disambiguation/bestbuy_review_2.3.17.11.16_merge_top_100.json","image_score",0.09)            