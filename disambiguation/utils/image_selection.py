


import os 
import pandas as pd 
import pickle
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
from util.common_util import json_to_product_id_dict
from util.env_config import * 
import concurrent.futures
import random 
from util.env_config import * 
from tqdm import tqdm
from ast import literal_eval
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import torch
import logging 

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def gen_review_product_json_dict(test_dir):
    product_dataset_path = Path(test_dir)
    with open(product_dataset_path, 'r', encoding='utf-8') as fp:
        review_product_json_list = json.load(fp)
        
        
    with open(products_path_str, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        product_json_dict=json_to_product_id_dict(new_crawled_products_url_json_array)
    return review_product_json_list,product_json_dict

def find_best_pair(review_image_path_list,product_image_path_list,review_image_dir,product_image_dir,model,entity_img_path_list_list, img_emb_list):
    
        
    review_image_list=[Image.open(os.path.join(review_image_dir,review_image)) for review_image in review_image_path_list]
    product_image_embed_list=[]
    for product_image in product_image_path_list:
        product_image_path=os.path.join(product_image_dir,product_image)
        product_image_position=entity_img_path_list_list.index(product_image_path)
        product_image_embed=img_emb_list[product_image_position]
        product_image_embed_list.append(product_image_embed)
    product_image_embed_list=torch.stack(product_image_embed_list)
    review_image_embed_list=model.encode(review_image_list, convert_to_tensor=True, show_progress_bar=True )
    # product_image_embed_list=model.encode(product_image_list, convert_to_tensor=True, show_progress_bar=True )
    product_image_embed_list=product_image_embed_list.to(review_image_embed_list.device)
    cos_scores = util.cos_sim(review_image_embed_list, product_image_embed_list)
    max_id=torch.argmax(cos_scores).item()
    num_per_row=len(product_image_path_list)
    review_image_relative_idx=max_id//num_per_row
    product_image_relative_idx=max_id%num_per_row
    review_image_relative_path= review_image_path_list[review_image_relative_idx] 
    product_image_relative_path= product_image_path_list[product_image_relative_idx] 
    return review_image_relative_path,product_image_relative_path,review_image_relative_idx,cos_scores
    
def select_review_image_and_gold_product_image(review_product_json,product_json_dict,model,entity_img_path_list_list, img_emb_list):
    
    review_image_path_list=review_product_json["review_image_path"]
    product_id=review_product_json["gold_entity_info"]["id"]
    product_json=product_json_dict[product_id]
    product_image_path_list=product_json["image_path"]
    review_image_relative_path,product_image_relative_path,_,_=find_best_pair(review_image_path_list,product_image_path_list,review_image_dir,
                                                                              product_image_dir,model,entity_img_path_list_list, img_emb_list)
    review_product_json["review_image_path_before_select"]=review_product_json["review_image_path"]
    review_product_json["review_image_path"]=[review_image_relative_path]
    review_product_json["review_special_product_info"]={}
    review_product_json["review_special_product_info"][product_id]={}
    review_product_json["review_special_product_info"][product_id]["image_path"]=[product_image_relative_path]
    return review_product_json
      


def select_image_for_product_candidate_based_on_review_image(review_product_json,product_json_dict,model):
    
    gold_product_id=review_product_json["gold_entity_info"]["id"]
    review_image_path_list=review_product_json["review_image_path"]
    candidate_product_id=review_product_json["fused_candidate_list"][:10]
    for product_id in candidate_product_id:
        if product_id !=gold_product_id:
            product_json=product_json_dict[product_id]
            product_image_path_list=product_json["image_path"] 
            _,product_image_relative_path,_,_=find_best_pair(review_image_path_list,product_image_path_list,review_image_dir,product_image_dir,model)
            review_product_json["review_special_product_info"][product_id]={}
            review_product_json["review_special_product_info"][product_id]["image_path"]=[product_image_relative_path]
    return review_product_json 

def select_one_image_based_on_similarity_for_train(review_path,out_path):
    model=  SentenceTransformer('clip-ViT-L-14')
    review_product_json_list,product_json_dict=gen_review_product_json_dict(review_path)
    for review_product_json in tqdm(review_product_json_list):
        if "review_special_product_info" not in review_product_json:
        
            review_product_json=select_one_image_for_one_review_product_json_for_train(review_product_json,product_json_dict,model)
         
    with open(out_path, 'w', encoding='utf-8') as fp:
        json.dump(review_product_json_list, fp, indent=4)
     

def select_one_image_for_one_review_product_json_for_train(review_product_json,product_json_dict,model,
                                    entity_img_path_list_list, img_emb_list):
    # if "review_special_product_info" in review_product_json:
    #     return review_product_json
    review_product_json=select_review_image_and_gold_product_image(review_product_json,product_json_dict,model,entity_img_path_list_list, img_emb_list)
    review_product_json=select_one_image_for_one_review_product_json_based_on_candidates(review_product_json,product_json_dict,model,entity_img_path_list_list, img_emb_list,is_allow_gold=False,mode="train")
    # review_product_json=select_image_for_product_candidate_based_on_review_image(review_product_json,product_json_dict,model)
    return review_product_json


# class SimilarityComparisor:
#     def __init__(self,score_matrix,best_review_relative_idx,max_score,product_image_path_list) -> None:
#         self.score_matrix=score_matrix
#         self.best_review_relative_idx=best_review_relative_idx 
#         self.max_score=max_score
#         self.product_image_path_list=product_image_path_list

def select_one_image_for_one_review_product_json_based_on_candidates(review_product_json,product_json_dict,model,
                                    entity_img_path_list_list, img_emb_list,is_allow_gold=True,mode="test"):
    if   mode in [ "test","mp_test"]:
        review_product_json["review_image_path"]=review_product_json["review_image_path_before_select"]
    gold_product_id=review_product_json["gold_entity_info"]["id"]
    review_image_path_list=review_product_json["review_image_path"]
    candidate_product_id=review_product_json["fused_candidate_list"][:100]
    # similarity_comparisor_list=[]
    total_product_image_len_list=[]
    total_product_image_path_list=[]
    candidate_product_id_to_check=[]
    for product_id in candidate_product_id:
        if not is_allow_gold and product_id==gold_product_id:
            continue
        else:
            candidate_product_id_to_check.append(product_id)
        product_json=product_json_dict[product_id]
        product_image_path_list=product_json["image_path"] 
        total_product_image_path_list.extend( product_image_path_list) 
        total_product_image_len_list.append(len(product_image_path_list))
    if len(candidate_product_id_to_check)>0:
        review_image_relative_path,_,review_image_relative_idx,cos_scores =find_best_pair(review_image_path_list,total_product_image_path_list,
                                                                                          review_image_dir,product_image_dir,model,entity_img_path_list_list, img_emb_list)
        if "review_image_path_before_select" not in review_product_json or mode=="test":
            review_product_json["review_image_path_before_select"]=review_product_json["review_image_path"]
            review_product_json["review_image_path"]=[review_image_relative_path]
            review_product_json["review_special_product_info"]={}
        product_image_similarity_scores=cos_scores[review_image_relative_idx]
        start_product_image_relative_position_in_total_product=0
        end_product_image_relative_position_in_total_product=0
        for idx,product_id in enumerate(candidate_product_id_to_check):
            
            end_product_image_relative_position_in_total_product+=total_product_image_len_list[idx]
            product_image_relative_idx_in_current_product=torch.argmax(product_image_similarity_scores[start_product_image_relative_position_in_total_product:end_product_image_relative_position_in_total_product]).item()
            product_image_relative_path=total_product_image_path_list[start_product_image_relative_position_in_total_product:end_product_image_relative_position_in_total_product][product_image_relative_idx_in_current_product]
            review_product_json["review_special_product_info"][product_id]={}
            review_product_json["review_special_product_info"][product_id]["image_path"]=[product_image_relative_path]
            start_product_image_relative_position_in_total_product+=total_product_image_len_list[idx]
    return review_product_json
def obtain_image_path_list(entity_dict):
    total_image_path_list=[]
    for _,product_json in entity_dict.items():
        image_path_list=product_json.image_path_list
        total_image_path_list.extend(image_path_list)
    return total_image_path_list

def mp_main(args):
    start_id=args.start_idx
    end_id=args.end_idx
    incomplete_products_path = Path(
        args.data_path
    )
    complete_products_path = Path(
        f"{args.out_path}_from_{start_id}_to_{end_id}.json"
    )
    incomplete_dict_list,product_json_dict=gen_review_product_json_dict(incomplete_products_path)
    emb_filename = 'corpus_image_embeddings.pkl'
    emb_dir=os.path.join("/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/embed/finetuned_27",emb_filename)
    with open(emb_dir, 'rb') as fIn:
        emb_file =  pickle.load(fIn) #torch.load(fIn,map_location={'cuda':'cpu'}   )#pickle.load(fIn) 
        entity_dict,entity_image_num_list,img_emb,entity_name_list,entity_img_path_list,entity_name_list_in_image_level=emb_file["entity_dict"],emb_file["entity_image_num_list"],emb_file["img_emb"],emb_file["entity_name_list"] ,emb_file["entity_img_path_list"],emb_file["entity_name_list_in_image_level"]
    entity_img_path_list=obtain_image_path_list(entity_dict)
    select_one_image_for_one_review_product_json_function=select_one_image_for_one_review_product_json_for_train if args.dataset in [ "train","val"] else select_one_image_for_one_review_product_json_based_on_candidates
    
    step_size =  args.step_size
    output_list = []
    print_ctr = 0
    result = []
    save_step=100
    world_size=torch.cuda.device_count()
    gpu_list=[i for i in range(world_size)]
    
    entity_img_path_list_list=[entity_img_path_list for i in range(step_size)]
    img_emb_list=[img_emb for i in range(step_size)]
    model_list=[  SentenceTransformer(args.image_model,device=torch.device('cuda',gpu_list[i%world_size]))  for i in range(step_size)]
    product_dict_list=[product_json_dict for i in range(step_size)] 
    
                                    
    for i in tqdm( range(0, len(incomplete_dict_list), step_size)):
        if i>=start_id :
            if   i<end_id:
                print(100 * '=')
                print(f'starting at the {print_ctr}th value')
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    try:
                        result = list(
                            tqdm( 
                                executor.map(
                                    select_one_image_for_one_review_product_json_function,
                                    incomplete_dict_list[i: i + step_size],
                                    product_dict_list,
                                    model_list ,
                                    entity_img_path_list_list,
                                    img_emb_list
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
    
     
import argparse
def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',type=str,help=" ",default=data_dir+"train/disambiguation/bestbuy_review_2.3.17.11.16_merge_top_100.json") 
    parser.add_argument('--out_path',type=str,help=" ",default=data_dir+"train/disambiguation/bestbuy_review_2.3.17.11.17_select_image_100")
    parser.add_argument('--image_model',type=str,help=" ",default= "/home/menglong/workspace/code/multimodal/multimodal_entity_linking/multimodal_entity_linking/output/retrieval/train/00027-train_bi-encoder-clip-ViT-L-14-2023-05-30_17-44-02/models")
    # parser.add_argument('--example_type',type=str,help=" ",default="verification")
    parser.add_argument('--mode',type=str,help=" ",default="mp")
    parser.add_argument('--dataset',type=str,help=" ",default="train")
    parser.add_argument('--step_size',type=int,help=" ",default=2)
    parser.add_argument('--start_idx',type=int,help=" ",default=0)
    parser.add_argument('--end_idx',type=int,help=" ",default=1000000)
    
    args = parser.parse_args()
    return args


  
  
  
  
if __name__ == '__main__':
    args = parser_args()     
    
    if args.mode=="mp":
        mp_main(args)
    else:
        select_one_image_based_on_similarity_for_train(args.data_path,args.out_path)