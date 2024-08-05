from transformers import CLIPTokenizer
import torch
from torch.utils.data import Dataset  
from torch.utils.data import DataLoader 
from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm 
import os

from util.data_util.entity_linking_data_util import json_to_dict 



import argparse

def gen_product_json_dict(product_json_path):
    with open(product_json_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        product_json_dict=json_to_dict(new_crawled_products_url_json_array)
    return product_json_dict


def add_positive_image(rel_docs_list,query_image_name,product_images):
    for corpus_image in product_images:
        rel_docs_list.append([query_image_name,0,corpus_image,1])
    return rel_docs_list



def generate_image_qrel(query_path,corpus_product_json_path,qrel_path, mode,media="img"):
    rel_docs_list=[] 
    product_json_dict=gen_product_json_dict(corpus_product_json_path)
    # cur_product_id=-111111
     
    corpus_product_json_path = Path(
        query_path
    )      
    with open(corpus_product_json_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        for review_dataset_json   in tqdm(new_crawled_products_url_json_array):
            product_id=review_dataset_json["gold_entity_info"]["id"]
            product_json=product_json_dict[product_id]
            product_images=product_json["image_path"]
            gold_product_category=product_json["product_category"]
            for query_image_name in review_dataset_json[ "review_image_path" ] : 
                rel_docs_list=add_positive_image(rel_docs_list,query_image_name,product_images)
                rel_docs_list=add_negative_corpus_id( rel_docs_list,query_image_name,product_id,product_json_dict,gold_product_category,mode)
               
                
    save(rel_docs_list,qrel_path,media)


def generate_text_qrel(query_path,corpus_product_json_path,qrel_path, mode,media="img"):
    rel_docs_list=[] 
    product_json_dict=gen_product_json_dict(corpus_product_json_path)
    # cur_product_id=-111111
     
    corpus_product_json_path = Path(
        query_path
    )      
    with open(corpus_product_json_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        for review_dataset_json   in tqdm(new_crawled_products_url_json_array):
            product_id=review_dataset_json["gold_entity_info"]["id"]
            gold_product_category=review_dataset_json["gold_entity_info"]["product_category"]
            review_id=review_dataset_json["review_id"]
            rel_docs_list.append([review_id,0,product_id,1])
            rel_docs_list=add_negative_corpus_id_for_text(rel_docs_list,review_id,product_id,product_json_dict,gold_product_category,mode)
             
                
    save(rel_docs_list,qrel_path,media)


def save(rel_docs_list,data_path,media):
    if media=="img":
        qrel_file_name="image_qrels_not_category_v2.csv"
    else:
        qrel_file_name="text_qrels.csv"#img_evidence_relevant_document_mapping.csv
   
    df = pd.DataFrame(rel_docs_list, columns = ['TOPIC', 'ITERATION','DOCUMENT#','RELEVANCY'])
    df.to_csv(os.path.join(data_path,qrel_file_name),index=False)#qrels.csv
    from datetime import datetime
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    
# def record_negative_corpus_id( relevant_dic):
#     negative_corpus_id_list=[]
#     length=100 if len(relevant_dic.keys())>100 else len(relevant_dic.keys())
#     negative_corpus_id_list.extend(list(relevant_dic.keys())[:length])
     
#     return negative_corpus_id_list
from random import randrange 
                   
def add_negative_corpus_id(rel_docs_list,query_image_name,product_id,product_json_dict,gold_product_category,mode):
    product_id_list=list(product_json_dict.keys())
    max_negative_num=200
    cur_negative_num=0
    while True:
        random_product_id=randrange(len(product_id_list))     
        product_json=product_json_dict[product_id_list[random_product_id]] 
        cur_product_category=product_json["product_category"]
        if gold_product_category != cur_product_category:
            
            product_images=product_json["image_path"]
            for negative_corpus_id in product_images:
                rel_docs_list.append([query_image_name,0,negative_corpus_id,0])
            
            if mode =="not_category":
                cur_negative_num+=1
                if cur_negative_num>=max_negative_num:
                    break 
    return rel_docs_list
                    
  
def add_negative_corpus_id_for_text(rel_docs_list,review_id,product_id,product_json_dict,gold_product_category,mode):
    product_id_list=list(product_json_dict.keys())
    max_negative_num=1000
    cur_negative_num=0
    while True:
        random_product_id=randrange(len(product_id_list))     
        product_json=product_json_dict[product_id_list[random_product_id]] 
        cur_product_id=product_json["id"]
        cur_product_category=product_json["product_category"]
        if gold_product_category != cur_product_category:
            rel_docs_list.append([review_id,0,cur_product_id,0])
                
            if mode =="not_category":
                cur_negative_num+=1
                if cur_negative_num>=max_negative_num:
                    break  
    return rel_docs_list
                                        
                    

def get_args():
   


    parser = argparse.ArgumentParser() 
    # parser.add_argument("--lr", default=1e-7, type=float)
    # parser.add_argument("--step_size", default=10, type=int)
    # parser.add_argument("--start_id", default=0, type=int)
    # parser.add_argument("--end_id", default=6000, type=int)
    # parser.add_argument("--use_pre_trained_model", default=True, action="store_false") 
    parser.add_argument("--dataset_split",default='train')
    parser.add_argument("--mode",default='not_category')
    parser.add_argument("--media",default='txt')
    parser.add_argument("--file",default='/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v4/bestbuy_products_40000_3.4.16_all_text_image_similar.json', type=str  ) 
    parser.add_argument("--review_file",default='/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v4/bestbuy_review_2.3.16.27_train_dev_test.json', type=str  ) 
    parser.add_argument("--review_parent_dir",default='/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v4', type=str  ) 
    args = parser.parse_args()

    print(args)
    return args

 
if __name__ == "__main__":  
    args=get_args()
                    
    query_path=f"/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/{args.dataset_split}/retrieval/bestbuy_review_2.3.17.2_clean_missed_product.json"
    product_json_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/bestbuy_products_40000_3.4.19_final_format.json"
    qrel_path=f"/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/{args.dataset_split}/retrieval/"
    generate_image_qrel(query_path,product_json_path,qrel_path,args.mode )
    # generate_text_qrel(query_path,product_json_path,qrel_path,args.mode ,args.media)