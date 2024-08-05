

import sys
import json
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, LoggingHandler, util, models, evaluation, losses, InputExample
import logging
from datetime import datetime
import gzip
import numpy as np 
import os
import pandas as pd 
from torchvision import transforms,datasets
import tarfile
from util.common_util import json_to_dict
from util.env_config import * 
from collections import defaultdict
from torch.utils.data import IterableDataset
from tqdm import tqdm
from torch.utils.data import Dataset
import random
import pickle
import argparse
import torch
from PIL import Image
from retrieval.data_util.compute_average_score_diff import load_candidates_over_threshold
from retrieval.data_util.core.dataset import TripletDictDataset
 
from util.read_example import get_father_dir 
from transformers import CLIPTokenizer
import torch
from torch.utils.data import Dataset  
from torch.utils.data import DataLoader 
from pathlib import Path
import json
import pandas as pd


# from util.read_example import get_relevant_document_dir
def get_father_dir(splited_dir):
    from pathlib import Path
    path = Path(splited_dir)
    whole_dataset_dir=path.parent.absolute()
    return whole_dataset_dir
  
  

def get_train_queries(queries,positive_rel_docs,negative_rel_docs):
    train_queries = {}
    #  get the first from dataset
    # default_negative_set=set()
    # default_negative_set.add(21799)  #['Perry–Castañeda_Library']
    for qid,pid_set in positive_rel_docs.items():
        # gold_product_id= list(pid_set)[0]
        if qid in negative_rel_docs:
            negative_set=negative_rel_docs[qid]
          
        else:
            exit(1)
        #     negative_set=default_negative_set
        # default_negative_set=pid_set 
        # if qid!=  4346 :
        #     continue #TODO 
        train_queries[qid] = {'qid': qid, 'query': queries[qid], 'pos': pid_set, 'neg': negative_set}
    return train_queries    



# def get_queries_with_content(queries,positive_rel_docs,negative_rel_docs,corpus,media,mode):
#     train_queries = {}
#     for idx,(qid,pid_set) in enumerate(positive_rel_docs.items()):
#         negative_corpus_id_list=negative_rel_docs[qid]
#         if media=="img":
#             query_content=Image.open(queries[qid])
#         else:
#             query_content=queries[qid]
        
#         train_queries[qid] = {'qid': qid, 'query': query_content, 'positive': get_content(corpus,pid_set,media)
#                               , 'negative': get_content(corpus,negative_corpus_id_list,media)}
#         if mode=="dry_run" and idx>6:
#             break
#     return train_queries  

def load_qrels_online(data_folder,media="txt",mode=None,query_file_name=None, ce_score_margin=None): 
    product_json_path = Path(
        products_path_str
    )      
    with open(product_json_path, 'r', encoding='utf-8') as fp:
        product_json_array=new_crawled_products_url_json_array = json.load(fp)
        product_json_dict=json_to_dict(product_json_array)
    query_path=os.path.join(data_folder,query_file_name)
    review_json_path = Path(
        query_path
    )      
    needed_pids = set()     #Passage IDs we need
    needed_qids = set()     #Query IDs we need
    negative_rel_docs={}
    dev_rel_docs = {}
    with open(review_json_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        for review_dataset_json   in tqdm(new_crawled_products_url_json_array):
            review_id=review_dataset_json["review_id"]
            fused_score_dict=review_dataset_json["fused_score_dict"]
            gold_product_id=review_dataset_json["gold_entity_info"]["id"]
            if media =="txt":
                candidate_id_list=load_candidates_over_threshold(review_dataset_json,product_json_dict,"bi_score",ce_score_margin)
            elif media =="img":
                candidate_id_list=load_candidates_over_threshold(review_dataset_json,product_json_dict,"image_score",ce_score_margin)
            elif media=="txt_img":
                candidate_id_list=load_candidates_over_threshold(review_dataset_json,product_json_dict,"fused_score",ce_score_margin)
            if review_id not in dev_rel_docs:
                dev_rel_docs[review_id] = set()
            dev_rel_docs[review_id].add(gold_product_id)
            needed_pids.add(review_id)
            needed_qids.add(gold_product_id)
            for candidate_id in candidate_id_list:
                if review_id not in negative_rel_docs:
                    negative_rel_docs[review_id] = set()
                negative_rel_docs[review_id].add(candidate_id)
                if mode=="dry_run":
                    break
 
    return dev_rel_docs,needed_pids,needed_qids,negative_rel_docs



def load_top_10_qrel(data_folder,media="txt",mode=None,query_file_name=None, ce_score_margin=None): 
    print("use top 10")
    query_path=os.path.join(data_folder,query_file_name)
    review_json_path = Path(
        query_path
    )      
    needed_pids = set()     #Passage IDs we need
    needed_qids = set()     #Query IDs we need
    negative_rel_docs={}
    dev_rel_docs = {}
    with open(review_json_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        for review_dataset_json   in tqdm(new_crawled_products_url_json_array):
            review_id=review_dataset_json["review_id"]
            candidate_id_list=[]
          
            fused_candidate_list=review_dataset_json["fused_candidate_list"]
            gold_product_id=review_dataset_json["gold_entity_info"]["id"]
            for candidate_id in fused_candidate_list[:10]:
                if candidate_id!=gold_product_id:
                    candidate_id_list.append(candidate_id)
            if review_id not in dev_rel_docs:
                dev_rel_docs[review_id] = set()
            dev_rel_docs[review_id].add(gold_product_id)
            needed_pids.add(review_id)
            needed_qids.add(gold_product_id)
            if review_id not in negative_rel_docs:
                negative_rel_docs[review_id] = set()
            for candidate_id in candidate_id_list:
                
                negative_rel_docs[review_id].add(candidate_id)
                if mode=="dry_run":
                    break
 
    return dev_rel_docs,needed_pids,needed_qids,negative_rel_docs



def load_qrels(data_folder,query_file_name,candidate_mode,media="txt",mode=None,  ce_score_margin=0.08):   #="dif_category"
    if candidate_mode=="over_score" :
        return load_qrels_online(data_folder,media ,mode,query_file_name, ce_score_margin)
    elif candidate_mode=="top_10" :
        return load_top_10_qrel(data_folder,media ,mode,query_file_name, ce_score_margin)
    if media=="img":
        qrel_file_name="retrieval/image_qrels_not_category_v2.csv"
    else:
        qrel_file_name="retrieval/text_qrels.csv"#img_evidence_relevant_document_mapping.csv
    qrels_filepath = os.path.join(data_folder, qrel_file_name)
    df_news = pd.read_csv(qrels_filepath ,encoding="utf8")
    needed_pids = set()     #Passage IDs we need
    needed_qids = set()     #Query IDs we need
    negative_rel_docs={}
    dev_rel_docs = {}       #Mapping qid => set with relevant pids
    print(f"qrel_num:{len(df_news)}")
    # Load which passages are relevant for which queries
    for _,row in tqdm(df_news.iterrows()):
        
        qid,  pid, relevance= row["TOPIC"],row["DOCUMENT#"],row["RELEVANCY"]

        if relevance==1:

            if qid not in dev_rel_docs:
                dev_rel_docs[qid] = set()
            dev_rel_docs[qid].add(pid)

            needed_pids.add(pid)
            needed_qids.add(qid)
        else:
            if qid not in negative_rel_docs:
                negative_rel_docs[qid] = set()
            negative_rel_docs[qid].add(pid)
            if mode=="dry_run":
                break
    return dev_rel_docs,needed_pids,needed_qids,negative_rel_docs

def balance(positive_train_samples,negative_train_samples,train_samples):
    new_float_ratio=0
    if len(positive_train_samples)>0:
        ratio= int(len(negative_train_samples)/len(positive_train_samples))
        
        if ratio>1:
            for i in range(ratio-1):
                train_samples.extend(positive_train_samples)
        
            new_float_ratio=len(negative_train_samples)/(len(positive_train_samples)*ratio)    
        else:
            new_float_ratio=len(negative_train_samples)/len(positive_train_samples)
    return train_samples,new_float_ratio
 
 

def get_train_queries_for_cross_encoder(queries,positive_rel_docs,negative_rel_docs):
    train_queries = {}
    #  get the first from dataset
    default_negative_set=[21799]  #['Perry–Castañeda_Library']
    for qid,pid_set in positive_rel_docs.items():
        if qid in negative_rel_docs:
            negative_set=negative_rel_docs[qid]
        else:
            negative_set=default_negative_set
        default_negative_set=pid_set 
        train_queries[qid] = {'qid': qid, 'query': queries[qid], 'pos': pid_set, 'neg': negative_set }
    return train_queries     

def load_cross_encoder_samples(data_folder,queries,corpus,num_max_negatives ,media,mode):    
     
    if media=="img":
        qrel_file_name="retrieval/image_qrels_not_category.csv"
    else:
        qrel_file_name="text_qrels.csv"#img_evidence_relevant_document_mapping.csv
   
    qrels_filepath = os.path.join(data_folder, qrel_file_name)
    df_news = pd.read_csv(qrels_filepath ,encoding="utf8")
    needed_pids = set()     #Passage IDs we need
    needed_qids = set()     #Query IDs we need
    negative_rel_docs={}
    dev_rel_docs = {}       #Mapping qid => set with relevant pids
    train_samples=[]
    negative_num_dict={}
    negative_train_samples=[]
    positive_train_samples=[]
    
    # Load which passages are relevant for which queries
    for idx,row in tqdm(df_news.iterrows()):
        
        qid,  pid, relevance= row["TOPIC"],row["DOCUMENT#"],row["RELEVANCY"]
        
        if media=="txt":
            query = queries[qid]
            corpus_content=corpus[pid]
        else:
            query = queries[qid] #Image.open()
            corpus_content= corpus[pid] 
        input_example=InputExample(texts=[query,corpus_content], label=relevance)
        if relevance==0:
            if qid in negative_num_dict:
                cur_num=negative_num_dict[qid]
                if cur_num>num_max_negatives:
                    continue
                else:
                    negative_num_dict[qid]+=1
            else:
                negative_num_dict[qid]=1
            negative_train_samples.append(input_example)
        else:
            positive_train_samples.append(input_example)
        train_samples.append(input_example)
        if mode=="dry_run" and idx>6:
            break

    train_samples,pos_weight=balance(positive_train_samples,negative_train_samples,train_samples)
         
    return train_samples ,pos_weight
# def load_train_samples(sub_data_folder,args,data_dealer,corpus,num_max_negatives):
#     positive_rel_docs,needed_pids,needed_qids,negative_rel_docs=load_qrels(sub_data_folder,args.media)
#     queries=data_dealer.load_queries(sub_data_folder,needed_qids)
#     train_samples,pos_weight=load_cross_encoder_samples(sub_data_folder,queries,corpus ,num_max_negatives)
#     return train_samples,pos_weight
    
# def load_val_query_dic(sub_data_folder, args,data_dealer,corpus):
#     positive_rel_docs,needed_pids,needed_qids,negative_rel_docs=load_qrels(sub_data_folder,args.media)
#     queries=data_dealer.load_queries(sub_data_folder,needed_qids)
#     queries_dict=get_queries_with_content(queries,positive_rel_docs,negative_rel_docs,corpus)
#     return  queries_dict




def load_val_image_query_dic(sub_data_folder, args,data_dealer,corpus,mode):
    positive_rel_docs,needed_pids,needed_qids,negative_rel_docs=load_qrels(sub_data_folder,args.media)
    queries=data_dealer.load_queries(sub_data_folder,args.query_file_name, args.query_image_dir)
    train_queries=get_train_queries(queries,positive_rel_docs,negative_rel_docs)
    val_dict_dataset=TripletDictDataset(train_queries,corpus)
    # queries_dict=get_queries_with_content(queries,positive_rel_docs,negative_rel_docs,corpus,args.media,mode)
    return  val_dict_dataset
                
                
