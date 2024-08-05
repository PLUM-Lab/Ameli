

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
from collections import defaultdict
from torch.utils.data import IterableDataset
from tqdm import tqdm
from torch.utils.data import Dataset
import random
import pickle
import argparse
import torch
from PIL import Image
from retrieval.data_util.core.data_dealer import *
from retrieval.data_util.core.data_dealer import AmeliEntityLevelImageDataDealer
from retrieval.data_util.core.dataset import *
from retrieval.data_util.core.dataset import TripletEntityLevelImageDataset
from retrieval.data_util.core.help_util import *
from retrieval.utils.retrieval_util import gen_data_dealer
from util.read_example import get_father_dir 
from transformers import CLIPTokenizer
import torch
from torch.utils.data import Dataset  
from torch.utils.data import DataLoader 
from pathlib import Path
import json
import pandas as pd

 
    

 

def  load_ameli_data_to_train_bi_encoder( args,ce_score_margin,num_negs_per_system):
    data_folder = args.train_data_folder
    corpus_dir=args.corpus_dir
    # os.makedirs(data_folder, exist_ok=True)
    # dataset=SnopesForRetrievalDataset( data_folder,"text_evidence")
    data_dealer=gen_data_dealer(args.media,args.level)
    corpus=data_dealer.load_corpus(corpus_dir, args.corpus_image_dir)
    positive_rel_docs,needed_pids,needed_qids,negative_rel_docs=load_qrels(data_folder,args.query_file_name,args.candidate_mode, args.media,
                                                                           args.mode,ce_score_margin=args.ce_score_margin)
    queries=data_dealer.load_queries(data_folder,args.query_file_name, args.query_image_dir)
    train_queries=get_train_queries(queries,positive_rel_docs,negative_rel_docs)
    # ce_scores=load_ce_score_to_prevent_false_negative(data_folder)
    # train_queries=form_train_queries_with_hard_negative(data_folder,args,ce_scores,ce_score_margin,num_negs_per_system,queries)
    if args.media=="txt":
        train_dataset = TripletDataset(train_queries ,corpus,args.mode)
    elif args.media=="txt_img":
        train_dataset=TripletEntityLevelImageTextDataset(train_queries ,corpus,args.mode,data_folder,args.query_file_name,args.corpus_image_dir)
    elif args.level=="entity":
        train_dataset=TripletEntityLevelImageDataset(train_queries ,corpus,args.mode,data_folder,args.query_file_name,args.corpus_image_dir)
    else:
        train_dataset = TripletImageDataset(train_queries ,corpus,args.mode)
    logging.info("Train queries: {}".format(len(train_dataset)))
    return train_dataset ,corpus
    
     

def load_ameli_data_to_train_cross_encoder(data_folder, val_data_folder,pos_neg_ration,max_train_samples,args,media,mode):
    data_folder = args.train_data_folder
    train_dataset,corpus,data_dealer,positive_rel_docs,needed_qids,queries=_load_data_for_cross_encoder(data_folder,args,mode)
    val_dict_dataset=load_val_image_query_dic( val_data_folder, args,data_dealer,corpus,mode)
    return train_dataset,val_dict_dataset,1      

def _load_data_for_cross_encoder(data_folder,args,mode,corpus=None):
    
    # os.makedirs(data_folder, exist_ok=True)
    # dataset=SnopesForRetrievalDataset( data_folder,"text_evidence")
    if args.media=="txt":
        data_dealer=AmeliTextDataDealer()
    else:
        data_dealer=AmeliImageDataDealer()
    if corpus is None:
        corpus_dir=args.corpus_dir
        corpus=data_dealer.load_corpus(corpus_dir, args.corpus_image_dir)
    num_max_negatives=50
    positive_rel_docs,needed_pids,needed_qids,negative_rel_docs=load_qrels(data_folder,args.media,mode)
    queries=data_dealer.load_queries(data_folder,args.query_file_name, args.query_image_dir)
    train_queries=get_train_queries(queries,positive_rel_docs,negative_rel_docs)
    # train_samples,pos_weight=load_cross_encoder_samples( data_folder,queries,corpus ,num_max_negatives,media,mode)
    if args.media=="txt":
        train_dataset = PairDataset(train_queries ,corpus)
    else:
        train_dataset = PairImageDataset(train_queries ,corpus,args.mode)
    return train_dataset,corpus,data_dealer,positive_rel_docs,needed_qids,queries
        
    
def load_ameli_test_data_for_cross_encoder(data_folder,args):
    dataset,corpus,data_dealer,positive_rel_docs,needed_qids,queries=_load_data_for_cross_encoder(data_folder,args,args.mode)
    
    #Read which passages are relevant
    relevant_docs = defaultdict(lambda: defaultdict(int))
    for qid, dev_rel in positive_rel_docs.items():
        for pid in dev_rel:
            relevant_docs[str(qid)][str(pid)]=1
    return queries,relevant_docs,needed_qids,corpus 
 

 

# def read_queries(data_folder):
#     ### Read the train queries, store in queries dict
#     queries = {}        #dict in the format: query_id -> query. Stores all training queries
#     queries_filepath = os.path.join(data_folder, 'queries.train.tsv')
#     if not os.path.exists(queries_filepath):
#         tar_filepath = os.path.join(data_folder, 'queries.tar.gz')
#         if not os.path.exists(tar_filepath):
#             logging.info("Download queries.tar.gz")
#             util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz', tar_filepath)

#         with tarfile.open(tar_filepath, "r:gz") as tar:
#             tar.extractall(path=data_folder)


#     with open(queries_filepath, 'r', encoding='utf8') as fIn:
#         for line in fIn:
#             qid, query = line.strip().split("\t")
#             qid = int(qid)
#             queries[qid] = query
#     return queries
 

# def gen_first_negative(qid,negative_rel_docs,positive_rel_docs):
#     if qid in negative_rel_docs:
#         negative_set=negative_rel_docs[qid]
#     else:
#         for query_id, positive_rel  in positive_rel_docs.items():
#             if query_id != qid:
#                 return positive_rel






    
    

# def load_corpus(data_folder, corpus_max_size):
#     corpus = {}             #Our corpus pid => passage
    
#     # df_news = pd.read_csv(collection_filepath ,encoding="utf8")
#     corpus_df=load_corpus_df(data_folder)
#     # Read passages
#     for _,row in tqdm(corpus_df.iterrows()):
#         pid=str(row["claim_id"])+"-"+str(row["relevant_document_id"])+"-"+str(row["paragraph_id"])
#         passage=row["paragraph"]
#         if   corpus_max_size <= 0 or len(corpus) <= corpus_max_size:
#             corpus[pid] = passage.strip()
#     return corpus
    
            
# def load_queries(data_folder,needed_qids):
    
#     dev_queries_file = os.path.join(data_folder, 'Corpus2.csv')
#     df_news = pd.read_csv(dev_queries_file ,encoding="utf8")
    
#     dev_queries = {}        #Our dev queries. qid => query
#     for _,row in tqdm(df_news.iterrows()):
#         claim_id=row["claim_id"]
#         claim=row['Claim']
#         if claim_id in needed_qids:
#             dev_queries[claim_id]=claim.strip() 
#     ### Load data
 
#     return dev_queries
 
 
 
 

# def load_data( args,ce_score_margin,num_negs_per_system):
#     data_folder = args.train_data_folder
#     # os.makedirs(data_folder, exist_ok=True)
#     # dataset=SnopesForRetrievalDataset( data_folder,"text_evidence")
#     if args.media=="txt":
#         data_dealer=TextDataDealer()
#     else:
#         data_dealer=ImageDataDealer()
#     corpus=data_dealer.load_corpus(data_folder, 0)
#     positive_rel_docs,needed_pids,needed_qids,negative_rel_docs=load_qrels(data_folder,args.media)
#     queries=data_dealer.load_queries(data_folder,needed_qids)
#     train_queries=get_train_queries(queries,positive_rel_docs,negative_rel_docs)
#     # ce_scores=load_ce_score_to_prevent_false_negative(data_folder)
#     # train_queries=form_train_queries_with_hard_negative(data_folder,args,ce_scores,ce_score_margin,num_negs_per_system,queries)
#     if args.media=="txt":
#         train_dataset = MSMARCODataset(train_queries ,corpus)
#     else:
#         train_dataset = MSMARCOImageDataset(train_queries ,corpus)
#     logging.info("Train queries: {}".format(len(train_dataset)))
#     return train_dataset ,corpus








  
        
# def load_data_for_cross_encoder(data_folder,pos_neg_ration,max_train_samples,args):
#     data_dealer=TextDataDealer()
#     corpus=data_dealer.load_corpus(data_folder, 0) 
#     num_max_negatives=20
#     train_samples,pos_weight=load_train_samples(data_folder, args,data_dealer,corpus,num_max_negatives)
#     dev_queries_dict=load_val_query_dic(args.val_data_folder, args,data_dealer,corpus)
    
     
 
  
            
#     return train_samples,dev_queries_dict,pos_weight