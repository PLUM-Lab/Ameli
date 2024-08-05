

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
from disambiguation.data_util.inner_util import gen_gold_attribute_without_key_section
from util.common_util import gen_review_text_rich
 
 
from util.read_example import get_father_dir 
from transformers import CLIPTokenizer
import torch
from torch.utils.data import Dataset  
from torch.utils.data import DataLoader 
from pathlib import Path
import json
import pandas as pd

class DataDealer:
    def load_corpus(self,data_folder, corpus_max_size):
        pass 

class ImageDataDealer:
    def __init__(self)  :
       
        # self.model._first_module().max_seq_length =77
        self.tokenizer_for_truncation=  CLIPTokenizer.from_pretrained("sentence-transformers/clip-ViT-L-14")#clip-vit-base-patch32 openai/clip-ViT-L-14
        
        
    def load_corpus(self,data_folder, corpus_max_size):
        corpus={}
        news_dict={}
        relevant_document_dir=get_father_dir(data_folder)
      
        news_dict,relevant_document_img_list=read_image(relevant_document_dir,news_dict,content="img")
        image_corpus=os.path.join(relevant_document_dir,"images")
        for relevant_document_img in relevant_document_img_list:
            corpus[relevant_document_img]=os.path.join(image_corpus,relevant_document_img)
        return corpus
    
    def load_queries(self,data_folder,needed_qids): 
        dev_queries_file = os.path.join(data_folder, 'Corpus2_for_retrieval.csv')
        df_news = pd.read_csv(dev_queries_file ,encoding="utf8")
        
        dev_queries = {}        #Our dev queries. qid => query
        for _,row in tqdm(df_news.iterrows()):
            claim_id=row["claim_id"]
            query=row['Claim']
            if claim_id in needed_qids:
                query=self.truncate_text(query)
                dev_queries[claim_id]=query.strip() 
        ### Load data
    
        return dev_queries
    
    def truncate_text(self,text):
        tokens=self.tokenizer_for_truncation([text])
        decoded_text=self.tokenizer_for_truncation.decode(tokens.input_ids[0][:75],skip_special_tokens =True)
        # decoded_text=decoded_text.replace("<|startoftext|>","")
        # decoded_text=decoded_text.replace("<|endoftext|>","")
        # sequence = self.tokenizer_for_truncation.encode_plus(text, add_special_tokens=False,   
        #                                        max_length=77, 
        #                                        truncation=True, 
        #                                        return_tensors='pt' )
        # return self.tokenizer_for_truncation.decode(sequence.input_ids.detach().cpu().numpy().tolist()[0])
        return decoded_text 


             
class TextDataDealer(DataDealer):
     
    
    def load_corpus(self,data_folder, corpus_max_size):
        corpus = {}             #Our corpus pid => passage
       
        # df_news = pd.read_csv(collection_filepath ,encoding="utf8")
        corpus_df=load_corpus_df(data_folder)
        # Read passages
        for _,row in tqdm(corpus_df.iterrows()):
            pid=str(row["claim_id"])+"-"+str(row["relevant_document_id"])+"-"+str(row["paragraph_id"])
            passage=row["paragraph"]
            if   corpus_max_size <= 0 or len(corpus) <= corpus_max_size:
                corpus[pid] = passage.strip()
        return corpus
          
    def load_queries(self,data_folder,needed_qids):
        
        dev_queries_file = os.path.join(data_folder, 'Corpus2.csv')
        df_news = pd.read_csv(dev_queries_file ,encoding="utf8")
        
        dev_queries = {}        #Our dev queries. qid => query
        for _,row in tqdm(df_news.iterrows()):
            claim_id=row["claim_id"]
            claim=row['Claim']
            if claim_id in needed_qids:
                dev_queries[claim_id]=claim.strip() 
        ### Load data
    
        return dev_queries



def _attribute_value_to_text( attributes):
    return ". ".join([ i   for i in attributes.values()])   
             
class AmeliTextDataDealer :
     
    
    def load_corpus(self,corpus_path,image_dir):
        corpus = {}             #Our corpus pid => passage
        product_json_path = Path(
            corpus_path
        )      
        with open(product_json_path, 'r', encoding='utf-8') as fp:
            new_crawled_products_url_json_array = json.load(fp)
            for review_dataset_json   in tqdm(new_crawled_products_url_json_array):
                product_name=review_dataset_json["product_name"]
                desc=review_dataset_json["overview_section"]["description"]
                title=review_dataset_json["product_name"]
                total_attribute_json_in_review=gen_gold_attribute_without_key_section( review_dataset_json["Spec"],review_dataset_json["product_name"],{},is_allow_multiple_attribute=False)
                attribute=_attribute_value_to_text(total_attribute_json_in_review)
                text=title+". "+desc+". "+attribute
                product_id=review_dataset_json["id"]
                corpus[product_id]=text.strip()
        return corpus
          
    def load_queries(self,data_folder,query_file_name, query_image_dir):
        dev_queries = {}        #Our dev queries. qid => query
        query_path=os.path.join(data_folder,query_file_name)
        product_json_path = Path(
            query_path
        )      
        with open(product_json_path, 'r', encoding='utf-8') as fp:
            new_crawled_products_url_json_array = json.load(fp)
            for review_dataset_json   in tqdm(new_crawled_products_url_json_array):
                review_id=review_dataset_json["review_id"]
                review_text=review_dataset_json["header"]+". "+review_dataset_json["body"]
                # review_text=gen_review_text_rich(review_dataset_json)
                dev_queries[review_id]=review_text.strip() 
        return dev_queries




class AmeliImageDataDealer:
    def __init__(self)  :
       
        # self.model._first_module().max_seq_length =77
        # self.tokenizer_for_truncation=  CLIPTokenizer.from_pretrained("sentence-transformers/clip-ViT-L-14")#clip-vit-base-patch32 openai/clip-ViT-L-14
        pass 
        
    def load_corpus(self,corpus_path, corpus_image_dir):
        corpus={}
        product_json_path = Path(
            corpus_path
        )      
        with open(product_json_path, 'r', encoding='utf-8') as fp:
            new_crawled_products_url_json_array = json.load(fp)
            for review_dataset_json   in tqdm(new_crawled_products_url_json_array):
                for image_name in review_dataset_json[ "image_path" ] :
                    if image_name   in corpus:
                        print(f"ERROR. duplicate image name {image_name},{review_dataset_json}")
                        return 
                    
                    corpus[image_name]= os.path.join(corpus_image_dir,image_name) 
        return corpus
    
    def load_queries(self,query_dir,query_file_name, query_image_dir): 
        query={}
        query_path=os.path.join(query_dir,query_file_name)
        product_json_path = Path(
            query_path
        )      
        with open(product_json_path, 'r', encoding='utf-8') as fp:
            new_crawled_products_url_json_array = json.load(fp)
            for review_dataset_json   in tqdm(new_crawled_products_url_json_array):
                for image_name in review_dataset_json[ "review_image_path" ] :
                    if image_name   in query:
                        print(f"ERROR. duplicate image name {image_name},{review_dataset_json}")
                        return 
                    
                    query[image_name]= os.path.join(query_image_dir,image_name) 
        return query
      
      
class AmeliEntityLevelImageTextDataDealer:
    def __init__(self)  :
       
        # self.model._first_module().max_seq_length =77
        # self.tokenizer_for_truncation=  CLIPTokenizer.from_pretrained("sentence-transformers/clip-ViT-L-14")#clip-vit-base-patch32 openai/clip-ViT-L-14
        pass 
        
    def load_corpus(self,corpus_path, corpus_image_dir):
        corpus={}
        product_json_path = Path(
            corpus_path
        )      
        with open(product_json_path, 'r', encoding='utf-8') as fp:
            new_crawled_products_url_json_array = json.load(fp)
            for product_dataset_json   in tqdm(new_crawled_products_url_json_array):
                product_id=product_dataset_json["id"]
                for image_name in product_dataset_json[ "image_path" ] :
                    if image_name   in corpus:
                        print(f"ERROR. duplicate image name {image_name},{product_dataset_json}")
                        return 
                    product_text=product_dataset_json["product_name"]
                    
                    corpus[product_id]=[product_text,os.path.join(corpus_image_dir,image_name)]
                    break
        return corpus
    
    def load_queries(self,query_dir,query_file_name, query_image_dir): 
        query={}
        query_path=os.path.join(query_dir,query_file_name)
        product_json_path = Path(
            query_path
        )      
        with open(product_json_path, 'r', encoding='utf-8') as fp:
            new_crawled_products_url_json_array = json.load(fp)
            
            for review_dataset_json   in tqdm(new_crawled_products_url_json_array):
                review_id=review_dataset_json["review_id"]
                for image_name in review_dataset_json[ "review_image_path" ] :
                
                    if image_name   in query:
                        print(f"ERROR. duplicate image name {image_name},{review_dataset_json}")
                        return 
                    review_text=gen_review_text_rich(review_dataset_json)
                    query[review_id]=[review_text,os.path.join(query_image_dir,image_name)]
                     
        return query
            
      
      
class AmeliEntityLevelImageDataDealer:
    def __init__(self)  :
       
        # self.model._first_module().max_seq_length =77
        # self.tokenizer_for_truncation=  CLIPTokenizer.from_pretrained("sentence-transformers/clip-ViT-L-14")#clip-vit-base-patch32 openai/clip-ViT-L-14
        pass 
        
    def load_corpus(self,corpus_path, corpus_image_dir):
        corpus={}
        product_json_path = Path(
            corpus_path
        )      
        with open(product_json_path, 'r', encoding='utf-8') as fp:
            new_crawled_products_url_json_array = json.load(fp)
            for product_dataset_json   in tqdm(new_crawled_products_url_json_array):
                product_id=product_dataset_json["id"]
                for image_name in product_dataset_json[ "image_path" ] :
                    if image_name   in corpus:
                        print(f"ERROR. duplicate image name {image_name},{product_dataset_json}")
                        return 
                    corpus[product_id]=os.path.join(corpus_image_dir,image_name)
                    break
        return corpus
    
    def load_queries(self,query_dir,query_file_name, query_image_dir): 
        query={}
        query_path=os.path.join(query_dir,query_file_name)
        product_json_path = Path(
            query_path
        )      
        with open(product_json_path, 'r', encoding='utf-8') as fp:
            new_crawled_products_url_json_array = json.load(fp)
            
            for review_dataset_json   in tqdm(new_crawled_products_url_json_array):
                review_id=review_dataset_json["review_id"]
                for image_name in review_dataset_json[ "review_image_path" ] :
                
                    if image_name   in query:
                        print(f"ERROR. duplicate image name {image_name},{review_dataset_json}")
                        return 
                    query[review_id]=os.path.join(query_image_dir,image_name)
        return query
            
      
      


class WikidiverseDataDealer:
    def __init__(self)  :
        
        pass 
    
    def load_qrels(self,data_folder,product_json_dict=None,media="txt"):    
        qrel_file_name="qrels.csv"#img_evidence_relevant_document_mapping.csv
        qrels_filepath = os.path.join(data_folder, qrel_file_name)
        df_news = pd.read_csv(qrels_filepath ,encoding="utf8")
        needed_pids = set()     #Passage IDs we need
        needed_qids = set()     #Query IDs we need
        negative_rel_docs={}
        dev_rel_docs = {}       #Mapping qid => set with relevant pids
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
        return dev_rel_docs,needed_pids,needed_qids,negative_rel_docs
        
    def load_corpus(self,data_folder):
        corpus={}
        relevant_document_dir=get_father_dir(data_folder)
        with open(os.path.join(relevant_document_dir, "wikipedia_entity.pickle"), 'rb') as handle:
            entity_dict = pickle.load(handle)
            for entity_name, entity in  entity_dict.items() :
                entity_img_path=gen_img_path(entity.img_path_list)
                corpus[entity_name]=[entity.text,entity_img_path]
         
        return corpus
    
    def load_queries(self,data_folder,needed_qids): 
        dev_queries_file = os.path.join(data_folder, 'w_10cands_with_path.json')
        dev_queries = {}    
        with open(dev_queries_file, 'r', encoding='utf-8') as fr:
            testData = json.load(fr)
            for datapoint  in testData:
                [caption, img, ment, ment_type, lctx, rctx, entity, cands, topic, start, end,img_path,is_img_downloaded,entity_name,candidate_entity_name_list,mention_id]=datapoint 
                dev_queries[mention_id]=[caption.strip() ,img_path]
        return dev_queries    
    

def gen_img_path( img_path_list):
    return img_path_list.split("[AND]")[0]