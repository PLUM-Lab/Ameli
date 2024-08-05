
from urllib.parse import unquote
import pandas as pd
import os 
import hashlib
import re
import json
from tqdm import tqdm
import pickle

from transformers import CLIPFeatureExtractor, CLIPProcessor
from disambiguation.data_util.inner_util import gen_gold_attribute_without_key_section, json_to_review_dict, review_json_to_product_dict
 
from retrieval.eval.eval_msmarco_mocheg import load_qrels
from retrieval.data_util.datapoint import Entity, Mention, gen_entity_name 

import torch
from torch.utils.data import Dataset  
from torch.utils.data import DataLoader 
from pathlib import Path
import json
import pandas as pd
from util.common_util import gen_review_text_rich

from util.data_util.data_util import NpEncoder


class Entity:
    def __init__(self,id,image_path_list,text,entity_category)  :
         
        self.id=id 
        self.image_path_list=image_path_list
        self.text=text 
        self.entity_candidate_id_list=[]
        self.entity_category=entity_category

class Query:
    def __init__(self,id,image_path_list,text,gold_entity_category,entity_candidate_name_list=[],gold_entity_id=None)  :
         
        self.id=id 
        self.image_path_list=image_path_list
        self.text=text 
        self.entity_candidate_id_list=entity_candidate_name_list
        self.entity_category=gold_entity_category
        self.gold_entity_id=gold_entity_id
        
class QueryDataset(Dataset):
    def __init__(self, query_dict):
         
        self.query_dict=query_dict
        

    def __len__(self):
        return len(self.query_dict)

    def __getitem__(self, idx):
        
        keys_list = list(self.query_dict)
        key = keys_list[idx]
        news=self.query_dict[key]
        text=news.text 
        id=news.id
        image_path=news.image_path_list
         
        entity_candidate_id_list=news.entity_candidate_id_list
        
        entity_category=news.entity_category
        return id,text, image_path, entity_candidate_id_list,entity_category
 

def load_entity_linking_data(query_dir, corpus_dir,corpus_pickle_dir, corpus_image_dir,dataset_image_dir, text_base):
    # query_entity_dict=read_entity(query_dir,None,image_dir,use_cache=False )
    corpus_entity_dict=read_entity(corpus_dir,corpus_pickle_dir,corpus_image_dir,text_base)
    
    query_entity_dict=read_mention(query_dir,dataset_image_dir)
    
    
    dataset=QueryDataset(query_entity_dict)
   
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)  
    return train_dataloader,corpus_entity_dict

 

 
def gen_review_text(review_json):
    return review_json["header"]+". "+review_json["body"]

def read_mention(corpus_path, image_dir  ):  
    
     
    entity_dict={}
    product_json_path = Path(
        corpus_path
    )      
    with open(product_json_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        for review_dataset_json   in tqdm(new_crawled_products_url_json_array):
            gold_product_id=review_dataset_json["gold_entity_info"]["id"]
            review_json=review_dataset_json
            image_path_list=review_json["review_image_path"]
          
            image_path_list=[os.path.join(image_dir,image_name) for image_name in image_path_list ]
            id=review_json["review_id"]
            # entity_category=review_dataset_json["product_category"]
            text=review_json["header"]+". "+review_json["body"]
            # text=gen_review_text_rich(review_json)#TODO 
            entity=Query(id,image_path_list,text,"",gold_entity_id=gold_product_id)
            entity_dict[id]=entity 
    
    return entity_dict


def _attribute_value_to_text( attributes):
    return ". ".join([ i   for i in attributes.values()])
    
def read_entity(corpus_path,corpus_pickle_dir,image_dir,text_base,use_cache=True ):  
    
    # if corpus_pickle_dir is not None:
    #     corpus_pickle_path=os.path.join(corpus_pickle_dir,"corpus_dict.pickle")
    # else:
    #     corpus_pickle_path =None 
    # if use_cache and os.path.exists(corpus_pickle_path):
         
    #     with open(corpus_pickle_path, 'rb') as handle:
    #         entity_dict = pickle.load(handle)
    # else:
    entity_dict={}
    product_json_path = Path(
        corpus_path
    )      
    with open(product_json_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        for review_dataset_json   in tqdm(new_crawled_products_url_json_array):
            if "image_path" in review_dataset_json and review_dataset_json["image_path"] is not None:
                image_path_list=[os.path.join(image_dir,image_name) for image_name in review_dataset_json[ 'image_path' ] ]
                id=review_dataset_json["id"]
                
                entity_category=review_dataset_json["product_category"]
                if text_base=="desc":
                    text=review_dataset_json["overview_section"]["description"]
                elif text_base=="title":
                    text=review_dataset_json["product_name"]
                elif text_base=="attribute":
                    total_attribute_json_in_review=gen_gold_attribute_without_key_section( review_dataset_json["Spec"],review_dataset_json["product_name"],{},is_allow_multiple_attribute=False)
                    text=_attribute_value_to_text(total_attribute_json_in_review)
                elif text_base=="desc_title":
                    desc=review_dataset_json["overview_section"]["description"]
                    title=review_dataset_json["product_name"]
                    text=desc+". "+title
                elif text_base=="title_desc":
                    desc=review_dataset_json["overview_section"]["description"]
                    title=review_dataset_json["product_name"]
                    text=title+". "+desc
                elif text_base=="attribute_title":
                    title=review_dataset_json["product_name"]
                    total_attribute_json_in_review=gen_gold_attribute_without_key_section( review_dataset_json["Spec"],review_dataset_json["product_name"],{},is_allow_multiple_attribute=False)
                    attribute=_attribute_value_to_text(total_attribute_json_in_review)
                    text=attribute+". "+title
                elif text_base=="title_attribute":
                    title=review_dataset_json["product_name"]
                    total_attribute_json_in_review=gen_gold_attribute_without_key_section( review_dataset_json["Spec"],review_dataset_json["product_name"],{},is_allow_multiple_attribute=False)
                    attribute=_attribute_value_to_text(total_attribute_json_in_review)
                    text=title+". "+attribute
                elif text_base=="attribute_desc":
                    desc=review_dataset_json["overview_section"]["description"]
                    total_attribute_json_in_review=gen_gold_attribute_without_key_section( review_dataset_json["Spec"],review_dataset_json["product_name"],{},is_allow_multiple_attribute=False)
                    attribute=_attribute_value_to_text(total_attribute_json_in_review)
                    text=attribute+". "+desc
                elif text_base=="desc_attribute":
                    desc=review_dataset_json["overview_section"]["description"]
                    total_attribute_json_in_review=gen_gold_attribute_without_key_section( review_dataset_json["Spec"],review_dataset_json["product_name"],{},is_allow_multiple_attribute=False)
                    attribute=_attribute_value_to_text(total_attribute_json_in_review)
                    text=desc+". "+attribute
                elif text_base=="desc_title_attribute":
                    desc=review_dataset_json["overview_section"]["description"]
                    title=review_dataset_json["product_name"]
                    total_attribute_json_in_review=gen_gold_attribute_without_key_section( review_dataset_json["Spec"],review_dataset_json["product_name"],{},is_allow_multiple_attribute=False)
                    attribute=_attribute_value_to_text(total_attribute_json_in_review)
                    text=desc+". "+title+". "+attribute
                elif text_base=="title_desc_attribute":
                    desc=review_dataset_json["overview_section"]["description"]
                    title=review_dataset_json["product_name"]
                    total_attribute_json_in_review=gen_gold_attribute_without_key_section( review_dataset_json["Spec"],review_dataset_json["product_name"],{},is_allow_multiple_attribute=False)
                    attribute=_attribute_value_to_text(total_attribute_json_in_review)
                    text=title+". "+desc+". "+attribute
                entity=Entity(id,image_path_list,text,entity_category)
                entity_dict[id]=entity 
        # if corpus_pickle_path is not None:
        #     with open(corpus_pickle_path, 'wb') as handle:
        #         pickle.dump(entity_dict, handle )
    return entity_dict
        
 
 
 
import pandas as pd
import numpy as np
 

class EntityLinkingSaver:
    def __init__(self,dataset_dir,top_k,text_base)  :
        with open(dataset_dir, 'r', encoding='utf-8') as fr:
            testData = json.load(fr)
            data_dict=review_json_to_product_dict(testData)
        self.data_dict=data_dict
        self.json_list=testData
        self.top_k=top_k
        self.text_base=text_base
         

    def add_retrieved_image(self,query_id ,query_image_path_list,semantic_results, retrieved_entity_name_list,corpus_dict ,score_dict):
        json_data=self.data_dict[query_id]
        
        json_data["image_score_dict"]=score_dict
        
        # if "product_id_with_similar_image" not in json_data or len(json_data["product_id_with_similar_image"])==0:
        # similar_ids_without_gold=[]
        json_data["product_id_with_similar_image_by_review"]=retrieved_entity_name_list
        self.data_dict[query_id]=json_data
    def add_retrieved_text(self,query_id,query_text,semantic_results,corpus_text_list,corpus_id_list,corpus_dict,corpus_text_corpus_id_dict):
        retrieved_corpus_id_list=[]
        score_dict={}
        for idx,hit in enumerate(semantic_results):
        
            corpus_text=corpus_text_list[hit['corpus_id']]
            corpus_id=corpus_id_list[hit['corpus_id']]
            if "cross-score" in hit:
                cross_score=hit['cross-score']
                cross_score_sigmoid=hit['cross_score_sigmoid']
            else:
                cross_score=None
                cross_score_sigmoid=None
            bi_score=hit['score']
            score_dict[corpus_id]={"idx":idx,"cross_score":cross_score,"bi_score":bi_score,"cross_score_sigmoid":cross_score_sigmoid}
            # corpus_id=corpus_text_corpus_id_dict[corpus_text]
            retrieved_corpus_id_list.append(corpus_id)
        json_data=self.data_dict[query_id]
        if self.text_base=="desc":
            json_data["product_id_with_similar_desc_by_review"]=retrieved_corpus_id_list
            json_data["desc_score_dict"]=score_dict
    
            
        else:
            json_data["product_id_with_similar_text_by_review"]=retrieved_corpus_id_list
            json_data["text_score_dict"]=score_dict
        
        self.data_dict[query_id]=json_data
            
    def save(self,out_path):
        # json_list=list(self.data_dict.values())
        
        # new_json_list=[]
        # for json_object in self.json_list:
 
        #     if "product_id_with_similar_image_by_review" in self.data_dict[review_id]:
        #         similar_product_ids_without_gold=self.data_dict[review_id]["product_id_with_similar_image_by_review"]
        #         json_object["product_id_with_similar_image_by_review"]=similar_product_ids_without_gold
        #     if "product_id_with_similar_text_by_review" in self.data_dict[review_id]:
        #         similar_product_ids_without_gold=self.data_dict[review_id]["product_id_with_similar_text_by_review"]
        #         json_object["product_id_with_similar_text_by_review"]=similar_product_ids_without_gold
        #     if "text_score_dict" in self.data_dict[review_id]:
        #         json_object["text_score_dict"]=self.data_dict[review_id]["text_score_dict"]
        #     if "image_score_dict" in self.data_dict[review_id]:
        #         json_object["image_score_dict"]=self.data_dict[review_id]["image_score_dict"]
           
        #     new_json_list.append(json_object)
        
        
        with open(out_path, 'w', encoding='utf-8') as fp:
            json.dump(list(self.data_dict.values()), fp, indent=4, cls=NpEncoder)  