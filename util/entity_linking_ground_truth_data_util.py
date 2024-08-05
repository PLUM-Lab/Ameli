
from urllib.parse import unquote
import pandas as pd
import os 
import hashlib
import re
import json
from tqdm import tqdm
import pickle

from transformers import CLIPFeatureExtractor, CLIPProcessor
 
from retrieval.eval.eval_msmarco_mocheg import load_qrels
from retrieval.data_util.datapoint import Entity, Mention, gen_entity_name 

import torch
from torch.utils.data import Dataset  
from torch.utils.data import DataLoader 
from pathlib import Path
import json
import pandas as pd
from util.common_util import json_to_dict

from util.data_util.data_util import NpEncoder


class Entity:
    def __init__(self,id,image_path_list,text,entity_category)  :
         
        self.id=id 
        self.image_path_list=image_path_list
        self.text=text 
        self.entity_candidate_id_list=[]
        self.entity_category=entity_category

class Query:
    def __init__(self,id,image_path_list,text,gold_entity_category,entity_candidate_name_list=[])  :
         
        self.id=id 
        self.image_path_list=image_path_list
        self.text=text 
        self.entity_candidate_id_list=entity_candidate_name_list
        self.gold_entity_category=gold_entity_category
        
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
 
 

def load_entity_linking_ground_truth_data(query_dir, corpus_dir,corpus_pickle_dir, image_dir):
    query_entity_dict=read_entity(query_dir,None,image_dir,use_cache=False )
    corpus_entity_dict=read_entity(corpus_dir,corpus_pickle_dir,image_dir)
    
    # mention_dict=read_mention(dataset_dir,entity_dict)
    
    
    dataset=QueryDataset(query_entity_dict)
   
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)  
    return train_dataloader,corpus_entity_dict

 
    
def read_entity(corpus_path,corpus_pickle_dir,image_dir,use_cache=True ):  
    
    if corpus_pickle_dir is not None:
        corpus_pickle_path=os.path.join(corpus_pickle_dir,"corpus_dict.pickle")
    else:
        corpus_pickle_path =None 
    if use_cache and os.path.exists(corpus_pickle_path):
         
        with open(corpus_pickle_path, 'rb') as handle:
            entity_dict = pickle.load(handle)
    else:
        entity_dict={}
        product_json_path = Path(
            corpus_path
        )      
        with open(product_json_path, 'r', encoding='utf-8') as fp:
            new_crawled_products_url_json_array = json.load(fp)
            for review_dataset_json   in tqdm(new_crawled_products_url_json_array):
                if "review_image_path" in review_dataset_json and review_dataset_json["review_image_path"] is not None:
                    image_path_list=[os.path.join(image_dir,image_name) for image_name in review_dataset_json[ 'review_image_path' ] ]
                    id=review_dataset_json["gold_entity_info"]["id"]
                    entity_category=review_dataset_json["product_category"]
                    text=review_dataset_json["overview_section"]["description"]
                    entity=Entity(id,image_path_list,text,entity_category)
                    entity_dict[id]=entity 
        if corpus_pickle_path is not None:
            with open(corpus_pickle_path, 'wb') as handle:
                pickle.dump(entity_dict, handle )
    return entity_dict
        
 
 
 
import pandas as pd
import numpy as np

class EntityLinkingGroundTruthSaver:
    def __init__(self,dataset_dir,top_k)  :
        with open(dataset_dir, 'r', encoding='utf-8') as fr:
            testData = json.load(fr)
            data_dict=json_to_dict(testData)
        self.data_dict=data_dict
        self.json_list=testData
        self.top_k=top_k
         

    def add_retrieved_image(self,query_id ,query_image_path_list,semantic_results, retrieved_entity_name_list,corpus_dict ):
        json_data=self.data_dict[query_id]
        product_id=json_data["id"]
        # if "product_id_with_similar_image" not in json_data or len(json_data["product_id_with_similar_image"])==0:
        similar_ids_without_gold=[]
        for retrieved_product_id in retrieved_entity_name_list:
            if retrieved_product_id!=product_id:
                similar_ids_without_gold.append(retrieved_product_id)
            if len(similar_ids_without_gold)==self.top_k-1:
                break 
        json_data["product_id_with_similar_image"]=similar_ids_without_gold
        self.data_dict[query_id]=json_data
   

    def save(self,out_path):
        # json_list=list(self.data_dict.values())
        
        new_json_list=[]
        for json_object in self.json_list:
            product_id=json_object["id"]
            if "product_id_with_similar_image" in self.data_dict[product_id]:
                similar_product_ids_without_gold=self.data_dict[product_id]["product_id_with_similar_image"]
                json_object["product_id_with_similar_image"]=similar_product_ids_without_gold
            new_json_list.append(json_object)
        
        
        with open(out_path, 'w', encoding='utf-8') as fp:
            json.dump(new_json_list, fp, indent=4, cls=NpEncoder)  