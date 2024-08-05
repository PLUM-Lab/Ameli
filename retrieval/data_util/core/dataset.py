

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
from transformers import BertTokenizer,AutoTokenizer
from transformers.image_transforms import convert_to_rgb
from util.common_util import review_json_to_product_dict
from util.read_example import get_father_dir  
from transformers import CLIPTokenizer
import torch
from torch.utils.data import Dataset  
from torch.utils.data import DataLoader 
from pathlib import Path
import json
import pandas as pd




def pad_to_max_num2(idx, max_candidate_num,entity_text_list,entity_img_list,image_list_for_one_entity,
                    attribute_list,negative_id_list,negative_entity_img_mask_list,negative_entity_image_mask_list_for_one_entity,positive_num):
    entity_mask = [1 for i in range(0,max_candidate_num-positive_num)]
 
    if idx< max_candidate_num-1-positive_num:
        for i in range(idx+1, max_candidate_num-positive_num):
            entity_text_list.append("pad")
            entity_img_list.extend(image_list_for_one_entity)   
            negative_entity_img_mask_list.extend(negative_entity_image_mask_list_for_one_entity)   
            attribute_list.append("pad")
            entity_mask[i]=0
            negative_id_list.append(-1)
    
    return entity_text_list,entity_img_list,entity_mask, attribute_list,negative_id_list,negative_entity_img_mask_list

def pad_to_max_num(idx, max_candidate_num,entity_text_list,entity_img_list,image, attribute_list,negative_id_list,positive_num,use_image=True):
    entity_mask = [1 for i in range(0,max_candidate_num-positive_num)]
 
    if idx< max_candidate_num-1-positive_num:
        for i in range(idx+1, max_candidate_num-positive_num):
            entity_text_list.append("pad")
            if use_image:    
                entity_img_list.append(image)   
            
            attribute_list.append("pad")
            entity_mask[i]=0
            negative_id_list.append(-1)
    
    return entity_text_list,entity_img_list,entity_mask, attribute_list,negative_id_list

class TripletDataset(Dataset):
    def __init__(self, queries, corpus,mode="train"):
        self.queries = queries
        self.queries_ids = list(queries.keys())
        self.corpus = corpus
        self.mode=mode

        for qid in self.queries:
            self.queries[qid]['pos'] = list(self.queries[qid]['pos'])
            self.queries[qid]['neg'] = list(self.queries[qid]['neg'])
            random.shuffle(self.queries[qid]['neg'])
        self.item_mapping_list=[i for i in range(len(self.queries_ids))]

    def __getitem__(self, item):
        query = self.queries[self.queries_ids[self.item_mapping_list[item]]]
        query_content=self.obtain_query(query['query'])
 
        
        pos_id = query['pos'].pop(0)    #Pop positive and add at end
        pos_object = self.obtain_corpus_by_idx(pos_id,query['qid']) 
        query['pos'].append(pos_id)

        neg_id = query['neg'].pop(0)    #Pop negative and add at end
        neg_object =self.obtain_corpus_by_idx(neg_id,query['qid'])    
        query['neg'].append(neg_id)

        return InputExample(texts=[query_content, pos_object, neg_object])
    def shuffle(self):
        random.shuffle(self.item_mapping_list)
        
    def obtain_corpus_by_idx(self,corpus_id,query_id):
        return self.corpus[corpus_id]
    def obtain_query(self,query_value):
        return query_value

    def __len__(self):
        if self.mode=="dry_run":
            return 10 
        else:
            return len(self.queries)    
        

class TripletImageDataset(TripletDataset):
    def __init__(self, queries, corpus,mode):
        super().__init__(queries,corpus,mode)
        
        
    def obtain_corpus_by_idx(self,corpus_id,query_id):
        
        image=Image.open(self.corpus[corpus_id])
        return image
    
    def obtain_query(self,query_image_path):
        query_image=Image.open(query_image_path)
        return query_image


class TripletEntityLevelImageDataset(TripletImageDataset):
    def __init__(self, queries, corpus,mode,data_folder,query_file_name,corpus_image_dir):
        super().__init__(queries,corpus,mode)
        self.corpus_image_dir=corpus_image_dir
        
        query_path=os.path.join(data_folder,query_file_name)
        product_json_path = Path(
            query_path
        )      
        with open(product_json_path, 'r', encoding='utf-8') as fp:
            new_crawled_products_url_json_array = json.load(fp)
            self.review_dataset_json_dict=review_json_to_product_dict(new_crawled_products_url_json_array)
        
        
    def obtain_corpus_by_idx(self,corpus_id,query_id):
        review_datast_json=self.review_dataset_json_dict[query_id]
        review_special_product_image_path=review_datast_json["review_special_product_info"]
        if str(corpus_id) in review_special_product_image_path:
            entity_image_path=os.path.join(self.corpus_image_dir, review_special_product_image_path[str(corpus_id)]["image_path"][0])
        else:
            entity_image_path=self.corpus[corpus_id]
        return Image.open(entity_image_path )
    
    

class TripletEntityLevelImageTextDataset(TripletImageDataset):
    def __init__(self, queries, corpus,mode,data_folder,query_file_name,corpus_image_dir):
        super().__init__(queries,corpus,mode)
        self.corpus_image_dir=corpus_image_dir
        
        query_path=os.path.join(data_folder,query_file_name)
        product_json_path = Path(
            query_path
        )      
        with open(product_json_path, 'r', encoding='utf-8') as fp:
            new_crawled_products_url_json_array = json.load(fp)
            self.review_dataset_json_dict=review_json_to_product_dict(new_crawled_products_url_json_array)
        
        
    def obtain_corpus_by_idx(self,corpus_id,query_id):
        review_datast_json=self.review_dataset_json_dict[query_id]
        review_special_product_image_path=review_datast_json["review_special_product_info"]
        text,image=self.corpus[corpus_id]
        if str(corpus_id) in review_special_product_image_path:
            entity_image_path=os.path.join(self.corpus_image_dir, review_special_product_image_path[str(corpus_id)]["image_path"][0])
        else:
            entity_image_path=image
        return [text,Image.open(entity_image_path )]
     
    def obtain_query(self,query_input):
        text,image=query_input
        query_image=Image.open(image)
        return [text,query_image]

class PairDataset(Dataset):
    def __init__(self, query_positive_negative_dict, corpus,mode="train"):
        self.query_positive_negative_dict = query_positive_negative_dict
        self.queries_ids = list(query_positive_negative_dict.keys())
        self.corpus = corpus
        self.mode=mode

        for qid in self.query_positive_negative_dict:
            self.query_positive_negative_dict[qid]['pos'] = list(self.query_positive_negative_dict[qid]['pos'])
            self.query_positive_negative_dict[qid]['neg'] = list(self.query_positive_negative_dict[qid]['neg'])
            random.shuffle(self.query_positive_negative_dict[qid]['neg'])

    def __getitem__(self, item):
        query = self.query_positive_negative_dict[self.queries_ids[item]]
        query_content=self.obtain_query(query['query'])
        relevance=random.randint(0, 1)
        if relevance==1:
            pos_id = query['pos'].pop(0)    #Pop positive and add at end
            train_object = self.obtain_corpus_by_idx(pos_id) 
            query['pos'].append(pos_id)
        else:
            neg_id = query['neg'].pop(0)    #Pop negative and add at end
            train_object =self.obtain_corpus_by_idx(neg_id)   
            query['neg'].append(neg_id)

        return InputExample(texts=[query_content, train_object], label=relevance)

    def obtain_corpus_by_idx(self,corpus_id):
        return self.corpus[corpus_id]
    def obtain_query(self,query_value):
        return query_value  
    def __len__(self):
        if self.mode=="dry_run":
            return 10  
        else:
            return len(self.query_positive_negative_dict)     
    
class PairImageDataset(PairDataset):
    
    def obtain_corpus_by_idx(self,corpus_id):
        
        image=Image.open(self.corpus[corpus_id])
        return image
    
    def obtain_query(self,query_image_path):
        query_image=Image.open(query_image_path)
        return query_image

class TripletDictDataset(Dataset):
    def __init__(self, query_positive_negative_dict, corpus,mode="train",media="img"):
        self.query_positive_negative_dict = query_positive_negative_dict
        self.queries_ids = list(query_positive_negative_dict.keys())
        self.corpus = corpus
        self.mode=mode
        self.media=media

        for qid in self.query_positive_negative_dict:
            self.query_positive_negative_dict[qid]['pos'] = list(self.query_positive_negative_dict[qid]['pos'])
            self.query_positive_negative_dict[qid]['neg'] = list(self.query_positive_negative_dict[qid]['neg'])
            random.shuffle(self.query_positive_negative_dict[qid]['neg'])

    def __getitem__(self, item):
        query = self.query_positive_negative_dict[self.queries_ids[item]]
        query_content=self.obtain_query(query['query'])
 
        pos_content_list=get_content(self.corpus,query['pos'],self.media)
        neg_content_list=get_content(self.corpus,query['neg'],self.media)
        

        return {'qid': self.queries_ids[item], 'query':query_content, "positive":pos_content_list,"negative": neg_content_list}

    def obtain_corpus_by_idx(self,corpus_id):
        return self.corpus[corpus_id]
    def obtain_query(self,query_value):
        if self.media=="txt":
            return query_value
        else:
            return Image.open(query_value)

    def __len__(self):
        if self.mode=="dry_run":
            return 10 
        else:
            return len(self.query_positive_negative_dict)    

def get_content(corpus,corpus_id_list,media):
    corpus_content_list=[]
    for corpus_id in corpus_id_list:
        if media=="img":
            corpus_content_list.append(Image.open(corpus[corpus_id]))
        else:
            corpus_content_list.append( corpus[corpus_id] )
    return corpus_content_list

# text tokenizer: bert, image tokenizer: clip
class MultimodalDatasetBERTCLIP(Dataset):
    def __init__(self, queries, corpus,tokenizer,max_len,processor,max_candidate_num,has_attribute,use_image,candidate_mode="standard",max_attribute_num=5,is_train=False,is_random_positive=True,is_train_many_negative_per_time=True):
        self.queries = queries
        self.queries_ids = list(queries.keys())
        self.corpus = corpus
        self.tokenizer=tokenizer
        self.is_train=is_train
        self.is_train_many_negative_per_time=is_train_many_negative_per_time
        self.use_image=use_image
        self.is_random_positive=is_random_positive
        self.max_candidate_num=max_candidate_num
        self.max_len=max_len
        self.has_attribute=has_attribute
        self.max_attribute_num= max_attribute_num
        
        for qid in self.queries:
            self.queries[qid]['pos'] = list(self.queries[qid]['pos'])
            self.queries[qid]['neg'] = list(self.queries[qid]['neg'])
            random.shuffle(self.queries[qid]['neg'])
        self.train_transformer = transforms.Compose([
            transforms.RandomCrop(224,pad_if_needed=True),#, padding=4
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
     
            ])
        self.test_transformer = transforms.Compose([
            transforms.RandomCrop(224, padding=4),#
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
     
            ])
        self.processor=processor
        self.candidate_mode=candidate_mode # ="end2end"
        
    def __join(self, d):
        return ". ".join([" ".join(i) for i in d.items()])
    
    def gen_image_tensor(self,mention_image_path,transfroms):
        mention_image=Image.open(mention_image_path )
        try:
            mention_image_tensor=self.processor(images=mention_image, return_tensors="pt")
        except Exception as e:
            print(f"{mention_image_path} {e}")
            raise e 
        return mention_image_tensor["pixel_values"] 
    def get_entity_image_tensor(self,entity_img_list,transfroms):
        # if self.use_image:
        return self.processor(images=entity_img_list, return_tensors="pt")["pixel_values"]
        # else:
        #     return torch.zeros(len(entity_img_list))
    def __getitem__(self, item):
        if self.is_train:
            transfroms=self.train_transformer
        else:
            transfroms=self.test_transformer
        query_id=self.queries_ids[item]
        query = self.queries[query_id]
        query_object = query['query']
        if len(query_object)==3:
            [mention_text,mention_image_path,mention_attributes]=query_object
        elif len(query_object)==4:
            [mention_text,mention_image_path,mention_attributes,mention_special_product_image_path]=query_object
        elif len(query_object)==7:
            [mention_text,mention_image_path,mention_attributes,surface_form,mention_special_product_image_path,nli_score_vector_dict,text_image_retrieval_score_list_dict]=query_object
            
        mention_tokens = self.tokenizer(mention_text,  
                                               add_special_tokens=True,   # Adds [CLS] and [SEP] token to every input text
                                               max_length=self.max_len, 
                                               truncation=True, 
                                               return_tensors='pt',
                                               padding="max_length")
        mention_image_pixel=self.gen_image_tensor(mention_image_path[0],transfroms)
        
        negative_entity_text_list,negative_entity_img_list,negative_entity_mask_list,negative_attribute_list,negative_id_list=self.gen_negative_list(query,mention_special_product_image_path)
        
        if self.candidate_mode!="end2end" or len(negative_id_list)==self.max_candidate_num-1:
            pos_id = query['pos'].pop(0)    #Pop positive and add at end
            pos_object = self.obtain_corpus_by_idx(pos_id) 
            query['pos'].append(pos_id)
            [positive_entity_text,positive_entity_image_path,positive_entity_attribute]=pos_object
            positive_entity_image=self.get_product_image(positive_entity_image_path,pos_id,mention_special_product_image_path)   
            entity_text_list,entity_img_list, entity_mask_tensor,label,entities_attribute_list ,entity_id_list= self.insert_positive_into_list(positive_entity_text,positive_entity_image,positive_entity_attribute,negative_entity_text_list,
                                                                                                                                                 negative_entity_img_list,negative_entity_mask_list,negative_attribute_list,negative_id_list,pos_id)
        else:
            entity_mask_tensor=self._convert_mask_to_tensor(negative_entity_mask_list)
            entity_id_list=torch.Tensor(negative_id_list)
            entity_text_list,entity_img_list,  entities_attribute_list =negative_entity_text_list,negative_entity_img_list,negative_attribute_list
            label=10
            
        entity_image_pixel=self.get_entity_image_tensor(entity_img_list,transfroms)
        entity_token_tensor = self.tokenizer(entity_text_list,  
                                               add_special_tokens=True,   # Adds [CLS] and [SEP] token to every input text
                                               max_length=self.max_len, 
                                               truncation=True, 
                                               return_tensors='pt',
                                               padding="max_length" )
        ( idx,_)=entity_token_tensor["input_ids"].shape
        
        if self.has_attribute:
            mention_attributes = self.__join(mention_attributes)
            mention_attribute_tokens = self.tokenizer(mention_attributes,  
                                                add_special_tokens=True,   # Adds [CLS] and [SEP] token to every input text
                                                max_length=self.max_len, 
                                                truncation=True, 
                                                return_tensors='pt',
                                                padding="max_length" )
            entities_attribute_tokens_list = self.tokenizer(entities_attribute_list,  
                                                add_special_tokens=True,   # Adds [CLS] and [SEP] token to every input text
                                                max_length=self.max_len, 
                                                truncation=True, 
                                                return_tensors='pt',
                                                padding="max_length" )
        else:
            mention_attribute_tokens=mention_tokens 
            entities_attribute_tokens_list=entity_token_tensor
        # if len(entity_token_tensor)>22 or len(entity_image_tensor)>0:
        #     print("ERROR")
        return  mention_tokens,mention_image_pixel ,entity_token_tensor,entity_image_pixel, entity_mask_tensor, label, mention_attribute_tokens, entities_attribute_tokens_list ,entity_id_list,query_id

    def insert_positive_into_list(self,positive_entity_text,positive_entity_image,positive_entity_attribute,negative_entity_text_list,
                                  negative_entity_img_list,negative_entity_mask,negative_attribute_list,negative_id_list,pos_id):
        if self.is_random_positive:
            positive_idx=random.randint(0, self.max_candidate_num-1)
        else:
            positive_idx=0
        negative_entity_text_list.insert(positive_idx,positive_entity_text)
        if self.use_image:
            negative_entity_img_list.insert(positive_idx,positive_entity_image)
        negative_attribute_list.insert(positive_idx,positive_entity_attribute)
        negative_entity_mask.insert(positive_idx,1)
        negative_entity_mask_tensor=self._convert_mask_to_tensor(negative_entity_mask)
        negative_id_list.insert(positive_idx,pos_id)
        negative_id_list=torch.Tensor(negative_id_list)
 
        return negative_entity_text_list,negative_entity_img_list ,negative_entity_mask_tensor ,positive_idx,negative_attribute_list,negative_id_list
    
    def _convert_mask_to_tensor(self,negative_entity_mask):
        negative_entity_mask_tensor=torch.Tensor(negative_entity_mask)
        negative_entity_mask_tensor=negative_entity_mask_tensor.bool()
        return negative_entity_mask_tensor 
    
    def _attribute_to_text(self,attributes):
        return ". ".join([": ".join(i) for i in attributes.items()])
    def get_product_image(self,entity_image_path_list,entity_id=None,review_special_product_image_path=None):
        # if self.use_image:
        return Image.open(entity_image_path_list[0] )
        # else:
        #     return None
    def gen_negative_list(self,query,mention_special_product_image_path=None):
        image=None
        negative_entity_text_list=[]
        negative_entity_img_list=[]
        if not self.is_train_many_negative_per_time:
            neg_id = query['neg'].pop(0)    #Pop negative and add at end   
            query['neg'].append(neg_id)
            neg_id_set=set()
            neg_id_set.add(neg_id)
        else:
            neg_id_set = query['neg']     #Pop negative and add at end
        if self.candidate_mode=="end2end" and len(neg_id_set)==self.max_candidate_num:
            positive_num=0
        else:
            positive_num=1
        negative_attribute_list=[]
        negative_id_list=[]
        idx=0
        if len(neg_id_set)>0:
            # neg_id_set=set() 
            # neg_id_set.add(24375) 
            for idx,neg_id in enumerate(neg_id_set):
                negative_id_list.append(neg_id)
                neg_object =self.obtain_corpus_by_idx(neg_id)   
                [negative_entity_text,negative_entity_image_path,negative_entity_attribute]=neg_object
                negative_entity_text_list.append(negative_entity_text)
                negative_attribute_list.append(self._attribute_to_text(negative_entity_attribute))
                if self.use_image:
                    image=self.get_product_image(negative_entity_image_path,neg_id,mention_special_product_image_path)   
                    negative_entity_img_list.append(image)      
                if idx >=self.max_candidate_num-positive_num-1:
                    break
        else:
            pad_entity_id=24375
            neg_object =self.obtain_corpus_by_idx(pad_entity_id)   
            [negative_entity_text,negative_entity_image_path,negative_entity_attribute]=neg_object
            image=Image.open(negative_entity_image_path[0] )
            idx=-1
        negative_entity_text_list,negative_entity_img_list,negative_entity_mask_list,negative_attribute_list,negative_id_list=pad_to_max_num(idx,self.max_candidate_num,negative_entity_text_list,negative_entity_img_list,image, negative_attribute_list,negative_id_list,positive_num,use_image=self.use_image)
        return negative_entity_text_list,negative_entity_img_list,negative_entity_mask_list ,negative_attribute_list,negative_id_list
    
    def obtain_corpus_by_idx(self,corpus_id):
        return self.corpus[corpus_id]

    def __len__(self):
        return len(self.queries)    
     

# text tokenizer: bert, image tokenizer: clip
class MultimodalDatasetBERTCrossEncoder(MultimodalDatasetBERTCLIP):
     def __getitem__(self, item):
        query_id=self.queries_ids[item]
        query = self.queries[query_id]
        query_object = query['query']
        [mention_text,mention_image_path,mention_attributes,surface_form,mention_special_product_image_path,nli_score_vector_dict,text_image_retrieval_score_list_dict]=query_object
        # if self.use_image:
        mention_image_pixel=self.gen_image_tensor(mention_image_path[0],None)
        # else:
        #     mention_image_pixel=torch.zeros(1)
        negative_entity_text_list,negative_entity_img_list,negative_entity_mask_list,negative_attribute_list,negative_id_list=self.gen_negative_list(query,mention_special_product_image_path)
        
        if self.candidate_mode!="end2end" or len(negative_id_list)==self.max_candidate_num-1:
            pos_id = query['pos'].pop(0)    #Pop positive and add at end
            pos_object = self.obtain_corpus_by_idx(pos_id) 
            query['pos'].append(pos_id)
            [positive_entity_text,positive_entity_image_path,positive_entity_attribute]=pos_object
            positive_entity_image=self.get_product_image(positive_entity_image_path,pos_id,mention_special_product_image_path)   
            entity_text_list,entity_img_list, entity_mask_tensor,label,entities_attribute_list ,entity_id_list= self.insert_positive_into_list(positive_entity_text,positive_entity_image,positive_entity_attribute,negative_entity_text_list,
                                                                                                                                                 negative_entity_img_list,negative_entity_mask_list,negative_attribute_list,negative_id_list,pos_id)
        else:
            entity_mask_tensor=self._convert_mask_to_tensor(negative_entity_mask_list)
            entity_id_list=torch.Tensor(negative_id_list)
            entity_text_list,entity_img_list,  entities_attribute_list =negative_entity_text_list,negative_entity_img_list,negative_attribute_list
            label=10
            
        mention_text_list=[mention_text for i in range(len(entity_text_list))]
        entity_image_pixel=self.get_entity_image_tensor(entity_img_list,None)
        entity_token_tensor = self.tokenizer(mention_text_list,entity_text_list,  
                                               add_special_tokens=True,   # Adds [CLS] and [SEP] token to every input text
                                               truncation=True, 
                                               return_tensors='pt',
                                               padding="max_length" )
        return  entity_token_tensor,mention_image_pixel ,entity_token_tensor,entity_image_pixel, entity_mask_tensor, label,entity_token_tensor,entity_token_tensor,entity_id_list,query_id
# separate tokenizer: bert for text, resnet for image
class MultimodalDatasetBERTResnet(MultimodalDatasetBERTCLIP):    
    def __init__(self, queries, corpus, tokenizer, max_len, processor, max_candidate_num, has_attribute, use_image, candidate_mode="standard", max_attribute_num=5):
        super().__init__(queries, corpus, tokenizer, max_len, processor, max_candidate_num, has_attribute, use_image, candidate_mode, max_attribute_num)
        self.train_transformer =  transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        self.test_transformer = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    def gen_image_tensor(self,mention_image_path,transfroms):
        # mention_image_path = "/content/drive/MyDrive/ANLP/multimodal_entity_linking/data/wikidiverse_cleaned/wikinewsImgs/c19b363713a701e7b5231ae31bd6fb3a.jpg"


        mention_image=Image.open(mention_image_path )
        mention_image =  convert_to_rgb(mention_image)  
        mention_image_tensor=transfroms(mention_image)
         
        return mention_image_tensor 
    def get_entity_image_tensor(self,entity_img_list,transfroms):
        mention_image_tensor_list=[]
        for entity_img in entity_img_list:
            entity_img =  convert_to_rgb(entity_img) 
            mention_image_tensor_list.append(transfroms(entity_img) )
             
        return torch.stack(mention_image_tensor_list)
    

# text tokenizer: bert, image tokenizer: resnet,    select image
class MultimodalDatasetSelectImageBERTResnet(MultimodalDatasetBERTResnet):        
    def get_product_image(self,entity_image_path_list,entity_id=None,review_special_product_image_path=None):
        if str(entity_id) in review_special_product_image_path:
            entity_image_path=review_special_product_image_path[str(entity_id)]
        else:
            entity_image_path=entity_image_path_list[0]
        return Image.open(entity_image_path )
    
    
# same tokenizer:  clip for image text
class MultimodalDatasetCLIP(MultimodalDatasetBERTCLIP):    
     
    
        
    def _gen_entity_list(self,query,mention_special_product_image_path):
        

        negative_entity_text_list,negative_entity_img_list,negative_entity_mask_list,negative_attribute_text_list,negative_id_list=self.gen_negative_list(query,mention_special_product_image_path)
        if self.candidate_mode!="end2end" or len(negative_id_list)==self.max_candidate_num-1:
            pos_id = query['pos'].pop(0)    
            pos_object = self.obtain_corpus_by_idx(pos_id) 
            query['pos'].append(pos_id)
            [positive_entity_text,positive_entity_image_path,positive_entity_attribute]=pos_object
            if self.use_image:
                positive_entity_image=self.get_product_image(positive_entity_image_path,pos_id,mention_special_product_image_path)   
            else:
                positive_entity_image=None
            positive_entity_attribute_text=self._attribute_to_text(positive_entity_attribute)
            entity_text_list,entity_img_list, entity_mask_tensor,label,entities_attribute_list ,entity_id_list= self.insert_positive_into_list(positive_entity_text,positive_entity_image,positive_entity_attribute_text,negative_entity_text_list,
                                                                                                                                               negative_entity_img_list,negative_entity_mask_list,negative_attribute_text_list,negative_id_list,pos_id)
        else:
            entity_mask_tensor=self._convert_mask_to_tensor(negative_entity_mask_list)
            entity_id_list=torch.Tensor(negative_id_list)
            entity_text_list,entity_img_list,  entities_attribute_list =negative_entity_text_list,negative_entity_img_list,negative_attribute_text_list
            label=self.max_candidate_num
        return entity_text_list,entity_img_list, entity_mask_tensor,label,entities_attribute_list,entity_id_list

    def _process(self,mention_entity_list_text_attribute_list,mention_entity_list_image_list):
        return self.processor(text=mention_entity_list_text_attribute_list, images=mention_entity_list_image_list, return_tensors="pt", padding="max_length",truncation=True)

    def __getitem__(self, item):
        mention_entity_list_text_list=[]
        mention_entity_list_attribute_list=[]
        mention_entity_list_image_list=[]
        query_id=self.queries_ids[item]
        query = self.queries[query_id]
        query_object = query['query']
        if len(query_object)==3:
            [mention_text,mention_image_path,mention_attributes]=query_object
            mention_special_product_image_path=None
        elif len(query_object)==7:
            [mention_text,mention_image_path,mention_attributes,surface_form,mention_special_product_image_path,nli_score_vector_dict,text_image_retrieval_score_list_dict]=query_object
        else:
            [mention_text,mention_image_path,mention_attributes,mention_special_product_image_path]=query_object
        mention_image=Image.open(mention_image_path[0] )
        mention_entity_list_text_list.append(mention_text)
        mention_entity_list_attribute_list.append(self._attribute_to_text(mention_attributes))
        mention_entity_list_image_list.append(mention_image)
         
        entity_text_list,entity_img_list, entity_mask_tensor,label,entity_list_attribute_list ,entity_id_list= self._gen_entity_list(query,mention_special_product_image_path)
        mention_entity_list_text_list.extend(entity_text_list)
        mention_entity_list_image_list.extend(entity_img_list)
        mention_entity_list_attribute_list.extend(entity_list_attribute_list)
        if self.has_attribute:
            mention_entity_list_text_list.extend(mention_entity_list_attribute_list)
        processed_inputs=self._process(mention_entity_list_text_list,mention_entity_list_image_list)
        
        return  processed_inputs,label,entity_mask_tensor ,entity_id_list,query_id
    
# text tokenizer: clip, image tokenizer: clip    , select image
class MultimodalDatasetSelectImage(MultimodalDatasetCLIP):
    def __init__(self, queries, corpus, tokenizer, max_len, processor, max_candidate_num, has_attribute, use_image, candidate_mode="standard", max_attribute_num=5, is_train=False, is_random_positive=True, is_train_many_negative_per_time=True,mode="train"):
        super().__init__(queries, corpus, tokenizer, max_len, processor, max_candidate_num, has_attribute, use_image, candidate_mode, max_attribute_num, is_train, is_random_positive, is_train_many_negative_per_time)
        self.mode=mode
        
    def get_product_image(self,entity_image_path_list,entity_id=None,review_special_product_image_path=None):
        if str(entity_id) in review_special_product_image_path:
            entity_image_path=review_special_product_image_path[str(entity_id)]
        else:
            entity_image_path=entity_image_path_list[0]
        return Image.open(entity_image_path )
    def __len__(self):
        if self.mode in ["dry_run"]:
            return 10
        else:
            return len(self.queries)   
    
from transformers.image_transforms import convert_to_rgb
   
class MultimodalDatasetSelectImageFLAVA(MultimodalDatasetSelectImage):
    def _process(self,mention_entity_list_text_attribute_list,mention_entity_list_image_list):
        cleaned_mention_entity_list_image_list=[convert_to_rgb(image) for image in mention_entity_list_image_list]
        return self.processor(text=mention_entity_list_text_attribute_list, images=cleaned_mention_entity_list_image_list, return_tensors="pt",padding="max_length", max_length=77,truncation=True)



#sbert_clip, sbert_bert
class MultimodalDatasetSBERTBERTCLIPAttribute(MultimodalDatasetSelectImage):
    def _process(self,mention_entity_list_text_attribute_list,mention_entity_list_image_list):
        text_processed=self.tokenizer(mention_entity_list_text_attribute_list,  
                                               add_special_tokens=True,   # Adds [CLS] and [SEP] token to every input text
                                               max_length=self.max_len, 
                                               truncation=True, 
                                               return_tensors='pt',
                                               padding="max_length")
        if self.use_image:
            image_processed= self.processor(images=mention_entity_list_image_list, return_tensors="pt", padding="max_length",truncation=True)
        else:
            image_processed={}
        image_processed["input_ids"]=text_processed["input_ids"]
        image_processed["attention_mask"]=text_processed["attention_mask"]
        return image_processed
    def __getitem__(self, item):
        mention_entity_list_text_list=[]
        mention_entity_list_attribute_list=[]
        mention_entity_list_image_list=[]
        query_id=self.queries_ids[item]
        query = self.queries[query_id]
        query_object = query['query']
        if len(query_object)==3:
            [mention_text,mention_image_path,mention_attributes]=query_object
            mention_special_product_image_path=None
        elif len(query_object)==7:
            [mention_text,mention_image_path,mention_attributes,surface_form,mention_special_product_image_path,nli_score_vector_dict,text_image_retrieval_score_list_dict]=query_object
        else:
            [mention_text,mention_image_path,mention_attributes,mention_special_product_image_path]=query_object
        if self.use_image:
            mention_image=Image.open(mention_image_path[0] )
            mention_entity_list_image_list.append(mention_image)
        mention_entity_list_text_list.append(mention_text)
        mention_entity_list_attribute_list.append(self._attribute_to_text(mention_attributes))
        
         
        entity_text_list,entity_img_list, entity_mask_tensor,label,entity_list_attribute_list ,entity_id_list= self._gen_entity_list(query,mention_special_product_image_path)
        mention_entity_list_text_list.extend(entity_text_list)
        mention_entity_list_image_list.extend(entity_img_list)
        mention_entity_list_attribute_list.extend(entity_list_attribute_list)
        if self.has_attribute:
            mention_entity_list_text_list.extend(mention_entity_list_attribute_list)
        processed_inputs=self._process(mention_entity_list_text_list,mention_entity_list_image_list)
        
        return  processed_inputs,label,entity_mask_tensor ,entity_id_list,query_id

    
     

        
# text tokenizer: clip, image tokenizer: clip,    to obtain text-to-image cross representations for relation    
class MultimodalDatasetSelectImageV2TEL(MultimodalDatasetSelectImage) :   
    def __getitem__(self, item):
        mention_entity_list_text_list=[]
        mention_entity_list_image_list=[]
        query_id=self.queries_ids[item]
        query = self.queries[query_id]
        query_object = query['query']
        if len(query_object)==3:
            [mention_text,mention_image_path,mention_attributes]=query_object
            mention_special_product_image_path=None
        else:
            [mention_text,mention_image_path,mention_attributes,mention_special_product_image_path]=query_object
        mention_image=Image.open(mention_image_path[0] )
        mention_entity_list_image_list.append(mention_image)
         
        entity_text_list,entity_img_list, entity_mask_tensor,label,entity_list_attribute_list ,entity_id_list= self._gen_entity_list(query,mention_special_product_image_path)
        mention_entity_list_text_list.extend(entity_text_list)
        processed_inputs=self._process(mention_entity_list_text_list,mention_entity_list_image_list)
        
        return  processed_inputs,label,entity_mask_tensor ,entity_id_list,query_id
    
#   image tokenizer: resnet    
class MultimodalDataset_resnet(MultimodalDatasetCLIP):
    def __init__(self, queries, corpus, tokenizer, max_len, processor, max_candidate_num, has_attribute, use_image, candidate_mode="standard", max_attribute_num=5):
        super().__init__(queries, corpus, tokenizer, max_len, processor, max_candidate_num, has_attribute, use_image, candidate_mode, max_attribute_num)
        self.processor= transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    def _process(self,mention_entity_list_text_attribute_list,mention_entity_list_image_list):
        pixel_value_list=[]
        for mention_entity_list_image in mention_entity_list_image_list:
            one_image =  convert_to_rgb(mention_entity_list_image)  
            pixel_value=self.processor(  one_image ) 
            pixel_value_list.append(pixel_value)
        pixel_value=torch.stack(pixel_value_list)
        processed_inputs={}
        processed_inputs["pixel_values"]=pixel_value
        processed_inputs["attention_mask"]=pixel_value
        processed_inputs["input_ids"]=pixel_value
    
        return processed_inputs

    
#multiple images
class MultimodalDatasetAttenBasedMultiImage(MultimodalDatasetCLIP):       
    def __init__(self, queries, corpus,tokenizer,max_len,processor,max_candidate_num,has_attribute,has_image,candidate_mode, max_attribute_num):
        super().__init__(queries,corpus,tokenizer,max_len,processor,max_candidate_num,has_attribute,has_image,candidate_mode, max_attribute_num)
        self.max_mention_image_num=2
        self.max_entity_image_num=5
        
    def _gen_image_list(self,mention_image_path_list,max_mention_image_num,is_to_tensor=False):
       
        mention_image_list=[]
        i=0
        entity_mask = [0 for i in range(0,max_mention_image_num )]
        for i,mention_image_path in enumerate(mention_image_path_list):
            if i>=max_mention_image_num:
                break
            mention_image=Image.open(mention_image_path)
            mention_image_list.append(mention_image)
            entity_mask[i]=1
        while i<max_mention_image_num-1:
            pad_image=mention_image
            mention_image_list.append(pad_image)
            i+=1
        if is_to_tensor:
            entity_mask=torch.Tensor(entity_mask)
            entity_mask=entity_mask.bool()
        return mention_image_list ,entity_mask
    
    def _gen_entity_list(self,query):
        max_entity_image_num=self.max_entity_image_num
        

        negative_entity_text_list,negative_entity_img_list,negative_entity_mask_list,negative_attribute_text_list,negative_id_list,negative_entity_img_mask_list=self.gen_negative_list(query)
        if self.candidate_mode!="end2end" or len(negative_id_list)==self.max_candidate_num-1:
            pos_id = query['pos'].pop(0)    
            pos_object = self.obtain_corpus_by_idx(pos_id) 
            query['pos'].append(pos_id)
            [positive_entity_text,positive_entity_image_path,positive_entity_attribute]=pos_object
            positive_entity_image_list,positive_entity_image_mask=self._gen_image_list(positive_entity_image_path,max_entity_image_num)
            # positive_entity_image=Image.open(positive_entity_image_path[0] )
            positive_entity_attribute_text=self._attribute_to_text(positive_entity_attribute)
            entity_text_list,entity_img_list, entity_mask_tensor,label,entities_attribute_list ,entity_id_list, entity_img_mask_list= self.insert_positive_into_list(positive_entity_text,
                                                                                                                                           positive_entity_image_list,positive_entity_attribute_text,negative_entity_text_list,
                                                                                                                                           negative_entity_img_list,negative_entity_mask_list,
                                                                                                                                           negative_attribute_text_list,negative_id_list,pos_id,negative_entity_img_mask_list,positive_entity_image_mask)
        else:
            entity_mask_tensor=self._convert_mask_to_tensor(negative_entity_mask_list)
            entity_id_list=torch.Tensor(negative_id_list)
            entity_text_list,entity_img_list,  entities_attribute_list =negative_entity_text_list,negative_entity_img_list,negative_attribute_text_list
            label=10
            entity_img_mask_list=negative_entity_img_mask_list

        
        
        return entity_text_list,entity_img_list, entity_mask_tensor,label,entities_attribute_list,entity_id_list,entity_img_mask_list

    def gen_negative_list(self,query):
        max_entity_image_num=self.max_entity_image_num
 

        negative_entity_text_list=[]
        negative_entity_img_list=[]
        negative_entity_img_mask_list=[]
        neg_id_set = query['neg']     #Pop negative and add at end
        if self.candidate_mode=="end2end" and len(neg_id_set)==self.max_candidate_num:
            positive_num=0
        else:
            positive_num=1
        negative_attribute_list=[]
        negative_id_list=[]
        idx=0
        if len(neg_id_set)>0:
            # neg_id_set=set() 
            # neg_id_set.add(24375) 
            for idx,neg_id in enumerate(neg_id_set):
                negative_id_list.append(neg_id)
                neg_object =self.obtain_corpus_by_idx(neg_id)   
                [negative_entity_text,negative_entity_image_path,negative_entity_attribute]=neg_object
                negative_entity_text_list.append(negative_entity_text)
                negative_attribute_list.append(self._attribute_to_text(negative_entity_attribute))
                negative_entity_image_list_for_one_entity,negative_entity_image_mask_list_for_one_entity=self._gen_image_list(negative_entity_image_path,max_entity_image_num)
                negative_entity_img_list.extend(negative_entity_image_list_for_one_entity)      
                negative_entity_img_mask_list.extend(negative_entity_image_mask_list_for_one_entity)
                if idx >=self.max_candidate_num-positive_num-1:
                    break
        else:
            pad_entity_id=24375
            neg_object =self.obtain_corpus_by_idx(pad_entity_id)   
            [negative_entity_text,negative_entity_image_path,negative_entity_attribute]=neg_object
            image=Image.open(negative_entity_image_path[0] )
            negative_entity_image_list_for_one_entity=[image for i in range(0,max_entity_image_num )]
            negative_entity_image_mask_list_for_one_entity=[0 for i in range(0,max_entity_image_num )]
            idx=-1
        negative_entity_text_list,negative_entity_img_list,negative_entity_mask_list,negative_attribute_list,negative_id_list,negative_entity_img_mask_list=pad_to_max_num2(idx,self.max_candidate_num,negative_entity_text_list,
                                                                                                                                             negative_entity_img_list,negative_entity_image_list_for_one_entity,
                                                                                                                                             negative_attribute_list,negative_id_list,negative_entity_img_mask_list,
                                                                                                                                             negative_entity_image_mask_list_for_one_entity,positive_num)
        return negative_entity_text_list,negative_entity_img_list,negative_entity_mask_list ,negative_attribute_list,negative_id_list,negative_entity_img_mask_list
    
    def __getitem__(self, item):
        mention_entity_list_text_list=[]
        mention_entity_list_attribute_list=[]
        mention_entity_list_image_list=[]
        mention_entity_list_image_mask_list=[]
        query_id=self.queries_ids[item]
        query = self.queries[query_id]
        query_object = query['query']
        [mention_text,mention_image_path_list,mention_attributes]=query_object
        mention_image_list,mention_image_mask=self._gen_image_list(mention_image_path_list, self.max_mention_image_num)
        mention_entity_list_text_list.append(mention_text)
        mention_entity_list_attribute_list.append(self._attribute_to_text(mention_attributes))
        mention_entity_list_image_list.extend(mention_image_list)
        mention_entity_list_image_mask_list.extend(mention_image_mask)
        
        entity_text_list,entity_img_list, entity_mask_tensor,label,entity_list_attribute_list ,entity_id_list, entity_img_mask_list= self._gen_entity_list(query)
        
        mention_entity_list_text_list.extend(entity_text_list)
        mention_entity_list_image_list.extend(entity_img_list)
        mention_entity_list_attribute_list.extend(entity_list_attribute_list)
        mention_entity_list_image_mask_list.extend(entity_img_mask_list)
        if self.has_attribute:
            mention_entity_list_text_list.extend(mention_entity_list_attribute_list)
        mention_entity_list_image_mask_list=torch.Tensor(mention_entity_list_image_mask_list)
        mention_entity_list_image_mask_list=mention_entity_list_image_mask_list.bool()
        processed_inputs=self._process(mention_entity_list_text_list,mention_entity_list_image_list)
        
        return  processed_inputs,label,entity_mask_tensor ,entity_id_list,query_id,mention_entity_list_image_mask_list
    
    def insert_positive_into_list(self,positive_entity_text,positive_entity_image,positive_entity_attribute,negative_entity_text_list,
                                  negative_entity_img_list,negative_entity_mask,negative_attribute_list,negative_id_list,pos_id,negative_entity_img_mask_list,positive_entity_image_mask):
        positive_idx=random.randint(0, self.max_candidate_num-1)
        negative_entity_text_list.insert(positive_idx,positive_entity_text)
        
        negative_attribute_list.insert(positive_idx,positive_entity_attribute)
        negative_entity_mask.insert(positive_idx,1)
        negative_entity_mask_tensor=torch.Tensor(negative_entity_mask)
        negative_entity_mask_tensor=negative_entity_mask_tensor.bool()
        negative_id_list.insert(positive_idx,pos_id)
        negative_id_list=torch.Tensor(negative_id_list)
        negative_entity_img_list[positive_idx*len(positive_entity_image_mask):positive_idx*len(positive_entity_image_mask)]=positive_entity_image
        negative_entity_img_mask_list[positive_idx*len(positive_entity_image_mask):positive_idx*len(positive_entity_image_mask)]=positive_entity_image_mask
        
        return negative_entity_text_list,negative_entity_img_list ,negative_entity_mask_tensor ,positive_idx,negative_attribute_list,negative_id_list,negative_entity_img_mask_list


 

#query-based multiple images. Query: use entity attributes to be the entity text for querying review
class MultimodalDatasetQueryBasedMultiImage(MultimodalDatasetAttenBasedMultiImage):   
    def __init__(self, queries, corpus,tokenizer,max_len,processor,max_candidate_num,has_attribute,has_image,candidate_mode, max_attribute_num):
        super().__init__(queries,corpus,tokenizer,max_len,processor,max_candidate_num,has_attribute,has_image,candidate_mode, max_attribute_num)
        self.max_mention_image_num=1
        self.max_entity_image_num=1 
    def _attribute_dict_to_text_list(self,attributes):
        return  [i for i in attributes.values()] 
    def pad_to_max_num(self,idx, max_candidate_num,entity_text_list,entity_img_list,image_list_for_one_entity,
                    attribute_list,negative_id_list,negative_entity_img_mask_list,negative_entity_image_mask_list_for_one_entity
                      ,positive_num  ):
        entity_mask = [1 for i in range(0,max_candidate_num-positive_num)]
    
        if idx< max_candidate_num-1-positive_num:
            for i in range(idx+1, max_candidate_num-positive_num):
                entity_text_list.append("pad")
                # entity_img_list.append(image)   
                entity_img_list.extend(image_list_for_one_entity)   
                negative_entity_img_mask_list.extend(negative_entity_image_mask_list_for_one_entity)   
                for j in range(self.max_attribute_num):
                    attribute_list.append("pad")
                entity_mask[i]=0
                negative_id_list.append(-1)
        
        return entity_text_list,entity_img_list,entity_mask, attribute_list,negative_id_list,negative_entity_img_mask_list
    def gen_negative_list(self,query,mention_attributes):
        
        negative_entity_text_list=[]
        negative_entity_img_list=[]
        negative_entity_img_mask_list=[]
        neg_id_set = query['neg']     #Pop negative and add at end
        if self.candidate_mode=="end2end" and len(neg_id_set)==self.max_candidate_num:
            positive_num=0
        else:
            positive_num=1
        # if len(neg_id_set)>9:
        #     print("stop")
        negative_attribute_list=[]
        negative_id_list=[]
        idx=0
        if len(neg_id_set)>0:
            # neg_id_set=set() 
            # neg_id_set.add(24375) 
            for idx,neg_id in enumerate(neg_id_set):
                negative_id_list.append(neg_id)
                neg_object =self.obtain_corpus_by_idx(neg_id)   
                [negative_entity_text,negative_entity_image_path,negative_entity_attribute]=neg_object
                # negative_entity_text_list.append(negative_entity_text)
                selected_attribute_dict=self._choose_attribute_by_mention_attributes(mention_attributes,negative_entity_attribute)
                
                negative_entity_text_list.append(self._attribute_to_text(selected_attribute_dict)+negative_entity_text)#"<|endoftext|>"+
                if self.use_image:
                    negative_entity_image_list_for_one_entity,negative_entity_image_mask_list_for_one_entity=self._gen_image_list(negative_entity_image_path,self.max_entity_image_num)
                    negative_entity_img_list.extend(negative_entity_image_list_for_one_entity)      
                    negative_entity_img_mask_list.extend(negative_entity_image_mask_list_for_one_entity)
                    # image=Image.open(negative_entity_image_path[0] )
                    # negative_entity_img_list.append(image)    
                else:
                    negative_entity_image_list_for_one_entity=[]
                    negative_entity_image_mask_list_for_one_entity=[0]
                if idx >=self.max_candidate_num-positive_num-1:
                    break
        else:
            pad_entity_id=24375
            neg_object =self.obtain_corpus_by_idx(pad_entity_id)   
            [negative_entity_text,negative_entity_image_path,negative_entity_attribute]=neg_object
            negative_entity_image_list_for_one_entity=[Image.open(negative_entity_image_path[0] )]
            negative_entity_image_mask_list_for_one_entity=[0]
            idx=-1
        negative_entity_text_list,negative_entity_img_list,negative_entity_mask_list,negative_attribute_list,negative_id_list,negative_entity_img_mask_list=self.pad_to_max_num(idx,self.max_candidate_num,negative_entity_text_list,negative_entity_img_list,negative_entity_image_list_for_one_entity, negative_attribute_list,negative_id_list,negative_entity_img_mask_list,negative_entity_image_mask_list_for_one_entity,positive_num)
        return negative_entity_text_list,negative_entity_img_list,negative_entity_mask_list ,negative_attribute_list,negative_id_list,negative_entity_img_mask_list
    def _choose_attribute_by_mention_attributes(self,mention_attributes, entity_attribute_dict):
        selected_attribute_dict={}
        valid_num=0
        for key in ["Brand","Product Name","Color"]:
            if key in entity_attribute_dict:
                valid_num+=1
                selected_attribute_dict[key]=entity_attribute_dict[key]
        for key in mention_attributes.keys():
            if valid_num<self.max_attribute_num:
                if key not in selected_attribute_dict and key in entity_attribute_dict:
                    selected_attribute_dict[key]=entity_attribute_dict[key]
                    valid_num+=1
        for key in entity_attribute_dict.keys():
            if valid_num<self.max_attribute_num:
                if key not in selected_attribute_dict :
                    selected_attribute_dict[key]=entity_attribute_dict[key]
                    valid_num+=1
            else:
                break
        for i in range(valid_num,self.max_attribute_num):
            selected_attribute_dict[str(i)]=list(entity_attribute_dict.values())[0]
        return selected_attribute_dict
    
    def insert_positive_into_list(self,positive_entity_text,positive_entity_image,positive_entity_attribute,negative_entity_text_list,
                                  negative_entity_img_list,negative_entity_mask,negative_attribute_list,negative_id_list,pos_id,negative_entity_img_mask_list,positive_entity_image_mask):
        positive_idx=random.randint(0, self.max_candidate_num-1)
        negative_entity_text_list.insert(positive_idx,positive_entity_text)
        if self.use_image:
            # negative_entity_img_list.insert(positive_idx,positive_entity_image)
            negative_entity_img_list[positive_idx*len(positive_entity_image_mask):positive_idx*len(positive_entity_image_mask)]=positive_entity_image
            negative_entity_img_mask_list[positive_idx*len(positive_entity_image_mask):positive_idx*len(positive_entity_image_mask)]=positive_entity_image_mask
        # negative_attribute_list.insert(positive_idx,positive_entity_attribute)
        negative_entity_mask.insert(positive_idx,1)
        negative_entity_mask_tensor=torch.Tensor(negative_entity_mask)
        negative_entity_mask_tensor=negative_entity_mask_tensor.bool()
        negative_id_list.insert(positive_idx,pos_id)
        negative_id_list=torch.Tensor(negative_id_list)
        negative_attribute_list[positive_idx*len(positive_entity_attribute):positive_idx*len(positive_entity_attribute)]=positive_entity_attribute
        return negative_entity_text_list,negative_entity_img_list ,negative_entity_mask_tensor ,positive_idx,negative_attribute_list,negative_id_list,negative_entity_img_mask_list
    def _attribute_to_text(self,attributes):
        return ". ".join([str(i) for i in attributes.values()])
        
    def _gen_entity_list(self,query,mention_attributes):
        max_entity_image_num=self.max_entity_image_num
        
        
        # negative_entity_text_list,negative_entity_img_list,negative_entity_mask_list,negative_attribute_text_list,negative_id_list,negative_entity_img_mask_list=self.gen_negative_list(query)
        # entity_text_list,entity_img_list, entity_mask_tensor,label,entities_attribute_list ,entity_id_list, entity_img_mask_list= self.insert_positive_into_list(positive_entity_text,
        #                                                                                                                                    positive_entity_image_list,positive_entity_attribute_text,negative_entity_text_list,
        #                                                                                                                                    negative_entity_img_list,negative_entity_mask_list,
        #                                                                                                                                    negative_attribute_text_list,negative_id_list,pos_id,negative_entity_img_mask_list,positive_entity_image_mask)

        negative_entity_text_list,negative_entity_img_list,negative_entity_mask_list,negative_attribute_text_list,negative_id_list,negative_entity_img_mask_list=self.gen_negative_list(query,mention_attributes)
        if self.candidate_mode!="end2end" or len(negative_id_list)==self.max_candidate_num-1:
            pos_id = query['pos'].pop(0)    
            pos_object = self.obtain_corpus_by_idx(pos_id) 
            query['pos'].append(pos_id)
            [positive_entity_text,positive_entity_image_path,positive_entity_attribute]=pos_object
            if self.use_image:
                positive_entity_image_list,positive_entity_image_mask=self._gen_image_list(positive_entity_image_path,max_entity_image_num)
                # positive_entity_image=Image.open(positive_entity_image_path[0] )
            else:
                positive_entity_image_list,positive_entity_image_mask=None,None
            selected_attribute_dict=self._choose_attribute_by_mention_attributes(mention_attributes,positive_entity_attribute)
            # positive_entity_attribute_list=self._attribute_to_text(selected_attribute_dict)
            positive_entity_attribute_list=self._attribute_dict_to_text_list(selected_attribute_dict)
            positive_entity_text=self._attribute_to_text(selected_attribute_dict)+positive_entity_text 
            entity_text_list,entity_img_list, entity_mask_tensor,label,entities_attribute_list ,entity_id_list,entity_img_mask_list= self.insert_positive_into_list(positive_entity_text,positive_entity_image_list,
                                                                                                                                           positive_entity_attribute_list,negative_entity_text_list,negative_entity_img_list,negative_entity_mask_list,negative_attribute_text_list,negative_id_list,pos_id,negative_entity_img_mask_list,positive_entity_image_mask)
        
        else:
            entity_mask_tensor=self._convert_mask_to_tensor(negative_entity_mask_list)
            entity_id_list=torch.Tensor(negative_id_list)
            entity_text_list,entity_img_list,  entities_attribute_list =negative_entity_text_list,negative_entity_img_list,negative_attribute_text_list
            label=10
            entity_img_mask_list=negative_entity_img_mask_list
        
        
        return entity_text_list,entity_img_list, entity_mask_tensor,label,entities_attribute_list,entity_id_list,entity_img_mask_list

    def _process(self,mention_entity_list_text_attribute_list,mention_entity_list_image_list):
        if self.use_image:
            return self.processor(text=mention_entity_list_text_attribute_list, images=mention_entity_list_image_list, return_tensors="pt", padding="max_length",truncation=True)
        else:
            return self.processor(text=mention_entity_list_text_attribute_list,  return_tensors="pt", padding="max_length",truncation=True)

    def __getitem__(self, item):
        mention_entity_list_text_list=[]
        mention_entity_list_attribute_list=[]
        mention_entity_list_image_list=[]
        mention_entity_list_image_mask_list=[]
        query_id=self.queries_ids[item]
        query = self.queries[query_id]
        query_object = query['query']
        [mention_text,mention_image_path,mention_attributes,surface_form]=query_object
        
        mention_text_input=surface_form+". "+self._attribute_to_text(mention_attributes)+". "+mention_text#+"<|endoftext|>"
        mention_entity_list_text_list.append(mention_text_input)
        if self.use_image:
            mention_image_list,mention_image_mask=self._gen_image_list(mention_image_path, self.max_mention_image_num)
            # mention_image=Image.open(mention_image_path[0] )
            mention_entity_list_image_list.extend(mention_image_list)
            mention_entity_list_image_mask_list.extend(mention_image_mask)
         
        entity_text_list,entity_img_list, entity_mask_tensor,label,entity_list_attribute_list ,entity_id_list,entity_img_mask_list= self._gen_entity_list(query,mention_attributes)
        mention_entity_list_text_list.extend(entity_text_list)
        mention_entity_list_image_list.extend(entity_img_list)
        mention_entity_list_attribute_list.extend(entity_list_attribute_list)
        if self.has_attribute:
            mention_entity_list_text_list.extend(mention_entity_list_attribute_list)
        
        processed_inputs=self._process(mention_entity_list_text_list,mention_entity_list_image_list)
        mention_entity_list_image_mask_list.extend(entity_img_mask_list) 
        mention_entity_list_image_mask_list=torch.Tensor(mention_entity_list_image_mask_list)
        mention_entity_list_image_mask_list=mention_entity_list_image_mask_list.bool()
        return  processed_inputs,label,entity_mask_tensor ,entity_id_list,query_id
    
    
    
#query based bert
class MultimodalDatasetQueryBasedBERT(MultimodalDatasetQueryBasedMultiImage):     
    def __init__(self, queries, corpus,tokenizer,max_len,processor,max_candidate_num,has_attribute,use_image,candidate_mode,args, max_attribute_num):
        super().__init__(queries, corpus,tokenizer,max_len,processor,max_candidate_num,has_attribute,use_image,candidate_mode, max_attribute_num)
        self.tokenizer=  AutoTokenizer.from_pretrained(args.pre_trained_dir)
    def _process(self,mention_entity_list_text_attribute_list,mention_entity_list_image_list):
        if self.use_image:
            return self.processor(text=mention_entity_list_text_attribute_list, images=mention_entity_list_image_list, return_tensors="pt", padding="max_length",truncation=True)
        else:
            return self.tokenizer(mention_entity_list_text_attribute_list,  
                                               add_special_tokens=True,   # Adds [CLS] and [SEP] token to every input text
                                               max_length=self.max_len, 
                                               truncation=True, 
                                               return_tensors='pt',
                                               padding="max_length" )
            
            
#image cross encoder
class MultimodalDatasetMultiImageCrossEncoder(MultimodalDatasetAttenBasedMultiImage):   
    def __init__(self, queries, corpus,tokenizer,max_len,processor,max_candidate_num,has_attribute,has_image,candidate_mode, max_attribute_num):
        super().__init__(queries,corpus,tokenizer,max_len,processor,max_candidate_num,has_attribute,has_image,candidate_mode, max_attribute_num)
        self.max_mention_image_num=2
        self.max_entity_image_num=2 
    def _attribute_dict_to_text_list(self,attributes):
        return  [i for i in attributes.values()] 
    def pad_to_max_num(self,idx, max_candidate_num,entity_text_list,entity_img_list,image_list_for_one_entity,
                    attribute_list,negative_id_list,negative_entity_img_mask_list,negative_entity_image_mask_list_for_one_entity
                       ,positive_num ):
        entity_mask = [1 for i in range(0,max_candidate_num-positive_num)]
    
        if idx< max_candidate_num-1-positive_num:
            for i in range(idx+1, max_candidate_num-1):
                entity_text_list.append("pad")
                # entity_img_list.append(image)   
                entity_img_list.extend(image_list_for_one_entity)   
                negative_entity_img_mask_list.extend(negative_entity_image_mask_list_for_one_entity)   
                for j in range(self.max_attribute_num):
                    attribute_list.append("pad")
                entity_mask[i]=0
                negative_id_list.append(-1)
        
        return entity_text_list,entity_img_list,entity_mask, attribute_list,negative_id_list,negative_entity_img_mask_list
    def gen_negative_list(self,query,mention_attributes):
  
        negative_entity_text_list=[]
        negative_entity_img_list=[]
        negative_entity_img_mask_list=[]
        neg_id_set = query['neg']     #Pop negative and add at end
        if self.candidate_mode=="end2end" and len(neg_id_set)==self.max_candidate_num:
            positive_num=0
        else:
            positive_num=1
        negative_attribute_list=[]
        negative_id_list=[]
        idx=0
        if len(neg_id_set)>0:
            # neg_id_set=set() 
            # neg_id_set.add(24375) 
            for idx,neg_id in enumerate(neg_id_set):
                negative_id_list.append(neg_id)
                neg_object =self.obtain_corpus_by_idx(neg_id)   
                [negative_entity_text,negative_entity_image_path,negative_entity_attribute]=neg_object
                # negative_entity_text_list.append(negative_entity_text)
                selected_attribute_dict=self._choose_attribute_by_mention_attributes(mention_attributes,negative_entity_attribute)
                
                negative_entity_text_list.append(self._attribute_to_text(selected_attribute_dict)+negative_entity_text)#"<|endoftext|>"+
                if self.use_image:
                    negative_entity_image_list_for_one_entity,negative_entity_image_mask_list_for_one_entity=self._gen_image_list(negative_entity_image_path,self.max_entity_image_num)
                    negative_entity_img_list.extend(negative_entity_image_list_for_one_entity)      
                    negative_entity_img_mask_list.extend(negative_entity_image_mask_list_for_one_entity)
                    # image=Image.open(negative_entity_image_path[0] )
                    # negative_entity_img_list.append(image)    
                else:
                    negative_entity_image_list_for_one_entity=[0,0]
                    negative_entity_image_mask_list_for_one_entity=[0,0]
                if idx >=self.max_candidate_num-positive_num-1:
                    break
        else:
            pad_entity_id=24375
            neg_object =self.obtain_corpus_by_idx(pad_entity_id)   
            [negative_entity_text,negative_entity_image_path,negative_entity_attribute]=neg_object
            negative_entity_image_list_for_one_entity=[Image.open(negative_entity_image_path[0] ),Image.open(negative_entity_image_path[0] )]
            negative_entity_image_mask_list_for_one_entity=[0,0]
            # print(f"no negative {query}")
            idx=-1
        negative_entity_text_list,negative_entity_img_list,negative_entity_mask_list,negative_attribute_list,negative_id_list,negative_entity_img_mask_list=self.pad_to_max_num(idx,self.max_candidate_num,negative_entity_text_list,negative_entity_img_list,negative_entity_image_list_for_one_entity, negative_attribute_list,negative_id_list,negative_entity_img_mask_list,negative_entity_image_mask_list_for_one_entity,positive_num)
        return negative_entity_text_list,negative_entity_img_list,negative_entity_mask_list ,negative_attribute_list,negative_id_list,negative_entity_img_mask_list
    def _choose_attribute_by_mention_attributes(self,mention_attributes, entity_attribute_dict):
        selected_attribute_dict={}
        valid_num=0
        for key in ["Brand","Product Name","Color"]:
            if key in entity_attribute_dict:
                valid_num+=1
                selected_attribute_dict[key]=entity_attribute_dict[key]
        for key in mention_attributes.keys():
            if valid_num<self.max_attribute_num:
                if key not in selected_attribute_dict and key in entity_attribute_dict:
                    selected_attribute_dict[key]=entity_attribute_dict[key]
                    valid_num+=1
        for key in entity_attribute_dict.keys():
            if valid_num<self.max_attribute_num:
                if key not in selected_attribute_dict :
                    selected_attribute_dict[key]=entity_attribute_dict[key]
                    valid_num+=1
            else:
                break
        for i in range(valid_num,self.max_attribute_num):
            selected_attribute_dict[str(i)]=list(entity_attribute_dict.values())[0]
        return selected_attribute_dict
    
    def insert_positive_into_list(self,positive_entity_text,positive_entity_image,positive_entity_attribute,negative_entity_text_list,
                                  negative_entity_img_list,negative_entity_mask,negative_attribute_list,negative_id_list,pos_id,negative_entity_img_mask_list,positive_entity_image_mask):
        positive_idx=random.randint(0, self.max_candidate_num-1)
        negative_entity_text_list.insert(positive_idx,positive_entity_text)
        if self.use_image:
            # negative_entity_img_list.insert(positive_idx,positive_entity_image)
            negative_entity_img_list[positive_idx*len(positive_entity_image_mask):positive_idx*len(positive_entity_image_mask)]=positive_entity_image
            negative_entity_img_mask_list[positive_idx*len(positive_entity_image_mask):positive_idx*len(positive_entity_image_mask)]=positive_entity_image_mask
        # negative_attribute_list.insert(positive_idx,positive_entity_attribute)
        negative_entity_mask.insert(positive_idx,1)
        negative_entity_mask_tensor=torch.Tensor(negative_entity_mask)
        negative_entity_mask_tensor=negative_entity_mask_tensor.bool()
        negative_id_list.insert(positive_idx,pos_id)
        negative_id_list=torch.Tensor(negative_id_list)
        negative_attribute_list[positive_idx*len(positive_entity_attribute):positive_idx*len(positive_entity_attribute)]=positive_entity_attribute
        return negative_entity_text_list,negative_entity_img_list ,negative_entity_mask_tensor ,positive_idx,negative_attribute_list,negative_id_list,negative_entity_img_mask_list
    def _attribute_to_text(self,attributes):
        return ". ".join([str(i) for i in attributes.values()])
        
    def _gen_entity_list(self,query,mention_attributes):
        max_entity_image_num=self.max_entity_image_num
        negative_entity_text_list,negative_entity_img_list,negative_entity_mask_list,negative_attribute_text_list,negative_id_list,negative_entity_img_mask_list=self.gen_negative_list(query,mention_attributes)
        if self.candidate_mode!="end2end" or len(negative_id_list)==self.max_candidate_num-1:
            pos_id = query['pos'].pop(0)    
            pos_object = self.obtain_corpus_by_idx(pos_id) 
            query['pos'].append(pos_id)
            [positive_entity_text,positive_entity_image_path,positive_entity_attribute]=pos_object
            if self.use_image:
                positive_entity_image_list,positive_entity_image_mask=self._gen_image_list(positive_entity_image_path,max_entity_image_num)
                # positive_entity_image=Image.open(positive_entity_image_path[0] )
            else:
                positive_entity_image_list,positive_entity_image_mask=None,None
            selected_attribute_dict=self._choose_attribute_by_mention_attributes(mention_attributes,positive_entity_attribute)
            # positive_entity_attribute_list=self._attribute_to_text(selected_attribute_dict)
            positive_entity_attribute_list=self._attribute_dict_to_text_list(selected_attribute_dict)
            positive_entity_text=self._attribute_to_text(selected_attribute_dict)+positive_entity_text 
            
            entity_text_list,entity_img_list, entity_mask_tensor,label,entities_attribute_list ,entity_id_list,entity_img_mask_list= self.insert_positive_into_list(positive_entity_text,positive_entity_image_list,
                                                                                                                                            positive_entity_attribute_list,negative_entity_text_list,negative_entity_img_list,negative_entity_mask_list,negative_attribute_text_list,negative_id_list,pos_id,negative_entity_img_mask_list,positive_entity_image_mask)
        else:
            entity_mask_tensor=self._convert_mask_to_tensor(negative_entity_mask_list)
            entity_id_list=torch.Tensor(negative_id_list)
            entity_text_list,entity_img_list,  entities_attribute_list =negative_entity_text_list,negative_entity_img_list,negative_attribute_text_list
            label=self.max_candidate_num
            entity_img_mask_list=negative_entity_img_mask_list
        return entity_text_list,entity_img_list, entity_mask_tensor,label,entities_attribute_list,entity_id_list,entity_img_mask_list

    def _process(self,mention_entity_list_text_attribute_list,mention_entity_list_image_list):
        if self.use_image:
            return self.processor(text=mention_entity_list_text_attribute_list, images=mention_entity_list_image_list, return_tensors="pt", padding="max_length",truncation=True)
        else:
            return self.processor(text=mention_entity_list_text_attribute_list,  return_tensors="pt", padding="max_length",truncation=True)

    def __getitem__(self, item):
        mention_entity_list_text_list=[]
        mention_entity_list_attribute_list=[]
        mention_entity_list_image_list=[]
        mention_entity_list_image_mask_list=[]
        query_id=self.queries_ids[item]
        query = self.queries[query_id]
        query_object = query['query']
        [mention_text,mention_image_path,mention_attributes,surface_form]=query_object
        
        mention_text_input=surface_form+". "+self._attribute_to_text(mention_attributes)+". "+mention_text#+"<|endoftext|>"
        mention_entity_list_text_list.append(mention_text_input)
        if self.use_image:
            mention_image_list,mention_image_mask=self._gen_image_list(mention_image_path, self.max_mention_image_num)
            # mention_image=Image.open(mention_image_path[0] )
            mention_entity_list_image_list.extend(mention_image_list)
            mention_entity_list_image_mask_list.extend(mention_image_mask)
         
        entity_text_list,entity_img_list, entity_mask_tensor,label,entity_list_attribute_list ,entity_id_list,entity_img_mask_list= self._gen_entity_list(query,mention_attributes)
        mention_entity_list_text_list.extend(entity_text_list)
        mention_entity_list_image_list.extend(entity_img_list)
        mention_entity_list_attribute_list.extend(entity_list_attribute_list)
        if self.has_attribute:
            mention_entity_list_text_list.extend(mention_entity_list_attribute_list)
        
        processed_inputs=self._process(mention_entity_list_text_list,mention_entity_list_image_list)
        mention_entity_list_image_mask_list.extend(entity_img_mask_list) 
        mention_entity_list_image_mask_list=torch.Tensor(mention_entity_list_image_mask_list)
        mention_entity_list_image_mask_list=mention_entity_list_image_mask_list.bool()
        return  processed_inputs,label,entity_mask_tensor ,entity_id_list,query_id,mention_entity_list_image_mask_list
    

     
                
                

    
# bert based (review, attribute) pairs 
class ReviewAttributePairDataset(MultimodalDatasetCLIP):    
  
    def _attribute_dict_to_text_list(self,attributes):
        return  [i for i in attributes.values()] 
    def pad_to_max_num(self,idx, max_candidate_num,entity_text_list,entity_img_list,image, attribute_list,negative_id_list,positive_num):
        entity_mask = [1 for i in range(0,max_candidate_num-positive_num)]
    
        if idx< max_candidate_num-1-positive_num:
            for i in range(idx+1, max_candidate_num-positive_num):
                # entity_text_list.append("pad")
                entity_img_list.append(image)   
                # entity_img_list.extend(image_list_for_one_entity)   
                # negative_entity_img_mask_list.extend(negative_entity_image_mask_list_for_one_entity)   
                for j in range(self.max_attribute_num+1):
                    entity_text_list.append("pad")
                entity_mask[i]=0
                negative_id_list.append(-1)
        
        return entity_text_list,entity_img_list,entity_mask, attribute_list,negative_id_list
    def gen_negative_list(self,query,mention_attributes,mention_text_input,surface_form):
        neg_id_set = query['neg']  
        if self.candidate_mode=="end2end" and len(neg_id_set)==self.max_candidate_num:
            positive_num=0
        else:
            positive_num=1
        negative_entity_text_list=[]
        negative_entity_img_list=[]
        negative_entity_img_mask_list=[]
           #Pop negative and add at end
        negative_attribute_list=[]
        negative_id_list=[]
        idx=0
        if len(neg_id_set)>0:
      
            for idx,neg_id in enumerate(neg_id_set):
                negative_id_list.append(neg_id)
                neg_object =self.obtain_corpus_by_idx(neg_id)   
                [negative_entity_text,negative_entity_image_path,negative_entity_attribute]=neg_object
                # if len(negative_entity_attribute)==0:
                #     print("error")
                review_attribute_pair_list=self._gen_review_attribute_pair_list( negative_entity_text,negative_entity_attribute,mention_attributes,mention_text_input,surface_form)  
                negative_entity_text_list.extend(review_attribute_pair_list)
                if self.use_image:
        
                    image=Image.open(negative_entity_image_path[0] )
                    negative_entity_img_list.append(image)    
                else:
                    image=None
                if idx >=self.max_candidate_num-positive_num-1:
                    break
        else:
            pad_entity_id=24375
            neg_object =self.obtain_corpus_by_idx(pad_entity_id)   
            [negative_entity_text,negative_entity_image_path,negative_entity_attribute]=neg_object
            # review_attribute_pair_list=self._gen_review_attribute_pair_list( negative_entity_text,negative_entity_attribute,mention_attributes,mention_text_input)  
            image=Image.open(negative_entity_image_path[0] )
            
            idx=-1
        negative_entity_text_list,negative_entity_img_list,negative_entity_mask_list,negative_attribute_list,negative_id_list=self.pad_to_max_num(idx,self.max_candidate_num,negative_entity_text_list,negative_entity_img_list,image, negative_attribute_list,negative_id_list,positive_num)
        return negative_entity_text_list,negative_entity_img_list,negative_entity_mask_list ,negative_attribute_list,negative_id_list
    def _choose_attribute_by_mention_attributes(self,mention_attributes, entity_attribute_dict):
        selected_attribute_dict={}
        valid_num=0
        if self.max_attribute_num==1: #TODO for no attribute setting
            if valid_num<self.max_attribute_num:
                for key in [ "Product Title"]:
                    if key in entity_attribute_dict:
                        selected_attribute_dict[key]=entity_attribute_dict[key]
        else:
            for key in ["Product Name","Brand"]:
                if valid_num<self.max_attribute_num:
                    if key in entity_attribute_dict:
                        valid_num+=1
                        selected_attribute_dict[key]=entity_attribute_dict[key]
            
            for key in mention_attributes.keys():
                if valid_num<self.max_attribute_num:
                    if key not in selected_attribute_dict and key in entity_attribute_dict:
                        selected_attribute_dict[key]=entity_attribute_dict[key]
                        valid_num+=1
            for key in entity_attribute_dict.keys():
                if valid_num<self.max_attribute_num:
                    if key not in selected_attribute_dict :
                        selected_attribute_dict[key]=entity_attribute_dict[key]
                        valid_num+=1
                else:
                    break
            for i in range(valid_num,self.max_attribute_num):
                # if len(entity_attribute_dict.values())==0:
                #     print(f"error")
                selected_attribute_dict[str(i)]=list(entity_attribute_dict.values())[0]
        return selected_attribute_dict
    
    def insert_positive_into_list(self,positive_review_attribute_pair_list,positive_entity_image,positive_entity_attribute,negative_entity_text_list,
                                  negative_entity_img_list,negative_entity_mask,negative_attribute_list,negative_id_list,pos_id):
        positive_idx=random.randint(0, self.max_candidate_num-1)
        # negative_entity_text_list.insert(positive_idx,positive_entity_text)
        negative_entity_text_list[positive_idx*len(positive_review_attribute_pair_list):positive_idx*len(positive_review_attribute_pair_list)]=positive_review_attribute_pair_list
        if self.use_image:
            negative_entity_img_list.insert(positive_idx,positive_entity_image)
        # negative_attribute_list.insert(positive_idx,positive_entity_attribute)
        negative_entity_mask.insert(positive_idx,1)
        negative_entity_mask_tensor=torch.Tensor(negative_entity_mask)
        negative_entity_mask_tensor=negative_entity_mask_tensor.bool()
        negative_id_list.insert(positive_idx,pos_id)
        negative_id_list=torch.Tensor(negative_id_list)
        negative_attribute_list[positive_idx*len(positive_entity_attribute):positive_idx*len(positive_entity_attribute)]=positive_entity_attribute
        return negative_entity_text_list,negative_entity_img_list ,negative_entity_mask_tensor ,positive_idx,negative_attribute_list,negative_id_list
    
    def _gen_positive_entity_list(self,query,mention_attributes,mention_text_input,surface_form):
        pos_id = query['pos'].pop(0)    
        pos_object = self.obtain_corpus_by_idx(pos_id) 
        query['pos'].append(pos_id)
        review_attribute_pair_list=[]
        [positive_entity_text,positive_entity_image_path,positive_entity_attribute]=pos_object
        if self.use_image:
            positive_entity_image=Image.open(positive_entity_image_path[0] )
        else:
            positive_entity_image=None
        review_attribute_pair_list=self._gen_review_attribute_pair_list(positive_entity_text,positive_entity_attribute,mention_attributes,mention_text_input,surface_form) 
        return review_attribute_pair_list,pos_id,positive_entity_text,positive_entity_image,[]
    
    def _gen_review_attribute_pair_list(self,positive_entity_text,positive_entity_attribute,mention_attributes,mention_text_input,surface_form)    :
        review_attribute_pair_list=[]
        selected_attribute_dict=self._choose_attribute_by_mention_attributes(mention_attributes,positive_entity_attribute)
        for attribute_key,attribute_value in  selected_attribute_dict.items():
            review_attribute_pair_list.append(mention_text_input+"[SEP]"+attribute_key+" of "+surface_form+" is "+attribute_value)
        review_attribute_pair_list.append(positive_entity_text+"[SEP]"+mention_text_input)
        return review_attribute_pair_list
    
    def _gen_entity_list(self,query,mention_attributes,mention_text_input,surface_form):
        review_attribute_pair_list=[]
        
        negative_entity_text_list,negative_entity_img_list,negative_entity_mask_list,negative_attribute_text_list,negative_id_list=self.gen_negative_list(query,mention_attributes,mention_text_input,surface_form)
        if self.candidate_mode!="end2end" or len(negative_id_list)==self.max_candidate_num-1:
            positive_review_attribute_pair_list,pos_id,positive_entity_text,positive_entity_image,positive_entity_attribute_list=self._gen_positive_entity_list(query,mention_attributes,mention_text_input,surface_form)
            entity_text_list,entity_img_list, entity_mask_tensor,label,entities_attribute_list ,entity_id_list= self.insert_positive_into_list(positive_review_attribute_pair_list,positive_entity_image,positive_entity_attribute_list,negative_entity_text_list,negative_entity_img_list,negative_entity_mask_list,negative_attribute_text_list,negative_id_list,pos_id)    
        else:
            entity_mask_tensor=self._convert_mask_to_tensor(negative_entity_mask_list)
            entity_id_list=torch.Tensor(negative_id_list)
            entity_text_list,entity_img_list,  entities_attribute_list =negative_entity_text_list,negative_entity_img_list,negative_attribute_text_list
            label=self.max_candidate_num
            
        
        return entity_text_list,entity_img_list, entity_mask_tensor,label,entities_attribute_list,entity_id_list

    
    def _process(self,mention_entity_list_text_attribute_list,mention_entity_list_image_list):
        if self.use_image:
            return self.processor(text=mention_entity_list_text_attribute_list, images=mention_entity_list_image_list, return_tensors="pt", padding="max_length",truncation=True)
        else:
            return self.tokenizer(mention_entity_list_text_attribute_list,  
                                               add_special_tokens=True,   # Adds [CLS] and [SEP] token to every input text
                                               max_length=self.max_len, 
                                               truncation=True, 
                                               return_tensors='pt',
                                               padding="max_length" )
    def __getitem__(self, item):
        mention_entity_list_text_list=[]
        mention_entity_list_image_list=[]
        query_id=self.queries_ids[item]
        query = self.queries[query_id]
        query_object = query['query']
        [mention_text,mention_image_path,mention_attributes,surface_form]=query_object
        if self.has_attribute:
            mention_text_input=surface_form+". \n"+self._attribute_to_text(mention_attributes)+". \n"+ mention_text
        else:
            mention_text_input=surface_form+". \n"  + mention_text
        entity_text_list,entity_img_list, entity_mask_tensor,label,entity_list_attribute_list ,entity_id_list= self._gen_entity_list(query,mention_attributes,mention_text_input,surface_form)
        mention_entity_list_text_list.extend(entity_text_list)
        mention_entity_list_image_list.extend(entity_img_list)
   
        
        processed_inputs=self._process(mention_entity_list_text_list,mention_entity_list_image_list)
        
        return  processed_inputs,label,entity_mask_tensor ,entity_id_list,query_id                
    
    
class ReviewAttributePairWithImageDataset(ReviewAttributePairDataset):        
    def __init__(self, queries, corpus,tokenizer,max_len,processor,max_candidate_num,has_attribute,has_image,candidate_mode, max_attribute_num,mode=None):
        super().__init__(queries,corpus,tokenizer,max_len,processor,max_candidate_num,has_attribute,has_image,candidate_mode, max_attribute_num)
        self.max_mention_image_num=1
        self.max_entity_image_num=1 
        self.mode=mode
        
    def pad_to_max_num(self,idx, max_candidate_num,entity_text_list,entity_img_list,image_list_for_one_entity, attribute_list,
                       negative_id_list,negative_entity_img_mask_list,negative_entity_image_mask_list_for_one_entity
                       ,positive_num ,negative_retrieval_nli_score_list_list,text_image_retrieval_nli_score_list):
        entity_mask = [1 for i in range(0,max_candidate_num-positive_num)]
    
        if idx< max_candidate_num-1-positive_num:
            for i in range(idx+1, max_candidate_num-positive_num):
                # entity_text_list.append("pad")
                entity_img_list.extend(image_list_for_one_entity)   
                negative_entity_img_mask_list.extend(negative_entity_image_mask_list_for_one_entity) 
    
                for j in range(self.max_attribute_num+1):
                    entity_text_list.append("pad")
                entity_mask[i]=0
                negative_id_list.append(-1)
                negative_retrieval_nli_score_list_list.append(text_image_retrieval_nli_score_list)
        
        return entity_text_list,entity_img_list,entity_mask, attribute_list,negative_id_list,negative_entity_img_mask_list,negative_retrieval_nli_score_list_list
    def gen_negative_list(self,query,mention_attributes,mention_text_input,surface_form,review_special_product_image_path,nli_score_vector_dict,text_image_retrieval_score_list_dict):
        neg_id_set = query['neg']  
        if self.candidate_mode=="end2end" and len(neg_id_set)==self.max_candidate_num:
            positive_num=0
        else:
            positive_num=1
        negative_retrieval_nli_score_list_list=[]
        negative_entity_text_list=[]
        negative_entity_img_list=[]
        negative_entity_img_mask_list=[]
           #Pop negative and add at end
        negative_attribute_list=[]
        negative_id_list=[]
        idx=0
        if len(neg_id_set)>0:
      
            for idx,neg_id in enumerate(neg_id_set):
                negative_id_list.append(neg_id)
                neg_object =self.obtain_corpus_by_idx(neg_id)   
                [negative_entity_text,negative_entity_image_path,negative_entity_attribute]=neg_object
                # if len(negative_entity_attribute)==0:
                #     print("error")
                review_attribute_pair_list=self._gen_review_attribute_pair_list( negative_entity_text,negative_entity_attribute,mention_attributes,mention_text_input,surface_form)  
                negative_entity_text_list.extend(review_attribute_pair_list)
                if self.use_image:
                    negative_entity_image_list_for_one_entity,negative_entity_image_mask_list_for_one_entity=self._gen_image_list(negative_entity_image_path,self.max_entity_image_num,neg_id,review_special_product_image_path)
                    negative_entity_img_list.extend(negative_entity_image_list_for_one_entity)      
                    negative_entity_img_mask_list.extend(negative_entity_image_mask_list_for_one_entity)
                    
                else:
                    negative_entity_image_list_for_one_entity=[0 for i in range(self.max_entity_image_num)]
                    negative_entity_image_mask_list_for_one_entity=[0 for i in range(self.max_entity_image_num)]
                text_image_retrieval_nli_score_list=self.gen_retrieval_nli_score_vector( nli_score_vector_dict,text_image_retrieval_score_list_dict,neg_id)
                negative_retrieval_nli_score_list_list.append(text_image_retrieval_nli_score_list)
                if idx >=self.max_candidate_num-positive_num-1:
                    break
        else:
            pad_entity_id=24375
            neg_object =self.obtain_corpus_by_idx(pad_entity_id)   
            [negative_entity_text,negative_entity_image_path,negative_entity_attribute]=neg_object
            # review_attribute_pair_list=self._gen_review_attribute_pair_list( negative_entity_text,negative_entity_attribute,mention_attributes,mention_text_input)  
            negative_entity_image_list_for_one_entity=[Image.open(negative_entity_image_path[0] ) for i in range(self.max_entity_image_num)]
            negative_entity_image_mask_list_for_one_entity=[0 for i in range(self.max_entity_image_num)]
            idx=-1
            
            text_image_retrieval_nli_score_list=[]
            if len(nli_score_vector_dict)>0:
                nli_score_list=list(nli_score_vector_dict.values())[0]
                text_image_retrieval_score_list=list(text_image_retrieval_score_list_dict.values())[0] 
                text_image_retrieval_nli_score_list.extend(text_image_retrieval_score_list)
                text_image_retrieval_nli_score_list.extend(nli_score_list)
             
        negative_entity_text_list,negative_entity_img_list,negative_entity_mask_list,negative_attribute_list,negative_id_list,negative_entity_img_mask_list,negative_retrieval_nli_score_list_list=self.pad_to_max_num(idx,self.max_candidate_num,negative_entity_text_list,negative_entity_img_list,
                                                                                                                                                                                negative_entity_image_list_for_one_entity, negative_attribute_list,negative_id_list,
                                                                                                                                                                                negative_entity_img_mask_list,negative_entity_image_mask_list_for_one_entity,
                                                                                                                                                                                positive_num,negative_retrieval_nli_score_list_list,text_image_retrieval_nli_score_list)
        return negative_entity_text_list,negative_entity_img_list,negative_entity_mask_list ,negative_attribute_list,negative_id_list,negative_entity_img_mask_list,negative_retrieval_nli_score_list_list
    
    
    def insert_positive_into_list(self,positive_review_attribute_pair_list,positive_entity_image,positive_entity_attribute,negative_entity_text_list,
                                  negative_entity_img_list,negative_entity_mask,negative_attribute_list,negative_id_list,pos_id,
                                  negative_entity_img_mask_list,positive_entity_image_mask,negative_retrieval_nli_score_list_list,gold_retrieval_nli_score_list):
        positive_idx=random.randint(0, self.max_candidate_num-1)
        # negative_entity_text_list.insert(positive_idx,positive_entity_text)
        negative_entity_text_list[positive_idx*len(positive_review_attribute_pair_list):positive_idx*len(positive_review_attribute_pair_list)]=positive_review_attribute_pair_list
        if self.use_image:
            negative_entity_img_list[positive_idx*len(positive_entity_image_mask):positive_idx*len(positive_entity_image_mask)]=positive_entity_image
            negative_entity_img_mask_list[positive_idx*len(positive_entity_image_mask):positive_idx*len(positive_entity_image_mask)]=positive_entity_image_mask
        # negative_attribute_list.insert(positive_idx,positive_entity_attribute)
        negative_entity_mask.insert(positive_idx,1)
        negative_entity_mask_tensor=torch.Tensor(negative_entity_mask)
        negative_entity_mask_tensor=negative_entity_mask_tensor.bool()
        negative_id_list.insert(positive_idx,pos_id)
        negative_id_list=torch.Tensor(negative_id_list)
        negative_attribute_list[positive_idx*len(positive_entity_attribute):positive_idx*len(positive_entity_attribute)]=positive_entity_attribute
        negative_retrieval_nli_score_list_list.insert(positive_idx,gold_retrieval_nli_score_list)
        negative_retrieval_nli_score_list_list=torch.Tensor(negative_retrieval_nli_score_list_list)
        return negative_entity_text_list,negative_entity_img_list ,negative_entity_mask_tensor ,positive_idx,negative_attribute_list,negative_id_list,negative_entity_img_mask_list,negative_retrieval_nli_score_list_list
    
    def _gen_positive_entity_list(self,query,mention_attributes,mention_text_input,surface_form,review_special_product_image_path,nli_score_vector_dict,text_image_retrieval_score_list_dict):
        pos_id = query['pos'].pop(0)    
        max_entity_image_num=self.max_entity_image_num
        pos_object = self.obtain_corpus_by_idx(pos_id) 
        query['pos'].append(pos_id)
        review_attribute_pair_list=[]
        [positive_entity_text,positive_entity_image_path,positive_entity_attribute]=pos_object
        if self.use_image:
            positive_entity_image_list,positive_entity_image_mask=self._gen_image_list(positive_entity_image_path,max_entity_image_num,pos_id,review_special_product_image_path)
            # positive_entity_image=Image.open(positive_entity_image_path[0] )
        else:
            positive_entity_image_list,positive_entity_image_mask=None, [0 for i in range(0,self.max_mention_image_num )]
        
        review_attribute_pair_list=self._gen_review_attribute_pair_list(positive_entity_text,positive_entity_attribute,mention_attributes,mention_text_input,surface_form) 
        text_image_retrieval_nli_score_list=self.gen_retrieval_nli_score_vector( nli_score_vector_dict,text_image_retrieval_score_list_dict,pos_id)
        return review_attribute_pair_list,pos_id,positive_entity_text,positive_entity_image_list,[],positive_entity_image_mask,text_image_retrieval_nli_score_list
    
    
    def gen_retrieval_nli_score_vector(self,nli_score_vector_dict,text_image_retrieval_score_list_dict,entity_id):
        text_image_retrieval_nli_score_list=[]
        if entity_id in nli_score_vector_dict:
            nli_score_list=nli_score_vector_dict[entity_id]
            text_image_retrieval_score_list=text_image_retrieval_score_list_dict[entity_id]
            text_image_retrieval_nli_score_list.extend(text_image_retrieval_score_list)
            text_image_retrieval_nli_score_list.extend(nli_score_list)
        return text_image_retrieval_nli_score_list
    
    def _gen_image_list(self,mention_image_path_list,max_mention_image_num,entity_id,review_special_product_image_path=None,is_to_tensor=False):
       
        mention_image_list=[]
        i=0
        entity_mask = [0 for i in range(0,max_mention_image_num )]
        for i,mention_image_path in enumerate(mention_image_path_list):
            if i>=max_mention_image_num:
                break
            mention_image=Image.open(mention_image_path)
            mention_image_list.append(mention_image)
            entity_mask[i]=1
        while i<max_mention_image_num-1:
            pad_image=mention_image
            mention_image_list.append(pad_image)
            i+=1
        if is_to_tensor:
            entity_mask=torch.Tensor(entity_mask)
            entity_mask=entity_mask.bool()
        return mention_image_list ,entity_mask
    
    def gen_review_information(self,query):
        query_object = query['query']
        nli_score_vector_dict,text_image_retrieval_score_list_dict={},{}
        if len(query_object)==4:
            [mention_text,mention_image_path,mention_attributes,surface_form]=query_object
            review_special_product_image_path=None
        elif len(query_object)==5:
            [mention_text,mention_image_path,mention_attributes,surface_form,review_special_product_image_path]=query_object
        else:
            [mention_text,mention_image_path,mention_attributes,surface_form,review_special_product_image_path,nli_score_vector_dict,text_image_retrieval_score_list_dict]=query_object
        if self.has_attribute:
            mention_text_input=surface_form+". \n"+self._attribute_to_text(mention_attributes)+". \n"+ mention_text
        else:
            mention_text_input=surface_form+". \n"  + mention_text
        if self.use_image:
            mention_image_list,mention_image_mask=self._gen_image_list(mention_image_path, self.max_mention_image_num,None)
            # mention_image=Image.open(mention_image_path[0] )
        else:
            mention_image_list,mention_image_mask=[],  [0 for i in range(0,self.max_mention_image_num )]
        return mention_image_list,mention_image_mask, mention_attributes,surface_form,mention_text_input,review_special_product_image_path,nli_score_vector_dict,text_image_retrieval_score_list_dict
    def _gen_entity_list(self,query,mention_attributes,mention_text_input,surface_form,review_special_product_image_path,nli_score_vector_dict,text_image_retrieval_score_list_dict):
        review_attribute_pair_list=[]
        
        negative_entity_text_list,negative_entity_img_list,negative_entity_mask_list,negative_attribute_text_list,negative_id_list,negative_entity_img_mask_list,negative_retrieval_nli_score_list_list=self.gen_negative_list(query,mention_attributes,mention_text_input,surface_form,review_special_product_image_path,nli_score_vector_dict,text_image_retrieval_score_list_dict)
        if self.candidate_mode!="end2end" or len(negative_id_list)==self.max_candidate_num-1:
            positive_review_attribute_pair_list,pos_id,positive_entity_text,positive_entity_image,positive_entity_attribute_list,positive_entity_image_mask,gold_retrieval_nli_score_list=self._gen_positive_entity_list(query,mention_attributes,mention_text_input,surface_form,review_special_product_image_path,nli_score_vector_dict,text_image_retrieval_score_list_dict)
            entity_text_list,entity_img_list, entity_mask_tensor,label,entities_attribute_list ,entity_id_list , entity_img_mask_list,retrieval_nli_score_list_tensor= self.insert_positive_into_list(positive_review_attribute_pair_list,positive_entity_image,positive_entity_attribute_list,negative_entity_text_list,negative_entity_img_list,
                                                                                                                                               negative_entity_mask_list,negative_attribute_text_list,negative_id_list,pos_id,negative_entity_img_mask_list,positive_entity_image_mask,negative_retrieval_nli_score_list_list,gold_retrieval_nli_score_list)    
        else:
            entity_mask_tensor=self._convert_mask_to_tensor(negative_entity_mask_list)
            entity_id_list=torch.Tensor(negative_id_list)
            entity_text_list,entity_img_list,  entities_attribute_list =negative_entity_text_list,negative_entity_img_list,negative_attribute_text_list
            label=self.max_candidate_num
            entity_img_mask_list=negative_entity_img_mask_list
            retrieval_nli_score_list_tensor=torch.Tensor(negative_retrieval_nli_score_list_list)
        
        return entity_text_list,entity_img_list, entity_mask_tensor,label,entities_attribute_list,entity_id_list,entity_img_mask_list,retrieval_nli_score_list_tensor

    def __getitem__(self, item):
        mention_entity_list_text_list=[]
        mention_entity_list_image_list=[]
        mention_entity_list_image_mask_list=[]
        query_id=self.queries_ids[item]
        query = self.queries[query_id]
        
        mention_image_list,mention_image_mask, mention_attributes,surface_form,mention_text_input,review_special_product_image_path,nli_score_vector_dict,text_image_retrieval_score_list_dict=self.gen_review_information(query)
        mention_entity_list_image_list.extend(mention_image_list)
        mention_entity_list_image_mask_list.extend(mention_image_mask)
    
        entity_text_list,entity_img_list, entity_mask_tensor,label,entity_list_attribute_list ,entity_id_list,entity_img_mask_list,retrieval_nli_score_list_tensor= self._gen_entity_list(query,mention_attributes,mention_text_input,surface_form,review_special_product_image_path,nli_score_vector_dict,text_image_retrieval_score_list_dict)
        mention_entity_list_text_list.extend(entity_text_list)
        mention_entity_list_image_list.extend(entity_img_list)
        mention_entity_list_image_mask_list.extend(entity_img_mask_list) 
        mention_entity_list_image_mask_list=torch.Tensor(mention_entity_list_image_mask_list)
        mention_entity_list_image_mask_list=mention_entity_list_image_mask_list.bool()
        
        
        tokenize_output=self.tokenizer(mention_entity_list_text_list,  
                                               add_special_tokens=True,   # Adds [CLS] and [SEP] token to every input text
                                               max_length=self.max_len, 
                                               truncation=True, 
                                               return_tensors='pt',
                                               padding="max_length" )
        if self.use_image:
            processed_inputs=self.processor( images=mention_entity_list_image_list, return_tensors="pt", padding="max_length",truncation=True)
        else:
            processed_inputs={}
            processed_inputs["pixel_values"]=tokenize_output["input_ids"] 
            mention_entity_list_image_mask_list=tokenize_output["input_ids"]
        processed_inputs["input_ids"]=tokenize_output["input_ids"]
        processed_inputs["attention_mask"]=tokenize_output["attention_mask"]
        processed_inputs["token_type_ids"]=tokenize_output["token_type_ids"]
        
        return  processed_inputs,label,entity_mask_tensor ,entity_id_list,query_id ,mention_entity_list_image_mask_list
    
    
    
class ReviewAttributePairWithImageDatasetSelectImage(ReviewAttributePairWithImageDataset):
    def _gen_image_list(self,entity_image_path_list,max_image_num,entity_id,review_special_product_image_path=None,is_to_tensor=False):
        if entity_id is None:
            return super()._gen_image_list(entity_image_path_list,max_image_num,entity_id,review_special_product_image_path,is_to_tensor)
        image_list=[]
        if review_special_product_image_path is   None:
            print(f"{entity_id} no review_special_product_image_path")
        elif   str(entity_id) in review_special_product_image_path:
            entity_image_path=review_special_product_image_path[str(entity_id)]
            if not os.path.exists(entity_image_path):
                entity_image_path=entity_image_path_list[0]
        else:
            entity_image_path=entity_image_path_list[0]
    
        image_list.append( Image.open(entity_image_path ))
        entity_mask = [1 ]
        if is_to_tensor:
            entity_mask=torch.Tensor(entity_mask)
            entity_mask=entity_mask.bool()
        return image_list ,entity_mask
     
     
    def __len__(self):
        if self.mode!="dry_run":
            return len(self.queries)    
        else:
            return 10
        
        

class ReviewAttributePairWithImageDatasetSelectImageAndRetrievalNLIScore(ReviewAttributePairWithImageDatasetSelectImage):

    def __getitem__(self, item):
        mention_entity_list_text_list=[]
        mention_entity_list_image_list=[]
        mention_entity_list_image_mask_list=[]
        query_id=self.queries_ids[item]
        query = self.queries[query_id]
        
        mention_image_list,mention_image_mask, mention_attributes,surface_form,mention_text_input,review_special_product_image_path,nli_score_vector_dict,text_image_retrieval_score_list_dict=self.gen_review_information(query)
        mention_entity_list_image_list.extend(mention_image_list)
        mention_entity_list_image_mask_list.extend(mention_image_mask)
    
        entity_text_list,entity_img_list, entity_mask_tensor,label,entity_list_attribute_list ,entity_id_list,entity_img_mask_list,retrieval_nli_score_list_tensor= self._gen_entity_list(query,mention_attributes,mention_text_input,surface_form,review_special_product_image_path,nli_score_vector_dict,text_image_retrieval_score_list_dict)
        mention_entity_list_text_list.extend(entity_text_list)
        mention_entity_list_image_list.extend(entity_img_list)
        mention_entity_list_image_mask_list.extend(entity_img_mask_list) 
        mention_entity_list_image_mask_list=torch.Tensor(mention_entity_list_image_mask_list)
        mention_entity_list_image_mask_list=mention_entity_list_image_mask_list.bool()
        
        
        tokenize_output=self.tokenizer(mention_entity_list_text_list,  
                                               add_special_tokens=True,   # Adds [CLS] and [SEP] token to every input text
                                               max_length=self.max_len, 
                                               truncation=True, 
                                               return_tensors='pt',
                                               padding="max_length" )
        if self.use_image:
            processed_inputs=self.processor( images=mention_entity_list_image_list, return_tensors="pt", padding="max_length",truncation=True)
        else:
            processed_inputs={}
            processed_inputs["pixel_values"]=tokenize_output["input_ids"] 
            mention_entity_list_image_mask_list=tokenize_output["input_ids"]
        processed_inputs["input_ids"]=tokenize_output["input_ids"]
        processed_inputs["attention_mask"]=tokenize_output["attention_mask"]
        processed_inputs["token_type_ids"]=tokenize_output["token_type_ids"]
        
        return  processed_inputs,label,entity_mask_tensor ,entity_id_list,query_id ,mention_entity_list_image_mask_list,retrieval_nli_score_list_tensor
    