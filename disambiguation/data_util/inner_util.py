
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
import tqdm
from torch.utils.data import Dataset
import random
import pickle
import argparse
import torch
from PIL import Image 
import copy
from transformers import CLIPTokenizer
from pathlib import Path
from attribute.util.util import extract_model_version 

from disambiguation.data_util.choose_candidate import CandidateChooser, EasyCandidateChooser, End2EndCandidateChooser, Hard10easy9
import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset

from util.common_util import json_to_dict,json_to_review_dict,review_json_to_product_dict,gen_extracted_attributes_except_ocr_str, gen_ocr_info


def spec_to_json_attribute(spec_object,is_key_attribute=False,total_attribute_json_in_review={},section_list=None,is_allow_multiple_attribute=True):
    total_attribute_json_in_review,has_brand,has_color,has_product_name=spec_to_json_attribute_w_special_attribute_success_flag(spec_object,is_key_attribute,total_attribute_json_in_review,section_list,is_allow_multiple_attribute)
    return total_attribute_json_in_review



def spec_to_json_attribute_w_special_attribute_success_flag(spec_object,is_key_attribute=False,total_attribute_json_in_review={},section_list=None,is_allow_multiple_attribute=True):
    merged_attribute_json ={}
    has_brand,has_color,has_product_name=False,False,False
    if is_key_attribute:
        important_attribute_json_list=spec_object[:2]
    else:
        important_attribute_json_list=spec_object
    for attribute_subsection in important_attribute_json_list:
        attribute_list_in_one_section=attribute_subsection["text"]
        section_key=attribute_subsection["subsection"]
        if section_list is not None and section_key not in section_list:
            continue 
        for attribute_json  in attribute_list_in_one_section:
            attribute_key=attribute_json["specification"]
            attribute_value=attribute_json["value"].lower()
            if attribute_key=="Brand":
                has_brand=True
            if attribute_key=="Color":
                has_color=True
            if attribute_key=="Product Name":
                has_product_name=True
            if attribute_value.lower()=="yes":
                continue
            elif attribute_value.lower()=="no"  :
                continue 
            elif attribute_value.lower()=="other":
                continue 
            elif attribute_value.lower()=="none":
                continue 
            elif len(attribute_value)==1 or attribute_value.isdigit() :
                continue  
            # merged_attribute_json[attribute_key]=attribute_value_to_check
            if is_allow_multiple_attribute:
                if attribute_key in total_attribute_json_in_review:
                    if attribute_value not in total_attribute_json_in_review[attribute_key]:
                        total_attribute_json_in_review[attribute_key].append(attribute_value)
                else:
                    total_attribute_json_in_review[attribute_key]=[attribute_value]
                
            else:
              
                total_attribute_json_in_review[attribute_key]=attribute_value.lower()
    return total_attribute_json_in_review,has_brand,has_color,has_product_name

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self,device):
        # if torch.cuda.is_available():
        #     device = torch.device("cuda")
        # elif torch.backends.mps.is_available():
        #     device = torch.device("mps")
        # else:
        #     device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)

def gen_review_text_rich(review_product_json):
    predicted_attribute_dict=review_product_json["predicted_attribute"]
     
    review_text=review_product_json["header"]+". "+review_product_json["body"]
    ocr_raw_text,ocr_attributes=gen_ocr_info(review_product_json)
    sentence1="\n Review: "+review_text+". \n"+ ocr_attributes+ocr_raw_text
  
    extracted_attributes_except_ocr_str=gen_extracted_attributes_except_ocr_str(review_product_json,"test")
    sentence1+=extracted_attributes_except_ocr_str
    return sentence1

def gen_real_path(image_dir,relative_image_path_list):
    return [os.path.join(image_dir,relative_image_path) for relative_image_path in relative_image_path_list]

def is_nan_or_miss(json,key):
    if key  not in json:
        return True
    elif json[key] is None or len( json[key]) == 0:          
        return True 
    else:
        return False   





class MultimodalEntityLinkingDataDealer:
    def __init__(self,review_path_str, review_image_dir,candidate_base,candidate_mode,max_candidate_num, dataset_class,entity_text_source="desc",train_max_candidate_num_in_corpus=None)  :
        # products_path = Path(
        #     products_path_str
        # ) 
        # with open(products_path, 'r', encoding='utf-8') as fp:
        #     self.product_json_array = json.load(fp)
        # self.product_json_dict=json_to_dict(self.product_json_array)
        if train_max_candidate_num_in_corpus is None:
            self.train_max_candidate_num_in_corpus=max_candidate_num
        else:
            self.train_max_candidate_num_in_corpus=train_max_candidate_num_in_corpus    
        self.dataset_class= dataset_class
        review_path = Path(
            review_path_str
        ) 
        self.entity_text_source=entity_text_source
        self.candidate_base=candidate_base
        with open(review_path, 'r', encoding='utf-8') as fp:
            self.review_product_json_array = json.load(fp)
        self.review_id_product_json_dict=review_json_to_product_dict(self.review_product_json_array)
        self.review_image_dir=review_image_dir
        self.candidate_mode=candidate_mode
        if candidate_mode=="easy":
            self.candidate_chooser=EasyCandidateChooser(candidate_base,candidate_mode,self.train_max_candidate_num_in_corpus)
        elif candidate_mode=="standard":
            self.candidate_chooser= CandidateChooser(candidate_base,candidate_mode,self.train_max_candidate_num_in_corpus)
        elif candidate_mode=="10hard9easy":
            self.candidate_chooser= Hard10easy9(candidate_base,candidate_mode,self.train_max_candidate_num_in_corpus)
        elif candidate_mode=="end2end":
            self.candidate_chooser= End2EndCandidateChooser(candidate_base,candidate_mode,self.train_max_candidate_num_in_corpus)
        # elif candidate_mode=="fuse":
        #     self.candidate_chooser= CandidateChooser(candidate_base,candidate_mode,max_candidate_num)
        # self.product_image_dir=product_image_dir
        
    def load_qrels(self,product_json_dict,data_folder,max_candidate_num):
        needed_pids = set()     #Passage IDs we need
        needed_qids = set()     #Query IDs we need
        negative_rel_docs={}
        empty_negative_example_num=0
        positive_rel_docs = {}       #Mapping qid => set with relevant pids
        for review_product_json in self.review_product_json_array:
            gold_product_id=review_product_json["gold_entity_info"]["id"]
            review_id=review_product_json["review_id"]
            if review_id not in positive_rel_docs:
                positive_rel_docs[review_id] = set()
            positive_rel_docs[review_id].add(gold_product_id)
            needed_pids.add(gold_product_id)
            needed_qids.add(review_id)
            # if review_id==28954:
            #     print("error")
            negative_rel_docs,empty_negative_example_num=self.candidate_chooser.choose_candidate( product_json_dict,self.review_id_product_json_dict,gold_product_id,
                                    review_id,negative_rel_docs,empty_negative_example_num)
            # if review_id in negative_rel_docs and len(negative_rel_docs[review_id])==0 and review_id in positive_rel_docs:
            #TODO     del positive_rel_docs[review_id]
        print(f"empty negative example number: {empty_negative_example_num}")
        return positive_rel_docs,needed_pids,needed_qids,negative_rel_docs
            
    def load_queries(self,data_folder,needed_qids): 
        
        dev_queries = {}    
     
        for review_product_json  in self.review_product_json_array:
            review_id=review_product_json["review_id"]
            review_text=gen_review_text(review_product_json)
            review_image_paths=gen_real_path(self.review_image_dir,review_product_json["review_image_path"])
            review_attributes=review_product_json["attribute"]
            # [caption, img, ment, ment_type, lctx, rctx, entity, cands, topic, start, end,img_path,is_img_downloaded,entity_name,candidate_entity_name_list,mention_id]=datapoint 
            dev_queries[review_id]=[review_text ,review_image_paths,review_attributes]
        return dev_queries    
    

    def load_corpus( self,products_path_str ,product_image_dir):
        corpus={}
            
        products_path = Path(
            products_path_str
        ) 
        with open(products_path, 'r', encoding='utf-8') as fp:
            product_json_array = json.load(fp)
        product_json_dict=json_to_dict( product_json_array)
        
        for  product_json  in  product_json_array:
            product_id=product_json["id"]
            if self.entity_text_source=="desc":
                product_text=product_json["overview_section"]["description"]
            elif self.entity_text_source=="title":
                product_text=product_json["product_name"]
            elif self.entity_text_source=="title_desc":
                product_text=product_json["product_name"]+". "+product_json["overview_section"]["description"]
            product_image_paths=gen_real_path( product_image_dir,product_json["image_path"])
            temp_product_attributes=gen_gold_attribute(product_json["Spec"],product_json["product_name"],self.dataset_class,is_list=False,total_attribute_json_in_review={} )
            product_attributes=copy.deepcopy(temp_product_attributes)
            
            # product_attributes=gen_gold_attribute(product_json["Spec"],product_json["product_name"],self.dataset_class,is_list=False)
            
            corpus[product_id]=[product_text,product_image_paths,product_attributes]
            
        return corpus,product_json_array,product_json_dict


class MultimodalEntityLinkingDataDealerSelectImage(MultimodalEntityLinkingDataDealer):
    def __init__(self, review_path_str, review_image_dir, candidate_base, candidate_mode, max_candidate_num, dataset_class , entity_text_source,product_image_dir):
        super().__init__(review_path_str, review_image_dir, candidate_base, candidate_mode, max_candidate_num, dataset_class,  entity_text_source)
        self.product_image_dir=product_image_dir
        
    def load_queries(self,data_folder,needed_qids): 
        
        dev_queries = {}    
     
        for review_product_json  in self.review_product_json_array:
            review_id=review_product_json["review_id"]
            
            review_text=gen_review_text(review_product_json)
            review_image_paths=gen_real_path(self.review_image_dir,review_product_json["review_image_path"])
            review_attributes=review_product_json["attribute"]
            review_special_product_info=review_product_json["review_special_product_info"]
            review_special_product_image_path={}
            for product_id_str, review_special_one_product_info in review_special_product_info.items():
                product_relative_path=review_special_one_product_info["image_path"]
                product_real_path=os.path.join(self.product_image_dir,product_relative_path[0])
                
                review_special_product_image_path[product_id_str]=product_real_path
            # [caption, img, ment, ment_type, lctx, rctx, entity, cands, topic, start, end,img_path,is_img_downloaded,entity_name,candidate_entity_name_list,mention_id]=datapoint 
            dev_queries[review_id]=[review_text ,review_image_paths,review_attributes,review_special_product_image_path]
        return dev_queries    
    

 
def gen_review_text(review_json):
    return review_json["header"]+". "+review_json["body"]

    
class MultimodalEntityLinkingDataDealerSurfaceForm(MultimodalEntityLinkingDataDealer):
    
 
    def load_queries(self,data_folder,needed_qids): 
        
        dev_queries = {}    
     
        for review_product_json  in self.review_product_json_array:
            review_id=review_product_json["review_id"]
            review_text=gen_review_text(review_product_json)
            review_image_paths=gen_real_path(self.review_image_dir,review_product_json["review_image_path"])
            review_attributes=review_product_json["attribute"]
            surface_form=review_product_json["mention"]
            # [caption, img, ment, ment_type, lctx, rctx, entity, cands, topic, start, end,img_path,is_img_downloaded,entity_name,candidate_entity_name_list,mention_id]=datapoint 
            dev_queries[review_id]=[review_text ,review_image_paths,review_attributes,surface_form]
        return dev_queries        
    # def choose_attribute(self):
class MultimodalEntityLinkingDataDealerSurfaceForm(MultimodalEntityLinkingDataDealer):
    
 
    def load_queries(self,data_folder,needed_qids): 
        
        dev_queries = {}    
     
        for review_product_json  in self.review_product_json_array:
            review_id=review_product_json["review_id"]
            review_text=gen_review_text(review_product_json)
            review_image_paths=gen_real_path(self.review_image_dir,review_product_json["review_image_path"])
            review_attributes=review_product_json["attribute"] 
            surface_form=review_product_json["mention"]
            # [caption, img, ment, ment_type, lctx, rctx, entity, cands, topic, start, end,img_path,is_img_downloaded,entity_name,candidate_entity_name_list,mention_id]=datapoint 
            dev_queries[review_id]=[review_text ,review_image_paths,review_attributes,surface_form]
        return dev_queries                

def pick_one_attribute_value_from_attribute_list(attribute_dict):
    attribute_dict_with_one_attribute_value={}
    for attribute_key,attribute_value_list in attribute_dict.items():
        attribute_dict_with_one_attribute_value[attribute_key]=attribute_value_list[0]
    return attribute_dict_with_one_attribute_value

class MultimodalEntityLinkingDataDealerSurfaceFormChooseAttribute(MultimodalEntityLinkingDataDealer):
    def __init__(self, review_path_str, review_image_dir, candidate_base, candidate_mode, max_candidate_num, dataset_class, entity_text_source="desc", train_max_candidate_num_in_corpus=None,is_gold_attribute=False):
        super().__init__(review_path_str, review_image_dir, candidate_base, candidate_mode, max_candidate_num, dataset_class, entity_text_source, train_max_candidate_num_in_corpus)
        self.is_gold_attribute=is_gold_attribute
 
    def load_queries(self,data_folder,needed_qids): 
        
        dev_queries = {}    
     
        for review_product_json  in self.review_product_json_array:
            review_id=review_product_json["review_id"]
            review_text=gen_review_text(review_product_json)
            review_image_paths=gen_real_path(self.review_image_dir,review_product_json["review_image_path"])
            if self.is_gold_attribute:
                review_attributes=pick_one_attribute_value_from_attribute_list(review_product_json["gold_attribute_for_predicted_category"] )
            else:
                review_attributes=review_product_json["predicted_attribute"] 
            surface_form=review_product_json["mention"]
            # [caption, img, ment, ment_type, lctx, rctx, entity, cands, topic, start, end,img_path,is_img_downloaded,entity_name,candidate_entity_name_list,mention_id]=datapoint 
            dev_queries[review_id]=[review_text ,review_image_paths,review_attributes,surface_form]
        return dev_queries   



class MultimodalEntityLinkingDataDealerSurfaceFormSelectImage(MultimodalEntityLinkingDataDealer):
    def __init__(self, review_path_str, review_image_dir, candidate_base, candidate_mode, max_candidate_num, dataset_class , entity_text_source,product_image_dir,train_max_candidate_num_in_corpus=None,review_text_mode="rich"):
        super().__init__(review_path_str, review_image_dir, candidate_base, candidate_mode, max_candidate_num, dataset_class , entity_text_source,train_max_candidate_num_in_corpus=train_max_candidate_num_in_corpus)
        self.product_image_dir=product_image_dir
        self.review_text_mode=review_text_mode
        
    def load_queries(self,data_folder,needed_qids): 
        
        dev_queries = {}    
     
        for review_product_json  in self.review_product_json_array:
            review_id=review_product_json["review_id"]
            if self.review_text_mode=="rich":
                review_text=gen_review_text_rich(review_product_json)
            else:
                review_text=gen_review_text(review_product_json)
            review_image_paths=gen_real_path(self.review_image_dir,review_product_json["review_image_path"])
            review_attributes=review_product_json["attribute"]
            surface_form=review_product_json["mention"]
            if "review_special_product_info" in review_product_json:
                review_special_product_info=review_product_json["review_special_product_info"]
            else:
                review_special_product_info={}
            review_special_product_image_path={}
            for product_id_str, review_special_one_product_info in review_special_product_info.items():
                product_relative_path=review_special_one_product_info["image_path"]
                product_real_path=os.path.join(self.product_image_dir,product_relative_path[0])
                
                review_special_product_image_path[product_id_str]=product_real_path
            # [caption, img, ment, ment_type, lctx, rctx, entity, cands, topic, start, end,img_path,is_img_downloaded,entity_name,candidate_entity_name_list,mention_id]=datapoint 
            dev_queries[review_id]=[review_text ,review_image_paths,review_attributes,surface_form,review_special_product_image_path]
        return dev_queries    
   
class MultimodalEntityLinkingDataDealerSurfaceFormSelectImageChooseAttribute(MultimodalEntityLinkingDataDealerSurfaceFormSelectImage):
    def __init__(self, review_path_str, review_image_dir, candidate_base, candidate_mode, max_candidate_num, dataset_class, entity_text_source, product_image_dir, train_max_candidate_num_in_corpus=None, review_text_mode="rich",is_gold_attribute=False):
        super().__init__(review_path_str, review_image_dir, candidate_base, candidate_mode, max_candidate_num, dataset_class, entity_text_source, product_image_dir, train_max_candidate_num_in_corpus, review_text_mode)
        self.is_gold_attribute=is_gold_attribute
    def load_queries(self,data_folder,needed_qids): 
        
        dev_queries = {}    
     
        for review_product_json  in self.review_product_json_array:
            review_id=review_product_json["review_id"]
            if self.review_text_mode=="rich":
                review_text=gen_review_text_rich(review_product_json)
            else:
                review_text=gen_review_text(review_product_json)
            review_image_paths=gen_real_path(self.review_image_dir,review_product_json["review_image_path"])
            if self.is_gold_attribute:
                review_attributes=pick_one_attribute_value_from_attribute_list(review_product_json["gold_attribute_for_predicted_category"] )
            else:
                review_attributes=review_product_json["predicted_attribute"] 
            surface_form=review_product_json["mention"]
            if "review_special_product_info" in review_product_json:
                review_special_product_info=review_product_json["review_special_product_info"]
            else:
                review_special_product_info={}
            review_special_product_image_path={}
            for product_id_str, review_special_one_product_info in review_special_product_info.items():
                product_relative_path=review_special_one_product_info["image_path"]
                product_real_path=os.path.join(self.product_image_dir,product_relative_path[0])
                
                review_special_product_image_path[product_id_str]=product_real_path
            # [caption, img, ment, ment_type, lctx, rctx, entity, cands, topic, start, end,img_path,is_img_downloaded,entity_name,candidate_entity_name_list,mention_id]=datapoint 
            dev_queries[review_id]=[review_text ,review_image_paths,review_attributes,surface_form,review_special_product_image_path]
        return dev_queries  
       
def gen_nli_vector_for_one_product(nli_score_product_level_json,all_attribute_key_list,use_nli_score_num=-1,empty_value=0.25)   :
    max_nli_idx_num=use_nli_score_num if use_nli_score_num>0 else len(all_attribute_key_list)
    
    nli_score_vector=[empty_value for i in range(max_nli_idx_num)]
    for attribute_key,nli_score_json in nli_score_product_level_json.items():
        nli_score=nli_score_json["score"]*2
        attribute_idx_in_nli_vector=all_attribute_key_list.index(attribute_key)
        if attribute_idx_in_nli_vector<max_nli_idx_num:
            nli_score_vector[attribute_idx_in_nli_vector]=nli_score
    return nli_score_vector
    
    
def gen_nli_score_dict_non_empty(review_product_json,nli_score_field_name,  path_to_all_attribute_key_list,fused_candidate_id_list,
                       gold_product_id,empty_nli_score,use_nli_score_num,allow_empty_nli)     :
    fused_candidate_id_with_gold_list=[]
    fused_candidate_id_with_gold_list.append(gold_product_id)
    fused_candidate_id_with_gold_list.extend(fused_candidate_id_list)
    if nli_score_field_name in review_product_json:
        nli_score_dict=review_product_json[nli_score_field_name]
    else:
        nli_score_dict={}
    predicted_attribute_json=review_product_json["predicted_attribute"]
    nli_score_vector_dict={}
    
    for candidate_id in fused_candidate_id_with_gold_list:
        nli_score_list=[]
        nli_score_num=0
        for idx,predicted_attribute_key in enumerate(predicted_attribute_json):
            if nli_score_num>=use_nli_score_num:
                break
            if str(candidate_id) in nli_score_dict:
                if predicted_attribute_key in nli_score_dict[str(candidate_id)]:
                    nli_score=nli_score_dict[str(candidate_id)][predicted_attribute_key]["score"]*2
                else:
                    nli_score=0
            else:
                nli_score=0
            nli_score_list.append(nli_score)
            nli_score_num+=1
        for i in range(  nli_score_num,use_nli_score_num):
            nli_score_list.append(0)
        nli_score_vector_dict[candidate_id]=nli_score_list
    return nli_score_vector_dict
                
        
    
def gen_nli_score_dict( review_product_json,nli_score_field_name,  path_to_all_attribute_key_list,fused_candidate_id_list,
                       gold_product_id,empty_nli_score,use_nli_score_num,allow_empty_nli):
    if not allow_empty_nli:
        return gen_nli_score_dict_non_empty(review_product_json,nli_score_field_name,  path_to_all_attribute_key_list,fused_candidate_id_list,
                       gold_product_id,empty_nli_score,use_nli_score_num,allow_empty_nli)
    with open(  path_to_all_attribute_key_list, 'r', encoding='utf-8') as fp:
        all_attribute_key_list = json.load(fp)    
    nli_score_vector_dict={}
    if nli_score_field_name in review_product_json:
        nli_score_dict=review_product_json[nli_score_field_name]
        fused_candidate_id_with_gold_list=[]
        fused_candidate_id_with_gold_list.append(gold_product_id)
        fused_candidate_id_with_gold_list.extend(fused_candidate_id_list)
        for entity_id  in fused_candidate_id_with_gold_list:
            if str(entity_id) in nli_score_dict:
                nli_score_product_level_json=nli_score_dict[str(entity_id)]
                nli_vector_for_cur_product=gen_nli_vector_for_one_product(nli_score_product_level_json,all_attribute_key_list,use_nli_score_num,empty_nli_score)
                nli_score_vector_dict[int(entity_id)]=nli_vector_for_cur_product
            else:
                contradiction_nli_vector=gen_nli_vector_for_one_product({},all_attribute_key_list,use_nli_score_num,empty_value=0) 
                nli_score_vector_dict[int(entity_id)]=contradiction_nli_vector
            
    else:
        empty_nli_vector=gen_nli_vector_for_one_product({},all_attribute_key_list,use_nli_score_num,empty_nli_score) 
        for entity_id  in fused_candidate_id_list:
            nli_score_vector_dict[entity_id]=empty_nli_vector
        nli_score_vector_dict[gold_product_id]=empty_nli_vector
    return nli_score_vector_dict

def gen_text_image_retrieval_score(fused_score_dict):
    retrieval_score_list_dict={}
    for product_id,retrieval_score_dict in fused_score_dict.items():
        if "bi_score" in retrieval_score_dict:
            text_retrieval_score=retrieval_score_dict["bi_score"]
        else:
            text_retrieval_score=0
        if "image_score" in retrieval_score_dict:
            image_retrieval_score=retrieval_score_dict["image_score"]
        else:
            image_retrieval_score=0
        retrieval_score_list_dict[int(product_id)]=[text_retrieval_score,image_retrieval_score]
    return retrieval_score_list_dict

class MultimodalEntityLinkingDataDealerSurfaceFormSelectImageWithNLIRetrievalScore(MultimodalEntityLinkingDataDealerSurfaceFormSelectImage):
    def __init__(self, review_path_str, review_image_dir, candidate_base, candidate_mode, max_candidate_num, dataset_class , entity_text_source,product_image_dir,path_to_all_attribute_key_list,empty_nli_score, use_nli_score_num,allow_empty_nli,train_max_candidate_num_in_corpus=None):
        super().__init__(review_path_str, review_image_dir, candidate_base, candidate_mode, max_candidate_num, dataset_class , entity_text_source,product_image_dir,train_max_candidate_num_in_corpus=train_max_candidate_num_in_corpus)
        self.path_to_all_attribute_key_list=path_to_all_attribute_key_list
        
        self. use_nli_score_num= use_nli_score_num
        self.empty_nli_score=empty_nli_score
        self.allow_empty_nli=allow_empty_nli
        
    def load_queries(self,data_folder,needed_qids): 
        dev_queries = {}    
        filter_num=0
        for review_product_json  in self.review_product_json_array:
            review_id=review_product_json["review_id"]
            review_text=gen_review_text(review_product_json)
            review_image_paths=gen_real_path(self.review_image_dir,review_product_json["review_image_path"])
            review_attributes=review_product_json["attribute"]
            fused_candidate_id_list,gold_product_id=review_product_json["fused_candidate_list"],review_product_json["gold_entity_info"]["id"]
            surface_form=review_product_json["mention"]
            if "review_special_product_info" in review_product_json:
                review_special_product_info=review_product_json["review_special_product_info"]
            else:
                review_special_product_info={}
            review_special_product_image_path={}
            for product_id_str, review_special_one_product_info in review_special_product_info.items():
                product_relative_path=review_special_one_product_info["image_path"]
                product_real_path=os.path.join(self.product_image_dir,product_relative_path[0])
                
                review_special_product_image_path[product_id_str]=product_real_path
            # [caption, img, ment, ment_type, lctx, rctx, entity, cands, topic, start, end,img_path,is_img_downloaded,entity_name,candidate_entity_name_list,mention_id]=datapoint 
            
            nli_score_vector_dict=gen_nli_score_dict(review_product_json,"nli_score", self.path_to_all_attribute_key_list,fused_candidate_id_list,gold_product_id,self.empty_nli_score,
                                                     self.use_nli_score_num,self.allow_empty_nli)
            text_image_retrieval_score_list_dict=gen_text_image_retrieval_score(review_product_json["fused_score_dict"])
            dev_queries[review_id]=[review_text ,review_image_paths,review_attributes,surface_form,review_special_product_image_path,nli_score_vector_dict,text_image_retrieval_score_list_dict]
          
        print(f"no nli_score {filter_num}")
        return dev_queries    


class MultimodalEntityLinkingDataDealerSurfaceFormSelectImageWithNLIRetrievalScoreRichReview(MultimodalEntityLinkingDataDealerSurfaceFormSelectImageWithNLIRetrievalScore):
    def load_queries(self,data_folder,needed_qids): 
        dev_queries = {}    
        filter_num=0
        for review_product_json  in self.review_product_json_array:
            review_id=review_product_json["review_id"]
            review_text=gen_review_text_rich(review_product_json)
            review_image_paths=gen_real_path(self.review_image_dir,review_product_json["review_image_path"])
            review_attributes=review_product_json["attribute"]
            fused_candidate_id_list,gold_product_id=review_product_json["fused_candidate_list"],review_product_json["gold_entity_info"]["id"]
            surface_form=review_product_json["mention"]
            if "review_special_product_info" in review_product_json:
                review_special_product_info=review_product_json["review_special_product_info"]
            else:
                review_special_product_info={}
            review_special_product_image_path={}
            for product_id_str, review_special_one_product_info in review_special_product_info.items():
                product_relative_path=review_special_one_product_info["image_path"]
                product_real_path=os.path.join(self.product_image_dir,product_relative_path[0])
                
                review_special_product_image_path[product_id_str]=product_real_path
            # [caption, img, ment, ment_type, lctx, rctx, entity, cands, topic, start, end,img_path,is_img_downloaded,entity_name,candidate_entity_name_list,mention_id]=datapoint 
            
            nli_score_vector_dict=gen_nli_score_dict(review_product_json,"nli_score", self.path_to_all_attribute_key_list,fused_candidate_id_list,gold_product_id,self.empty_nli_score,
                                                     self.use_nli_score_num,self.allow_empty_nli)
            text_image_retrieval_score_list_dict=gen_text_image_retrieval_score(review_product_json["fused_score_dict"])
            dev_queries[review_id]=[review_text ,review_image_paths,review_attributes,surface_form,review_special_product_image_path,nli_score_vector_dict,text_image_retrieval_score_list_dict]
          
        print(f"no nli_score {filter_num}")
        return dev_queries    



class MultimodalEntityLinkingDataDealerSurfaceFormSelectImageWithAttributeHashRichReview(MultimodalEntityLinkingDataDealerSurfaceFormSelectImageWithNLIRetrievalScore):
    def load_queries(self,data_folder,needed_qids): 
        dev_queries = {}    
        filter_num=0
        for review_product_json  in self.review_product_json_array:
            review_id=review_product_json["review_id"]
            review_text=gen_review_text_rich(review_product_json)
            review_image_paths=gen_real_path(self.review_image_dir,review_product_json["review_image_path"])
            review_attributes=review_product_json["attribute"]
            fused_candidate_id_list,gold_product_id=review_product_json["fused_candidate_list"],review_product_json["gold_entity_info"]["id"]
            surface_form=review_product_json["mention"]
            if "review_special_product_info" in review_product_json:
                review_special_product_info=review_product_json["review_special_product_info"]
            else:
                review_special_product_info={}
            review_special_product_image_path={}
            for product_id_str, review_special_one_product_info in review_special_product_info.items():
                product_relative_path=review_special_one_product_info["image_path"]
                product_real_path=os.path.join(self.product_image_dir,product_relative_path[0])
                
                review_special_product_image_path[product_id_str]=product_real_path
            # [caption, img, ment, ment_type, lctx, rctx, entity, cands, topic, start, end,img_path,is_img_downloaded,entity_name,candidate_entity_name_list,mention_id]=datapoint 
            
            nli_score_vector_dict=gen_nli_score_dict(review_product_json,"nli_score", self.path_to_all_attribute_key_list,fused_candidate_id_list,gold_product_id,self.empty_nli_score,
                                                     self.use_nli_score_num,self.allow_empty_nli)
            text_image_retrieval_score_list_dict=gen_text_image_retrieval_score(review_product_json["fused_score_dict"])
            dev_queries[review_id]=[review_text ,review_image_paths,review_attributes,surface_form,review_special_product_image_path,nli_score_vector_dict,text_image_retrieval_score_list_dict]
          
        print(f"no nli_score {filter_num}")
        return dev_queries    

def gen_gold_attribute_without_key_section(spec_json,product_title,total_attribute_json_in_review,section_list=None,dataset_class="v6",is_allow_multiple_attribute=True):
    dataset_class="v6"
    
    attribute_json=gen_gold_attribute(spec_json,product_title,dataset_class,is_allow_multiple_attribute,section_list ,total_attribute_json_in_review  )
    return attribute_json   

def gen_gold_attribute(spec_json,product_title,dataset_class="v6",is_list=True,section_list=["Key Specs","General","All"],total_attribute_json_in_review={} ):
    is_allow_multiple_attribute=is_list
    is_key_attribute=False
    attribute_json,has_brand,has_color,has_product_name=spec_to_json_attribute_w_special_attribute_success_flag(spec_json,is_key_attribute,total_attribute_json_in_review,section_list,is_allow_multiple_attribute)
    if len(attribute_json)==0 and section_list is not None:
        attribute_json,has_brand,has_color,has_product_name=spec_to_json_attribute_w_special_attribute_success_flag(spec_json,is_key_attribute,total_attribute_json_in_review,None,is_allow_multiple_attribute)
        
    product_title_segmentation=product_title.split(" - ")
    if len(product_title_segmentation)==3 and dataset_class not in ["v2","v3","v4"]:
        if "Brand" not in attribute_json and is_list:    
            attribute_json["Brand"]=[]
        if not has_brand:
            if is_list:
                attribute_json["Brand"].append(product_title_segmentation[0].lower())
            else:
                attribute_json["Brand"]= product_title_segmentation[0].lower()
        
        if "Color" not in attribute_json and is_list:  
            attribute_json["Color"]=[]
        if not has_color:
            if is_list:
                attribute_json["Color"].append(product_title_segmentation[2].lower())
            else:
                attribute_json["Color"]=product_title_segmentation[2].lower()
        if "Product Name" not in attribute_json and is_list:  
            attribute_json["Product Name"]=[]
        if not has_product_name:
            if is_list:
                attribute_json["Product Name"].append(product_title_segmentation[1].lower())
            else:
                attribute_json["Product Name"]= product_title_segmentation[1].lower()
    
    if "Product Title" not in attribute_json and is_list:  
        attribute_json["Product Title"]=[ ]
    if is_list:
        attribute_json["Product Title"].append(product_title)
    else:
        attribute_json["Product Title"]=product_title
         
    # if "Product Name" not in attribute_json:
    #     if is_list:
    #         attribute_json["Product Name"]=[]
    #         attribute_json["Product Name"].append(product_title)
    #     else:
    #         attribute_json["Product Name"]=product_title
   
    model_version_list=extract_model_version(product_title)
    if "Model Version" not in attribute_json and is_list:  
        attribute_json["Model Version"]=[ ]
    if is_list:
         
        attribute_json["Model Version"].extend(model_version_list)
    elif len(model_version_list)>0:
        attribute_json["Model Version"]=model_version_list[0]
    else:
        attribute_json["Model Version"]=""
        
    
        
    return attribute_json
     
    
# def spec_to_json_attribute(spec_object,is_key_attribute):
#     merged_attribute_json ={}
#     if is_key_attribute:
#         important_attribute_json_list=spec_object[:2]
#     else:
#         important_attribute_json_list=spec_object
#     for attribute_subsection in important_attribute_json_list:
#         attribute_list_in_one_section=attribute_subsection["text"]
#         for attribute_json  in attribute_list_in_one_section:
#             attribute_key=attribute_json["specification"]
#             attribute_value=attribute_json["value"]
#             if attribute_value.lower()=="yes":
#                 continue
#             elif attribute_value.lower()=="no"  :
#                 continue 
#             elif attribute_value.lower()=="other":
#                 continue 
#             elif len(attribute_value)==1:
#                 continue 
#             merged_attribute_json[attribute_key]=attribute_value
             
#     return merged_attribute_json