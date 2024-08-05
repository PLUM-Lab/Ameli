
from torch.utils.data import DataLoader
import math
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from disambiguation.model.nli_evaluator import CESoftmaxAccuracyEvaluator
from sentence_transformers.readers import InputExample
import logging
from datetime import datetime
import os
import torch
from PIL import Image
import os
import wandb
import numpy as np
from tqdm.contrib import tzip
import pickle
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from time import sleep
import warnings
import torch.nn.functional as F
 
import gzip
from torch import nn
import csv
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, LoggingHandler, util, models, evaluation, losses, InputExample
import logging
from datetime import datetime
import gzip
import os
import tarfile
from collections import defaultdict
from attribute.attribution_extraction import gen_candidate_attribute
from disambiguation.data_util.inner_util import gen_gold_attribute_without_key_section
from util.common_util import append_one_kind_of_attribute, gen_extracted_attributes_except_ocr_str, gen_ocr_info, json_to_dict, review_json_to_product_dict
from util.env_config import * 
import tqdm
import json
import random
import pickle
import argparse
import torch
#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)
#### /print debug information to stdout

label2int = {"contradiction": 0, "entailment": 1}#, "neutral": 2}
#As dataset, we use SNLI + MultiNLI
#Check if dataset exsist. If not, download and extract  it
def add_one_example(sentence1,attribute_key,surface_form,attribute_value,label_id,subset_samples,candidate_product_id="",review_id=""):
    sentence2=attribute_key+" of "+surface_form+" is "+attribute_value
    subset_samples.append(InputExample(texts=[sentence1,sentence2], label=label_id,guid=f"{review_id}_{candidate_product_id}_{attribute_key}"))
    return subset_samples
    
import copy 

def find_neutral_attribute_value_list(   product_json_list, attribute_key,gold_product,subset_samples,surface_form,sentence1,label_statistic):
    random_product_list=random.sample(product_json_list,500)
    
    
    temp_gold_product_attribute_json=gen_gold_attribute_without_key_section(gold_product["Spec"],gold_product["product_name"] ,{})
    gold_product_attribute_json=copy.deepcopy(temp_gold_product_attribute_json)
    for one_product_json in random_product_list:
        if one_product_json["product_category"] !=gold_product["product_category"]:
            temp_product_attribute_json=gen_gold_attribute_without_key_section(one_product_json["Spec"],one_product_json["product_name"] ,{})
            product_attribute_json=copy.deepcopy(temp_product_attribute_json)
            for idx,(attribute_key,attribute_value_list) in enumerate(product_attribute_json.items()):
                if idx<5 and  attribute_key not in gold_product_attribute_json:
                    subset_samples=add_one_example(sentence1,attribute_key,surface_form,attribute_value_list[0],label2int["neutral"],subset_samples)
                    label_statistic[label2int["neutral"]]+=1
            break
    return subset_samples,label_statistic

def find_negative_attribute_value_list( gold_attribute_json,entity_candidate_attribute_json,
                                                                                 attribute_key ):
    if attribute_key in gold_attribute_json:
        gold_attribute=gold_attribute_json[attribute_key]
    else:
        gold_attribute=[]
    if attribute_key in entity_candidate_attribute_json:
        candidate_attribute_list=entity_candidate_attribute_json[attribute_key]
    else:
        candidate_attribute_list=[]
    negative_attribute_value_list=[]
    for candidate_attribute in candidate_attribute_list:
        if candidate_attribute not in gold_attribute:
            negative_attribute_value_list.append(candidate_attribute)    
    return negative_attribute_value_list
    
def load_one_dataset(data_path,mode):
    with open(products_path_str, 'r', encoding='utf-8') as fp:
        products_json_array = json.load(fp)
        product_dict=json_to_dict(products_json_array)
    label_statistic=[0,0,0]
    with open(  data_path, 'r', encoding='utf-8') as fp:
        product_dataset_json_array = json.load(fp)
        subset_samples = []
        for idx,review_product_json   in enumerate(product_dataset_json_array ):
            predicted_attribute_dict=review_product_json["predicted_attribute"]
            gold_product_id=review_product_json["gold_entity_info"]["id"]
            surface_form=review_product_json["mention"]
            review_id=review_product_json["review_id"]
            review_text=review_product_json["header"]+". "+review_product_json["body"]
            ocr_raw_text,ocr_attributes=gen_ocr_info(review_product_json)
            sentence1=surface_form+". \n Review: "+review_text+". \n"+ ocr_attributes+ocr_raw_text
            if mode=="test":
                extracted_attributes_except_ocr_str=gen_extracted_attributes_except_ocr_str(review_product_json,mode)
                sentence1+=extracted_attributes_except_ocr_str
            entity_candidate_attribute_json=gen_candidate_attribute(review_id,product_dict,review_product_json)
            temp_gold_attribute_json=gen_gold_attribute_without_key_section(product_dict[gold_product_id]["Spec"],product_dict[gold_product_id]["product_name"] ,{})
            gold_attribute_json=copy.deepcopy(temp_gold_attribute_json)
            for attribute_key,predicted_attribute_value in predicted_attribute_dict.items():
                negative_attribute_value_list=find_negative_attribute_value_list(gold_attribute_json,entity_candidate_attribute_json,
                                                                                 attribute_key)
                subset_samples=add_one_example(sentence1,attribute_key,surface_form,predicted_attribute_value,label2int["entailment"],subset_samples)
                label_statistic[label2int["entailment"]]+=1
                for negative_attribute_value in negative_attribute_value_list:
                    subset_samples=add_one_example(sentence1,attribute_key,surface_form,negative_attribute_value,label2int["contradiction"],subset_samples)
                    label_statistic[label2int["contradiction"]]+=1
                subset_samples,label_statistic=find_neutral_attribute_value_list(   products_json_array, attribute_key,product_dict[gold_product_id],subset_samples,surface_form,sentence1,label_statistic)
            if mode=="dry_run" and idx>128:
                break
    return subset_samples, label_statistic


def gen_extracted_attributes_except_ocr_str(review_product_json,dataset):
    exact_attribute_str=append_one_kind_of_attribute("predicted_attribute_exact",review_product_json,"EXACT")
    gpt2_attribute_str=append_one_kind_of_attribute("predicted_attribute_gpt2",review_product_json,"GPT2")
    model_version_attribute_str=append_one_kind_of_attribute("predicted_attribute_model_version",review_product_json,"VERSION",is_allow_multiple=True)
    # model_version_to_product_title_attribute_str=append_one_kind_of_attribute("predicted_attribute_model_version_to_product_title",review_product_json,"PRODUCT")
    if dataset=="test" and "predicted_attribute_chatgpt" in review_product_json:
        chatgpt_attribute_str=append_one_kind_of_attribute("predicted_attribute_chatgpt",review_product_json,"CHATGPT")
    else:
        chatgpt_attribute_str=""
    return exact_attribute_str+model_version_attribute_str+ gpt2_attribute_str+chatgpt_attribute_str
    
    

def gen_review_text_rich(review_product_json):
    predicted_attribute_dict=review_product_json["predicted_attribute"]
     
    review_text=review_product_json["header"]+". "+review_product_json["body"]
    ocr_raw_text,ocr_attributes=gen_ocr_info(review_product_json)
    sentence1="\n Review: "+review_text+". \n"+ ocr_attributes+ocr_raw_text
  
    extracted_attributes_except_ocr_str=gen_extracted_attributes_except_ocr_str(review_product_json,"test")
    sentence1+=extracted_attributes_except_ocr_str
    return sentence1

def load_one_dataset2(data_path,mode):
    with open(products_path_str, 'r', encoding='utf-8') as fp:
        products_json_array = json.load(fp)
        product_dict=json_to_dict(products_json_array)
    label_statistic=[0,0]
    with open(  data_path, 'r', encoding='utf-8') as fp:
        product_dataset_json_array = json.load(fp)
        subset_samples = []
        for idx,review_product_json   in enumerate(product_dataset_json_array ):
            rich_review=gen_review_text_rich(review_product_json)
            gold_product_id=review_product_json["gold_entity_info"]["id"]
            fused_candidate_list=review_product_json["fused_candidate_list"]
            for candidate_id in fused_candidate_list:
                candidate_product_json=product_dict[candidate_id]
                product_title=candidate_product_json["product_name"]
                label_id=1 if candidate_id==gold_product_id else 0
                subset_samples.append(InputExample(texts=[rich_review,product_title], label=label_id ))
                label_statistic[label_id]+=1
            
            if mode=="dry_run" and idx>128:
                break
    return subset_samples, label_statistic


def load_dataset(args):
  
            
    train_samples,train_label_statistic= load_one_dataset(args.train_data_path,args.mode)
    dev_samples,_= load_one_dataset(args.val_data_path,args.mode)
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.train_batch_size)
    return train_dataloader,dev_samples,label2int,train_label_statistic

def load_dataset_for_cross_encoder(args):
  
            
    train_samples,train_label_statistic= load_one_dataset2(args.train_data_path,args.mode)
    dev_samples,_= load_one_dataset2(args.val_data_path,args.mode)
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.train_batch_size)
    return train_dataloader,dev_samples,label2int,train_label_statistic
    
def create_example_for_one_candidate_one_attribute(product_dict,subset_samples,predicted_attribute_dict,sentence1,candidate_product_id,
                                                   gold_attribute_json,attribute_key,candidate_attribute_json,surface_form,review_id):
    if attribute_key in candidate_attribute_json and len(candidate_attribute_json[attribute_key])>0:
        candidate_attribute_value=candidate_attribute_json[attribute_key][0]
        if attribute_key in gold_attribute_json and len(gold_attribute_json[attribute_key])>0:
            gold_attribute_value =gold_attribute_json[attribute_key][0]
        else:
            gold_attribute_value=""
        if candidate_attribute_value==gold_attribute_value:
            nli_label="entailment"
        else:
            nli_label="contradiction"
        subset_samples=add_one_example(sentence1,attribute_key,surface_form,candidate_attribute_value,label2int[nli_label],
                                       subset_samples,candidate_product_id,review_id)
    return subset_samples
 
def create_example_for_one_candidate(product_dict,subset_samples,predicted_attribute_dict,sentence1,candidate_product_id,gold_attribute_json,
                                     surface_form,review_id):
    candidate_product_json=product_dict[candidate_product_id]
    temp_candidate_attribute_json= gen_gold_attribute_without_key_section(candidate_product_json["Spec"],candidate_product_json["product_name"] ,{})
    candidate_attribute_json=copy.deepcopy(temp_candidate_attribute_json)
    for attribute_key,predicted_attribute_value in predicted_attribute_dict.items():
        subset_samples=create_example_for_one_candidate_one_attribute(product_dict,subset_samples,predicted_attribute_dict,sentence1,
                                                                      candidate_product_id,gold_attribute_json,attribute_key,
                                                                      candidate_attribute_json,surface_form,review_id)
    return subset_samples
         
def load_one_dataset_for_disambiguation(data_path,mode,dataset):
    with open(products_path_str, 'r', encoding='utf-8') as fp:
        products_json_array = json.load(fp)
        product_dict=json_to_dict(products_json_array)
    label_statistic=[0,0,0]
    with open(  data_path, 'r', encoding='utf-8') as fp:
        product_dataset_json_array = json.load(fp)
        subset_samples = []
        for idx,review_product_json   in enumerate(product_dataset_json_array ):
            predicted_attribute_dict=review_product_json["predicted_attribute"]
            gold_product_id=review_product_json["gold_entity_info"]["id"]
            surface_form=review_product_json["mention"]
            review_id=review_product_json["review_id"]
            review_text=review_product_json["header"]+". "+review_product_json["body"]
            ocr_raw_text,ocr_attributes=gen_ocr_info(review_product_json)
            sentence1=surface_form+". \n Review: "+review_text+". \n"+ ocr_attributes+ocr_raw_text
            if dataset=="test":
                extracted_attributes_except_ocr_str=gen_extracted_attributes_except_ocr_str(review_product_json,dataset)
                sentence1+=extracted_attributes_except_ocr_str
            temp_gold_attribute_json=gen_gold_attribute_without_key_section(product_dict[gold_product_id]["Spec"],product_dict[gold_product_id]["product_name"] ,{})
            gold_attribute_json=copy.deepcopy(temp_gold_attribute_json)
            fused_candidate_list=review_product_json["fused_candidate_list"]
            if gold_product_id not in fused_candidate_list:
                fused_candidate_list.append(gold_product_id)
            for candidate_product_id in fused_candidate_list:
                subset_samples=create_example_for_one_candidate(product_dict,subset_samples,predicted_attribute_dict,sentence1,candidate_product_id,
                                                                gold_attribute_json,surface_form,review_id)
             
            if mode in [ "dry_run","test_dry_run"] and idx>64:
                break
    return subset_samples, label_statistic


""" 
score_dict={"pred_scores":pred_scores,"pred_labels":pred_labels,"gold_label":self.labels,
                            "guid":self.guid_list}
"""
from sklearn.utils.extmath import softmax
def convert_three_score_to_one(pred_score):
    normed_pred_score = softmax([pred_score])[0]
    nli_score=0*normed_pred_score[0]+0.5*normed_pred_score[1]+1*normed_pred_score[2]
    return nli_score
    
def convert_to_one_score(pickle_path, data_path,out_path):
    with open(  data_path, 'r', encoding='utf-8') as fp:
        product_dataset_json_array = json.load(fp)
        review_dataset_json_dict=review_json_to_product_dict(product_dataset_json_array)
        
    with open(pickle_path, 'rb') as handle:
        score_dict = pickle.load(handle)
        pred_scores=score_dict["pred_scores"]
        pred_scores=pred_scores.tolist()
        guid_list=score_dict["guid"]
        pred_labels=score_dict["pred_labels"]
        pred_labels=pred_labels.tolist()
        labels=score_dict["gold_label"]
        out_list=[]
        for pred_label,label,guid,pred_score in tzip(pred_labels,labels,guid_list,pred_scores):
            guid_token_list=guid.split("_")
            review_id,product_id=guid_token_list[0],guid_token_list[1]
            attribute_key=guid[len(review_id)+len(product_id)+2:]
            nli_score=convert_three_score_to_one(pred_score)
            review_dataset_json=review_dataset_json_dict[int(review_id)]
            if "nli_score" not in review_dataset_json:
                review_dataset_json["nli_score"]={}
            if product_id not in review_dataset_json["nli_score"]:
                review_dataset_json["nli_score"][product_id]={}
            review_dataset_json["nli_score"][product_id][attribute_key]={}
            review_dataset_json["nli_score"][product_id][attribute_key]["score"]=nli_score
            review_dataset_json["nli_score"][product_id][attribute_key]["nli_label"]=label
            review_dataset_json["nli_score"][product_id][attribute_key]["predicted_nli_label"]=pred_label
    review_dataset_json_list=list(review_dataset_json_dict.values())
    with open(out_path, 'w', encoding='utf-8') as fp:
        json.dump(review_dataset_json_list, fp, indent=4)    
    
def gen_total_attribute_key_list_for_one_subset(data_path,total_attribute_key_num_dict):
    with open(  data_path, 'r', encoding='utf-8') as fp:
        product_dataset_json_array = json.load(fp)    
        for product_dataset_json in product_dataset_json_array:
            predicted_attribute_json=product_dataset_json["predicted_attribute"] 
            for predicted_attribute_key in predicted_attribute_json:
                if predicted_attribute_key not in total_attribute_key_num_dict:
                    total_attribute_key_num_dict[predicted_attribute_key]=0
                total_attribute_key_num_dict[predicted_attribute_key]+=1
    return total_attribute_key_num_dict
                
                
def gen_total_attribute_key_list(out_path):
    total_attribute_key_num_dict={}
    for mode in ["test","val","train"]:
        data_path=f"/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/{mode}/disambiguation/bestbuy_review_2.3.17.11.13_add_nli_score.json"
        
        total_attribute_key_num_dict=gen_total_attribute_key_list_for_one_subset(data_path,total_attribute_key_num_dict)
    sorted_total_attribute_key_num_dict= sorted(total_attribute_key_num_dict.items(), key=lambda x:x[1],reverse=True)
    sorted_total_attribute_key_list=[attribute_num_dict[0] for attribute_num_dict in sorted_total_attribute_key_num_dict]
    with open(out_path, 'w', encoding='utf-8') as fp:
        json.dump(sorted_total_attribute_key_list, fp, indent=4)    
    
if __name__=="__main__":            
    convert_to_one_score()