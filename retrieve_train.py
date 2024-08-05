import sys
import json
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, LoggingHandler, util, models, evaluation, losses, InputExample
import logging
from datetime import datetime
import gzip
import os
import tarfile
from collections import defaultdict
 
import tqdm
 
import random
import pickle
import argparse
import torch
from retrieval.eval.eval_cross_encoder_mocheg import test_cross_encoder
from retrieval.eval.eval_msmarco_mocheg import test

from retrieval.training.train_bi_encoder_mnrl_mocheg import train
from retrieval.training.train_cross_encoder_mocheg import train_cross_encoder
from retrieval.utils.enums import TrainingAttribute 
from util.env_config import *
from sentence_transformers import models, losses, datasets
def set_seed(seed_value):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value) 
    
def get_args():
    
    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    #### /print debug information to stdout


    parser = argparse.ArgumentParser()
    parser.add_argument("--train_batch_size" , type=int)#480
    parser.add_argument("--media", type=str  ) #txt,img 
    parser.add_argument("--max_seq_length",  type=int)#100
    parser.add_argument("--model_name" )# 'clip-ViT-B-32','multi-qa-MiniLM-L6-cos-v1'
    parser.add_argument("--text_model_name"  ,type=str , default="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/multimodal_entity_linking/output/retrieval/train/00026-train_bi-encoder-/models" )# 'clip-ViT-B-32','multi-qa-MiniLM-L6-cos-v1'
    parser.add_argument("--image_model_name", type=str , default="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/multimodal_entity_linking/output/retrieval/train/00027-train_bi-encoder-clip-ViT-L-14-2023-05-30_17-44-02/models"  )# 'clip-ViT-B-32','multi-qa-MiniLM-L6-cos-v1'
    parser.add_argument("--max_passages", default=0, type=int)
    parser.add_argument("--epochs",   type=int)
    parser.add_argument("--seed_value",   type=int,default=None)
    parser.add_argument("--pooling", default="mean")
    parser.add_argument("--negs_to_use", default=None, help="From which systems should negatives be used? Multiple systems seperated by comma. None = all")
    parser.add_argument("--warmup_steps", default=1000, type=int)#1000
    parser.add_argument("--lr", default=1e-5, type=float)#1e-5
    parser.add_argument("--num_negs_per_system", default=50, type=int)#5
    parser.add_argument("--use_pre_trained_model", default=True, action="store_false")
    parser.add_argument("--use_all_queries", default=False, action="store_true")
    parser.add_argument("--ce_score_margin", default=0.08, type=float)
    parser.add_argument("--use_cached_train_queries", default=False, action="store_true")
    parser.add_argument('--train_data_folder', help='input',   default=data_dir+"train")#'retrieval/input/msmarco-data'
    parser.add_argument('--test_data_folder', help='input',   default=data_dir+"test") 
    parser.add_argument('--val_data_folder', help='input',   default=data_dir+"val") 
    parser.add_argument('--query_file_name', help='input',   default= "retrieval/bestbuy_review_2.3.17.2_clean_missed_product.json")#bestbuy_review_2.3.17.11.17_select_image_100.json_from_0_to_1000000.json
    parser.add_argument('--corpus_dir', help='input',   default=products_path_str) 
    parser.add_argument('--corpus_image_dir', help='input',   default=data_dir+"product_images") 
    parser.add_argument('--query_image_dir', help='input',   default=data_dir+"cleaned_review_images") 
    parser.add_argument("--do_val", default=False, action="store_true")
    parser.add_argument("--desc", type=str  ) #txt,img 
    parser.add_argument('--text_source', help='input',   default= "title") #desc, title_desc
    parser.add_argument("--mode", type=str , default="train" )  #dry_run
    parser.add_argument("--level", type=str , default="entity" )  #dry_run
    parser.add_argument("--candidate_mode", type=str , default="other" )#over_score
    parser.add_argument("--train_config", type=str,default="IMAGE_MODEL" )
    parser.add_argument("--use_precomputed_corpus_embeddings", default=False, action="store_false")
    parser.add_argument("--weight_decay", default=0.001, type=float)  
    parser.add_argument("--freeze_text_layer_num", default=17, type=int)
    parser.add_argument("--freeze_img_layer_num", default=20, type=int)
    parser.add_argument('--top_candidate_corpus_path', help='input',   default="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/multimodal_entity_linking/retrieval/output/runs_3/00054-train_bi-encoder-clip-ViT-L-14-2023-02-17_13-01-53/query_result_img.pkl") 
    parser.add_argument("--save_predict_in_data_dir", default=True, action="store_false")
    #test
    parser.add_argument("--feature_extractor",default="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/multimodal_entity_linking/retrieval/output/runs_3/00119-train_cross-encoder-clip-ViT-L-14-2023-03-02_19-03-28/feature_extractor.pt" )# 'clip-ViT-B-32','multi-qa-MiniLM-L6-cos-v1'
    
    args = parser.parse_args()

    print(args)
    return args

def main():
    args=get_args()
    if args.seed_value is not None :
        seed_value=args.seed_value
    else:
        seed_value=random.seed(10)
    if seed_value is not None:
        set_seed(seed_value)
    train_attribute=TrainingAttribute[args.train_config]
    if args.model_name is None:
        args.model_name =train_attribute.model_name
    if args.train_batch_size is None:
        args.train_batch_size = train_attribute.batch_size
    if args.epochs is None:
        args.epochs  =train_attribute.epoch
    if args.media is None:
        args.media=train_attribute.media 
    if args.max_seq_length is None:
        args.max_seq_length=train_attribute.max_seq_length
        
    if args.mode in ["train","dry_run"]:
        if args.train_config in [ "CROSS_ENCODER","IMAGE_CROSS_ENCODER" ]:
            train_cross_encoder(args)
        else:
            train(args)
    elif args.mode=="test":
        if args.train_config in [ "CROSS_ENCODER","IMAGE_CROSS_ENCODER"]:
            test_cross_encoder(args)
        else:
            test(args)
 
        

if __name__ == "__main__":
    
    main() 