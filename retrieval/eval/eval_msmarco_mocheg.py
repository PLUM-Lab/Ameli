"""
This script runs the evaluation of an SBERT msmarco model on the
MS MARCO dev dataset and reports different performances metrices for cossine similarity & dot-product.
Usage:
python eval_msmarco.py model_name [max_corpus_size_in_thousands]
"""
from PIL import Image
import pandas as pd 
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
from torch.utils.data import IterableDataset
import tqdm
from torch.utils.data import Dataset
import random
import pickle
import argparse
import torch
 
from sentence_transformers import  LoggingHandler, SentenceTransformer, evaluation, util, models
import logging
import sys
import os
import tarfile
from retrieval.data_util.core.data_dealer import AmeliTextDataDealer
from retrieval.eval.evaluator import MultiMediaInformationRetrievalEvaluator
from retrieval.model.model import MultiMediaSentenceTransformer
from retrieval.utils.enums import TrainingAttribute
from retrieval.utils.retrieval_util import WandbLog, gen_data_dealer
from util.common_util import setup_with_args

from retrieval.data_util.core.gen_data import AmeliImageDataDealer, ImageDataDealer, TextDataDealer, load_qrels 
from util.read_example import get_father_dir
 
def test_retrieval(model_name,data_folder,corpus,model_save_path,media,use_precomputed_corpus_embeddings,query_image_dir,query_file_name,
                   candidate_mode,ce_score_margin,level): 
    ####  Load model
    model = MultiMediaSentenceTransformer(model_name)
    logging.info("Start Test" )
    ir_evaluator=gen_evaluator(data_folder,corpus,media,model,use_precomputed_corpus_embeddings,model_save_path,query_image_dir,query_file_name,
                               candidate_mode,ce_score_margin=ce_score_margin,level=level)
    
    score=ir_evaluator(model, output_path=model_save_path )
    WandbLog()(score, -1, -1)
    
def corpus_dict_to_corpus(entity_dict):
    corpus={}
    for entity_id,entity_json   in entity_dict.items():
        for image_path in entity_json.image_path_list:
            image_name=image_path.split("/")[-1]
            corpus[image_name]=image_path
    return corpus

def gen_corpus_embedding_with_cache(corpus,model,data_folder,use_precomputed_corpus_embeddings,media):
    corpus_folder= get_father_dir(data_folder)
    emb_folder=os.path.join(corpus_folder,f"embed")
    emb_filename = 'corpus_image_embeddings.pkl'
    emb_dir=os.path.join(emb_folder,emb_filename)
    corpus_dict_path=os.path.join(emb_folder,"corpus_dict.pickle")
    if use_precomputed_corpus_embeddings and os.path.exists(emb_dir): 
        with open(emb_dir, 'rb') as fIn:
            emb_file = pickle.load(fIn)  
            entity_dict,entity_image_num_list,img_emb,entity_name_list,entity_img_path_list=emb_file["entity_dict"],emb_file["entity_image_num_list"],emb_file["img_emb"],emb_file["entity_name_list"] ,emb_file["entity_img_path_list"]
            # img_emb, img_names =emb_file["img_emb"],emb_file["img_names"] 
            corpus=corpus_dict_to_corpus(entity_dict)
        print("Images:", len( img_emb))
        # if os.path.exists(corpus_dict_path):
        #     with open(corpus_dict_path, 'rb') as fIn:
        #         corpus = pickle.load(fIn)  
        #     print("cached corpus_dict:", len( corpus))
        # else:
        #     print(f"Error! must have corpus_dict.pkl in {emb_folder} while use img_corpus_emb.pkl in {emb_folder}")
        #     exit()
    else:
        img_emb=gen_corpus_embedding(corpus,model)
        emb_file = { "img_emb":  img_emb, "img_names": list(corpus.keys()) }            #,"img_folder":img_folder
        pickle.dump( emb_file, open(emb_dir , "wb" ) )
        pickle.dump( corpus, open(corpus_dict_path , "wb" ) )
        
    
    return img_emb,corpus

def gen_corpus_embedding(corpus,model):
    batch_size=480 #480
    live_num_in_current_batch=0
    live_num=0
    current_image_batch=[]
    total_img_emb= torch.tensor([],device= torch.device('cuda'))
    corpus_len=len(corpus)
    for corpus_id,corpus_img_path in corpus.items():
        image=Image.open(corpus_img_path)
        current_image_batch.append(image)
        live_num_in_current_batch+=1
        
        if live_num_in_current_batch%batch_size==0:
            img_emb = model.encode(current_image_batch, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True)
            total_img_emb=torch.cat([total_img_emb,img_emb],0)
            live_num_in_current_batch=0
            current_image_batch=[]
            live_num+=batch_size
            print(live_num/corpus_len)
    return total_img_emb

def gen_evaluator(data_folder,corpus,media,model,use_precomputed_corpus_embeddings,model_save_path,query_image_dir, query_file_name,candidate_mode,
                  ce_score_margin, 
                  mode=None,level=None):
    dev_rel_docs,needed_pids,needed_qids,_=load_qrels(data_folder, query_file_name, candidate_mode, media,mode,ce_score_margin=ce_score_margin)
    data_dealer=gen_data_dealer( media, level)
    dev_queries=data_dealer.load_queries(data_folder,query_file_name,query_image_dir)
    
    
    ## Run evaluator
    logging.info("Start evaluation. Queries: {}".format(len(dev_queries)))
    logging.info("Corpus: {}".format(len(corpus)))
    save_query_result_path=os.path.join(model_save_path,f"query_result_{media}.pkl")
    if media=="img":
        # corpus_embedding,corpus=gen_corpus_embedding_with_cache(corpus,model,data_folder,use_precomputed_corpus_embeddings,media)
        ir_evaluator = MultiMediaInformationRetrievalEvaluator(dev_queries, corpus, dev_rel_docs,
                                                            show_progress_bar=True,
                                                            corpus_chunk_size=100000,
                                                            precision_recall_at_k=[1,10,20,50,100,1000  ],
                                                            name="msmarco dev",
                                                            # corpus_embed=corpus_embedding,
                                                            score_functions ={'cos_sim': util.cos_sim },
                                                            save_query_result_path=save_query_result_path,
                                                            map_at_k=[5,10],
                                                            ndcg_at_k=[5,10],
                                                            media= media)
    else:
        ir_evaluator = MultiMediaInformationRetrievalEvaluator(dev_queries, corpus, dev_rel_docs,
                                                            show_progress_bar=True,
                                                            corpus_chunk_size=100000,
                                                            precision_recall_at_k=[1,10,20,50,100,1000  ],
                                                            name="msmarco dev",
                                                            score_functions ={'cos_sim': util.cos_sim },
                                                            save_query_result_path=save_query_result_path,
                                                            map_at_k=[5,10],
                                                            ndcg_at_k=[5,10],
                                                            media= media)
    
    return ir_evaluator

def get_args():
    
    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    #### /print debug information to stdout


    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_max_size", default=0, type=int)
    parser.add_argument("--model_name",default='multi-qa-MiniLM-L6-cos-v1') 
    parser.add_argument("--data_folder",default='/home/menglong/workspace/code/referred/conll2019-snopes-crawling/final_corpus/mocheg2/test') #retrieval/input/msmarco-data
    args = parser.parse_args()

    print(args)
    return args


def test(args):
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
    data_dealer=gen_data_dealer( args.media, args.level)
    corpus_max_size=0
    corpus=data_dealer.load_corpus(args.corpus_dir,  args.corpus_image_dir)
    model_save_path,args=setup_with_args(args,'retrieval/output/runs_3','test_bi-encoder-{}-{}'.format(args.model_name.replace("/", "-"), datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    
 
    test_retrieval(args.model_name,args.test_data_folder,corpus,args.run_dir,args.media,
                   args.use_precomputed_corpus_embeddings,args.query_image_dir,args.query_file_name,
                   args.candidate_mode,args.ce_score_margin,args.level)

def main():
    args=get_args()
     
    test(args)

if __name__ == "__main__":
    
    main() 