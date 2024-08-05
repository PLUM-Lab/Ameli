import os
import click
import re
import json
import tempfile
import torch
from retrieval.utils.retrieval_util import setup_config
from retrieval.scorer import evaluate
from retrieval.training import training_loop
from util.env_config import *
 
import numpy as np
import logging 
logging.basicConfig(filename="log.txt",
                    filemode='w',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)
#----------------------------------------------------------------------------



# example


@click.command()
@click.pass_context
@click.option('--outdir', help='Where to save the results', required=True, metavar='DIR',default="output/retrieval/runs") 
@click.option('--csv_out_dir', help='Where to save the results',default=data_dir+"temp/bestbuy_review_2.3.17.3_pre_trained_image.json",   metavar='DIR' ) 
@click.option('--dataset_dir', help='input', required=True, metavar='DIR',default=data_dir+"val/retrieval/bestbuy_review_2.3.17.2_clean_missed_product.json") 
@click.option('--dataset_image_dir', help='input', required=True, metavar='DIR',default=data_dir+"cleaned_review_images") 
@click.option('--corpus_image_dir', help='input', required=True, metavar='DIR',default=data_dir+"product_images") 
@click.option('--corpus_pickle_dir', help='input',  metavar='DIR') #,default=data_dir+"embed"
@click.option('--corpus_dir', help='input', required=True, metavar='DIR',default=products_path_str ) 
# @click.option('--KB_image_dir', help='input', required=True, metavar='DIR',default="data/BLINK_benchmark/wikidiverse_questions.jsonl") 
# @click.option('--in_dir', help='input', required=True, metavar='DIR',default="/home/menglong/workspace/code/referred/conll2019-snopes-crawling/final_corpus/mode3_latest_v2")
@click.option('--top_k', help='top_k', type=int,default=1000, metavar='INT')  
@click.option('--metric', type=str,default="similarity" )  
@click.option('--sent_num',   type=int,default=1, metavar='INT')
@click.option('--media', type=str,default="img" ) #txt,img_txt
@click.option('--use_precomputed_embeddings', type=str,default="y" )   # #image_search use precomputed_embeddings for images
@click.option('--bi_encoder_checkpoint',  metavar='DIR',default="all-mpnet-base-v2")#"/home/menglong/workspace/code/misinformation_detection/retrieval/output/runs_2/train_bi-encoder-mnrl-multi-qa-MiniLM-L6-cos-v1-margin_3.0-2022-05-30_00-53-08"
@click.option('--cross_encoder_checkpoint',  metavar='DIR',default="cross-encoder/ms-marco-MiniLM-L-12-v2")#'/home/menglong/workspace/code/misinformation_detection/retrieval/output/runs_3/00063-train_cross-encoder-cross-encoder-ms-marco-MiniLM-L-6-v2-2022-06-02_21-59-58-latest'
@click.option('--image_encoder_checkpoint',  metavar='DIR',default="clip-ViT-L-14")#retrieval/output/runs_3/00081-train_bi-encoder-clip-ViT-B-32-2022-06-07_07-42-24/models
@click.option('--image_cross_encoder_checkpoint',  metavar='DIR',default="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/multimodal_entity_linking/retrieval/output/runs_3/00119-train_cross-encoder-clip-ViT-L-14-2023-03-02_19-03-28/match.pt")
@click.option('--image_feature_extractor_checkpoint',  metavar='DIR',default="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/multimodal_entity_linking/retrieval/output/runs_3/00119-train_cross-encoder-clip-ViT-L-14-2023-03-02_19-03-28/feature_extractor.pt")
@click.option('--no_rerank',  is_flag=True, show_default=True, default=False   ) #txt,img_txt
@click.option('--score_by_fine_tuned',  is_flag=True,  default=True   ) #txt,img_txt
@click.option('--mode', type=str,default="retrieve_by_review" )#retrieve_by_gold
@click.option('--dataset', type=str,default="entity_linking" )
@click.option('--text_base', type=str,default="title" )#title_desc_attribute
@click.option('--is_only_in_category',  is_flag=True,  default=False   )
@click.option('--img_fuse_mode', type=str,default="max" )#title
@click.option('--candidate_field', type=str,default="product_id_with_similar_text_by_review" )#title
def main(ctx,  **config_kwargs):
    outdir=config_kwargs["outdir"]
    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    config_kwargs["cur_run_id"]=cur_run_id
    args,logger=setup_config(config_kwargs)
    
    logging.info("start")
    # args.img_in_dir=os.path.join(args.in_dir,"images")
    # if args.csv_out_dir is None:
    #     args.csv_out_dir=os.path.join(args.dataset_dir,"retrieval/retrieval_result.csv")
    # args.relevant_document_dir=get_relevant_document_dir(args.txt_in_dir)
    # args.relevant_document_image_dir=os.path.join(args.relevant_document_dir,"images")
    training_loop.training_loop(args,logger,rank=0)
  
        
    evaluate(args.csv_out_dir,False,args.candidate_field) 


if __name__ == "__main__":
    
    main() 