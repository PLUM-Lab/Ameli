from PIL import Image
import os

import numpy as np
import click
import pickle
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from time import sleep
from disambiguation.mp_inference import mp_inference 
from util.env_config import * 
from functools import partial 
from disambiguation.train_verify import ddp_train_loop, train_loop 
from util.env_config import *
import logging 
import re
import torch.multiprocessing as mp
logging.basicConfig(filename="image.txt",
                    filemode='w',
                  
                    level=logging.INFO)
  
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial


def hyper_search(config_kwargs):
    config = {
 
        "lr":   tune.choice([   0.01,0.0001, 0.001,0.005,0.05,0.0005]), # , 0.0001 ,0.00001
        "batch_size":    tune.choice([   40]),#,
        "accum_iter":  tune.choice([   10,4])#,12,8,1
        
    }
    gpus_per_trial = 1
    num_samples=config_kwargs["num_samples_for_hyper_search"]
    max_num_epochs=10
    metric_name= "val_f1"
    cpus_per_trial=7

    scheduler = ASHAScheduler(
        metric= metric_name,
        mode="max",
        max_t=max_num_epochs,
        grace_period=4,
        reduction_factor=2)         
    reporter = CLIReporter(
        parameter_columns=[  "accum_iter","lr", "batch_size"],
        metric_columns=[  metric_name , "training_iteration"],max_progress_rows=num_samples)    #"reconstruct_loss",               
    result = tune.run(
        partial(train_loop ,config_kwargs=config_kwargs ),
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        config=config,
        # stop=stopper,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        checkpoint_at_end=False) 
        

    best_trial = result.get_best_trial(metric_name, "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final score: {}".format(
        best_trial.last_result[ metric_name]))     
  
@click.command()
@click.pass_context
#Data
@click.option('--outdir', help='Where to save the results', required=True, metavar='DIR',default="output/runs")
@click.option('--train_dir', help='input', required=True, metavar='DIR',default=os.path.join(data_dir,"train/disambiguation/bestbuy_review_2.3.17.11.19.16_update_gold_attribute.json")) 
@click.option('--val_dir', help='input', required=True, metavar='DIR',default=os.path.join(data_dir,"val/disambiguation/bestbuy_review_2.3.17.11.19.16.1_update_candidate.json")) 
@click.option('--test_dir', help='input', required=True, metavar='DIR',default=os.path.join(data_dir,"test/disambiguation/bestbuy_review_2.3.17.11.20.1_update_annotated_gold_attribute_update_candidate.json"))  
@click.option('--products_path_str', help='input', required=True, metavar='DIR',default=products_path_str) 
@click.option('--review_image_dir', help='input', required=True, metavar='DIR',default=review_image_dir) 
@click.option('--product_image_dir', help='input', required=True, metavar='DIR',default=product_image_dir) 
@click.option('--path_to_all_attribute_key_list', help='input', required=True, metavar='DIR',default=os.path.join(data_dir,"test/disambiguation/sorted_all_attribute_key.json")) 
@click.option('--fused_score_data_path', help='input', required=True, metavar='DIR',default=os.path.join(data_dir,"test/disambiguation/postprocess/bestbuy_review_2.3.17.11.20_add_disambiguation_score.json"))  
@click.option('--weighted_score_data_path', help='input', required=True, metavar='DIR',default=os.path.join(data_dir,"test/disambiguation/postprocess/bestbuy_review_2.3.17.11.21_fuse_disambiguation_score"))  
@click.option('--filtered_data_path', help='input', required=True, metavar='DIR',default=os.path.join(data_dir,"test/disambiguation/postprocess/bestbuy_review_2.3.17.11.22_filter_by_attribute.json"))  
@click.option('--dataset', type=str,default="entity_linking" )  
@click.option('--train_max_candidate_num_in_corpus', default=10, type=int, help='grad accum step')
@click.option('--train_max_candidate_num', default=10, type=int, help='grad accum step')
@click.option('--max_candidate_num', default=10, type=int, help='grad accum step') #20
@click.option('--max_attribute_num', default=5, type=int, help='grad accum step') #20
@click.option('--candidate_base', type=str,default="by_review_fused" ) #fuse
@click.option('--candidate_mode', type=str,default="standard" ) #easy 10hard9easy
@click.option('--test_candidate_mode', type=str,default="end2end" )
@click.option('--dev_candidate_mode', type=str,default="standard" )
@click.option('--dataset_class', type=str,default="select_image_sbert_attribute" ) #v2
@click.option('--datadealer_class', type=str,default="select_image_with_attribute_hash_and_retrieval_score_rich_review" )
@click.option('--entity_text_source', type=str,default="desc" ) #title
@click.option('--is_random_positive/--no_random_positive', default=True, is_flag=True, help='use attribute information?')
@click.option('--is_train_many_negative_per_time/--no_train_many_negative_per_time', default=True, is_flag=True, help='use attribute information?')
@click.option('--review_text_mode', type=str,default="standard" ) #rich standard
#Model
@click.option('--is_freeze_clip', default=True)
@click.option('--is_outer_loss/--is_direct_loss', default=True, is_flag=True, help='use attribute information?')
@click.option('--is_freeze_bert', default=True)
@click.option('--model_attribute', type=str,default="A_SBERT_ATTRIBUTE" )  #B6 A4
@click.option('--max_len', type=int,default=512, metavar='INT') 
@click.option('--model_embed_dim', type=int,default=768, metavar='INT') 
@click.option('--pre_trained_dir', type=str,default='cross-encoder/nli-deberta-base')  #all-mpnet-base-v2 /home/menglong/workspace/code/multimodal/multimodal_entity_linking/multimodal_entity_linking/output/retrieval/train/00026-train_bi-encoder-/models   cross-encoder/nli-deberta-base   bert-large-uncased cross-encoder/nli-deberta-base
@click.option('--pre_trained_image_model_dir', type=str,default="openai/clip-vit-large-patch14")  #/home/menglong/workspace/code/multimodal/multimodal_entity_linking/multimodal_entity_linking/output/retrieval/train/00027-train_bi-encoder-clip-ViT-L-14-2023-05-30_17-44-02/models
@click.option('--freeze_bert_layer_number', default=195, type=int )#16 parameters per layer
@click.option('--freeze_text_layer_num', default=17, type=int )
@click.option('--freeze_img_layer_num', default=20, type=int ) 
@click.option('--use_attributes/--no_use_attributes', default=True, is_flag=True, help='use attribute information?')#False
@click.option('--use_image/--no_use_image', default=True, is_flag=True, help='use attribute information?')#False
@click.option('--use_text/--no_use_text', default=True, is_flag=True, help='use attribute information?')#False
@click.option('--is_filter_by_attribute/--no_filter_by_attribute', default=True, is_flag=True, help='use attribute information?')#False
@click.option('--fuse', default="concatenate", help='use attribute information?')
@click.option('--n_tok', type=int,default=1000000, metavar='INT')
@click.option('--d_tok', type=int,default=768, metavar='INT')# 
@click.option('--d_image', type=int,default=768, metavar='INT')# 
@click.option('--d_text_tok', type=int,default=512, metavar='INT')# 
@click.option('--d_hid', type=int,default=768, metavar='INT')
@click.option('--n_head', type=int,default=2, metavar='INT')
@click.option('--dr', type=float,default=0.5 )
@click.option('--model_group', default="all", help='use attribute information?')
@click.option('--x_layers', type=int,default=2, metavar='INT')
@click.option('--best_text_weight', type=float,default=None)
@click.option('--search_retrieval_score_field', type=str,default=None)

#Training
@click.option('--best_retrieval_weight', type=float,default=None)#False
# @click.option('--search_retrieval_score_field', type=str,default="fused_score")#False
@click.option('--train_config', type=str,default="BI_ENCODER" )  #CLAIM_IMAGE
@click.option('--train_batch_size', type=int,default=16, metavar='INT') 
@click.option('--batch_size', type=int,default=16, metavar='INT') 
@click.option('--epoch', default=10, type=int, help='force stop at specified epoch') 
@click.option('--verbos', type=str,default="y" )  
@click.option('--mode', type=str,default="train" ) 
@click.option('--lr', type=float,default=1e-5 )
@click.option('--empty_nli_score', type=float,default=0.25 )
@click.option('--use_nli_score_num', type=int,default=5 )
@click.option('--allow_empty_nli/--no_allow_empty_nli', default=True, is_flag=True, help='use attribute information?')#False
@click.option('--is_gpu/--is_not_gpu', default=True, is_flag=True, help='use attribute information?')#False
@click.option('--clip_grad', type=int,default=1)
@click.option('--early_stop', type=int,default=10, metavar='INT')
@click.option('--loss_weight_power', type=int,default=2, metavar='INT')
@click.option('--is_wandb', type=str,default="y" )  #CLAIM_IMAGE
@click.option('--accum_iter', default=1, type=int, help='grad accum step')
@click.option('--num_samples_for_hyper_search', type=int,default=20, metavar='INT')
@click.option('--num_processes_per_gpu', type=int,default=4, metavar='INT')
@click.option('--desc', type=str  ) 
@click.option('--parallel', type=str ,default="data_parallel" ) #ddp
@click.option('--gpu_list')#,default=[0,1,2]
@click.option('--world_size',  type=int ) 
@click.option('--num_worker',  type=int,default=10 ) 
@click.option('--disambiguation_result_postfix', type=str ,default="" ) 
#for inference
@click.option('--checkpoint_dir', help='input',  metavar='DIR')  #,default='verification/output/runs/00121-'
@click.option('--save_predict', type=str,default="y" ) 
@click.option('--top_k', type=int,default=1, metavar='INT')
@click.option('--subset_to_check', default="Test")#Train
@click.option('--load_mode', default="one_to_one")#ddp_to_one ddp_to_ddp
def main(ctx,  **config_kwargs):
    if config_kwargs["world_size"] is None :
        world_size=torch.cuda.device_count()
        config_kwargs["world_size"]=world_size
    else:
        world_size=config_kwargs["world_size"]
    print(f"world_size:{world_size}")
    mode=config_kwargs["mode"]
    outdir=config_kwargs["outdir"]
    if mode !="mp_test":
        # Pick output directory.
        prev_run_dirs = []
        if os.path.isdir(outdir):
            prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        config_kwargs["cur_run_id"]=cur_run_id
    if mode=="hyper_search":
        hyper_search(config_kwargs)
    elif mode=="mp_test":
        mp_inference(config_kwargs)
    else:
        if config_kwargs["parallel"]=="ddp":
            mp.spawn(ddp_train_loop,
                args=(world_size,config_kwargs),
                nprocs=world_size,
                join=True) 
        else:
            
            train_loop(None,config_kwargs)
     

    
     




if __name__ == "__main__":
    
    main() 