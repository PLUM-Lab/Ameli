"""
This examples trains a CrossEncoder for the NLI task. A CrossEncoder takes a sentence pair
as input and outputs a label. Here, it learns to predict the labels: "contradiction": 0, "entailment": 1, "neutral": 2.

It does NOT produce a sentence embedding and does NOT work for individual sentences.

Usage:
python training_nli.py
"""

from torch.utils.data import DataLoader
import math
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from disambiguation.data_util.nli_data_util import load_dataset_for_cross_encoder, load_one_dataset,label2int, load_one_dataset_for_disambiguation
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
from util.common_util import json_to_dict
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


def get_loss_weights(device,train_label_statistic,power=1):
    normed_weights=get_loss_weight_fun1(3,np.array(train_label_statistic),device,power)
    
    return normed_weights 
 
    
    
def get_loss_weight_fun1(num_of_classes,samples_per_class,device,power=1):
    weights_for_samples=1.0/np.array(np.power(samples_per_class,power))
    weights_for_samples=weights_for_samples/np.sum(weights_for_samples)*num_of_classes
    weights_for_samples=torch.tensor(weights_for_samples, dtype=torch.float32,device=device)
    return weights_for_samples    


class WandbLog:
    
        
    def __call__(self, score, epoch, steps):
        wandb.log({f"val/score": score,
             f"val/epoch": epoch,
            "val/steps": steps})
           
     
   
def test_cross_encoder(args,model_name):
    run_name='test_ameli_nli-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_save_path = 'output/test_nli/'+run_name
    print(model_save_path)
    os.makedirs(model_save_path,exist_ok=True)
    if args.attribute_level=="candidate":
        test_samples,_= load_one_dataset_for_disambiguation(args.test_data_path,args.mode,args.dataset)
    else:
        test_samples,_= load_one_dataset(args.test_data_path,args.mode)
    logging.info("Queries: {}".format(len(test_samples)))
    model = CrossEncoder(model_name, num_labels=len(label2int))
    evaluator = CESoftmaxAccuracyEvaluator.from_input_examples(test_samples, name='AMELINLI-dev',show_progress_bar=True,
                                                               batch_size=args.train_batch_size,is_save_score=True )
    results =evaluator(model,   epoch=args.epoch+1, steps=-1,output_path=model_save_path)
 
    
    
def main(args):
    
    train_dataloader,dev_samples,label2int,train_label_statistic=load_dataset_for_cross_encoder(args)
    device=torch.device("cuda")
    num_epochs = args.epoch
    run_name='training_ameli_nli-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_save_path = 'output/train_nli/'+run_name
    wandb.init(project=f'entity-linking-nli',config=args)
    wandb.run.name=f"{run_name}-{wandb.run.name}"
    #Define our CrossEncoder model. We use distilroberta-base as basis and setup it up to predict 3 labels
    model = CrossEncoder(args.model_name, num_labels=len(label2int))
    #During training, we use CESoftmaxAccuracyEvaluator to measure the accuracy on the dev set.
    evaluator = CESoftmaxAccuracyEvaluator.from_input_examples(dev_samples, name='AMELINLI-dev',show_progress_bar=True,batch_size=args.train_batch_size)
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
    logger.info("Warmup-steps: {}".format(warmup_steps))
    # Train the model
    normed_weights=get_loss_weights(device,train_label_statistic,args.loss_weight_power)
    loss_function=nn.CrossEntropyLoss(weight=normed_weights)
    model.fit(train_dataloader=train_dataloader,
            evaluator=evaluator,
            epochs=num_epochs,
            evaluation_steps=10000,
            warmup_steps=warmup_steps,
            output_path=model_save_path,
            optimizer_params = {'lr': args.lr},
            loss_fct=loss_function,
            callback=WandbLog(),
            save_best_model=True,
            use_amp=True)

    
def get_args():
    
    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    #### /print debug information to stdout


    parser = argparse.ArgumentParser()
    # parser.add_argument('--output_path', help='input',   default=data_dir+"train/disambiguation/bestbuy_review_2.3.17.11.12_add_raw_ocr.json")#'retrieval/input/msmarco-data'
    parser.add_argument("--train_batch_size" , type=int,default=48)#480
    parser.add_argument("--epoch",   type=int,default=20)
    parser.add_argument("--attribute_level",type=str,default="candidate")
    parser.add_argument("--dataset", type=str  ,default="test") #distilroberta-base
    parser.add_argument("--mode", type=str  ,default="train") #distilroberta-base
    parser.add_argument("--loss_weight_power",   type=int,default=2)
    parser.add_argument("--seed_value",   type=int,default=None)
    parser.add_argument("--model_name", type=str  ,default="distilroberta-base") #cross-encoder/nli-deberta-v3-base
    parser.add_argument("--lr", default=1e-5, type=float)#1e-5
    parser.add_argument('--train_data_path', help='input',   default=data_dir+"train/disambiguation/bestbuy_review_2.3.17.11.14_fix_nli.json")#'retrieval/input/msmarco-data'
    parser.add_argument('--test_data_path', help='input',   default=data_dir+"test/disambiguation/bestbuy_review_2.3.17.11.15_add_predict_attribute_contain_all.json") 
    parser.add_argument('--val_data_path', help='input',   default=data_dir+"val/disambiguation/bestbuy_review_2.3.17.11.14_fix_nli.json") 
    
    # parser.add_argument("--save_predict_in_data_dir", default=True, action="store_false")

    args = parser.parse_args()

    print(args)
    return args

        

if __name__ == "__main__":
    
    args=get_args()
    if args.mode in [ "train","dry_run"]:
        main(args)
    else:
        test_cross_encoder(args,args.model_name)