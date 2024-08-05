# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Miscellaneous utility classes and functions."""

import ctypes
import fnmatch
import importlib
import inspect
import numpy as np
import os
import shutil
import sys
import types
import io
import pickle
import re
import requests
import html
import hashlib
import glob
import tempfile
import urllib
import urllib.request
import uuid

from distutils.util import strtobool
from typing import Any, List, Tuple, Union
import os 
import click
import re
import json

from attribute.attribution_extraction import spec_to_json_attribute

# Util classes
# ------------------------------------------------------------------------------------------
import wandb

from retrieval.data_util.core.data_dealer import AmeliEntityLevelImageDataDealer, AmeliEntityLevelImageTextDataDealer, AmeliImageDataDealer, AmeliTextDataDealer 
def log_callback_st(train_ix, epoch,training_steps, current_lr, loss_value):
    wandb.log({f"train/contrastive_loss_loss": loss_value,
             f"train/contrastive_loss_lr": current_lr[-1],
             f"train/contrastive_loss_lr_previous": current_lr[-2] if len(current_lr)>1 else -1,
            "train/steps": training_steps})

class WandbLog:
    
        
    def __call__(self, score, epoch, steps):
        wandb.log({f"val/score": score,
             f"val/epoch": epoch,
            "val/steps": steps})
            
            
def setup_training_loop_kwargs(config_kwargs):
    args=EasyDict()
    for key,value in config_kwargs.items():
        setattr(args,key,value)
 
    run_desc=""
    
    return run_desc,args

import torch 
from torch import nn
sigmoid_method = nn.Sigmoid()



def freeze_part_bert(model,freeze_text_layer_num,freeze_img_layer_num):
    count = 0
 
    for p in model[0].model.text_model.named_parameters():
        
        if (count<=freeze_text_layer_num):
            p[1].requires_grad=False    
        else:
            break
              
        count=count+1
        print(p[0], p[1].requires_grad)
        
    count=0
    for p in model[0].model.vision_model.named_parameters():
        
        if (count<=freeze_img_layer_num):
            p[1].requires_grad=False    
        else:
            break
              
        count=count+1
        print(p[0], p[1].requires_grad)
        
    return model 

def gen_data_dealer(media,level):
    if  media=="txt":
        data_dealer=AmeliTextDataDealer()
    elif media=="txt_img":
        data_dealer=AmeliEntityLevelImageTextDataDealer()
    elif level=="entity":
        data_dealer=AmeliEntityLevelImageDataDealer()
    else:
        data_dealer=AmeliImageDataDealer()
    return data_dealer








def cross_score_to_sigmoid(score):
     
    input=torch.tensor(score)
    score_sigmoid=sigmoid_method(input)
    return score_sigmoid.item()

def old_spec_to_json_attribute(spec_object,is_key_attribute,section_list=None):
    return spec_to_json_attribute(spec_object,is_key_attribute,{},section_list,False)

    # merged_attribute_json ={}
    # if is_key_attribute:
    #     important_attribute_json_list=spec_object[:2]
    # else:
    #     important_attribute_json_list=spec_object
    # for attribute_subsection in important_attribute_json_list:
    #     attribute_list_in_one_section=attribute_subsection["text"]
    #     section_key=attribute_subsection["subsection"]
    #     if section_list is not None and section_key not in section_list:
    #         continue 
    #     for attribute_json  in attribute_list_in_one_section:
    #         attribute_key=attribute_json["specification"]
    #         attribute_value=attribute_json["value"]
    #         merged_attribute_json[attribute_key]=attribute_value.lower()
    # return merged_attribute_json

def create_new_folder(args,run_desc):
    outdir=args.outdir
    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
   
    return outdir,cur_run_id


def setup_config(config_kwargs,rank=0,has_log=True):
     
 
    # Setup training options.
    run_desc, args = setup_training_loop_kwargs(config_kwargs)

    if config_kwargs["mode"]=="mp_test":
        outdir,cur_run_id=create_new_folder(args,run_desc)
        args.cur_run_id=cur_run_id
    else:
        outdir=args.outdir
        
    args.run_dir = os.path.join(outdir, f'{args.cur_run_id:05d}-{run_desc}')
    
    
    # Create output directory.
    
    if   rank==0:
        assert not os.path.exists(args.run_dir)
        print('Creating output directory...')
        os.makedirs(args.run_dir)
        print()
        print(f'Training options:  ')
        print(json.dumps(args, indent=2))
        print()
        print(f'Output directory:   {args.run_dir}')
        with open(os.path.join(args.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(args, f, indent=2)
        # if args.use_attributes:
        #     print(f"args.use_attributes:{args.use_attributes}")
    if has_log:
        # Launch processes.
        
        logger= Logger(file_mode='a', should_flush=False) #file_name=os.path.join(args.run_dir, 'log.txt'), ,file_name=os.path.join(args.run_dir, 'log.txt')
        if   rank==0:
            print('Launching processes...')
            # logger.write(text=json.dumps(args, indent=2))
    else:
        logger=None
    return args,logger


def show_model(model):
    for p in model.named_parameters():
      
        print(p[0], p[1].requires_grad)
        
import pandas as pd 
def remove_img_evidence(data_path):
    relevant_doc_corpus=os.path.join(data_path,"retrieval/retrieval_result.csv")
    df_news = pd.read_csv(relevant_doc_corpus ,encoding="utf8")
    df=df_news.drop(columns=['img_evidences'])
    df.to_csv(relevant_doc_corpus)

class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


class Logger(object):
    """Redirect stderr to stdout, optionally print stdout to a file, and optionally force flushing on both stdout and the file."""

    def __init__(self, file_name: str = None, file_mode: str = "w", should_flush: bool = True):
        self.file = None

        if file_name is not None:
            self.file = open(file_name, file_mode)

        self.should_flush = should_flush
        self.stdout = sys.stdout
        self.stderr = sys.stderr

        sys.stdout = self
        # sys.stderr = self

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def write(self, text: Union[str, bytes]) -> None:
        """Write text to stdout (and a file) and optionally flush."""
        if isinstance(text, bytes):
            text = text.decode()
        if len(text) == 0: # workaround for a bug in VSCode debugger: sys.stdout.write(''); sys.stdout.flush() => crash
            return

        if self.file is not None:
            self.file.write(text)

        self.stdout.write(text)

        if self.should_flush:
            self.flush()

    def flush(self) -> None:
        """Flush written text to both stdout and a file, if open."""
        if self.file is not None:
            self.file.flush()

        self.stdout.flush()

    def close(self) -> None:
        """Flush, close possible files, and remove stdout/stderr mirroring."""
        self.flush()

        # if using multiple loggers, prevent closing in wrong order
        if sys.stdout is self:
            sys.stdout = self.stdout
        if sys.stderr is self:
            sys.stderr = self.stderr

        if self.file is not None:
            self.file.close()
            self.file = None
 