

 # encoding=utf-8
from io import BytesIO
import math
import torch.nn.functional as F
import torchvision.models as cv_models
import torch.nn as nn
import torch
import os
import sys
from typing import Optional, Tuple, Union
from transformers import CLIPProcessor, DebertaV2ForSequenceClassification ,BertForSequenceClassification,DebertaForSequenceClassification,RobertaForSequenceClassification
import transformers
 
from transformers import   BertTokenizer, BertModel,AutoModel

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss
import inspect
from collections import OrderedDict, UserDict
from collections.abc import MutableMapping
from contextlib import ExitStack
from dataclasses import fields
from enum import Enum
from typing import Any, ContextManager, List, Tuple


        
def masked_softmax(  tensor, mask):
    input_tensor = tensor.masked_fill((~mask), value=torch.tensor(-1e9))
    result = nn.functional.softmax(input_tensor, dim=-1)
    return input_tensor,result

def gen_contrastive_labels(output,labels):
    (batch_size, batch_size_times_cand_num) = output.shape
    cand_num = int(batch_size_times_cand_num/batch_size)
    label_int_list = []
    for i in range(batch_size):
        original_label=labels[i]
        label_int_list.append(i*cand_num+original_label)
    labels = torch.tensor(label_int_list, device=output.device)
    return labels


def myPrinter(args, str_to_be_printed):
    if args.device == 'cuda' or args.device == 'cpu' or args.rank == 0:
        print(str_to_be_printed)
        
        
        
def gen_attention_mask_for_multihead_attn( evidence_bert_mask):
    attention_mask = evidence_bert_mask.to(dtype=torch.bool )
    return ~attention_mask        
    
            
def repeat_mention_in_entity_shape(mention_embed, entity_list_embed):
    entity_num=entity_list_embed.shape[1]
    duplicated_mention_embed=mention_embed.expand(-1,entity_num, -1,-1 ) 
    return duplicated_mention_embed
def repeat_mention_mask_in_entity_shape(mention_embed, entity_list_embed):
    entity_num=entity_list_embed.shape[1]
    duplicated_mention_embed=mention_embed.expand(-1,entity_num, -1) 
    return duplicated_mention_embed
            