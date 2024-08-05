# encoding=utf-8
from io import BytesIO
import math
import torch.nn.functional as F
import torchvision.models as cv_models
import torch.nn as nn
import torch
import os
import sys
from transformers import AutoProcessor, FlavaModel
from transformers import CLIPProcessor
import transformers
from retrieval.model.model import MultiMediaSentenceTransformer

from retrieval.utils.retrieval_util import freeze_part_bert
  
  


# import oss2 as oss


class CLIPBERT(nn.Module):
    def __init__(self, args, text_model_path,image_model_path):
        super().__init__()
        self.args = args
        self.mode = "txt_img"
        # self.device = device
        self.text_model= MultiMediaSentenceTransformer(text_model_path)
        self.image_model= MultiMediaSentenceTransformer(image_model_path) 
        self.image_model=freeze_part_bert(self.image_model,args.freeze_text_layer_num,args.freeze_img_layer_num)
    
    def forward(self,features):
       #2,3,224,224  #2,384
        text_features,image_features=features
        text_embed=self.text_model(text_features)["sentence_embedding"]
        image_embed=self.image_model(image_features)["sentence_embedding"]
        final_embed=torch.cat((text_embed,image_embed),-1)
        return  {'sentence_embedding':final_embed}
    

    def tokenize(self,cur_input):
        text=[one_list[0] for one_list in cur_input]
        image=[one_list[1] for one_list in cur_input]
        text_feature=self.text_model.tokenize(text)
        image_feature=self.image_model.tokenize(image)
        return text_feature,image_feature

    def save(self, output_path: str):
        self.image_model.save(output_path)
        
        self.text_model.save(output_path)