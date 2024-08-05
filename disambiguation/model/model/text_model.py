# encoding=utf-8
from io import BytesIO
import math
import torch.nn.functional as F
import torchvision.models as cv_models
import torch.nn as nn
import torch
import os
import sys

from transformers import CLIPProcessor
import transformers

from disambiguation.model.layer.data_specific_encoder import   ImageLevelEncoder 
from disambiguation.model.layer.basic_layer  import   PositionalEncoding,myPrinter,ClassifierWithoutReshape
from disambiguation.model.layer.adapt import CustomCLIP
from disambiguation.model.high_level_encoder.image_encoder import ImageRepresentor
from disambiguation.model.high_level_encoder.text_encoder import TextRepresentor
from disambiguation.model.layer.function import gen_contrastive_labels,masked_softmax

 
class AMELITextModel(nn.Module):
    def __init__(self, args,   model_attribute,feature_model,device):
        super().__init__()
        self.args = args
        self.model_attribute=model_attribute
        self.text_representor=TextRepresentor( args,   model_attribute,feature_model,device)
        self.max_mention_image_num=model_attribute.max_mention_image_num
        self.max_entity_image_num=model_attribute.max_entity_image_num
        model_embed_dim=args.model_embed_dim
        self.device=device
        self.final_one_modality_classifier=   ClassifierWithoutReshape( model_embed_dim,dim_feedforward=2048,dropout=args.dr,output_dim=1)
            
        self.criterion = nn.CrossEntropyLoss( )
        
    def masked_softmax( self, tensor, mask):
        input_tensor = tensor.masked_fill((~mask), value=torch.tensor(-1e9))
        result = nn.functional.softmax(input_tensor, dim=-1)#log_
        return input_tensor,result
    
    def forward(self, input_ids,attention_mask ,labels,entity_mask_tensor    ,token_type_ids=None,is_train=False):
      
        batch_size=input_ids.shape[0]
        representation=self.text_representor(input_ids,attention_mask,token_type_ids)
        logics=self.final_one_modality_classifier(representation,batch_size)
        logics=logics.reshape(batch_size,-1)
        logics,logics_after_softmax=self.masked_softmax(logics,entity_mask_tensor)

        if is_train:
            loss=self.compute_loss(logics,labels)
        else:
            loss=logics
        return logics, labels ,loss,logics_after_softmax
    
    def compute_loss(self,score,labels):
        loss=self.criterion(score, labels)
        return loss 
                    