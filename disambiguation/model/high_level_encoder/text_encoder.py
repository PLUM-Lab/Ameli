#mention: bert-base-uncased
#entity: wiki 2vec
#image: CLIP


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
 
from disambiguation.model.layer.basic_layer import ClassifierWithoutReshape 
from disambiguation.model.layer.data_specific_encoder import EncoderForReviewAttributePair 



class TextRepresentor(nn.Module):
    def __init__(self, args,   model_attribute,feature_model,device):
        super().__init__()
        self.args = args
        self.model_attribute=model_attribute
         
        self.encoder=EncoderForReviewAttributePair(args,   model_attribute,feature_model,device,args.use_attributes,args.use_image)
        
        
        text_embed_dim=args.d_text_tok
        image_embed_dim=768
        self.max_mention_image_num=model_attribute.max_mention_image_num
        self.max_entity_image_num=model_attribute.max_entity_image_num
        model_embed_dim=args.model_embed_dim
        self.device=device
        modality_num=0
        self.classifier=   ClassifierWithoutReshape( (args.max_attribute_num+1)*model_embed_dim,dim_feedforward=2048,dropout=args.dr,
                                                    output_dim=model_embed_dim)
       
        
 
    def gen_text_feature(self,input_ids,attention_mask,token_type_ids):
        mention_attribute_pair_embeds,batch_size=self.encoder( input_ids,attention_mask,token_type_ids)
        text_representation=self.classifier(mention_attribute_pair_embeds,batch_size)
        batch_size,entity_num,hidden_dim=text_representation.shape
        text_representation=text_representation.reshape(batch_size*entity_num,-1)
        return text_representation
    
    def forward(self,   input_ids,attention_mask  ,token_type_ids=None):
      
        batch_size=input_ids.shape[0]
        text_representation=self.gen_text_feature(input_ids,attention_mask,token_type_ids)
 
 
        return text_representation
                
                
                                
                