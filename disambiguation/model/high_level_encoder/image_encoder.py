
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

from retrieval.model.matcher import MatchERT
from disambiguation.model.layer.data_specific_encoder import ImageEncoder
from disambiguation.model.layer.basic_layer  import   PositionalEncoding,myPrinter
from disambiguation.model.layer.function import  repeat_mention_in_entity_shape
 

    
class ImageRepresentor(nn.Module):
    def __init__(self, args,   model_attribute,image_feature_model,device):
        super().__init__()
        self.args = args
        self.model_attribute=model_attribute
         
        
        self.image_encoder= ImageEncoder(args,   model_attribute,image_feature_model,device,args.use_attributes,model_attribute.max_mention_image_num,
                                  model_attribute.max_entity_image_num)
        
        text_embed_dim=args.d_text_tok
        image_embed_dim=768
        self.max_mention_image_num=model_attribute.max_mention_image_num
        self.max_entity_image_num=model_attribute.max_entity_image_num
        model_embed_dim=args.model_embed_dim
        self.device=device
        modality_num=0
         
        self.match=MatchERT(output_classification=False)
      
        
        
    def gen_image_feature(self,pixel_values,mention_entity_image_mask):
        mention_image_embed,entity_image_embeds,batch_size,mention_image_mask,entity_image_mask=self.image_encoder(pixel_values,mention_entity_image_mask)
        reshaped_mention_embed=repeat_mention_in_entity_shape(mention_image_embed, entity_image_embeds)
        batch_size, sequence_num,sequence_len,hiddle_dim=entity_image_embeds.shape 
        reshaped_entity_list_embed=entity_image_embeds.reshape(batch_size*sequence_num,sequence_len,hiddle_dim)
        reshaped_mention_embed=reshaped_mention_embed.reshape(batch_size*sequence_num,-1,hiddle_dim)
        image_representation=self.match(src_global=None, src_local=reshaped_mention_embed,  tgt_global=None, tgt_local=reshaped_entity_list_embed)
        return image_representation
     
    
    def forward(self,  pixel_values,mention_entity_image_mask=None
                ):
      
        
        image_representation=self.gen_image_feature(pixel_values,mention_entity_image_mask)
 

        return image_representation
                
                
                                
                