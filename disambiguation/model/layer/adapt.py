
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
     

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x
    
    

class CustomCLIP(nn.Module):

    def __init__(self,   clip_model,image_embed_dim=768,is_resnet=False):
        super().__init__()
        self.image_encoder = clip_model 
        # self.text_encoder = TextEncoder(cfg, classnames, clip_model)
        if not is_resnet:
            self.logit_scale = clip_model.logit_scale
        # if is_resnet:
        #     self.dtype=clip_model.feature_extractor.dtype
        # else:
        #     self.dtype = clip_model.dtype
        self.adapter = Adapter(image_embed_dim, 4)#.to(self.dtype)

            
    # def forward(self, pixel_values, input_ids, attention_mask):
        
    #     image_features = self.image_encoder.get_image_features(pixel_values)
        
    #     x = self.adapter(image_features)

    #     ratio = 0.2
    #     image_features = ratio * x + (1 - ratio) * image_features

    #     # text_features = self.text_encoder()

    #     image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    #     # text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    #     # logit_scale = self.logit_scale.exp()
    #     # logits = logit_scale * image_features @ text_features.t()

    #     return image_features#logits    
  
        
    def get_image_features(self,pixel_values):
        image_features = self.image_encoder.get_image_features(pixel_values)
        x = self.adapter(image_features)

        ratio = 0.2
        image_features = ratio * x + (1 - ratio) * image_features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features
     
class CustomMultimodalCLIP(nn.Module):

    def __init__(self,   clip_model,image_embed_dim=768,is_resnet=False,has_adapter=True):
        super().__init__()
        self.encoder = clip_model 
        # self.text_encoder = TextEncoder(cfg, classnames, clip_model)
        if not is_resnet:
            self.logit_scale = clip_model.logit_scale
        # if is_resnet:
        #     self.dtype=clip_model.feature_extractor.dtype
        # else:
        #     self.dtype = clip_model.dtype
        if has_adapter:
            self.text_adapter = Adapter(image_embed_dim, 4)#.to(self.dtype)
            self.image_adapter = Adapter(image_embed_dim, 4)#.to(self.dtype)
        self.has_adapter=has_adapter
            
    def forward(self, pixel_values, input_ids, attention_mask):
        
        clip_output = self.encoder(input_ids=input_ids,attention_mask=attention_mask, pixel_values=pixel_values)# 
        text_embeds=clip_output.text_embeds
        image_emeds=clip_output.image_embeds
        
        if self.has_adapter:
            image_x = self.image_adapter(image_emeds)
            text_x = self.text_adapter(text_embeds)

            ratio = 0.2
            text_embeds = ratio * text_x + (1 - ratio) * text_embeds
            image_emeds = ratio * image_x + (1 - ratio) * image_emeds

        

        image_features = image_emeds / image_emeds.norm(dim=-1, keepdim=True)
        text_features = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return image_features,text_features,logits    
  
        
    def get_image_features(self,pixel_values):
        image_features = self.image_encoder.get_image_features(pixel_values)
        x = self.adapter(image_features)

        ratio = 0.2
        image_features = ratio * x + (1 - ratio) * image_features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features    