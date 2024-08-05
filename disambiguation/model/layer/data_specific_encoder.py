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
from disambiguation.model.layer.basic_layer  import   PositionalEncoding,myPrinter
#clip         
class Encoder(nn.Module):
    def __init__(self, args,   model_attribute,clip_model,device, use_attributes):
        super().__init__()
        self.encoder = clip_model
        self.device=device
        self.use_attributes=use_attributes
        self.parallel= args.parallel 
        
    def split_text_attribute_embed(self,mention_entity_text_attribute_embeds,batch_size,double_sequence_num,sequence_len,sequence_num):
        original_shape_mention_entity_text_attribute_embeds=mention_entity_text_attribute_embeds.reshape(batch_size,double_sequence_num,sequence_len,-1)
        mention_entity_attribute_embeds=original_shape_mention_entity_text_attribute_embeds[:,sequence_num:,:,:]
        mention_entity_text_embeds=original_shape_mention_entity_text_attribute_embeds[:,:sequence_num,:,:]
        return mention_entity_text_embeds,mention_entity_attribute_embeds
    
    def split_mention_entity_embed(self,mention_entity_embeds):
        mention_embed=mention_entity_embeds[:,0:1,:,:]
        entity_embeds=mention_entity_embeds[:,1:,:,:]
        return mention_embed,entity_embeds
    def split_attention(self,attention_mask):
        #batch_size,double_sequence_num,sequence_len
        if self.use_attributes:
            sequence_num=int(attention_mask.shape[1]/2)
        else:
            sequence_num=int(attention_mask.shape[1])
        mention_entity_text_attention_mask=attention_mask[:,:sequence_num,:]
        mention_text_attention_mask=mention_entity_text_attention_mask[:,0:1,:]
        entity_text_attention_mask=mention_entity_text_attention_mask[:,1:,:]
        if self.use_attributes:
            mention_entity_attribute_attention_mask=attention_mask[:,sequence_num:,:]
            mention_attribute_attention_mask=mention_entity_attribute_attention_mask[:,0:1,:]
            entity_attribute_attention_mask=mention_entity_attribute_attention_mask[:,1:,:]
            
            return mention_text_attention_mask,entity_text_attention_mask,mention_attribute_attention_mask,entity_attribute_attention_mask
        else:
            return mention_text_attention_mask,entity_text_attention_mask
    def encode_input(self,reshaped_pixel_values,reshaped_input_ids,reshaped_attention_mask):
        output_clip=self.encoder(input_ids=reshaped_input_ids,attention_mask=reshaped_attention_mask , pixel_values=reshaped_pixel_values)
        mention_entity_text_attribute_embeds=output_clip.text_model_output.last_hidden_state  #(batch_size*textual_sequence_num,sequence_length, hidden_size) 44,77,512
        mention_entity_img_embed=output_clip.vision_model_output.last_hidden_state  #(batch_size*sequence_num,patch_num, hidden_size) 22,50,768
        return mention_entity_text_attribute_embeds,mention_entity_img_embed
    
    def forward( self,pixel_values,input_ids,attention_mask ):
        #batch_size,double_sequence_num,sequence_len
        # pixel_values=mention_entity_list_text_attribute_image_processed_input['pixel_values'] 
        # input_ids=mention_entity_list_text_attribute_image_processed_input['input_ids'] 
        # attention_mask=mention_entity_list_text_attribute_image_processed_input['attention_mask'] 
        # if self.parallel !="data_parallel":
        #     pixel_values=pixel_values.to(self.device)
        #     input_ids=input_ids.to(self.device)
        #     attention_mask=attention_mask.to(self.device)
        # else:
        #     pixel_values=pixel_values.cuda()
        #     input_ids=input_ids.cuda()
        #     attention_mask=attention_mask.cuda()
            
        batch_size,textual_sequence_num,sequence_len=input_ids.shape 
        batch_size,sequence_num,num_channels, height, width=pixel_values.shape
        reshaped_input_ids=input_ids.reshape(batch_size*textual_sequence_num,-1)
        reshaped_attention_mask=attention_mask.reshape(batch_size*textual_sequence_num,-1)
        reshaped_pixel_values=pixel_values.reshape(batch_size*sequence_num,num_channels, height, width)
        mention_entity_text_attribute_embeds,mention_entity_img_embed=self.encode_input(reshaped_pixel_values,reshaped_input_ids,reshaped_attention_mask)
        
        patch_num=mention_entity_img_embed.shape[-2]
        original_shape_mention_entity_img_embed=mention_entity_img_embed.reshape(batch_size,sequence_num,patch_num,-1)
        mention_image_embed,entity_image_embeds=self.split_mention_entity_embed(original_shape_mention_entity_img_embed)
        if self.use_attributes:
            mention_entity_text_embeds,mention_entity_attribute_embeds=self.split_text_attribute_embed( mention_entity_text_attribute_embeds,batch_size,textual_sequence_num,sequence_len,sequence_num)
            mention_attribute_embed,entity_attribute_embeds=self.split_mention_entity_embed(mention_entity_attribute_embeds)
            mention_text_embed,entity_text_embeds=self.split_mention_entity_embed(mention_entity_text_embeds)
            mention_text_attention_mask,entity_text_attention_mask,mention_attribute_attention_mask,entity_attribute_attention_mask=self.split_attention(attention_mask)
            # img_evidences_embed=img_evidences_embed[:,1:,:] #  remove CLS token  
            #(batch_size, sequence_num,sequence_length, hidden_size), (batch_size,sequence_num,patch_num, hidden_size)
            return mention_text_embed,entity_text_embeds,mention_image_embed,entity_image_embeds,mention_attribute_embed,entity_attribute_embeds,mention_text_attention_mask,entity_text_attention_mask,mention_attribute_attention_mask,entity_attribute_attention_mask,batch_size

        else:
            original_shape_mention_entity_text_embed=mention_entity_text_attribute_embeds.reshape(batch_size,textual_sequence_num,sequence_len,-1)
            mention_text_embed,entity_text_embeds=self.split_mention_entity_embed(original_shape_mention_entity_text_embed)
            mention_text_attention_mask,entity_text_attention_mask =self.split_attention(attention_mask)
            return mention_text_embed,entity_text_embeds,mention_image_embed,entity_image_embeds,mention_text_attention_mask,entity_text_attention_mask,batch_size
        
#clip 
class  Encoder2(Encoder):        
    def __init__(self, args,   model_attribute,clip_model,device, use_attributes,max_mention_image_num,max_entity_image_num):
        super().__init__(args,   model_attribute,clip_model,device, use_attributes)
        self.max_mention_image_num=max_mention_image_num
        self.max_entity_image_num=max_entity_image_num
        
    def split_mention_entity_image_embed(self,mention_entity_embeds,batch_size,sequence_num,patch_num):
        mention_embed=mention_entity_embeds[:,0:self.max_mention_image_num,1:,:]
        entity_embeds=mention_entity_embeds[:,self.max_mention_image_num:,1:,:]
        remaining_path_num=patch_num-1
        real_sequence_num=int((sequence_num-self.max_mention_image_num)/self.max_entity_image_num)
        mention_embed=mention_embed.reshape(batch_size,1,self.max_mention_image_num*remaining_path_num,-1)
        entity_embeds=entity_embeds.reshape(batch_size,real_sequence_num,self.max_entity_image_num*remaining_path_num,-1)
        return mention_embed ,entity_embeds
    def split_mention_entity_image_mask(self,mention_entity_image_mask,batch_size,sequence_num,patch_num):
        mention_entity_image_mask=torch.unsqueeze(mention_entity_image_mask,-1)
        mention_entity_image_mask=mention_entity_image_mask.expand(-1,-1,patch_num)
        mention_embed=mention_entity_image_mask[:,0:self.max_mention_image_num,:]
        entity_embeds=mention_entity_image_mask[:,self.max_mention_image_num:,:]
        real_sequence_num=int((sequence_num-self.max_mention_image_num)/self.max_entity_image_num)
        mention_embed=mention_embed.reshape(batch_size,1,self.max_mention_image_num*patch_num)
        entity_embeds=entity_embeds.reshape(batch_size,real_sequence_num,self.max_entity_image_num*patch_num)
        return mention_embed,entity_embeds
    def forward( self,pixel_values,input_ids,attention_mask ,mention_entity_image_mask):
        batch_size,textual_sequence_num,sequence_len=input_ids.shape 
        batch_size,entity_image_num,num_channels, height, width=pixel_values.shape
        reshaped_input_ids=input_ids.reshape(batch_size*textual_sequence_num,-1)
        reshaped_attention_mask=attention_mask.reshape(batch_size*textual_sequence_num,-1)
        reshaped_pixel_values=pixel_values.reshape(batch_size*entity_image_num,num_channels, height, width)
        output_clip=self.encoder(input_ids=reshaped_input_ids,attention_mask=reshaped_attention_mask , pixel_values=reshaped_pixel_values)
        mention_entity_text_attribute_embeds=output_clip.text_model_output.last_hidden_state  #(batch_size*textual_sequence_num,sequence_length, hidden_size) 44,77,512
        mention_entity_img_embed=output_clip.vision_model_output.last_hidden_state  #(batch_size*sequence_num,patch_num, hidden_size) 22,50,768
        patch_num=mention_entity_img_embed.shape[-2]
        original_shape_mention_entity_img_embed=mention_entity_img_embed.reshape(batch_size,entity_image_num,patch_num,-1)
        mention_image_embed,entity_image_embeds=self.split_mention_entity_image_embed(original_shape_mention_entity_img_embed,batch_size,entity_image_num,patch_num)
        mention_image_mask,entity_image_mask=self.split_mention_entity_image_mask(mention_entity_image_mask,batch_size,entity_image_num,patch_num)
        if self.use_attributes:
            attribute_num=int(textual_sequence_num/2)
            mention_entity_text_embeds,mention_entity_attribute_embeds=self.split_text_attribute_embed( mention_entity_text_attribute_embeds,batch_size,textual_sequence_num,sequence_len,attribute_num)
            mention_attribute_embed,entity_attribute_embeds=self.split_mention_entity_embed(mention_entity_attribute_embeds)
            mention_text_embed,entity_text_embeds=self.split_mention_entity_embed(mention_entity_text_embeds)
            mention_text_attention_mask,entity_text_attention_mask,mention_attribute_attention_mask,entity_attribute_attention_mask=self.split_attention(attention_mask)
            # img_evidences_embed=img_evidences_embed[:,1:,:] #  remove CLS token  
            #(batch_size, sequence_num,sequence_length, hidden_size), (batch_size,sequence_num,patch_num, hidden_size)
            return mention_text_embed,entity_text_embeds,mention_image_embed,entity_image_embeds,mention_attribute_embed,entity_attribute_embeds,mention_text_attention_mask,entity_text_attention_mask,mention_attribute_attention_mask,entity_attribute_attention_mask,batch_size,mention_image_mask,entity_image_mask

        else:
            original_shape_mention_entity_text_embed=mention_entity_text_attribute_embeds.reshape(batch_size,textual_sequence_num,sequence_len,-1)
            mention_text_embed,entity_text_embeds=self.split_mention_entity_embed(original_shape_mention_entity_text_embed)
            mention_text_attention_mask,entity_text_attention_mask =self.split_attention(attention_mask)
            return mention_text_embed,entity_text_embeds,mention_image_embed,entity_image_embeds,mention_text_attention_mask,entity_text_attention_mask,batch_size,mention_image_mask,entity_image_mask


#clip   
class Encoder3(nn.Module):
    def __init__(self, args,   model_attribute,clip_model,device, use_attributes,use_image):
        super().__init__()
        self.encoder = clip_model
        self.device=device
        self.use_attributes=use_attributes
        self.parallel= args.parallel 
        self.use_image=use_image
        self.max_attribute_num=args.max_attribute_num
        self.max_entity_num=args.max_candidate_num
        
    def split_text_attribute_embed(self,mention_entity_text_attribute_embeds,batch_size,double_sequence_num,sequence_len,sequence_num):
        original_shape_mention_entity_text_attribute_embeds=mention_entity_text_attribute_embeds.reshape(batch_size,double_sequence_num,sequence_len,-1)
        mention_entity_attribute_embeds=original_shape_mention_entity_text_attribute_embeds[:,sequence_num+1:,:,:]
        mention_entity_text_embeds=original_shape_mention_entity_text_attribute_embeds[:,:sequence_num+1,:,:]
        return mention_entity_text_embeds,mention_entity_attribute_embeds
    
    def split_mention_entity_embed(self,mention_entity_embeds):
        mention_embed=mention_entity_embeds[:,0:1,:,:]
        entity_embeds=mention_entity_embeds[:,1:,:,:]
        return mention_embed,entity_embeds
    def split_attention(self,attention_mask):
        #batch_size,double_sequence_num,sequence_len
        if self.use_attributes:
            sequence_num=int(attention_mask.shape[1]/2)
        else:
            sequence_num=int(attention_mask.shape[1])
        mention_entity_text_attention_mask=attention_mask[:,:sequence_num,:]
        mention_text_attention_mask=mention_entity_text_attention_mask[:,0:1,:]
        entity_text_attention_mask=mention_entity_text_attention_mask[:,1:,:]
        if self.use_attributes:
            mention_entity_attribute_attention_mask=attention_mask[:,sequence_num:,:]
            mention_attribute_attention_mask=mention_entity_attribute_attention_mask[:,0:1,:]
            entity_attribute_attention_mask=mention_entity_attribute_attention_mask[:,1:,:]
            
            return mention_text_attention_mask,entity_text_attention_mask,mention_attribute_attention_mask,entity_attribute_attention_mask
        else:
            return mention_text_attention_mask,entity_text_attention_mask
    def encode_input(self,reshaped_input_ids,reshaped_attention_mask):
        if self.use_image:
            output_clip=self.encoder(input_ids=reshaped_input_ids,attention_mask=reshaped_attention_mask)# , pixel_values=reshaped_pixel_values
        else:
            output_clip=self.encoder.text_model(input_ids=reshaped_input_ids,attention_mask=reshaped_attention_mask) 
            mention_entity_text_attribute_embeds=output_clip.last_hidden_state  #(batch_size*textual_sequence_num,sequence_length, hidden_size) 44,77,512  
        return mention_entity_text_attribute_embeds
    
    def forward( self, input_ids,attention_mask ):
 
        sequence_num=self.max_entity_num
        batch_size,textual_sequence_num,sequence_len=input_ids.shape 
        # batch_size,sequence_num,num_channels, height, width=pixel_values.shape
        reshaped_input_ids=input_ids.reshape(batch_size*textual_sequence_num,-1)
        reshaped_attention_mask=attention_mask.reshape(batch_size*textual_sequence_num,-1)
        # reshaped_pixel_values=pixel_values.reshape(batch_size*sequence_num,num_channels, height, width)
        mention_entity_text_attribute_embeds=self.encode_input(reshaped_input_ids,reshaped_attention_mask)
        # mention_entity_img_embed=output_clip.vision_model_output.last_hidden_state  #(batch_size*sequence_num,patch_num, hidden_size) 22,50,768
        # patch_num=mention_entity_img_embed.shape[-2]
        # original_shape_mention_entity_img_embed=mention_entity_img_embed.reshape(batch_size,sequence_num,patch_num,-1)
        # mention_image_embed,entity_image_embeds=self.split_mention_entity_embed(original_shape_mention_entity_img_embed)
        if self.use_attributes:
            mention_entity_text_embeds,mention_entity_attribute_embeds=self.split_text_attribute_embed( mention_entity_text_attribute_embeds,batch_size,textual_sequence_num,sequence_len,sequence_num)
            mention_attribute_embed,entity_attribute_embeds=self.split_mention_entity_embed(mention_entity_attribute_embeds)
            mention_text_embed,entity_text_embeds=self.split_mention_entity_embed(mention_entity_text_embeds)
            mention_text_attention_mask,entity_text_attention_mask,mention_attribute_attention_mask,entity_attribute_attention_mask=self.split_attention(attention_mask)
            # img_evidences_embed=img_evidences_embed[:,1:,:] #  remove CLS token  
            #(batch_size, sequence_num,sequence_length, hidden_size), (batch_size,sequence_num,patch_num, hidden_size)
            return mention_text_embed,entity_text_embeds, mention_attribute_embed,entity_attribute_embeds,mention_text_attention_mask,entity_text_attention_mask,mention_attribute_attention_mask,entity_attribute_attention_mask,batch_size

        else:
            original_shape_mention_entity_text_embed=mention_entity_text_attribute_embeds.reshape(batch_size,textual_sequence_num,sequence_len,-1)
            mention_text_embed,entity_text_embeds=self.split_mention_entity_embed(original_shape_mention_entity_text_embed)
            mention_text_attention_mask,entity_text_attention_mask =self.split_attention(attention_mask)
            return mention_text_embed,entity_text_embeds, mention_text_attention_mask,entity_text_attention_mask,batch_size
        
def freeze_part_bert(bert_model,freeze_layer_num):
    count = 0
    for p in bert_model.named_parameters():
        
        if (count<=freeze_layer_num):
            p[1].requires_grad=False    
        else:
            break
              
        count=count+1
        print(p[0], p[1].requires_grad)
        
    return bert_model 

#bert 
class Encoder4( Encoder3):
    def __init__(self, args,   model_attribute,feature_model,device, use_attributes,use_image):
        super().__init__(args,   model_attribute,feature_model,device, use_attributes,use_image)
        if feature_model is None:
            feature_model =   AutoModel.from_pretrained(args.pre_trained_dir)
        
        freeze_part_bert(feature_model,args.freeze_bert_layer_number)
        self.encoder=feature_model
        
    def encode_input(self,reshaped_input_ids,reshaped_attention_mask):
    
        output_clip=self.encoder(input_ids=reshaped_input_ids,attention_mask=reshaped_attention_mask) 
        mention_entity_text_attribute_embeds=output_clip.last_hidden_state  #(batch_size*textual_sequence_num,sequence_length, hidden_size) 44,77,512  
        return mention_entity_text_attribute_embeds
    



#clip   
class ImageLevelEncoder(nn.Module):
    def __init__(self, args,   model_attribute,clip_model,device, use_attributes,use_image):
        super().__init__()
        self.encoder = clip_model
        self.device=device
        self.use_attributes=use_attributes
        self.parallel= args.parallel 
        self.use_image=use_image
        self.max_attribute_num=args.max_attribute_num
        self.max_entity_num=args.max_candidate_num
        self.max_mention_image_num=model_attribute.max_mention_image_num
        self.max_entity_image_num=model_attribute.max_entity_image_num
        
    def split_text_attribute_embed(self,mention_entity_text_attribute_embeds,batch_size,double_sequence_num,sequence_len,sequence_num):
        original_shape_mention_entity_text_attribute_embeds=mention_entity_text_attribute_embeds.reshape(batch_size,double_sequence_num,sequence_len,-1)
        mention_entity_attribute_embeds=original_shape_mention_entity_text_attribute_embeds[:,sequence_num+1:,:,:]
        mention_entity_text_embeds=original_shape_mention_entity_text_attribute_embeds[:,:sequence_num+1,:,:]
        return mention_entity_text_embeds,mention_entity_attribute_embeds
    
    def split_mention_entity_embed(self,mention_entity_embeds):
        mention_embed=mention_entity_embeds[:,0:1,:,:]
        entity_embeds=mention_entity_embeds[:,1:,:,:]
        return mention_embed,entity_embeds
    def split_attention(self,attention_mask):
        #batch_size,double_sequence_num,sequence_len
        if self.use_attributes:
            sequence_num=int(attention_mask.shape[1]/2)
        else:
            sequence_num=int(attention_mask.shape[1])
        mention_entity_text_attention_mask=attention_mask[:,:sequence_num,:]
        mention_text_attention_mask=mention_entity_text_attention_mask[:,0:1,:]
        entity_text_attention_mask=mention_entity_text_attention_mask[:,1:,:]
        if self.use_attributes:
            mention_entity_attribute_attention_mask=attention_mask[:,sequence_num:,:]
            mention_attribute_attention_mask=mention_entity_attribute_attention_mask[:,0:1,:]
            entity_attribute_attention_mask=mention_entity_attribute_attention_mask[:,1:,:]
            
            return mention_text_attention_mask,entity_text_attention_mask,mention_attribute_attention_mask,entity_attribute_attention_mask
        else:
            return mention_text_attention_mask,entity_text_attention_mask
    def encode_input(self,reshaped_pixel_values):
        
        image_embeds=self.encoder.get_image_features(pixel_values=reshaped_pixel_values)# , pixel_values=reshaped_pixel_va
        # image_embeds=output_clip.image_embeds  #(batch_size*textual_sequence_num,sequence_length, hidden_size) 44,77,512  
        return image_embeds
    
    def split_mention_entity_image_embed(self,mention_entity_embeds,batch_size,sequence_num):
        mention_embed=mention_entity_embeds[:,0:self.max_mention_image_num,:]
        entity_embeds=mention_entity_embeds[:,self.max_mention_image_num:,:]
     
        real_sequence_num=int((sequence_num-self.max_mention_image_num)/self.max_entity_image_num)
        mention_embed=mention_embed.reshape(batch_size,1,      -1)
        entity_embeds=entity_embeds.reshape(batch_size,real_sequence_num, -1)
        return mention_embed ,entity_embeds
  
    def forward( self,pixel_values,input_ids,attention_mask ,mention_entity_image_mask=None):
 
        batch_size,entity_image_num,num_channels, height, width=pixel_values.shape
 
        reshaped_pixel_values=pixel_values.reshape(batch_size*entity_image_num,num_channels, height, width)
        mention_entity_img_embed=self.encode_input( reshaped_pixel_values)
         
        original_shape_mention_entity_img_embed=mention_entity_img_embed.reshape(batch_size,entity_image_num,-1)
        mention_image_embed,entity_image_embeds=self.split_mention_entity_image_embed(original_shape_mention_entity_img_embed,batch_size,entity_image_num)
          
 
        return None,None,mention_image_embed,entity_image_embeds,None,None,batch_size

 

class ImageLevelImageAndTextEncoder(nn.Module):
    def __init__(self, args,   model_attribute,clip_model,device, use_attributes,use_image):
        super().__init__()
        self.encoder = clip_model
        self.device=device
        self.use_attributes=use_attributes
        self.parallel= args.parallel 
        self.use_image=use_image
        self.max_attribute_num=args.max_attribute_num
        self.max_entity_num=args.max_candidate_num
        self.max_mention_image_num=model_attribute.max_mention_image_num
        self.max_entity_image_num=model_attribute.max_entity_image_num
        
    def split_text_attribute_embed(self,mention_entity_text_attribute_embeds,batch_size,double_sequence_num,sequence_len,sequence_num):
        original_shape_mention_entity_text_attribute_embeds=mention_entity_text_attribute_embeds.reshape(batch_size,double_sequence_num,sequence_len,-1)
        mention_entity_attribute_embeds=original_shape_mention_entity_text_attribute_embeds[:,sequence_num+1:,:,:]
        mention_entity_text_embeds=original_shape_mention_entity_text_attribute_embeds[:,:sequence_num+1,:,:]
        return mention_entity_text_embeds,mention_entity_attribute_embeds
    
    def split_mention_entity_embed(self,mention_entity_embeds):
        mention_embed=mention_entity_embeds[:,0:1,:,:]
        entity_embeds=mention_entity_embeds[:,1:,:,:]
        return mention_embed,entity_embeds
    def split_attention(self,attention_mask):
        #batch_size,double_sequence_num,sequence_len
        if self.use_attributes:
            sequence_num=int(attention_mask.shape[1]/2)
        else:
            sequence_num=int(attention_mask.shape[1])
        mention_entity_text_attention_mask=attention_mask[:,:sequence_num,:]
        mention_text_attention_mask=mention_entity_text_attention_mask[:,0:1,:]
        entity_text_attention_mask=mention_entity_text_attention_mask[:,1:,:]
        if self.use_attributes:
            mention_entity_attribute_attention_mask=attention_mask[:,sequence_num:,:]
            mention_attribute_attention_mask=mention_entity_attribute_attention_mask[:,0:1,:]
            entity_attribute_attention_mask=mention_entity_attribute_attention_mask[:,1:,:]
            
            return mention_text_attention_mask,entity_text_attention_mask,mention_attribute_attention_mask,entity_attribute_attention_mask
        else:
            return mention_text_attention_mask,entity_text_attention_mask
    def encode_input(self,reshaped_pixel_values,reshaped_input_ids,reshaped_attention_mask):
        image_embeds,text_embeds,logits =self.encoder(input_ids=reshaped_input_ids,attention_mask=reshaped_attention_mask, pixel_values=reshaped_pixel_values)# 
        # image_embeds=self.encoder.get_image_features(pixel_values=reshaped_pixel_values)# , pixel_values=reshaped_pixel_va
        # image_embeds=output_clip.image_embeds  #(batch_size*textual_sequence_num,sequence_length, hidden_size) 44,77,512  
        return image_embeds,text_embeds,logits
    
    def split_mention_entity_image_embed(self,mention_entity_embeds,batch_size,sequence_num):
        mention_embed=mention_entity_embeds[:,0:self.max_mention_image_num,:]
        entity_embeds=mention_entity_embeds[:,self.max_mention_image_num:,:]
     
        real_sequence_num=int((sequence_num-self.max_mention_image_num)/self.max_entity_image_num)
        mention_embed=mention_embed.reshape(batch_size,1,      -1)
        entity_embeds=entity_embeds.reshape(batch_size,real_sequence_num, -1)
        return mention_embed ,entity_embeds
  
    def forward( self,pixel_values,input_ids,attention_mask ,mention_entity_image_mask=None):
        batch_size,textual_sequence_num,sequence_len=input_ids.shape 
        # batch_size,sequence_num,num_channels, height, width=pixel_values.shape
        reshaped_input_ids=input_ids.reshape(batch_size*textual_sequence_num,-1)
        reshaped_attention_mask=attention_mask.reshape(batch_size*textual_sequence_num,-1)
        batch_size,entity_image_num,num_channels, height, width=pixel_values.shape
 
        reshaped_pixel_values=pixel_values.reshape(batch_size*entity_image_num,num_channels, height, width)
        mention_entity_img_embed,entity_text_embed,logits=self.encode_input( reshaped_pixel_values,reshaped_input_ids,reshaped_attention_mask)
         
        original_shape_mention_entity_img_embed=mention_entity_img_embed.reshape(batch_size,entity_image_num,-1)
        original_entity_text_embeds=entity_text_embed.reshape(batch_size,textual_sequence_num,-1)
          
 
        return original_shape_mention_entity_img_embed,original_entity_text_embeds,logits,batch_size




class ImageLevelSeparateImageAndTextEncoder(Encoder):
    def __init__(self, args, model_attribute, clip_model, device, use_attributes,use_image,text_model,use_text):
        super().__init__(args, model_attribute, clip_model, device, use_attributes)
        self.text_model=text_model
        self.use_text=use_text
        self.use_image=use_image
    
          
    def encode_input(self,reshaped_pixel_values,reshaped_input_ids,reshaped_attention_mask):
        if self.use_image and self.use_text:
            text_embeds=self.text_model({"input_ids":reshaped_input_ids,"attention_mask":reshaped_attention_mask})["sentence_embedding"]
            image_embeds =self.encoder( { "pixel_values":reshaped_pixel_values,"image_text_info":[0 for i in range(len(reshaped_pixel_values))]})["sentence_embedding"]# 
        elif self.use_image:
            image_embeds =self.encoder( { "pixel_values":reshaped_pixel_values,"image_text_info":[0 for i in range(len(reshaped_pixel_values))]})["sentence_embedding"]# 
            text_embeds=image_embeds
        else:
            text_embeds=self.text_model({"input_ids":reshaped_input_ids,"attention_mask":reshaped_attention_mask})["sentence_embedding"]
            image_embeds=text_embeds
        
        # image_embeds=self.encoder.get_image_features(pixel_values=reshaped_pixel_values)# , pixel_values=reshaped_pixel_va
        # image_embeds=output_clip.image_embeds  #(batch_size*textual_sequence_num,sequence_length, hidden_size) 44,77,512  
        return text_embeds, image_embeds
     
    def split_text_attribute_embed(self,mention_entity_text_attribute_embeds,batch_size,double_sequence_num,sequence_len,sequence_num):
        original_shape_mention_entity_text_attribute_embeds=mention_entity_text_attribute_embeds.reshape(batch_size,double_sequence_num,sequence_len,-1)
        mention_entity_attribute_embeds=original_shape_mention_entity_text_attribute_embeds[:,sequence_num:,:,:]
        mention_entity_text_embeds=original_shape_mention_entity_text_attribute_embeds[:,:sequence_num,:,:]
        return mention_entity_text_embeds,mention_entity_attribute_embeds
    
    def split_mention_entity_embed(self,mention_entity_embeds):
        mention_embed=mention_entity_embeds[:,0:1,:]
        entity_embeds=mention_entity_embeds[:,1:,:]
        return mention_embed,entity_embeds
    def split_attention(self,attention_mask):
        sequence_num=int(attention_mask.shape[1])
        mention_entity_text_attention_mask=attention_mask[:,:sequence_num,:]
        mention_text_attention_mask=mention_entity_text_attention_mask[:,0:1,:]
        entity_text_attention_mask=mention_entity_text_attention_mask[:,1:,:]
        return mention_text_attention_mask,entity_text_attention_mask
 
    
    def forward( self,pixel_values,input_ids,attention_mask ):
            
        batch_size,textual_sequence_num,sequence_len=input_ids.shape 
        
        
        if self.use_image and self.use_text:
            batch_size,sequence_num,num_channels, height, width=pixel_values.shape
            reshaped_input_ids=input_ids.reshape(batch_size*textual_sequence_num,-1)
            reshaped_attention_mask=attention_mask.reshape(batch_size*textual_sequence_num,-1)
            reshaped_pixel_values=pixel_values.reshape(batch_size*sequence_num,num_channels, height, width)
            mention_entity_text_attribute_embeds,mention_entity_img_embed=self.encode_input(reshaped_pixel_values,reshaped_input_ids,reshaped_attention_mask)
            original_shape_mention_entity_img_embed=mention_entity_img_embed.reshape(batch_size,sequence_num,-1)
            mention_image_embed,entity_image_embeds=self.split_mention_entity_embed(original_shape_mention_entity_img_embed)
            original_shape_mention_entity_text_embed=mention_entity_text_attribute_embeds.reshape(batch_size,textual_sequence_num,-1)
            mention_text_embed,entity_text_embeds=self.split_mention_entity_embed(original_shape_mention_entity_text_embed)
        elif self.use_text: 
            reshaped_input_ids=input_ids.reshape(batch_size*textual_sequence_num,-1)
            reshaped_attention_mask=attention_mask.reshape(batch_size*textual_sequence_num,-1)
            reshaped_pixel_values=None
            mention_entity_text_attribute_embeds,mention_entity_img_embed=self.encode_input(reshaped_pixel_values,reshaped_input_ids,reshaped_attention_mask)
            original_shape_mention_entity_text_embed=mention_entity_text_attribute_embeds.reshape(batch_size,textual_sequence_num,-1)
            mention_text_embed,entity_text_embeds=self.split_mention_entity_embed(original_shape_mention_entity_text_embed)
            mention_image_embed,entity_image_embeds=mention_text_embed,entity_text_embeds
        else:
            batch_size,sequence_num,num_channels, height, width=pixel_values.shape
            reshaped_input_ids=None
            reshaped_attention_mask=None
            reshaped_pixel_values=pixel_values.reshape(batch_size*sequence_num,num_channels, height, width)
            mention_entity_text_attribute_embeds,mention_entity_img_embed=self.encode_input(reshaped_pixel_values,reshaped_input_ids,reshaped_attention_mask)
            original_shape_mention_entity_img_embed=mention_entity_img_embed.reshape(batch_size,sequence_num,-1)
            mention_image_embed,entity_image_embeds=self.split_mention_entity_embed(original_shape_mention_entity_img_embed)
            mention_text_embed,entity_text_embeds=mention_image_embed,entity_image_embeds
        mention_text_attention_mask,entity_text_attention_mask =self.split_attention(attention_mask)
        return mention_text_embed,entity_text_embeds,mention_image_embed,entity_image_embeds,mention_text_attention_mask,entity_text_attention_mask,batch_size
        
# 
class ImageLevelSeparateImageAndTextEncoderText(ImageLevelSeparateImageAndTextEncoder):
    def forward( self, input_ids,attention_mask ):
            
        batch_size,textual_sequence_num,sequence_len=input_ids.shape 
         
        reshaped_input_ids=input_ids.reshape(batch_size*textual_sequence_num,-1)
        reshaped_attention_mask=attention_mask.reshape(batch_size*textual_sequence_num,-1)
        reshaped_pixel_values=None
        mention_entity_text_attribute_embeds,mention_entity_img_embed=self.encode_input(reshaped_pixel_values,reshaped_input_ids,reshaped_attention_mask)
        original_shape_mention_entity_text_embed=mention_entity_text_attribute_embeds.reshape(batch_size,textual_sequence_num,-1)
        mention_text_embed,entity_text_embeds=self.split_mention_entity_embed(original_shape_mention_entity_text_embed)
        
        mention_text_attention_mask,entity_text_attention_mask =self.split_attention(attention_mask)
        return mention_text_embed,entity_text_embeds, mention_text_attention_mask,entity_text_attention_mask,batch_size
    
    def encode_input(self,reshaped_pixel_values,reshaped_input_ids,reshaped_attention_mask):
        text_embeds=self.text_model({"input_ids":reshaped_input_ids,"attention_mask":reshaped_attention_mask})["sentence_embedding"]
        image_embeds=text_embeds
        
        # image_embeds=self.encoder.get_image_features(pixel_values=reshaped_pixel_values)# , pixel_values=reshaped_pixel_va
        # image_embeds=output_clip.image_embeds  #(batch_size*textual_sequence_num,sequence_length, hidden_size) 44,77,512  
        return text_embeds, image_embeds
    
class ImageLevelImageEncoder(nn.Module):
    def __init__(self, args,   model_attribute,clip_model,device, use_attributes,use_image):
        super().__init__()
        self.encoder = clip_model
        self.device=device
        self.use_attributes=use_attributes
        self.parallel= args.parallel 
        self.use_image=use_image
        self.max_attribute_num=args.max_attribute_num
        self.max_entity_num=args.max_candidate_num
        self.max_mention_image_num=model_attribute.max_mention_image_num
        self.max_entity_image_num=model_attribute.max_entity_image_num
          
    
    def encode_input(self,reshaped_pixel_values):
        
        image_embeds=self.encoder.get_image_features(pixel_values=reshaped_pixel_values)# , pixel_values=reshaped_pixel_va
        # image_embeds=output_clip.image_embeds  #(batch_size*textual_sequence_num,sequence_length, hidden_size) 44,77,512  
        return image_embeds
    
    def split_mention_entity_image_embed(self,mention_entity_embeds,batch_size,sequence_num):
        mention_embed=mention_entity_embeds[:,0:self.max_mention_image_num,:]
        entity_embeds=mention_entity_embeds[:,self.max_mention_image_num:,:]
     
        real_sequence_num=int((sequence_num-self.max_mention_image_num)/self.max_entity_image_num)
        mention_embed=mention_embed.reshape(batch_size,1,      -1)
        entity_embeds=entity_embeds.reshape(batch_size,real_sequence_num, -1)
        return mention_embed ,entity_embeds
  
    def forward( self,pixel_values, mention_entity_image_mask=None):
 
        batch_size,entity_image_num,num_channels, height, width=pixel_values.shape
 
        reshaped_pixel_values=pixel_values.reshape(batch_size*entity_image_num,num_channels, height, width)
        mention_entity_img_embed=self.encode_input( reshaped_pixel_values)
         
        original_shape_mention_entity_img_embed=mention_entity_img_embed.reshape(batch_size,entity_image_num,-1)
        mention_image_embed,entity_image_embeds=self.split_mention_entity_image_embed(original_shape_mention_entity_img_embed,batch_size,entity_image_num)
          
 
        return  mention_image_embed,entity_image_embeds, batch_size
                

class EncoderForReviewAttributePair(nn.Module):
    def __init__(self, args,   model_attribute,feature_model,device, use_attributes,use_image):
        super().__init__()
        freeze_part_bert(feature_model,args.freeze_bert_layer_number)
        self.encoder = feature_model
        self.device=device
        self.use_attributes=use_attributes
        self.parallel= args.parallel 
        self.use_image=use_image
        self.max_attribute_num=args.max_attribute_num
        self.max_entity_num=args.max_candidate_num
        self.pair_num=args.max_attribute_num+1
         
    def encode_input(self,reshaped_input_ids,reshaped_attention_mask,token_type_ids):
        if token_type_ids is not None:
            output_clip=self.encoder(input_ids=reshaped_input_ids,attention_mask=reshaped_attention_mask,token_type_ids=token_type_ids) 
        else:
            output_clip=self.encoder(input_ids=reshaped_input_ids,attention_mask=reshaped_attention_mask ) 
        mention_entity_text_attribute_embeds=output_clip.pooled_output  #(batch_size*textual_sequence_num,sequence_length, hidden_size) 44,77,512  
        return mention_entity_text_attribute_embeds
    
    def forward( self, input_ids,attention_mask ,token_type_ids ):
 
        sequence_num=self.max_entity_num
        batch_size,textual_sequence_num,sequence_len=input_ids.shape 
        # batch_size,sequence_num,num_channels, height, width=pixel_values.shape
        reshaped_input_ids=input_ids.reshape(batch_size*textual_sequence_num,-1)
        reshaped_attention_mask=attention_mask.reshape(batch_size*textual_sequence_num,-1)
        if token_type_ids is not None:
            reshaped_token_type_ids=token_type_ids.reshape(batch_size*textual_sequence_num,-1)
        else:
            reshaped_token_type_ids=None
        mention_entity_text_attribute_embeds=self.encode_input(reshaped_input_ids,reshaped_attention_mask,reshaped_token_type_ids)
        original_shape_mention_attribute_embeds=mention_entity_text_attribute_embeds.reshape(batch_size,self.max_entity_num,-1)
        
        return original_shape_mention_attribute_embeds,batch_size
       
class  ImageEncoder(Encoder):        
    def __init__(self, args,   model_attribute,clip_model,device, use_attributes,max_mention_image_num,max_entity_image_num):
        super().__init__(args,   model_attribute,clip_model,device, use_attributes)
        self.max_mention_image_num=max_mention_image_num
        self.max_entity_image_num=max_entity_image_num
        
    def split_mention_entity_image_embed(self,mention_entity_embeds,batch_size,sequence_num,patch_num):
        mention_embed=mention_entity_embeds[:,0:self.max_mention_image_num,1:,:]#1:,:]
        entity_embeds=mention_entity_embeds[:,self.max_mention_image_num:,1:,:]
        remaining_path_num=patch_num-1
        real_sequence_num=int((sequence_num-self.max_mention_image_num)/self.max_entity_image_num)
        mention_embed=mention_embed.reshape(batch_size,1,self.max_mention_image_num*remaining_path_num,-1)
        entity_embeds=entity_embeds.reshape(batch_size,real_sequence_num,self.max_entity_image_num*remaining_path_num,-1)
        return mention_embed ,entity_embeds
    def split_mention_entity_image_mask(self,mention_entity_image_mask,batch_size,sequence_num,patch_num):
        mention_entity_image_mask=torch.unsqueeze(mention_entity_image_mask,-1)
        mention_entity_image_mask=mention_entity_image_mask.expand(-1,-1,patch_num)
        mention_embed=mention_entity_image_mask[:,0:self.max_mention_image_num,1:]
        entity_embeds=mention_entity_image_mask[:,self.max_mention_image_num:,1:]
        remaining_path_num=patch_num-1
        real_sequence_num=int((sequence_num-self.max_mention_image_num)/self.max_entity_image_num)
        mention_embed=mention_embed.reshape(batch_size,1,self.max_mention_image_num*remaining_path_num)
        entity_embeds=entity_embeds.reshape(batch_size,real_sequence_num,self.max_entity_image_num*remaining_path_num)
        return mention_embed,entity_embeds
    def forward( self,pixel_values, mention_entity_image_mask):
        batch_size,image_num,num_channels, height, width=pixel_values.shape
        reshaped_pixel_values=pixel_values.reshape(batch_size*image_num,num_channels, height, width)
        output_clip=self.encoder.vision_model( pixel_values=reshaped_pixel_values)
        mention_entity_img_embed=output_clip.last_hidden_state  #(batch_size*image_num,patch_num, hidden_size) 22,50,768
        patch_num=mention_entity_img_embed.shape[-2]
        original_shape_mention_entity_img_embed=mention_entity_img_embed.reshape(batch_size,image_num,patch_num,-1)
        mention_image_embed,entity_image_embeds=self.split_mention_entity_image_embed(original_shape_mention_entity_img_embed,batch_size,image_num,patch_num)
        mention_image_mask,entity_image_mask=self.split_mention_entity_image_mask(mention_entity_image_mask,batch_size,image_num,patch_num)
        return mention_image_embed,entity_image_embeds,batch_size,mention_image_mask,entity_image_mask
  
       
       
class MultimodalSelfAttenEncoder(nn.Module):
    def __init__(self, config, model_attribute,text_embed_dim,model_embed_dim):
        super(MultimodalSelfAttenEncoder, self).__init__()
        myPrinter(
            config, '\t############OneStreamSelfAttenMMEncoder Model#####################')
        
        self.use_attributes = config.use_attributes
        self.device = config.device
        self.position_embeddings = PositionalEncoding(d_model=config.d_tok)
        self.seg_embeddings = nn.Embedding(4, config.d_tok, padding_idx=0)
        # self-attention layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=config.d_tok, nhead=config.n_head,
                                                 dropout=config.dr,
                                                   activation='gelu')
        encoder_norm = nn.LayerNorm(config.d_tok)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, config.x_layers, encoder_norm)
        self.class_embedding = nn.Parameter(torch.randn(config.d_tok))
        self.has_overall_cls_embed=model_attribute.has_overall_cls_embed
        self.text_projection = nn.Linear( text_embed_dim, model_embed_dim, bias=False)
        self.attribute_projection = nn.Linear( text_embed_dim, model_embed_dim, bias=False)

    def gen_multimodal_token_type_id(self,batch_size,patch_size, seq_len,att_seq_len,token_type_ids, att_token_type_ids):
        if token_type_ids is None:
            token_type_ids= torch.ones((batch_size, seq_len)).to(
            self.device) 
        image_segment = torch.ones((batch_size, patch_size)).to(
            self.device) 
        if self.use_attributes:
            if att_token_type_ids is None:
                att_token_type_ids= 2*torch.ones((batch_size, att_seq_len)).to(
                self.device) 
            if self.has_overall_cls_embed:
                overall_cls_segment=torch.zeros((batch_size, 1)).to(self.device) 
                multimodal_segment = torch.cat((overall_cls_segment,token_type_ids, att_token_type_ids, image_segment * 3),
                                dim=1).type(torch.long)
            else:
                multimodal_segment = torch.cat((token_type_ids, att_token_type_ids, image_segment * 3),
                                dim=1).type(torch.long)
        else:
            if self.has_overall_cls_embed:
                overall_cls_segment=torch.zeros((batch_size, 1)).to(self.device) 
                multimodal_segment = torch.cat((overall_cls_segment,token_type_ids, image_segment * 2),
                                dim=1).type(torch.long)
            else:
                multimodal_segment = torch.cat((token_type_ids, image_segment * 2),
                                dim=1).type(torch.long)
        return multimodal_segment
    
    def gen_multimodal_pad_mask(self,batch_size,patch_size, lang_pad_mask,  att_lang_pad_mask):
        if self.use_attributes:
            visn_pad_mask = torch.ones((batch_size, patch_size)).to(self.device)
            if self.has_overall_cls_embed:
                overall_cls_mask=torch.ones((batch_size, 1)).to(self.device)
                pad_mask = torch.cat(
                    [overall_cls_mask,lang_pad_mask,  att_lang_pad_mask, visn_pad_mask], dim=1).type(torch.bool)
            else:
                pad_mask = torch.cat(
                    [lang_pad_mask, att_lang_pad_mask, visn_pad_mask], dim=1).type(torch.bool)
            pad_mask = (~pad_mask)
        else:
            visn_pad_mask = torch.ones((batch_size, patch_size)).to(self.device)
            if self.has_overall_cls_embed:
                overall_cls_mask=torch.ones((batch_size, 1)).to(self.device)
                pad_mask = torch.cat(
                    [overall_cls_mask,lang_pad_mask, visn_pad_mask], dim=1).type(torch.bool)
            else:
                pad_mask = torch.cat(
                    [lang_pad_mask, visn_pad_mask], dim=1).type(torch.bool)
            pad_mask = (~pad_mask)
        return pad_mask
    
    def gen_multimodal_embedding(self,batch_size,lang_feats, att_lang_feats, visn_feats):
        if self.use_attributes:
            if self.has_overall_cls_embed:
                overall_cls_embed=  self.class_embedding.expand(batch_size, 1, -1)
                    
                unit_embeddings = torch.cat([overall_cls_embed,lang_feats, att_lang_feats, visn_feats], dim=1)
            else:
                unit_embeddings = torch.cat([lang_feats, att_lang_feats, visn_feats], dim=1)
        else:
            if self.has_overall_cls_embed:
                overall_cls_embed=  self.class_embedding.expand(batch_size, 1, -1)
                unit_embeddings = torch.cat([overall_cls_embed, lang_feats, visn_feats], dim=1)
            else:
                unit_embeddings = torch.cat([lang_feats, visn_feats], dim=1)
        return unit_embeddings 
    
    def gen_embedding_with_position(self,batch_size,patch_size,seq_len, att_seq_len,lang_feats, lang_pad_mask, token_type_ids, 
                                                      visn_feats, att_lang_feats, att_lang_pad_mask, att_token_type_ids):
        
        multimodal_segment=self.gen_multimodal_token_type_id(batch_size,patch_size,seq_len, att_seq_len,token_type_ids, att_token_type_ids)
        multimodal_attention_mask=self.gen_multimodal_pad_mask( batch_size,patch_size, lang_pad_mask,  att_lang_pad_mask)
        multimodal_embedding=self.gen_multimodal_embedding(batch_size,lang_feats, att_lang_feats, visn_feats)
        unit_embeddings = self.position_embeddings(multimodal_embedding)
        seg_embeddings = self.seg_embeddings(multimodal_segment)
        unit_embeddings = unit_embeddings + seg_embeddings
        return unit_embeddings,multimodal_attention_mask
    
 
    
    def forward(self, lang_feats, lang_pad_mask, token_type_ids, visn_feats, att_lang_feats=None, att_lang_pad_mask=None, att_token_type_ids=None):
        lang_feats=self.text_projection(lang_feats)
        att_lang_feats=self.attribute_projection(att_lang_feats)
        
        (batch_size, seq_len, d_tok) = lang_feats.size()
        if self.use_attributes:
            (_, att_seq_len, att_d_tok) = att_lang_feats.size()
            assert d_tok == att_d_tok
        else:
            att_seq_len=seq_len
        (_, patch_size, d_patch) = visn_feats.size()
        assert d_tok == d_patch
        
        unit_embeddings,pad_mask=self.gen_embedding_with_position( batch_size,patch_size,seq_len, att_seq_len,lang_feats, lang_pad_mask, token_type_ids, 
                                                      visn_feats, att_lang_feats, att_lang_pad_mask, att_token_type_ids)
        output = self.encoder(unit_embeddings.permute(
            1, 0, 2), src_key_padding_mask=pad_mask)
        #TODO for entity, we need to filter masked entities. e.g., max_entity_num=21, but we only have 15 entities.

        output = output.permute(1, 0, 2)
        
        return  output ,output[:, 0, :]
        