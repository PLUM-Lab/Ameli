
 # encoding=utf-8
from io import BytesIO
import math
import torch.nn.functional as F
import torchvision.models as cv_models
import torch.nn as nn
import torch
import os
from disambiguation.model.high_level_encoder.image_encoder import ImageRepresentor
from disambiguation.model.high_level_encoder.text_encoder import TextRepresentor
import sys

from transformers import CLIPProcessor
import transformers

from retrieval.model.matcher import MatchERT
from .layer.basic_layer import * 
from .layer.data_specific_encoder import * 
from .layer.function import masked_softmax

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

class AMELIMatch(nn.Module):
    def __init__(self, args,   model_attribute,clip_model,device):
        super().__init__()
        self.args = args
        self.model_attribute=model_attribute
        self.encoder=Encoder(args,   model_attribute,clip_model,device,args.use_attributes)
        text_embed_dim=512
        image_embed_dim=768
        model_embed_dim=args.model_embed_dim
        self.device=device
        self.text_matcher=Matcher2(text_embed_dim,model_embed_dim,dropout=args.dr,match_pooling_method=model_attribute.match_pooling)
        self.image_matcher=Matcher2(image_embed_dim,model_embed_dim,dropout=args.dr,match_pooling_method=model_attribute.match_pooling)
        
        if args.use_attributes:
            self.attribute_matcher=Matcher2(text_embed_dim,model_embed_dim,dropout=args.dr,match_pooling_method=model_attribute.match_pooling) 
            if model_attribute.match_pooling=="pooling3":
                input_dim=text_embed_dim*2+image_embed_dim
            else:
                input_dim=3*model_embed_dim
            self.cross_modal_fuser=Fuser2(input_dim,dropout=args.dr)
        else:
            self.cross_modal_fuser=Fuser2(2*model_embed_dim,dropout=args.dr)
        
     
    
    def forward(self, pixel_values,input_ids,attention_mask ,label,entity_mask_tensor  , is_train,   is_contrastive=True,mention_entity_image_mask=None,token_type_ids=None):
      
        # if self.args.parallel !="data_parallel":
        #     entity_mask_tensor=entity_mask_tensor.to(self.device)
        # else:
        #     entity_mask_tensor=entity_mask_tensor.cuda()
        if self.args.use_attributes:
            mention_text_embed,entity_text_embeds,mention_image_embed,entity_image_embeds,mention_attribute_embed,entity_attribute_embeds,mention_text_attention_mask,entity_text_attention_mask,mention_attribute_attention_mask,entity_attribute_attention_mask,batch_size=self.encoder(pixel_values,input_ids,attention_mask)
        else:
            mention_text_embed,entity_text_embeds,mention_image_embed,entity_image_embeds,mention_text_attention_mask,entity_text_attention_mask,batch_size=self.encoder(pixel_values,input_ids,attention_mask)
        entity_text_match=self.text_matcher(mention_text_embed,entity_text_embeds,entity_text_attention_mask,mention_text_attention_mask)
        entity_image_match=self.image_matcher(mention_image_embed,entity_image_embeds,None,None)
        if self.args.use_attributes:
            entity_attribute_match=self.attribute_matcher(mention_attribute_embed,entity_attribute_embeds,entity_attribute_attention_mask,mention_attribute_attention_mask)
            logics=self.cross_modal_fuser(entity_text_match, entity_image_match,entity_attribute_match,batch_size)
        else:
            logics=self.cross_modal_fuser(entity_text_match, entity_image_match,None,batch_size)
        logics=masked_softmax(logics,entity_mask_tensor)

        return logics,label,None,None,None ,None
    
        

class AMELIAtten(nn.Module):
    def __init__(self, args,   model_attribute,clip_model,device):
        super().__init__()
        self.args = args
        self.model_attribute=model_attribute
        self.encoder=Encoder(args,   model_attribute,clip_model,device,args.use_attributes)
        text_embed_dim=512
        image_embed_dim=768
        model_embed_dim=args.model_embed_dim
        self.device=device
        
        
        self.text_intra_modal_fuser=IntraModalityFuser(text_embed_dim,model_embed_dim,dropout=args.dr)
        self.image_intra_modal_fuser=IntraModalityFuser(image_embed_dim,model_embed_dim,dropout=args.dr)
        if args.use_attributes:
            self.attribute_intra_modal_fuser=IntraModalityFuser(text_embed_dim,model_embed_dim,dropout=args.dr)
        self.inter_modal_fuser=MultimodalSelfAttenEncoder(args, model_attribute,text_embed_dim,model_embed_dim)
       
        self.classifier=   Classifier( model_embed_dim,dropout=args.dr)
        
         
    
    def forward(self, pixel_values,input_ids,attention_mask ,label,entity_mask_tensor  , is_train,   is_contrastive=True,token_type_ids=None):
      
        # if self.args.parallel !="data_parallel":
        #     entity_mask_tensor=entity_mask_tensor.to(self.device)
        # else:
        #     entity_mask_tensor=entity_mask_tensor.cuda()
        if self.args.use_attributes:
            mention_text_embed,entity_text_embeds,mention_image_embed,entity_image_embeds,mention_attribute_embed,entity_attribute_embeds,mention_text_attention_mask,entity_text_attention_mask,mention_attribute_attention_mask,entity_attribute_attention_mask,batch_size=self.encoder(pixel_values,input_ids,attention_mask)
        else:
            mention_text_embed,entity_text_embeds,mention_image_embed,entity_image_embeds,mention_text_attention_mask,entity_text_attention_mask,batch_size=self.encoder(pixel_values,input_ids,attention_mask)
        entity_aware_mention_text_embed,mention_text_attention_mask=self.text_intra_modal_fuser(mention_text_embed,entity_text_embeds,entity_text_attention_mask,mention_text_attention_mask)
        
        entity_aware_mention_image_embed,_=self.image_intra_modal_fuser(mention_image_embed,entity_image_embeds,None,None)
        if self.args.use_attributes:
            entity_aware_mention_attribute_embed,mention_attribute_attention_mask=self.attribute_intra_modal_fuser(mention_attribute_embed,entity_attribute_embeds,entity_attribute_attention_mask,mention_attribute_attention_mask)
            
            _,fused_multimodal_entity_embed=self.inter_modal_fuser(entity_aware_mention_text_embed, mention_text_attention_mask,
                                                                 None, entity_aware_mention_image_embed, entity_aware_mention_attribute_embed , 
                                                                 mention_attribute_attention_mask,None)
        else:
            _,fused_multimodal_entity_embed=self.inter_modal_fuser(entity_aware_mention_text_embed, mention_text_attention_mask,  None, entity_aware_mention_image_embed, None , None,None)
        logics=self.classifier(fused_multimodal_entity_embed,batch_size)
        logics=    masked_softmax(logics,entity_mask_tensor)

        return logics,label      ,None,None,None   ,None
    
              

class AMELIMatchPooling2(nn.Module):
    def __init__(self, args,   model_attribute,clip_model,device):
        super().__init__()
        self.args = args
        self.model_attribute=model_attribute
        self.encoder=Encoder(args,   model_attribute,clip_model,device,args.use_attributes)
        text_embed_dim=512
        image_embed_dim=768
        self.max_mention_image_num=model_attribute.max_mention_image_num
        self.max_entity_image_num=model_attribute.max_entity_image_num
        model_embed_dim=args.model_embed_dim
        self.device=device
        modality_num=0
        if args.use_attributes: 
            modality_num+=1
            self.attribute_matcher=Matcher2(text_embed_dim,model_embed_dim,dropout=args.dr,match_pooling_method=model_attribute.match_pooling)
        if args.use_image:
            modality_num+=1
            self.image_matcher=Matcher2(image_embed_dim,model_embed_dim,dropout=args.dr,match_pooling_method=model_attribute.match_pooling)
        if args.use_text:
            modality_num+=1
            self.text_matcher=Matcher2(text_embed_dim,model_embed_dim,dropout=args.dr,match_pooling_method=model_attribute.match_pooling)  
              
        self.modality_weight = torch.nn.Parameter(data=torch.Tensor(modality_num), requires_grad=True)
        self.modality_weight.data.uniform_(-1, 1)
        

    
    def forward(self, pixel_values,input_ids,attention_mask ,label,entity_mask_tensor  , is_train,   is_contrastive=True,entity_image_mask=None,token_type_ids=None):
      
      
        if self.args.use_attributes:
            mention_text_embed,entity_text_embeds,mention_image_embed,entity_image_embeds,mention_attribute_embed,entity_attribute_embeds,mention_text_attention_mask,entity_text_attention_mask,mention_attribute_attention_mask,entity_attribute_attention_mask,batch_size=self.encoder(pixel_values,input_ids,attention_mask)
        else:
            mention_text_embed,entity_text_embeds,mention_image_embed,entity_image_embeds,mention_text_attention_mask,entity_text_attention_mask,batch_size=self.encoder(pixel_values,input_ids,attention_mask)
        match_score_list=[]
        if self.args.use_text:
            entity_text_match=self.text_matcher(mention_text_embed,entity_text_embeds,entity_text_attention_mask,mention_text_attention_mask)
            entity_text_match_score=torch.sum(entity_text_match,-1)
            match_score_list.append(entity_text_match_score)
        else:
            entity_text_match_score=None
        if self.args.use_image:
            entity_image_match=self.image_matcher(mention_image_embed,entity_image_embeds,None,None)
            entity_image_match_score=torch.sum(entity_image_match,-1)
            match_score_list.append(entity_image_match_score)
        else:
            entity_image_match_score=None
        #(batch_size*sequence_num,  sequence_len) 
        
        
        if self.args.use_attributes:
            entity_attribute_match=self.attribute_matcher(mention_attribute_embed,entity_attribute_embeds,entity_attribute_attention_mask,mention_attribute_attention_mask)
            entity_attribute_match_score=torch.sum(entity_attribute_match,-1)
            match_score_list.append(entity_attribute_match_score)
        else:
            entity_attribute_match_score=None
        entity_match_score=torch.stack(match_score_list,dim=-1)
        entity_match_score=entity_match_score*self.modality_weight
        entity_match_score=torch.sum(entity_match_score,-1)
        logics=entity_match_score.reshape(batch_size,-1)
        logics=masked_softmax(logics,entity_mask_tensor)

        return logics,label,entity_match_score,entity_text_match_score,entity_image_match_score,entity_attribute_match_score
                

class AMELIMatchMultiImage(nn.Module):
    def __init__(self, args,   model_attribute,clip_model,device):
        super().__init__()
        self.args = args
        self.model_attribute=model_attribute
        if model_attribute.encoder=="encoder1":
            self.encoder=Encoder(args,   model_attribute,clip_model,device,args.use_attributes)
        else:
            self.encoder=Encoder2(args,   model_attribute,clip_model,device,args.use_attributes,model_attribute.max_mention_image_num,
                                  model_attribute.max_entity_image_num)
        text_embed_dim=512
        image_embed_dim=768
        self.max_mention_image_num=model_attribute.max_mention_image_num
        self.max_entity_image_num=model_attribute.max_entity_image_num
        model_embed_dim=args.model_embed_dim
        self.device=device
        modality_num=0
        if args.use_attributes:     
            modality_num+=1
            self.attribute_matcher=Matcher2(text_embed_dim,model_embed_dim,dropout=args.dr,match_pooling_method=model_attribute.match_pooling)
        if args.use_image:
            modality_num+=1
            self.image_matcher=Matcher2(image_embed_dim,model_embed_dim,dropout=args.dr,match_pooling_method=model_attribute.match_pooling)
        if args.use_text:
            modality_num+=1
            self.text_matcher=Matcher2(text_embed_dim,model_embed_dim,dropout=args.dr,match_pooling_method=model_attribute.match_pooling)  
              
        self.modality_weight = torch.nn.Parameter(data=torch.Tensor(modality_num), requires_grad=True)
        self.modality_weight.data.uniform_(-1, 1)
     
    
    def forward(self, pixel_values,input_ids,attention_mask ,label,entity_mask_tensor  , is_train,   is_contrastive=True,mention_entity_image_mask=None,token_type_ids=None):
      
      
        if self.args.use_attributes:
            mention_text_embed,entity_text_embeds,mention_image_embed,entity_image_embeds,mention_attribute_embed,entity_attribute_embeds,mention_text_attention_mask,entity_text_attention_mask,mention_attribute_attention_mask,entity_attribute_attention_mask,batch_size,mention_image_mask,entity_image_mask=self.encoder(pixel_values,input_ids,attention_mask,mention_entity_image_mask)
        else:
            mention_text_embed,entity_text_embeds,mention_image_embed,entity_image_embeds,mention_text_attention_mask,entity_text_attention_mask,batch_size,mention_image_mask,entity_image_mask=self.encoder(pixel_values,input_ids,attention_mask,mention_entity_image_mask)
        match_score_list=[]
        if self.args.use_text:
            entity_text_match=self.text_matcher(mention_text_embed,entity_text_embeds,entity_text_attention_mask,mention_text_attention_mask)
            entity_text_match_score=torch.sum(entity_text_match,-1)
            match_score_list.append(entity_text_match_score)
        else:
            entity_text_match_score=None
        if self.args.use_image:
            entity_image_match=self.image_matcher(mention_image_embed,entity_image_embeds,entity_image_mask,mention_image_mask)
            entity_image_match_score=torch.sum(entity_image_match,-1)
            match_score_list.append(entity_image_match_score)
        else:
            entity_image_match_score=None
        #(batch_size*sequence_num,  sequence_len) 
        
        
        if self.args.use_attributes:
            entity_attribute_match=self.attribute_matcher(mention_attribute_embed,entity_attribute_embeds,entity_attribute_attention_mask,mention_attribute_attention_mask)
            entity_attribute_match_score=torch.sum(entity_attribute_match,-1)
            match_score_list.append(entity_attribute_match_score)
        else:
            entity_attribute_match_score=None
        entity_match_score=torch.stack(match_score_list,dim=-1)
        entity_match_score=entity_match_score*self.modality_weight
        entity_match_score=torch.sum(entity_match_score,-1)
        logics=entity_match_score.reshape(batch_size,-1)
        logics= masked_softmax(logics,entity_mask_tensor)

        return logics,label,entity_match_score,entity_text_match_score,entity_image_match_score,entity_attribute_match_score
                



class AMELIQuery(nn.Module):
    def __init__(self, args,   model_attribute,clip_model,device):
        super().__init__()
        self.args = args
        self.model_attribute=model_attribute
        if model_attribute.encoder=="encoder1":
            self.encoder=Encoder(args,   model_attribute,clip_model,device,args.use_attributes)
        elif model_attribute.encoder=="encoder2":
            self.encoder=Encoder2(args,   model_attribute,clip_model,device,args.use_attributes,model_attribute.max_mention_image_num,
                                  model_attribute.max_entity_image_num)
        elif model_attribute.encoder=="encoder3":
            self.encoder=Encoder3(args,   model_attribute,clip_model,device,args.use_attributes,args.use_image)
        elif model_attribute.encoder=="encoder4":
            self.encoder= Encoder4(args,   model_attribute,clip_model,device,args.use_attributes,args.use_image)
        else:
            raise Exception("wrong model_attribute.encoder")
        
        text_embed_dim=args.d_text_tok
        image_embed_dim=768
        self.max_mention_image_num=model_attribute.max_mention_image_num
        self.max_entity_image_num=model_attribute.max_entity_image_num
        model_embed_dim=args.model_embed_dim
        self.device=device
        modality_num=0
        if args.use_attributes:     
            modality_num+=1
            self.attribute_matcher=Matcher2(text_embed_dim,model_embed_dim,dropout=args.dr,match_pooling_method=model_attribute.match_pooling)
        if args.use_image:
            modality_num+=1
            self.image_matcher=Matcher2(image_embed_dim,model_embed_dim,dropout=args.dr,match_pooling_method=model_attribute.match_pooling)
        if args.use_text:
            modality_num+=1
            self.text_matcher=Matcher2(text_embed_dim,model_embed_dim,dropout=args.dr,match_pooling_method=model_attribute.match_pooling)  
              
        self.modality_weight = torch.nn.Parameter(data=torch.Tensor(modality_num), requires_grad=True)
        self.modality_weight.data.uniform_(-1, 1)
        self.classifier=   Classifier( model_embed_dim,dropout=args.dr)
    
    def forward(self,  input_ids,attention_mask ,label,entity_mask_tensor  , is_train,   is_contrastive=True,mention_entity_image_mask=None,token_type_ids=None):
      
      
        if self.args.use_attributes:
            mention_text_embed,entity_text_embeds,mention_image_embed,entity_image_embeds,mention_attribute_embed,entity_attribute_embeds,mention_text_attention_mask,entity_text_attention_mask,mention_attribute_attention_mask,entity_attribute_attention_mask,batch_size,mention_image_mask,entity_image_mask=self.encoder( input_ids,attention_mask)
        else:
            mention_text_embed,entity_text_embeds, mention_text_attention_mask,entity_text_attention_mask,batch_size=self.encoder(input_ids,attention_mask)
           
        
        match_score_list=[]
        if self.args.use_text:
            entity_text_match=self.text_matcher(mention_text_embed,entity_text_embeds,entity_text_attention_mask,mention_text_attention_mask)
            entity_text_match_score=torch.sum(entity_text_match,-1)
            # match_score_list.append(entity_text_match_score)
        else:
            entity_text_match_score=None
        if self.args.use_image:
            entity_image_match=self.image_matcher(mention_image_embed,entity_image_embeds,None,None)
            entity_image_match_score=torch.sum(entity_image_match,-1)
            match_score_list.append(entity_image_match_score)
        else:
            entity_image_match_score=None
        #(batch_size*sequence_num,  sequence_len) 
        
        
        if self.args.use_attributes:
            entity_attribute_match=self.attribute_matcher(mention_attribute_embed,entity_attribute_embeds,entity_attribute_attention_mask,mention_attribute_attention_mask)
            entity_attribute_match_score=torch.sum(entity_attribute_match,-1)
            match_score_list.append(entity_attribute_match_score)
        else:
            entity_attribute_match_score=None
        logics=self.classifier(entity_text_match,batch_size)
            
            
            
        # entity_match_score=torch.stack(match_score_list,dim=-1)
        # entity_match_score=entity_match_score*self.modality_weight
        # entity_match_score=torch.sum(entity_match_score,-1)
        logics=logics.reshape(batch_size,-1)
        logics=masked_softmax(logics,entity_mask_tensor)

        return logics,label,[],entity_text_match_score,entity_image_match_score,entity_attribute_match_score
                



class AMELICross(nn.Module):
    def __init__(self, args,   model_attribute,clip_model,device):
        super().__init__()
        
        self.match=MatchERT()
        
        self.args = args
        self.model_attribute=model_attribute
        self.encoder=Encoder2(args,   model_attribute,clip_model,device,args.use_attributes,model_attribute.max_mention_image_num,
                                  model_attribute.max_entity_image_num)
        
        text_embed_dim=512
        image_embed_dim=768
        self.max_mention_image_num=model_attribute.max_mention_image_num
        self.max_entity_image_num=model_attribute.max_entity_image_num
        model_embed_dim=args.model_embed_dim
        self.device=device
        modality_num=0
        # if args.use_attributes:     
        #     modality_num+=1
        #     self.attribute_matcher=Matcher2(text_embed_dim,model_embed_dim,dropout=args.dr,match_pooling_method=model_attribute.match_pooling)
        # if args.use_image:
        #     modality_num+=1
        #     self.image_matcher=Matcher2(image_embed_dim,model_embed_dim,dropout=args.dr,match_pooling_method=model_attribute.match_pooling)
        # if args.use_text:
        #     modality_num+=1
        #     self.text_matcher=Matcher2(text_embed_dim,model_embed_dim,dropout=args.dr,match_pooling_method=model_attribute.match_pooling)  
              
        self.modality_weight = torch.nn.Parameter(data=torch.Tensor(modality_num), requires_grad=True)
        self.modality_weight.data.uniform_(-1, 1)
     
    
    def forward(self, pixel_values,input_ids,attention_mask ,label,entity_mask_tensor  , is_train,   is_contrastive=True,mention_entity_image_mask=None):
      
      
        if self.args.use_attributes:
            mention_text_embed,entity_text_embeds,mention_image_embed,entity_image_embeds,mention_attribute_embed,entity_attribute_embeds,mention_text_attention_mask,entity_text_attention_mask,mention_attribute_attention_mask,entity_attribute_attention_mask,batch_size,mention_image_mask,entity_image_mask=self.encoder(pixel_values,input_ids,attention_mask,mention_entity_image_mask)
        else:
            mention_text_embed,entity_text_embeds,mention_image_embed,entity_image_embeds,mention_text_attention_mask,entity_text_attention_mask,batch_size,mention_image_mask,entity_image_mask=self.encoder(pixel_values,input_ids,attention_mask,mention_entity_image_mask)
        reshaped_mention_embed=repeat_mention_in_entity_shape(mention_image_embed, entity_image_embeds)
        batch_size, sequence_num,sequence_len,hiddle_dim=entity_image_embeds.shape 
        reshaped_entity_list_embed=entity_image_embeds.reshape(batch_size*sequence_num,sequence_len,hiddle_dim)
        reshaped_mention_embed=reshaped_mention_embed.reshape(batch_size*sequence_num,-1,hiddle_dim)
        model_predictions=self.match(src_global=None, src_local=reshaped_mention_embed,  tgt_global=None, tgt_local=reshaped_entity_list_embed)
        mention_image_mask,entity_image_mask
        
        logics=model_predictions.reshape(batch_size,-1)
        logics= masked_softmax(logics,entity_mask_tensor)

        return logics,label,logics,logics,logics,logics
    
    
class AMELINLI(nn.Module):
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
        # if args.use_attributes:     
        #     modality_num+=1
        #     self.attribute_matcher=Matcher2(text_embed_dim,model_embed_dim,dropout=args.dr,match_pooling_method=model_attribute.match_pooling)
        # if args.use_image:
        #     modality_num+=1
        #     self.image_matcher=Matcher2(image_embed_dim,model_embed_dim,dropout=args.dr,match_pooling_method=model_attribute.match_pooling)
        # if args.use_text:
        #     modality_num+=1
        #     self.text_matcher=Matcher2(text_embed_dim,model_embed_dim,dropout=args.dr,match_pooling_method=model_attribute.match_pooling)  
              
        # self.modality_weight = torch.nn.Parameter(data=torch.Tensor(modality_num), requires_grad=True)
        # self.modality_weight.data.uniform_(-1, 1)
        self.classifier=   ClassifierWithoutReshape( (args.max_attribute_num+1)*model_embed_dim,dim_feedforward=1024,dropout=args.dr)#2048
    
    def forward(self,  input_ids,attention_mask ,label,entity_mask_tensor  , is_train,   is_contrastive=True,mention_entity_image_mask=None
                ,token_type_ids=None):
      
   
        mention_attribute_pair_embeds,batch_size=self.encoder( input_ids,attention_mask,token_type_ids)
        logics=self.classifier(mention_attribute_pair_embeds,batch_size)
         
        logics=logics.reshape(batch_size,-1)
        logics=masked_softmax(logics,entity_mask_tensor)

        return logics,label,logics,logics,logics,logics
                
                
                
    
class AMELINLImage(nn.Module):
    def __init__(self, args,   model_attribute,feature_model,image_feature_model,device):
        super().__init__()
        self.args = args
        self.model_attribute=model_attribute
        if args.use_text:
            self.text_representor=TextRepresentor( args,   model_attribute,feature_model,device)
        if args.use_image:
            self.image_representor= ImageRepresentor(args,   model_attribute,image_feature_model,device)
        
        text_embed_dim=args.d_text_tok
        image_embed_dim=768
        self.max_mention_image_num=model_attribute.max_mention_image_num
        self.max_entity_image_num=model_attribute.max_entity_image_num
        model_embed_dim=args.model_embed_dim
        self.device=device
        modality_num=0
        if args.use_text and  args.use_image:
            self.final_classifier=   ClassifierWithoutReshape( 2*model_embed_dim,dim_feedforward=2048,dropout=args.dr,output_dim=1)
        else :
            self.final_one_modality_classifier=   ClassifierWithoutReshape( model_embed_dim,dim_feedforward=2048,dropout=args.dr,output_dim=1)
            
        
        
    
    
    def forward(self,  pixel_values,input_ids,attention_mask ,label,entity_mask_tensor  , is_train,   is_contrastive=True,mention_entity_image_mask=None
                ,token_type_ids=None):
      
        batch_size=input_ids.shape[0]
        
        
        representation_list=[]
        if self.args.use_text :
            text_representation=self.text_representor(input_ids,attention_mask,token_type_ids)
            representation_list.append(text_representation)
        if  self.args.use_image:
            image_representation=self.image_representor(pixel_values,mention_entity_image_mask)
            representation_list.append(image_representation)
        if len(representation_list)==2:
            representation=torch.cat(representation_list, 1)
        else:
            representation=representation_list[0]
        if self.args.use_text and self.args.use_image:
            logics=self.final_classifier(representation,batch_size)
        else:
            logics=self.final_one_modality_classifier(representation,batch_size)
         
        logics=logics.reshape(batch_size,-1)
        logics,score_after_softmax=masked_softmax(logics,entity_mask_tensor)

        return logics,score_after_softmax
                
                
                                


class TextCrossEncoder(nn.Module):
    def __init__(self, args, device, train_attribute,model_attribute,is_freeze_bert,is_freeze_clip):
        super().__init__()
        self.args = args

        self.mode = "txt_img"
        self.device = device
        num_labels=2
        '''score combination'''
        self.score_combine = nn.Linear(2, 1, bias=False)
        self.config = AutoConfig.from_pretrained(args.pre_trained_dir)
        classifier_trained = True
        if self.config.architectures is not None:
            classifier_trained = any([arch.endswith('ForSequenceClassification') for arch in self.config.architectures])

        if num_labels is None and not classifier_trained:
            num_labels = 1

        if num_labels is not None:
            self.config.num_labels = num_labels

        self.cross_encoder = AutoModelForSequenceClassification.from_pretrained(args.pre_trained_dir, config=self.config )
 

    def forward(self, mention_input_ids,mention_attention_mask,mention_token_type_ids,mention_image,entity_input_ids,entity_attention_masks,entity_token_type_ids, entity_images, e_mask, att_m_txt_info=None, att_e_txt_info=None, is_train=False, is_contrastive=False,labels=None):
       
        batch_size,textual_sequence_num,sequence_len=entity_input_ids.shape 
        # batch_size,sequence_num,num_channels, height, width=pixel_values.shape
        reshaped_input_ids=entity_input_ids.view(batch_size*textual_sequence_num,-1)
        reshaped_attention_mask=entity_attention_masks.view(batch_size*textual_sequence_num,-1)
        # if entity_token_type_ids is not None:
        #     reshaped_token_type_ids=entity_token_type_ids.view(batch_size*textual_sequence_num,-1)
        # else:
        #     reshaped_token_type_ids=None
        # model_predictions = self.cross_encoder(reshaped_input_ids,reshaped_attention_mask,reshaped_token_type_ids , return_dict=True)
        model_predictions = self.cross_encoder(reshaped_input_ids,reshaped_attention_mask , return_dict=True)
        cross_encoder_multiclass_logits =  model_predictions.logits  
        output_logic=cross_encoder_multiclass_logits[:][-1]
        output_logic=self.masked_softmax( output_logic, e_mask)
        return output_logic, labels,torch.zeros(1),torch.zeros(1)
     
    def masked_softmax(self, tensor, mask):
        input_tensor = tensor.masked_fill((~mask), value=torch.tensor(-1e9))
        # result = nn.functional.log_softmax(input_tensor, dim=-1)
        return input_tensor
                