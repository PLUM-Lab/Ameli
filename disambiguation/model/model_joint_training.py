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
from disambiguation.model.layer.basic_layer  import  SimpleMLP2, MLPResidual,  PositionalEncoding,myPrinter,ClassifierWithoutReshape
from disambiguation.model.layer.adapt import CustomCLIP
from disambiguation.model.high_level_encoder.image_encoder import ImageRepresentor
from disambiguation.model.high_level_encoder.text_encoder import TextRepresentor
from disambiguation.model.layer.function import gen_contrastive_labels,masked_softmax
from disambiguation.model.model.image_model import AMELIImageModel
from disambiguation.model.model.text_model import AMELITextModel

class AMELIJoint(nn.Module):
    def __init__(self, args, device, train_attribute,model_attribute,text_feature_model,clip_model,model_group="all"):
        super().__init__()
        self.args = args

        self.mode = "txt_img"
        self.device = device
        self.model_group=model_group
        '''score combination'''
        self.score_combine = nn.Linear(2, 1, bias=False)
        if model_group in [ "all","image"]:
            self.image_model=AMELIImageModel( args, device,  model_attribute,clip_model,has_adapter=model_attribute.has_adapter)
        if model_group in [ "all","text"]:
            self.text_model=AMELITextModel(args,   model_attribute,text_feature_model,device)
        
    # def forward(self, m_txt_info, m_img_info, e_txt_info, e_img_info, e_mask, att_m_txt_info=None, att_e_txt_info=None, is_train=False, is_contrastive=False,labels=None):
    def forward(self,pixel_values,input_ids,attention_mask,labels,entity_mask_tensor , is_train=None, 
                        is_contrastive=None,text_weight=0.5,token_type_ids=None):
        
        if self.model_group =="all":
            text_score,_,text_loss,text_score_after_softmax=self.text_model(input_ids,attention_mask,labels,entity_mask_tensor ,token_type_ids=token_type_ids,is_train=is_train )
            image_score,_,image_loss,image_score_after_softmax=self.image_model(pixel_values, labels,entity_mask_tensor , is_train , 
                            is_contrastive)
            loss=text_loss+image_loss
            score=text_score_after_softmax*text_weight+image_score_after_softmax*(1-text_weight)
            text_score_before_softmax,image_score_before_softmax=text_score,image_score
        elif self.model_group =="text":
            text_score,_,text_loss,text_score_after_softmax=self.text_model(input_ids,attention_mask,labels,entity_mask_tensor ,token_type_ids=token_type_ids,is_train=is_train )
            loss=text_loss 
            score=text_score_after_softmax 
            image_score_after_softmax=text_score_after_softmax
            text_score_before_softmax,image_score_before_softmax=text_score,text_score
        else:
            image_score,_,image_loss,image_score_after_softmax=self.image_model(pixel_values, labels,entity_mask_tensor , is_train , 
                            is_contrastive)
            loss= image_loss 
            score= image_score_after_softmax 
            text_score_after_softmax=image_score_after_softmax
            text_score_before_softmax,image_score_before_softmax=image_score,image_score
            # labels = gen_labels(score)
        # if torch.isnan(score.std()):
        #     print('output is nan after the masked log softmax')
        return score, labels ,loss,text_score_after_softmax,image_score_after_softmax,text_score_before_softmax,image_score_before_softmax
    
 
    def masked_softmax(self, tensor, mask):
        input_tensor = tensor.masked_fill((~mask), value=torch.tensor(-1e9))
        # result = nn.functional.log_softmax(input_tensor, dim=-1)
        return input_tensor



class AMELI_MLP(nn.Module):
    def __init__(self, args, device, train_attribute,model_attribute, model_group="all"):
        super().__init__()
        self.args = args

        self.mode = "txt_img"
        self.device = device
        self.model_group=model_group
        '''score combination'''
        # self.score_combine = nn.Linear(2, 1, bias=False)
        if model_attribute.mlp_type=="mlp":
            self.mlp=SimpleMLP2(input_dim=args.d_tok,dropout=args.dr)
        else:
            self.mlp=MLPResidual(input_dim=args.d_tok,dropout=args.dr)
        
    # def forward(self, m_txt_info, m_img_info, e_txt_info, e_img_info, e_mask, att_m_txt_info=None, att_e_txt_info=None, is_train=False, is_contrastive=False,labels=None):
    def forward(self, input_ids,attention_mask,labels,entity_mask_tensor ,retrieval_nli_score_list_tensor, is_train=None, 
                        is_contrastive=None,text_weight=0.5,token_type_ids=None):
         
        logics=self.mlp(retrieval_nli_score_list_tensor)
         
        batch_size=retrieval_nli_score_list_tensor.shape[0]
         
        logics=logics.reshape(batch_size,-1)
        logics=masked_softmax(logics,entity_mask_tensor)

        return logics,labels,logics,logics,logics    
        
 
    
 
    def masked_softmax(self, tensor, mask):
        input_tensor = tensor.masked_fill((~mask), value=torch.tensor(-1e9))
        # result = nn.functional.log_softmax(input_tensor, dim=-1)
        return input_tensor
