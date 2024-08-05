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

from disambiguation.model.layer.data_specific_encoder import   ImageLevelEncoder, ImageLevelImageEncoder 
from disambiguation.model.layer.basic_layer  import   PositionalEncoding,myPrinter,ClassifierWithoutReshape
from disambiguation.model.layer.adapt import CustomCLIP
from disambiguation.model.high_level_encoder.image_encoder import ImageRepresentor
from disambiguation.model.high_level_encoder.text_encoder import TextRepresentor
from disambiguation.model.layer.function import gen_contrastive_labels,masked_softmax

class AMELIImageModel(nn.Module):
    def __init__(self, args, device,  model_attribute,clip_model,has_adapter=True):
        super().__init__()
        self.args = args
        self.device = device
        if has_adapter:
            custom_clip=CustomCLIP(clip_model)
        else:
            custom_clip=clip_model
        self.encoder=ImageLevelImageEncoder(args,   model_attribute,custom_clip,device,args.use_attributes,args.use_image)
        self.criterion = nn.CrossEntropyLoss( )
    
    def forward(self,pixel_values, labels,entity_mask , is_train=None, 
                        is_contrastive=None  ):
        
        mention_image_embed,entity_image_embeds,batch_size=self.encoder(pixel_values)
        # bsc * n_cand 
        mention_image_embed = mention_image_embed.view(batch_size, -1, 1)  # bsz * d_hid * 1
        if is_train and is_contrastive:
            # batch_size * batch_size * n_cand * d_hid    batch_size * d_hid * 1
            simi = torch.einsum(
                'ijkl,ild->ijk', [entity_image_embeds.expand(batch_size, -1, -1, -1), mention_image_embed])
            simi = simi.view(batch_size, -1)  # bsz * (bsz * n_cand)
            score = self.masked_softmax(simi, entity_mask.expand(
                batch_size, -1, -1).view(batch_size, -1))
            labels = gen_contrastive_labels(score,labels)
        else:
            simi = torch.bmm(entity_image_embeds, mention_image_embed).view(
                batch_size, -1)  # bsz * n_cand
            score,score_after_softmax = self.masked_softmax(simi, entity_mask)
        if is_train:
            loss=self.compute_loss(score,labels)
        else:
            loss=score
        return score, labels ,loss,score_after_softmax
    
    def compute_loss(self,score,labels):
        loss=self.criterion(score, labels)
        return loss 
    def masked_softmax(self, tensor, mask):
        input_tensor = tensor.masked_fill((~mask), value=torch.tensor(-1e9))
        result = nn.functional.softmax(input_tensor, dim=-1)#log_
        return input_tensor,result

 