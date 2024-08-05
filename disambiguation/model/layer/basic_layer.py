

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
from disambiguation.model.layer.function import myPrinter,repeat_mention_in_entity_shape,repeat_mention_mask_in_entity_shape,gen_attention_mask_for_multihead_attn


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
def freeze(model):
    model.requires_grad_(False)
      
class FeatureExtractor(nn.Module):
    def __init__(self ,pre_trained_image_model_dir="resnet152",is_freeze_clip=False):
        super(FeatureExtractor, self).__init__()
        # res_net= ResNetMNIST.load_from_checkpoint(os.path.join(res_net_dir,"resnet18_mnist.pt"), map_location="cuda")
         
        # res_net=torch.hub.load('pytorch/vision:v0.10.0', pre_trained_image_model_dir, pretrained=True)
        res_net=cv_models.resnet50(pretrained=True) #  also need 
        self.feature_extractor  = nn.Sequential(*list(res_net.children())[:-3])
        # res_net.eval()
        # res_net.fc = nn.Linear(2048, 768)
        
        if is_freeze_clip:
            freeze(self.feature_extractor )
       
        # self.feature_extractor = torch.nn.Sequential(*list(res_net.children())[:-1])
        # print("")
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        resnet_feature = self.feature_extractor(x)
        # x = self.feature_extractor.bn1(x)
        # x = self.feature_extractor.relu(x)
        # x = self.feature_extractor.maxpool(x)
        # x = self.feature_extractor.layer1(x)
        # x = self.feature_extractor.layer2(x)
        # resnet_feature= self.feature_extractor.layer3(x)  #-> b,1024,14,14
        # resnet_feature = self.feature_extractor.layer4(x) 
        grid_feature=self.maxpool(resnet_feature) #-> b,1024,7,7
        grid_feature = torch.flatten(grid_feature, 2)
        grid_feature = grid_feature.permute(0, 2, 1)
        # feature = torch.squeeze(feature, -1)
        # feature = torch.unsqueeze(feature, 1)
        return grid_feature
    def get_image_features(self,pixel_values):
        return self.forward(pixel_values)

 

class FeatureExtractorResnet152(nn.Module):
    def __init__(self):
        super(FeatureExtractorResnet152, self).__init__()
        # res_net= ResNetMNIST.load_from_checkpoint(os.path.join(res_net_dir,"resnet18_mnist.pt"), map_location="cuda")
        # res_net = torch.hub.load(
        #     'pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        res_net=torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
        # res_net.eval()
        # res_net.fc = nn.Linear(2048, 768)
        self.feature_extractor = res_net
        # self.feature_extractor = torch.nn.Sequential(*list(res_net.children())[:-1])
        # print("")

    def forward(self, x):
        feature_in_4d = self.feature_extractor(x)
        feature = torch.squeeze(feature_in_4d, -1)
        feature = torch.squeeze(feature, -1)
        feature = torch.unsqueeze(feature, 1)
        return feature
    def get_image_features(self,pixel_values):
        return self.forward(pixel_values)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
    

#get CLS
class pooling1(nn.Module):
    def __init__(self ):
        super().__init__()
    def forward(self,compare_entity_to_mention,reshaped_mention_embed,reshaped_mention_attention_mask ):
        return compare_entity_to_mention[:,0,:] 
    
#vector -> 1 for match
class pooling2(nn.Module):
    def __init__(self ,model_embed_dim,  dim_feedforward=2048, dropout=0.1,  
                 layer_norm_eps=1e-5):
        super().__init__()
        self.linear1 = nn.Linear(model_embed_dim, dim_feedforward )
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, 1 )

        self.norm1 = nn.LayerNorm(model_embed_dim, eps=layer_norm_eps )
        # self.norm2 = nn.LayerNorm(model_embed_dim, eps=layer_norm_eps )
        # self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation=F.relu
        
    def forward(self,compare_entity_to_mention,reshaped_mention_embed,reshaped_mention_attention_mask ):
        #batch_size,sequence_num,  hid_dim
        # compare_entity_to_mention=self.dropout1(compare_entity_to_mention)
        compare_entity_to_mention = self.norm1(compare_entity_to_mention)
        compare_entity_to_mention = self.linear2(self.dropout(self.activation(self.linear1(compare_entity_to_mention))))
        # compare_entity_to_mention=self.dropout2(compare_entity_to_mention)
        # compare_entity_to_mention = self.norm2(compare_entity_to_mention)
        return compare_entity_to_mention.squeeze(-1)
        
#mention-based weighted sum for compare_entity_to_mention
class pooling3(nn.Module):
    def __init__(self ,embed_dim,model_embed_dim,  dim_feedforward=2048, dropout=0.1,  
                 layer_norm_eps=1e-5):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention( embed_dim, 8,kdim= embed_dim,  vdim= model_embed_dim, batch_first=True)
        
    def forward(self,compare_entity_to_mention,mention_embed,reshaped_mention_attention_mask):
        #batch_size,sequence_num,  hid_dim
        if reshaped_mention_attention_mask is not None:
            compare_entity_to_mention=self.multihead_attn(mention_embed,mention_embed,compare_entity_to_mention,key_padding_mask=reshaped_mention_attention_mask)[0]
        else:
            compare_entity_to_mention=self.multihead_attn(mention_embed,mention_embed,compare_entity_to_mention )[0]
        return compare_entity_to_mention[:,0,:] 
    
    
#self attention
class pooling4(nn.Module):
    def __init__(self ,embed_dim,model_embed_dim,  dim_feedforward=2048, dropout=0.1,  
                 layer_norm_eps=1e-5):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_embed_dim, nhead=2,
                                                 dropout=0.3,
                                                   activation='gelu')
        encoder_norm = nn.LayerNorm(model_embed_dim)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, 4, encoder_norm)
        # self.multihead_attn = nn.MultiheadAttention( embed_dim, 8,kdim= embed_dim,  vdim= model_embed_dim, batch_first=True)
        
    def forward(self,compare_entity_to_mention,mention_embed,reshaped_mention_attention_mask):
        #batch_size,sequence_num,  hid_dim
        # if reshaped_mention_attention_mask is not None:
        #     compare_entity_to_mention=self.multihead_attn(mention_embed,mention_embed,compare_entity_to_mention,key_padding_mask=reshaped_mention_attention_mask)[0]
        # else:
        compare_entity_to_mention=self.encoder(compare_entity_to_mention.permute(  1, 0, 2) )[0]
      
         

        # compare_entity_to_mention = compare_entity_to_mention.permute(1, 0, 2)
        return compare_entity_to_mention  
    
class Matcher(nn.Module):
    def __init__(self,embed_dim ):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention( embed_dim, 1,kdim= embed_dim,  vdim= embed_dim, batch_first=True)


    def forward(self,mention_embed, entity_list_embed):
        reshaped_mention_embed=repeat_mention_in_entity_shape(mention_embed, entity_list_embed)
        # attention_mask=self.gen_attention_mask( evidence_mask)
        batch_size, sequence_num,sequence_len,hiddle_dim=entity_list_embed.shape 
        entity_list_embed=entity_list_embed.reshape(batch_size*sequence_num,sequence_len,hiddle_dim)
        reshaped_mention_embed=reshaped_mention_embed.reshape(batch_size*sequence_num,sequence_len,hiddle_dim)
        mention_aware_entity_list_embed=self.multihead_attn(entity_list_embed, reshaped_mention_embed, reshaped_mention_embed )#, key_padding_mask =attention_mask
        return mention_aware_entity_list_embed
     
        
        
    def gen_attention_mask(self,evidence_bert_mask):
        attention_mask = evidence_bert_mask.to(dtype=torch.bool )
        return ~attention_mask    
    
 
         
class Matcher2(nn.Module):
    def __init__(self,embed_dim,model_embed_dim,  dim_feedforward=2048, dropout=0.1,  
                 layer_norm_eps=1e-5,match_pooling_method="pooling1" ) :
         
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention( embed_dim, 2,kdim= embed_dim,  vdim= embed_dim, batch_first=True)
        self.linear1 = nn.Linear(embed_dim*2, dim_feedforward )
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, model_embed_dim )

        self.norm1 = nn.LayerNorm(embed_dim*2, eps=layer_norm_eps )
        self.norm2 = nn.LayerNorm(model_embed_dim, eps=layer_norm_eps )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation=F.relu
        self.match_pooling_method=match_pooling_method
        if match_pooling_method=="pooling1":
            self.match_pooling=pooling1()
        elif match_pooling_method=="pooling2":
            self.match_pooling=pooling2(model_embed_dim)
        elif match_pooling_method=="pooling3":
            self.match_pooling=pooling3(embed_dim,model_embed_dim)
        elif match_pooling_method=="pooling4":
            self.match_pooling=pooling4(embed_dim,model_embed_dim)
        
    def match(self, entity_aware_mention_embed ,reshaped_mention_embed,reshaped_mention_attention_mask ):
        compare_entity_to_mention = torch.cat([reshaped_mention_embed - entity_aware_mention_embed, reshaped_mention_embed * entity_aware_mention_embed], -1)
        # use separate linear functions
        compare_entity_to_mention=self.dropout1(compare_entity_to_mention)
        compare_entity_to_mention = self.norm1(compare_entity_to_mention)
        compare_entity_to_mention = self.linear2(self.dropout(self.activation(self.linear1(compare_entity_to_mention))))
        compare_entity_to_mention=self.dropout2(compare_entity_to_mention)
        compare_entity_to_mention = self.norm2(compare_entity_to_mention)
        if reshaped_mention_attention_mask is not None:
            reshaped_mention_attention_mask=gen_attention_mask_for_multihead_attn( reshaped_mention_attention_mask)
            
        entity_match=self.match_pooling(compare_entity_to_mention,reshaped_mention_embed,reshaped_mention_attention_mask ) #pooling
        return entity_match

    def forward(self,mention_embed, entity_list_embed,entity_attention_mask,mention_attention_mask):
        #(batch_size, sequence_num,sequence_length, hid_dim) 
        reshaped_mention_embed=repeat_mention_in_entity_shape(mention_embed, entity_list_embed)
        batch_size, sequence_num,sequence_len,hiddle_dim=entity_list_embed.shape 
        if mention_attention_mask is not None:
            reshaped_mention_attention_mask=repeat_mention_mask_in_entity_shape(mention_attention_mask, entity_attention_mask)
            reshaped_mention_attention_mask=reshaped_mention_attention_mask.reshape(batch_size*sequence_num,-1)
        else:
            reshaped_mention_attention_mask=None
        
        entity_list_embed=entity_list_embed.reshape(batch_size*sequence_num,sequence_len,hiddle_dim)
        reshaped_mention_embed=reshaped_mention_embed.reshape(batch_size*sequence_num,-1,hiddle_dim)
        if entity_attention_mask is not None:
            entity_attention_mask=entity_attention_mask.reshape(batch_size*sequence_num,sequence_len)
            entity_attention_mask_for_attention=gen_attention_mask_for_multihead_attn( entity_attention_mask)
            entity_aware_mention_embed=self.multihead_attn( reshaped_mention_embed, entity_list_embed,entity_list_embed , key_padding_mask =entity_attention_mask_for_attention)[0]#
        else:
            entity_aware_mention_embed=self.multihead_attn( reshaped_mention_embed, entity_list_embed,entity_list_embed )[0]#
        entity_match=self.match( entity_aware_mention_embed ,reshaped_mention_embed,reshaped_mention_attention_mask )
        if self.match_pooling_method=="pooling2":
            if mention_attention_mask is not None:
                entity_match= entity_match * reshaped_mention_attention_mask   
        #(batch_size*sequence_num,  hid_dim) 
        return entity_match     
        


# class Matcher3(nn.Module):
#     def __init__(self,embed_dim,model_embed_dim,  dim_feedforward=2048, dropout=0.1,  
#                  layer_norm_eps=1e-5,match_pooling_method="pooling1" ) :
         
#         super().__init__()
#         self.multihead_attn = nn.MultiheadAttention( embed_dim, 2,kdim= embed_dim,  vdim= embed_dim, batch_first=True)
#         self.linear1 = nn.Linear(embed_dim*2, dim_feedforward )
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, model_embed_dim )

#         self.norm1 = nn.LayerNorm(embed_dim*2, eps=layer_norm_eps )
#         self.norm2 = nn.LayerNorm(model_embed_dim, eps=layer_norm_eps )
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.activation=F.relu
#         if match_pooling_method=="pooling1":
#             self.match_pooling=pooling1()
#         elif match_pooling_method=="pooling2":
#             self.match_pooling=pooling2(model_embed_dim)
#         elif match_pooling_method=="pooling3":
#             self.match_pooling=pooling3(embed_dim,model_embed_dim)
        
#     def match(self, entity_aware_mention_embed ,reshaped_mention_embed,reshaped_mention_attention_mask ):
#         compare_entity_to_mention = torch.cat([reshaped_mention_embed - entity_aware_mention_embed, reshaped_mention_embed * entity_aware_mention_embed], -1)
#         # use separate linear functions
#         compare_entity_to_mention=self.dropout1(compare_entity_to_mention)
#         compare_entity_to_mention = self.norm1(compare_entity_to_mention)
#         compare_entity_to_mention = self.linear2(self.dropout(self.activation(self.linear1(compare_entity_to_mention))))
#         compare_entity_to_mention=self.dropout2(compare_entity_to_mention)
#         compare_entity_to_mention = self.norm2(compare_entity_to_mention)
        
#         entity_match=self.match_pooling(compare_entity_to_mention,reshaped_mention_embed,reshaped_mention_attention_mask ) #pooling
#         return entity_match

#     def forward(self,mention_embed, entity_list_embed,entity_attention_mask,mention_attention_mask):
#         #(batch_size, sequence_num,sequence_length, hid_dim) 
#         reshaped_mention_embed=repeat_mention_in_entity_shape(mention_embed, entity_list_embed)
#         batch_size, sequence_num,sequence_len,hiddle_dim=entity_list_embed.shape 
#         if mention_attention_mask is not None:
#             reshaped_mention_attention_mask=repeat_mention_mask_in_entity_shape(mention_attention_mask, entity_attention_mask)
#             reshaped_mention_attention_mask=reshaped_mention_attention_mask.reshape(batch_size*sequence_num,-1)
        
#         entity_list_embed=entity_list_embed.reshape(batch_size*sequence_num,sequence_len,hiddle_dim)
#         reshaped_mention_embed=reshaped_mention_embed.reshape(batch_size*sequence_num,-1,hiddle_dim)
#         if entity_attention_mask is not None:
#             entity_attention_mask=entity_attention_mask.reshape(batch_size*sequence_num,sequence_len)
#             entity_attention_mask_for_attention=self.gen_attention_mask_for_multihead_attn( entity_attention_mask)
#             entity_aware_mention_embed=self.multihead_attn( reshaped_mention_embed, entity_list_embed,entity_list_embed , key_padding_mask =entity_attention_mask_for_attention)[0]#
#         else:
#             entity_aware_mention_embed=self.multihead_attn( reshaped_mention_embed, entity_list_embed,entity_list_embed )[0]#
#         entity_match=self.match( entity_aware_mention_embed ,reshaped_mention_embed,reshaped_mention_attention_mask )
 
#         #(batch_size*sequence_num,  hid_dim) 
#         return entity_match
         
        
#     def gen_attention_mask_for_multihead_attn(self,evidence_bert_mask):
#         attention_mask = evidence_bert_mask.to(dtype=torch.bool )
#         return ~attention_mask        
 
class IntraModalityFuser(Matcher2):
    def forward(self,mention_embed, entity_list_embed,entity_attention_mask,mention_attention_mask):
        batch_size, sequence_num,sequence_len,hiddle_dim=entity_list_embed.shape 
        if mention_attention_mask is not None:
            reshaped_mention_attention_mask=repeat_mention_mask_in_entity_shape(mention_attention_mask, entity_attention_mask)
            reshaped_mention_attention_mask=reshaped_mention_attention_mask.reshape(batch_size*sequence_num,-1)
        else:
            reshaped_mention_attention_mask=None
        #(batch_size, sequence_num,sequence_length, hid_dim) 
        reshaped_mention_embed=repeat_mention_in_entity_shape(mention_embed, entity_list_embed)
        
        
        entity_list_embed=entity_list_embed.reshape(batch_size*sequence_num,sequence_len,hiddle_dim)
        reshaped_mention_embed=reshaped_mention_embed.reshape(batch_size*sequence_num,sequence_len,hiddle_dim)
        if entity_attention_mask is not None:
                            
            entity_attention_mask=entity_attention_mask.reshape(batch_size*sequence_num,sequence_len)
            entity_attention_mask_for_attention= gen_attention_mask_for_multihead_attn( entity_attention_mask)
            #TODO add mask to take care of the mention attention mask
            entity_aware_mention_embed=self.multihead_attn( reshaped_mention_embed, entity_list_embed,entity_list_embed , key_padding_mask =entity_attention_mask_for_attention)[0]#
        else:
            entity_aware_mention_embed=self.multihead_attn( reshaped_mention_embed, entity_list_embed,entity_list_embed )[0]#
        
        
        #(batch_size*sequence_num,  hid_dim) 
        return entity_aware_mention_embed,reshaped_mention_attention_mask
          
    
class Fuser2(nn.Module):
    def __init__(self, input_dim,  dim_feedforward=2048, dropout=0.1, 
                 layer_norm_eps=1e-5  ) :
         
        super().__init__()
        self.linear1 = nn.Linear(input_dim, dim_feedforward )
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, 1 )

        self.norm1 = nn.LayerNorm(input_dim , eps=layer_norm_eps )
        # self.norm2 = nn.LayerNorm(int(input_dim/3), eps=layer_norm_eps )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation=F.relu
        
    def forward(self,entity_text_match, entity_image_match,entity_attribute_match,batch_size):
        #batch*sequence_num, hid_dim
        if entity_attribute_match is not None:
            match_embed=torch.cat([entity_image_match,entity_text_match, entity_attribute_match],dim=-1)
        else:
            match_embed=torch.cat([entity_image_match,entity_text_match ],dim=-1)
        # multimodal_match_embed=self.linear1(match_embed) #(batch*entity_num,3*hid_dim) -> (batch*entity_num,hid_dim)
        
        multimodal_match_embed=self.dropout1(match_embed)
        multimodal_match_embed = self.norm1(multimodal_match_embed)
        logics = self.linear2(self.dropout(self.activation(self.linear1(multimodal_match_embed))))
        #(batch*entity_num,hid_dim) -> (batch*entity_num,1)
        entity_num=int(logics.shape[0]/batch_size)
        logics=logics.reshape(batch_size,entity_num)
        
        return logics 
  
class Classifier(nn.Module):

    def __init__(self, input_dim,  dim_feedforward=2048, dropout=0.1, 
                 layer_norm_eps=1e-5  ) :
         
        super().__init__()
        self.linear1 = nn.Linear(input_dim, dim_feedforward )
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, 1 )

        self.norm1 = nn.LayerNorm(input_dim , eps=layer_norm_eps )
        # self.norm2 = nn.LayerNorm(int(input_dim/3), eps=layer_norm_eps )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation=F.relu
        
        
    def forward(self,match_embed,batch_size):
        #batch*sequence_num, hid_dim
        
        # multimodal_match_embed=self.linear1(match_embed) #(batch*entity_num,3*hid_dim) -> (batch*entity_num,hid_dim)
        
        multimodal_match_embed=self.dropout1(match_embed)
        multimodal_match_embed = self.norm1(multimodal_match_embed)
        logics = self.linear2(self.dropout(self.activation(self.linear1(multimodal_match_embed))))
        #(batch*entity_num,hid_dim) -> (batch*entity_num,1)
        entity_num=int(logics.shape[0]/batch_size)
        logics=logics.reshape(batch_size,entity_num)
        
        return logics   
class ClassifierWithoutReshape(nn.Module):

    def __init__(self, input_dim,  dim_feedforward=2048, dropout=0.1, 
                 layer_norm_eps=1e-5  ,output_dim=1) :
         
        super().__init__()
        self.linear1 = nn.Linear(input_dim, dim_feedforward )
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward,output_dim)

        self.norm1 = nn.LayerNorm(input_dim , eps=layer_norm_eps )
        # self.norm2 = nn.LayerNorm(int(input_dim/3), eps=layer_norm_eps )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation=F.relu
        
        
    def forward(self,match_embed,batch_size):
        #batch*sequence_num, hid_dim
        
        # multimodal_match_embed=self.linear1(match_embed) #(batch*entity_num,3*hid_dim) -> (batch*entity_num,hid_dim)
        
        multimodal_match_embed=self.dropout1(match_embed)
        multimodal_match_embed = self.norm1(multimodal_match_embed)
        logics = self.linear2(self.dropout(self.activation(self.linear1(multimodal_match_embed))))
        #(batch*entity_num,hid_dim) -> (batch*entity_num,1)
        
        
        
        return logics           
    
    
class SimpleMLP2(nn.Module):
    def __init__(self, input_dim, output_dim=1 ,dropout=0.1, 
                 layer_norm_eps=1e-5 ):
        super().__init__()

        self.input_fc = nn.Linear(input_dim, 250)
        self.hidden_fc = nn.Linear(250, 100)
        self.output_fc = nn.Linear(100, output_dim)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(input_dim , eps=layer_norm_eps )
        self.norm2 = nn.LayerNorm(int(input_dim/3), eps=layer_norm_eps )
        
        
    def forward(self, x):
 
      

        h_1 = F.relu(self.input_fc(x))
        h_1=self.dropout1(h_1)

        h_2 = F.relu(self.hidden_fc(h_1))
        h_2=self.dropout2(h_2)
        
        y_pred = self.output_fc(h_2)

        return y_pred  
    
    

class MLPResidual(nn.Module):

    def __init__(self, input_dim,  dim_feedforward=2048, dropout=0.1, 
                 layer_norm_eps=1e-5  ,output_dim=1) :
         
        super().__init__()
        self.linear1 = nn.Linear(input_dim, dim_feedforward )
        self.linear2 = nn.Linear(dim_feedforward,input_dim)
        self.linear3 = nn.Linear(input_dim,128)
        self.linear4 = nn.Linear(128,output_dim)
        
        self.norm1 = nn.LayerNorm(input_dim , eps=layer_norm_eps )
        self.norm2 = nn.LayerNorm(int(input_dim), eps=layer_norm_eps )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation=F.relu
        
        
    def forward(self,src):
        #batch*sequence_num, hid_dim
        
        # multimodal_match_embed=self.linear1(match_embed) #(batch*entity_num,3*hid_dim) -> (batch*entity_num,hid_dim)
 

        src2 = self.norm1(src)
        tgt2 =  self.linear2(self.dropout1(self.activation(self.linear1(src2))))
        tgt2 = src + self.dropout2(tgt2)
        tgt2 = self.norm2(tgt2)
        
        tgt4 = self.linear4(self.dropout3(self.activation(self.linear3(tgt2))))
     
        return tgt4           
        