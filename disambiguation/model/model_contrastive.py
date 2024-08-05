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

from disambiguation.model.layer.data_specific_encoder import   ImageLevelEncoder, ImageLevelImageAndTextEncoder, ImageLevelSeparateImageAndTextEncoder, ImageLevelSeparateImageAndTextEncoderText   
from disambiguation.model.layer.basic_layer  import   SimpleMLP2, FeatureExtractor, PositionalEncoding,myPrinter
from disambiguation.model.layer.adapt import CustomCLIP, CustomMultimodalCLIP
from disambiguation.model.layer.function import gen_contrastive_labels
from disambiguation.model.layer.mlp import MLP
from disambiguation.utils.disambiguation_util import freeze_part_bert
 
from retrieval.model.model import MultiMediaSentenceTransformer
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


# import oss2 as oss


        
def init_weights(module):
    m = module
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
    elif isinstance(m, nn.Embedding):
        nn.init.xavier_normal_(m.weight.data)
    elif isinstance(m, nn.LayerNorm):
        m.bias.data.zero_()
        m.weight.data.fill_(1.0)
    elif isinstance(m, nn.BatchNorm1d):
        m.bias.data.zero_()
        m.weight.data.fill_(1.0)



# More complex fusion
from transformers import LxmertTokenizer, LxmertModel
# class ThreeStreamFusion(nn.Module):
#     def __init__(self, config, model_attribute):
#         super(ThreeStreamFusion, self).__init__()
#         # cross attention between attribute and text vectors
#         self.cfg = config
#         self.cross_attention = nn.MultiheadAttention(
#             config.d_tok, config.n_head, dropout=config.dr)
#         self.lxmert = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")

#     def forward(self, lang_feats, lang_pad_mask, visn_feats, att_lang_feats=None, att_lang_pad_mask=None, att_token_type_ids = None, token_type_ids=None):
        
#         lang_feats = lang_feats.permute(1, 0, 2)
#         visn_feats = visn_feats.permute(1, 0, 2)
        
#         #attend over the langauge features guided by the attributes
#         if self.cfg.use_attributes:
#             att_lang_feats = att_lang_feats.permute(1, 0, 2)
#             lang_att_output ,attn_output_weights = self.cross_attention(query=lang_feats, key=att_lang_feats, value=att_lang_feats,
#                                                     key_padding_mask=att_lang_pad_mask.type(torch.bool))
#         else:
#             att_lang_feats = None


        # lang_att_output = lang_att_output.permute(1, 0, 2)
        # visn_feats = visn_feats.permute(1, 0, 2)
        # att_lang_feats = att_lang_feats.permute(1, 0, 2)
        
        # visn_pos = torch.ones(visn_feats.shape[0], visn_feats.shape[1], 4)
        # # visual_attention = visn_attention_mask,
        # outputs = self.lxmert(visual_feats = visn_feats, visual_pos = visn_pos, input_ids = lang_att_output, attention_mask = lang_pad_mask)
        # lang_feats, visn_feats, output = outputs["language_output"], outputs["vision_output"], outputs["pooled_output"]
        
        # return lang_feats, visn_feats, output, att_lang_feats
        

# End more complex fusion

# class SelfAttenFFNLayers(nn.Module):
#     def __init__(self, d_model, dim_feedforward, dropout, activation='gelu'):
#         super().__init__()
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)
#         self.norm = nn.LayerNorm(d_model)
#         self.activation = self._get_activation_fn(activation)

#     def forward(self, attention_output):
#         src = attention_output
#         attention_output = self.activation(self.linear1(attention_output))
#         attention_output = self.dropout(self.linear2(attention_output))
#         src = self.norm(src+attention_output)
#         return src

#     def _get_activation_fn(self, activation):
#         if activation == "relu":
#             return F.relu
#         elif activation == "gelu":
#             return F.gelu


class SimpleMLP(nn.Module):
    def __init__(self, in_dim,out_dim,  dim_feedforward=2048, dropout=0.1,  
                 layer_norm_eps=1e-5):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, dim_feedforward )
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, out_dim )

        self.norm1 = nn.LayerNorm(in_dim, eps=layer_norm_eps )
        self.dropout2 = nn.Dropout(dropout)
        self.activation=F.relu
        
         

    def forward(self, compare_entity_to_mention):
        compare_entity_to_mention = self.norm1(compare_entity_to_mention)
        src = self.linear2(self.dropout(self.activation(self.linear1(compare_entity_to_mention))))
        
        return src

    def _get_activation_fn(self, activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu


# class LXRTXLayer(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.cross_attention = nn.MultiheadAttention(
#             config.d_tok, config.n_head, dropout=config.dr)
#         # self.attention_output = AttenOutput(config)
#         self.lang_self_att = nn.MultiheadAttention(
#             config.d_tok, config.n_head, dropout=config.dr)
#         self.visn_self_att = nn.MultiheadAttention(
#             config.d_tok, config.n_head, dropout=config.dr)
#         self.lang_ffn = SelfAttenFFNLayers(
#             d_model=config.d_tok, dim_feedforward=config.d_hid, dropout=config.dr)
#         self.visn_ffn = SelfAttenFFNLayers(
#             d_model=config.d_tok, dim_feedforward=config.d_hid, dropout=config.dr)

#     def cross_att(self, lang_input, lang_attention_mask, visn_input, visn_attention_mask):
#         lang_input = lang_input.permute(1, 0, 2)
#         visn_input = visn_input.permute(1, 0, 2)
#         lang_att_output, _ = self.cross_attention(query=lang_input, key=visn_input, value=visn_input,
#                                                   key_padding_mask=visn_attention_mask)
#         visn_att_output, _ = self.cross_attention(query=visn_input, key=lang_input, value=lang_input,
#                                                   key_padding_mask=lang_attention_mask)
#         lang_att_output = lang_att_output.permute(1, 0, 2)
#         visn_att_output = visn_att_output.permute(1, 0, 2)
#         return lang_att_output, visn_att_output

#     def self_att(self, lang_input, lang_attention_mask, visn_input, visn_attention_mask):
#         lang_input = lang_input.permute(1, 0, 2)
#         visn_input = visn_input.permute(1, 0, 2)
#         lang_att_output, _ = self.lang_self_att(query=lang_input, key=lang_input, value=lang_input,
#                                                 key_padding_mask=lang_attention_mask)
#         visn_att_output, _ = self.visn_self_att(query=visn_input, key=visn_input, value=visn_input,
#                                                 key_padding_mask=visn_attention_mask)
#         lang_att_output = lang_att_output.permute(1, 0, 2)
#         visn_att_output = visn_att_output.permute(1, 0, 2)
#         return lang_att_output, visn_att_output

#     def output_fc(self, lang_input, visn_input):
#         lang_output = self.lang_ffn(lang_input)
#         visn_output = self.visn_ffn(visn_input)
#         return lang_output, visn_output

#     def forward(self, lang_feats, lang_attention_mask, visn_feats, visn_attention_mask):
#         lang_att_output, visn_att_output = lang_feats, visn_feats
#         lang_att_output, visn_att_output = self.cross_att(
#             lang_att_output, lang_attention_mask, visn_att_output, visn_attention_mask)
#         lang_att_output, visn_att_output = self.self_att(
#             lang_att_output, lang_attention_mask, visn_att_output, visn_attention_mask)
#         lang_output, visn_output = self.output_fc(
#             lang_att_output, visn_att_output)
#         return lang_output, visn_output


# class SelfAttenFuseLayer(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         # self-attention layer
#         encoder_layer = nn.TransformerEncoderLayer(d_model=config.d_tok, nhead=config.n_head,
#                                                    dim_feedforward=config.d_hid, dropout=config.dr,
#                                                    activation='gelu')
#         encoder_norm = nn.LayerNorm(config.d_tok)
#         self.encoder = nn.TransformerEncoder(
#             encoder_layer, config.x_layers, encoder_norm)

#     def forward(self, lang_feats, lang_attention_mask, visn_feats, visn_attention_mask):
#         (batch_size, seq_len, d_tok) = lang_feats.size()
#         input = torch.cat((lang_feats, visn_feats), dim=1)
#         pad_mask = torch.cat((lang_attention_mask, visn_attention_mask), dim=1)
#         output = self.encoder(input.permute(1, 0, 2),
#                               src_key_padding_mask=pad_mask)
#         # if torch.isnan(output.std()):
#         #     print('fused features are nan after self-attention layer!')
#         output = output.permute(1, 0, 2)
#         lang_feats = output[:, :seq_len - 1, :]
#         visn_feats = output[:, seq_len - 1:, :]
#         return lang_feats, visn_feats


class MMEncoder(nn.Module):
    def __init__(self, config, embedding_matrix=None, mode='lxr', use_one_stream=True,train_attribute=None,model_attribute=None):
        super().__init__()
        myPrinter(
            config, '\t############Mention Multimodal Encoder Model#####################')
        
        self.use_attributes = config.use_attributes
        
        self.mode = mode
        self.device = config.device
        self.use_one_stream = use_one_stream
        self.mention_text_embeddings = nn.Embedding(
            embedding_matrix.shape[0], embedding_matrix.shape[1])
        # fuse layer
        if config.fuse=="concatenate":
            self.encoder =  OneStreamSelfAttenMMEncoder(config,model_attribute)
        self.mlp=SimpleMLP(2*config.d_tok,config.d_tok)
        # else:
        #     self.encoder = ThreeStreamFusion(config,model_attribute)
        # else:
        #   if config.mode == 'x-lxrt' or config.mode=='x-self':
        #     self.encoder = DualStreamSelfAttenMMEncoder(config)
        #   elif con fig.mode == 'x-adapt':
        #     self.encoder = AdaptiveFuseMMEncoder(config)
        #   else:
        #     print('WRONG TYPE OF MODEL: {}'.format(config.mode))
        #     exit(0)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, visual_feats=None, att_input_ids=None, att_token_type_ids=None, att_attention_mask=None):
        # Word Embeddings
        lang_embeddings = self.mention_text_embeddings(input_ids)

        if self.use_attributes:
            att_lang_embeddings = self.mention_text_embeddings(att_input_ids)

            if torch.isnan(att_lang_embeddings.std()):
                print('attribute language features are nan after embedding layer!')
                raise exception
        
        else:
            att_lang_embeddings = None


        if torch.isnan(lang_embeddings.std()):
            print('language features are nan after embedding layer!')
            raise exception
        # Run LXRT backbone
        lang_feats, visn_feats, output, att_lang_feats = self.encoder(lang_feats=lang_embeddings, token_type_ids=token_type_ids,
                                                      lang_pad_mask=attention_mask,
                                                      visn_feats=visual_feats, att_lang_feats=att_lang_embeddings, att_lang_pad_mask=att_attention_mask, att_token_type_ids=att_token_type_ids)
        # pool
        first_token_hidden  = output[:, 0, :]
        last_token_hidden=output[:, -1, :]
        token_hidden=torch.cat((first_token_hidden, last_token_hidden), -1)
        final_output=self.mlp(token_hidden)
        final_output = final_output / final_output.norm(dim=-1, keepdim=True)
        return final_output


class EntMMEncoder(nn.Module): 
    def __init__(self, config, embedding_matrix=None, mode='lxr', use_one_stream=True,train_attribute=None,model_attribute=None):
        super().__init__()
        myPrinter(
            config, '\t############Entity Multimodal Encoder Model#####################')
        
        self.use_attributes = config.use_attributes
        
        self.mode = mode
        self.device = config.device
        self.use_one_stream = use_one_stream
        # fuse layer
        self.entity_text_embeddings = nn.Embedding(
            embedding_matrix.shape[0], embedding_matrix.shape[1])
        if config.fuse=="concatenate":
            self.encoder =  OneStreamSelfAttenMMEncoder(config,model_attribute)
        # else:
        #     self.encoder = ThreeStreamFusion(config,model_attribute)
        self.mlp=SimpleMLP(2*config.d_tok,config.d_tok)
        # else:
        #   if config.mode == 'x-lxrt' or config.mode=='x-self':
        #     self.encoder = DualStreamSelfAttenMMEncoder(config)
        #   elif config.mode == 'x-adapt':
        #     self.encoder = AdaptiveFuseMMEncoder(config)
        #   else:
        #     print('WRONG TYPE OF MODEL: {}'.format(config.mode))
        #     exit(0)

    def forward(self, input_ids, token_type_ids=None,
                attention_mask=None, visual_feats=None, att_input_ids=None, att_token_type_ids=None, att_attention_mask=None):
        
        lang_embeddings = self.entity_text_embeddings(input_ids)

        if self.use_attributes:
            att_lang_embeddings = self.entity_text_embeddings(att_input_ids)
        else:
            att_lang_embeddings = None

        # Run LXRT backbone
        lang_feats, visn_feats, output, att_lang_feats = self.encoder(lang_feats=lang_embeddings, token_type_ids=token_type_ids,
                                                      lang_pad_mask=attention_mask,
                                                      visn_feats=visual_feats, att_lang_feats=att_lang_embeddings, att_lang_pad_mask=att_attention_mask, att_token_type_ids=att_token_type_ids)
        # pool
        first_token_hidden  = output[:, 0, :]
        last_token_hidden=output[:, -1, :]
        token_hidden=torch.cat((first_token_hidden, last_token_hidden), -1)
        final_output=self.mlp(token_hidden)
        final_output = final_output / final_output.norm(dim=-1, keepdim=True)
        return final_output
        


# class TXTSimpleEncoder(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.device = config.device
#         # embedding layer
#         self.tok_embeddings = nn.Embedding(
#             config.n_tok, config.d_tok, padding_idx=0)
#         self.seg_embeddings = nn.Embedding(2, config.d_tok, padding_idx=0)
#         self.position_embeddings = PositionalEncoding(d_model=config.d_tok)
#         # self-attention layer
#         encoder_layer = nn.TransformerEncoderLayer(d_model=config.d_tok, nhead=config.n_head,
#                                                    dim_feedforward=config.d_hid, dropout=config.dr,
#                                                    activation='gelu')
#         encoder_norm = nn.LayerNorm(config.d_tok)
#         self.encoder = nn.TransformerEncoder(
#             encoder_layer, config.x_layers, encoder_norm)

#     def forward(self, input_ids, token_type_ids=None, pad_mask=None):
#         if pad_mask is None:
#             pad_mask = torch.ones_like(input_ids)
#         if token_type_ids is None:
#             token_type_ids = torch.ones_like(input_ids)

#         pad_mask = (~pad_mask).type(torch.bool)

#         # Positional Word Embeddings
#         lang_embeddings = self.tok_embeddings(input_ids)
#         # if torch.isnan(lang_embeddings.std()):
#         #     print('language features are nan after embedding layer!')
#         lang_embeddings = self.position_embeddings(lang_embeddings)
#         # if torch.isnan(lang_embeddings.std()):
#         #     print('language features are nan after position embedding layer!')
#         seg_embeddings = self.seg_embeddings(token_type_ids)
#         lang_embeddings = lang_embeddings + seg_embeddings
#         # if torch.isnan(lang_embeddings.std()):
#         #     print('language features are nan after segmentation embedding layer!')

#         # self-attention layer
#         # input: bsz * seq_len * d_tok
#         # output: bsz * seq_len * d_tok
#         output = self.encoder(lang_embeddings.permute(
#             1, 0, 2), src_key_padding_mask=pad_mask)
#         # if torch.isnan(output.std()):
#         #     print('language features are nan after self-attention layer!')
#         output = output.permute(1, 0, 2)

#         return output


# class IMGSimpleEncoder(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         myPrinter(config, '\t############Simple IMG Model#####################')
#         self.prjLayers = nn.Sequential(
#             nn.Linear(config.d_tok, config.d_tok, bias=False),
#             nn.LayerNorm(config.d_tok, eps=1e-12))

#     def forward(self, imgs):
#         imgs = torch.mean(imgs, dim=1)
#         # if torch.isnan(imgs.std()):
#         #     print('img features are nan after mean!')
#         imgs = self.prjLayers(imgs)
#         # if torch.isnan(imgs.std()):
#         #     print('img features are nan after prjLayer!')
#         return imgs
from transformers import CLIPProcessor, CLIPModel
from transformers.models.clip.modeling_clip import CLIPVisionEmbeddings 
from transformers import CLIPVisionConfig

def freeze(model):
    model.requires_grad_(False)
    
class FeatureExtractorPatch(nn.Module):
    def __init__(self,image_size,is_load_embedding,is_freeze_clip):
        super(FeatureExtractorPatch, self).__init__()
        
        # self.embed_dim = 768
        # self.image_size =image_size
        # self.patch_size = 32
        
        if is_load_embedding:
            self.embeddings=self.load_clip_embedding()
            if is_freeze_clip:
                freeze(self.embeddings)
        else:
            config = CLIPVisionConfig(image_size=image_size)
            self.embeddings = CLIPVisionEmbeddings(config)
        
    def load_clip_embedding(self):
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        return model.vision_model.embeddings
        
    def forward(self, pixel_values):
        return self.embeddings(pixel_values)


def get_bert_embed_matrix(bert_checkpoint):
    bert = transformers.BertModel.from_pretrained(bert_checkpoint)
    bert_embeddings = list(bert.children())[0]
    bert_word_embeddings = list(bert_embeddings.children())[0]
    mat = bert_word_embeddings.weight.data.numpy()
    return mat

 
    
class OneStreamSelfAttenMMEncoder(nn.Module):
    def __init__(self, config, model_attribute):
        super(OneStreamSelfAttenMMEncoder, self).__init__()
        myPrinter(
            config, '\t############OneStreamSelfAttenMMEncoder Model#####################')
        
        self.use_attributes = config.use_attributes
        
        self.device = config.device
        self.position_embeddings = PositionalEncoding(d_model=config.d_tok)
        self.seg_embeddings = nn.Embedding(3, config.d_tok, padding_idx=0)
        # self-attention layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=config.d_tok, nhead=config.n_head,
                                                   dim_feedforward=config.d_hid, dropout=config.dr,
                                                   activation='gelu')
        encoder_norm = nn.LayerNorm(config.d_tok)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, config.x_layers, encoder_norm)
        self.class_embedding = nn.Parameter(torch.randn(config.d_tok))
        self.has_overall_cls_embed=model_attribute.has_overall_cls_embed


    def gen_multimodal_token_type_id(self,batch_size,patch_size, token_type_ids, att_token_type_ids):
        image_segment = torch.ones((batch_size, patch_size),device=token_type_ids.device) 
        if self.use_attributes:
            if self.has_overall_cls_embed:
                overall_cls_segment=torch.zeros((batch_size, 1),device=token_type_ids.device) 
                multimodal_segment = torch.cat((overall_cls_segment,token_type_ids, att_token_type_ids, image_segment * 2),
                                dim=1).type(torch.long)
            else:
                multimodal_segment = torch.cat((token_type_ids, att_token_type_ids, image_segment * 2),
                                dim=1).type(torch.long)
        else:
            if self.has_overall_cls_embed:
                overall_cls_segment=torch.zeros((batch_size, 1),device=token_type_ids.device) 
                multimodal_segment = torch.cat((overall_cls_segment,token_type_ids, image_segment * 2),
                                dim=1).type(torch.long)
            else:
                multimodal_segment = torch.cat((token_type_ids, image_segment * 2),
                                dim=1).type(torch.long)
        return multimodal_segment
    
    def gen_multimodal_pad_mask(self,batch_size,patch_size, lang_pad_mask,  att_lang_pad_mask):
        if self.use_attributes:
            visn_pad_mask = torch.ones((batch_size, patch_size),device=lang_pad_mask.device) 
            if self.has_overall_cls_embed:
                overall_cls_mask=torch.ones((batch_size, 1),device=lang_pad_mask.device) 
                pad_mask = torch.cat(
                    [overall_cls_mask,lang_pad_mask,  att_lang_pad_mask, visn_pad_mask], dim=1).type(torch.bool)
            else:
                pad_mask = torch.cat(
                    [lang_pad_mask, att_lang_pad_mask, visn_pad_mask], dim=1).type(torch.bool)
            pad_mask = (~pad_mask)
        else:
            visn_pad_mask = torch.ones((batch_size, patch_size),device=lang_pad_mask.device) 
            if self.has_overall_cls_embed:
                overall_cls_mask=torch.ones((batch_size, 1),device=lang_pad_mask.device) 
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
    
    def gen_embedding_with_position(self,batch_size,patch_size, lang_feats, lang_pad_mask, token_type_ids, 
                                                      visn_feats, att_lang_feats, att_lang_pad_mask, att_token_type_ids):
        
        multimodal_segment=self.gen_multimodal_token_type_id(batch_size,patch_size, token_type_ids, att_token_type_ids)
        multimodal_attention_mask=self.gen_multimodal_pad_mask( batch_size,patch_size, lang_pad_mask,  att_lang_pad_mask)
        multimodal_embedding=self.gen_multimodal_embedding(batch_size,lang_feats, att_lang_feats, visn_feats)
        unit_embeddings = self.position_embeddings(multimodal_embedding)
        seg_embeddings = self.seg_embeddings(multimodal_segment)
        unit_embeddings = unit_embeddings + seg_embeddings
        return unit_embeddings,multimodal_attention_mask
    
 
    
    def forward(self, lang_feats, lang_pad_mask, token_type_ids, visn_feats, att_lang_feats=None, att_lang_pad_mask=None, att_token_type_ids=None):
        (batch_size, seq_len, d_tok) = lang_feats.size()
        if self.use_attributes:
            (_, att_seq_len, att_d_tok) = att_lang_feats.size()
            assert d_tok == att_d_tok
        (_, patch_size, d_patch) = visn_feats.size()
        assert d_tok == d_patch
        
        unit_embeddings,pad_mask=self.gen_embedding_with_position( batch_size,patch_size, lang_feats, lang_pad_mask, token_type_ids, 
                                                      visn_feats, att_lang_feats, att_lang_pad_mask, att_token_type_ids)
        output = self.encoder(unit_embeddings.permute(
            1, 0, 2), src_key_padding_mask=pad_mask)
        #  for entity, we need to filter masked entities. e.g., max_entity_num=21, but we only have 15 entities.

        output = output.permute(1, 0, 2)
        
        lang_feats = output[:, seq_len-1, :]
        if self.use_attributes:
            att_lang_feats = output[:, (seq_len+att_seq_len-1), :]
        else:
            att_lang_feats = None
        visn_feats = output[:, -1, :]
         
        return lang_feats, visn_feats, output, att_lang_feats




def change_shape(e_embed):
    (batch_size, candidate_num, seq_len) = e_embed.shape
    reshaped_e_embed = e_embed.reshape(batch_size*candidate_num, seq_len)
    return reshaped_e_embed, batch_size, candidate_num




def gen_labels(output):
    (batch_size, batch_size_times_cand_num) = output.shape
    cand_num = int(batch_size_times_cand_num/batch_size)
    label_int_list = []
    for i in range(batch_size):
        label_int_list.append(0)
    labels = torch.tensor(label_int_list, device=output.device)
    return labels





class UNITER(nn.Module):
    def __init__(self, args, device, train_attribute,model_attribute,is_freeze_bert,is_freeze_clip):
        super().__init__()
        self.args = args

        self.mode = "txt_img"
        self.device = device
        '''
        img img_feats_extractor 
        '''
        if self.mode == 'txt':
            pass
        elif model_attribute.image_embed_level == "image":
            self.img_feats_extractor = FeatureExtractor(args.pre_trained_image_model_dir,is_freeze_clip)
        else:
            self.img_feats_extractor = FeatureExtractorPatch(model_attribute.image_size,model_attribute.is_load_clip_embedding,args.is_freeze_clip)
        
        embedding_matrix = get_bert_embed_matrix(args.pre_trained_dir)
        '''
        mention encoder 
        '''
        self.m_encoder = MMEncoder(
            config=args, embedding_matrix=embedding_matrix,train_attribute=train_attribute,model_attribute=model_attribute)
        
        '''entity encoder'''
        self.e_encoder = EntMMEncoder(
            config=args, embedding_matrix=embedding_matrix,train_attribute=train_attribute,model_attribute=model_attribute)

        '''score combination'''
        self.score_combine = nn.Linear(2, 1, bias=False)

        self.apply(init_weights)
      
        for n, p in self.named_parameters():
            if n.endswith('mention_text_embeddings.weight'):
                p.data.copy_(torch.from_numpy(embedding_matrix))
                if is_freeze_bert:
                    # print(f"debug: is_freeze_bert {is_freeze_bert}")
                    p.requires_grad = False
                else:
                    p.requires_grad = True
            if n.endswith('entity_text_embeddings.weight'):
                p.data.copy_(torch.from_numpy(embedding_matrix))
                if is_freeze_bert:
                    p.requires_grad = False
                else:
                    p.requires_grad = True

    def forward(self, mention_input_ids,mention_attention_mask,mention_token_type_ids,mention_image,entity_input_ids,entity_attention_masks,entity_token_type_ids, entity_images, e_mask, att_m_txt_info=None, att_e_txt_info=None, is_train=False, is_contrastive=False,labels=None):
        '''
        1. a_m_txt_info, a_e_txt_info added to param and to constructor and forward
        '''
    

        m_embedding=self.gen_mention_embedding(mention_input_ids,mention_attention_mask,mention_token_type_ids,mention_image, att_m_txt_info)#bsc * d_hid 
        entity_embedding_list,batch_size=self.gen_entity_embedding(entity_input_ids,entity_attention_masks,entity_token_type_ids, entity_images, att_e_txt_info) #bsc * n_cand * d_hid
        
        e_mask = e_mask   # bsc * n_cand 
        m_embedding = m_embedding.view(batch_size, -1, 1)  # bsz * d_hid * 1
        if   (is_train and is_contrastive):
            score, labels=None,None
        else:
            simi = torch.bmm(entity_embedding_list, m_embedding).view(
                batch_size, -1)  # bsz * n_cand
            score = self.masked_softmax(simi, e_mask)
            # labels = gen_labels(score)
        # if torch.isnan(score.std()):
        #     print('output is nan after the masked log softmax')
        return score, labels,m_embedding,entity_embedding_list
    
    def gen_entity_embedding(self,entity_input_ids,entity_attention_masks,entity_token_type_ids, entity_images, att_e_txt_info):
 
        e_seqs =entity_input_ids
        e_seq_pad_masks =entity_attention_masks
        e_seq_segment =entity_token_type_ids
        e_img_info = entity_images # bsc * n_cand * 3 * 256 * 256

        if self.args.use_attributes:
            
            att_e_seqs = att_e_txt_info["input_ids"]  # bsc * n_cand * seq_len
            att_e_seq_segment = att_e_txt_info["token_type_ids"]  # bsc * n_cand * seq_len
            att_e_seq_pad_masks = att_e_txt_info["attention_mask"]
            att_e_seqs = att_e_seqs 
            att_e_seq_pad_masks = att_e_seq_pad_masks 
            att_e_seq_segment = att_e_seq_segment 
        
        image_size=e_img_info.shape[-1]
        ''' entity '''
        if self.mode == 'txt':
            e_output = self.e_encoder(input_ids=e_seqs, token_type_ids=e_seq_segment,
                                      attention_mask=e_seq_pad_masks)
        elif self.mode == 'img':
             # bsz * n_cand * 3 * 256 *256
            e_imgs = e_img_info.view(-1, 3, image_size, image_size) # (bsz * n_cand) * 3 * 256 *256
            # (bsz * n_cand) * patch_num * 100
            e_imgs = self.img_feats_extractor(e_imgs)
            e_output = self.e_encoder(input_ids=None, token_type_ids=None,
                                      attention_mask=None, visual_feats=e_imgs)
        else:
            # e_img_info = e_img_info[:,:,0,:,:,:] # bsz * n_cand * 3 * 256 *256
            # (bsz * n_cand) * 3 * 256 *256
            '''
            entity encoder, [(batch_size *n_cand* patch_num * d_hidden),(batch_size*n_cand*seq_len)]-> (batch_size *n_cand* d_hid )
            '''
            e_imgs = e_img_info.view(-1, 3, image_size, image_size)
            (batch_size, candidate_num, seq_len) = e_seqs.shape
            e_seqs = e_seqs.view(batch_size*candidate_num, seq_len)
            e_seq_segment = e_seq_segment.view(
                batch_size*candidate_num, seq_len)
            e_seq_pad_masks = e_seq_pad_masks.view(
                batch_size*candidate_num, seq_len)
             
             
            if self.args.use_attributes:
                # (batch_size, candidate_num, seq_len) = att_e_seqs.shape
                att_e_seqs = att_e_seqs.view(batch_size*candidate_num, seq_len)
                att_e_seq_segment = att_e_seq_segment.view(
                    batch_size*candidate_num, seq_len)
                att_e_seq_pad_masks = att_e_seq_pad_masks.view(
                    batch_size*candidate_num, seq_len)
            else:
                att_e_seqs, att_e_seq_segment, att_e_seq_pad_masks = None, None, None
            
            e_imgs_feature = self.img_feats_extractor(
                e_imgs)  # (bsz * n_cand) * patch_num * 1024
            e_output = self.e_encoder(input_ids=e_seqs, token_type_ids=e_seq_segment,
                                      attention_mask=e_seq_pad_masks, visual_feats=e_imgs_feature, att_input_ids=att_e_seqs, att_token_type_ids=att_e_seq_segment, att_attention_mask=att_e_seq_pad_masks)  # (bsz * n_cand) * d_hid
            e_output = e_output.view(batch_size, candidate_num, -1)
        return e_output,batch_size
        
    def gen_mention_embedding(self,mention_input_ids,mention_attention_mask,mention_token_type_ids,mention_image, att_txt_info):
 
        m_seqs = torch.squeeze( mention_input_ids,1)  # bsz * seq_len
        m_seq_pad_masks =torch.squeeze( mention_attention_mask,1)    # bsz * seq_len
        m_seq_segment =torch.squeeze( mention_token_type_ids,1)    # bsz * seq_len
        m_imgs =torch.squeeze( mention_image,1)    # bsz * 3 * 256 *256

        if self.args.use_attributes:
            att_m_seqs = att_txt_info["input_ids"]
            att_m_seq_segment = att_txt_info["token_type_ids"]
            att_m_seq_pad_masks = att_txt_info["attention_mask"] 
            att_m_seqs = torch.squeeze(att_m_seqs)   # bsz * seq_len
            att_m_seq_pad_masks = torch.squeeze(att_m_seq_pad_masks)   # bsz * seq_len
            att_m_seq_segment = torch.squeeze(att_m_seq_segment)  # bsz * seq_len
        else:
            att_m_seqs, att_m_seq_segment, att_m_seq_pad_masks = None, None, None


        ''' mention '''
        if self.mode == 'txt':
            m_output = self.m_encoder(
                m_seqs, m_seq_segment, m_seq_pad_masks)  # bsz * 1 * d_hid
        elif self.mode == 'img':
            # bsz * patch_num * d_hidden  patch_size=64
            m_imgs = self.img_feats_extractor(m_imgs)
            m_output = self.m_encoder(input_ids=None, token_type_ids=None,
                                      attention_mask=None, visual_feats=m_imgs)
        else:
            '''
            img img_feats_extractor, (batch_size,3,256,256)->(batch_size * patch_num * d_hidden)
            '''
            m_imgs_feature = self.img_feats_extractor(
                m_imgs)  # bsz * patch_num * d_hidden
            '''
            mention encoder, [(batch_size * patch_num * d_hidden),(batch_size*seq_len)]-> (batch_size * d_hid )
            '''
            m_output = self.m_encoder(input_ids=m_seqs, token_type_ids=m_seq_segment,
                                      attention_mask=m_seq_pad_masks, visual_feats=m_imgs_feature,att_input_ids=att_m_seqs, att_token_type_ids=att_m_seq_segment, att_attention_mask=att_m_seq_pad_masks)

        return m_output 
    
    def masked_softmax(self, tensor, mask):
        input_tensor = tensor.masked_fill((~mask), value=torch.tensor(-1e9))
        # result = nn.functional.log_softmax(input_tensor, dim=-1)
        return input_tensor
    
    
class V2VELCLIP(nn.Module):
    def __init__(self, args, device, train_attribute,model_attribute,clip_model,is_resnet=False,has_adapter=True):
        super().__init__()
        self.args = args

        self.mode = "txt_img"
        self.device = device
 
        '''score combination'''
        self.score_combine = nn.Linear(2, 1, bias=False)
        if has_adapter:
            custom_clip=CustomCLIP(clip_model,args.d_image,is_resnet)
        else:
            custom_clip=clip_model
        self.encoder=ImageLevelEncoder(args,   model_attribute,custom_clip,device,args.use_attributes,args.use_image)
    # def forward(self, m_txt_info, m_img_info, e_txt_info, e_img_info, e_mask, att_m_txt_info=None, att_e_txt_info=None, is_train=False, is_contrastive=False,labels=None):
    def forward(self,pixel_values,input_ids,attention_mask,labels,entity_mask , is_train=None, 
                        is_contrastive=None):
        mention_text_embed,entity_text_embeds,mention_image_embed,entity_image_embeds,mention_text_attention_mask,entity_text_attention_mask,batch_size=self.encoder(pixel_values,input_ids,attention_mask)
        
          # bsc * n_cand 
        mention_image_embed = mention_image_embed.view(batch_size, -1, 1)  # bsz * d_hid * 1
        if is_train and is_contrastive:
            # batch_size * batch_size * n_cand * d_hid    batch_size * d_hid * 1
            simi = torch.einsum(
                'ijkl,ild->ijk', [entity_image_embeds.expand(batch_size, -1, -1, -1), mention_image_embed])
            simi = simi.view(batch_size, -1)  # bsz * (bsz * n_cand)
            score,score_after_softmax = self.masked_softmax(simi, entity_mask.expand(
                batch_size, -1, -1).view(batch_size, -1))
            labels = gen_contrastive_labels(score,labels)
        else:
            simi = torch.bmm(entity_image_embeds, mention_image_embed).view(
                batch_size, -1)  # bsz * n_cand
            score,score_after_softmax = self.masked_softmax(simi, entity_mask)
            # labels = gen_labels(score)
        # if torch.isnan(score.std()):
        #     print('output is nan after the masked log softmax')
        return score, labels ,score_after_softmax
    
 
    def masked_softmax(self, tensor, mask):
        input_tensor = tensor.masked_fill((~mask), value=torch.tensor(-1e9))
        result = nn.functional.softmax(input_tensor, dim=-1)
        return input_tensor,result
    
def masked_softmax(  tensor, mask):
    input_tensor = tensor.masked_fill((~mask), value=torch.tensor(-1e9))
    # result = nn.functional.log_softmax(input_tensor, dim=-1)
    return input_tensor
class V2TEL(nn.Module):
    def __init__(self, args, device, train_attribute,model_attribute,clip_model,is_resnet=False,has_adapter=True):
        super().__init__()
        self.args = args

        self.mode = "txt_img"
        self.device = device
 
        '''score combination'''
        self.score_combine = nn.Linear(2, 1, bias=False)
        
        custom_clip=CustomMultimodalCLIP(clip_model,args.d_image,is_resnet,has_adapter)
       
        self.encoder=ImageLevelImageAndTextEncoder(args,   model_attribute,custom_clip,device,args.use_attributes,args.use_image)
    # def forward(self, m_txt_info, m_img_info, e_txt_info, e_img_info, e_mask, att_m_txt_info=None, att_e_txt_info=None, is_train=False, is_contrastive=False,labels=None):
    def forward(self,pixel_values,input_ids,attention_mask,labels,entity_mask , is_train=None, 
                        is_contrastive=None,token_type_ids=None):
        mention_image_embed, entity_text_embed, logits ,batch_size=self.encoder(pixel_values,input_ids,attention_mask)
        
          # bsc * n_cand 
        mention_image_embed = mention_image_embed.view(batch_size, -1, 1)  # bsz * d_hid * 1
        if   (is_train and is_contrastive):
            score, labels=None,None
        else:
            simi = torch.bmm(entity_text_embed, mention_image_embed).view(
                batch_size, -1)  # bsz * n_cand
            score =  masked_softmax(simi, entity_mask)
            # labels = gen_labels(score)
        # if torch.isnan(score.std()):
        #     print('output is nan after the masked log softmax')
        return score, labels  ,mention_image_embed,entity_text_embed
        
        
        
def freeze_flava(flava):
    flava.image_model.requires_grad_(False)
    flava.text_model.requires_grad_(False)
    return flava
    
class FLAVAEL(nn.Module):
    def __init__(self, args, device, train_attribute,model_attribute,clip_model,is_resnet=False,has_adapter=True):
        super().__init__()
        self.args = args

        self.mode = "txt_img"
        self.device = device
 
        '''score combination'''
        self.score_combine = nn.Linear(2, 1, bias=False)
        


        self.encoder= FlavaModel.from_pretrained("facebook/flava-full")
        self.encoder=freeze_flava(self.encoder)
        # self.encoder.model.requires_grad_(False)
       
        
        
    # def forward(self, m_txt_info, m_img_info, e_txt_info, e_img_info, e_mask, att_m_txt_info=None, att_e_txt_info=None, is_train=False, is_contrastive=False,labels=None):
    def forward(self,pixel_values,input_ids,attention_mask,labels,entity_mask , is_train=None, 
                        is_contrastive=None ,token_type_ids=None):
        
        batch_size,entity_image_num,num_channels, height, width=pixel_values.shape
        reshaped_pixel_values=pixel_values.reshape(batch_size*entity_image_num,num_channels, height, width)
        batch_size,textual_sequence_num,sequence_len=input_ids.shape 
        # batch_size,sequence_num,num_channels, height, width=pixel_values.shape
        reshaped_input_ids=input_ids.reshape(batch_size*textual_sequence_num,-1)
        reshaped_attention_mask=attention_mask.reshape(batch_size*textual_sequence_num,-1)
        if token_type_ids is not None:
            reshaped_token_type_ids=token_type_ids.reshape(batch_size*textual_sequence_num,-1)
        else:
            reshaped_token_type_ids=None
            
        outputs =self.encoder(input_ids=reshaped_input_ids,attention_mask=reshaped_attention_mask,token_type_ids=reshaped_token_type_ids, pixel_values=reshaped_pixel_values)
        multimodal_embeddings = outputs.multimodal_embeddings
        cls_multimodal_embeddings=multimodal_embeddings[:,0]
        original_shape_multimodal_embeddings=cls_multimodal_embeddings.reshape(batch_size,textual_sequence_num,-1)
        mention_multimodal_embeddings=original_shape_multimodal_embeddings[:,0,:]
        entity_multimodal_embeddings=original_shape_multimodal_embeddings[:,1:,:]
        batch_size=original_shape_multimodal_embeddings.shape[0]
        
          # bsc * n_cand 
        mention_multimodal_embeddings = mention_multimodal_embeddings.view(batch_size, -1, 1)  # bsz * d_hid * 1
        if not (is_train and is_contrastive):
            simi = torch.bmm(entity_multimodal_embeddings, mention_multimodal_embeddings).view(
                batch_size, -1)  # bsz * n_cand
            score = self.masked_softmax(simi, entity_mask)
        else:
            score=labels
        return score, labels   ,mention_multimodal_embeddings,entity_multimodal_embeddings
    
 
    def masked_softmax(self, tensor, mask):
        input_tensor = tensor.masked_fill((~mask), value=torch.tensor(-1e9))
        # result = nn.functional.log_softmax(input_tensor, dim=-1)
        return input_tensor
             
             
class V2VELCLIPAttributeFromSBERT(nn.Module):
    def __init__(self, args, device, train_attribute,model_attribute,text_encoder,image_encoder,is_resnet=False,has_adapter=True):
        super().__init__()
        self.args = args

        self.mode = "txt_img"
        self.device = device
        self.use_image=args.use_image
        self.use_attributes=args.use_attributes
        self.use_text=args.use_text
        '''score combination'''
        # self.score_combine = nn.Linear(2, 1, bias=False)
        # custom_clip=clip_model
        self.text_encoder=text_encoder
        
        self.image_encoder=image_encoder
        image_encoder=freeze_part_bert( image_encoder,args.freeze_text_layer_num,args.freeze_img_layer_num)
        self.encoder=ImageLevelSeparateImageAndTextEncoder(args,   model_attribute,image_encoder,device,args.use_attributes,args.use_image,text_encoder,args.use_text)
        self.text_project= MLP(
            in_dim=768,
            out_dim=768,
            hidden_dims=768 
        )
        self.image_project= MLP(
            in_dim=768,
            out_dim=768,
            hidden_dims=768 
        )
        self.multimodal_fuser= MLP(
            in_dim=1536,
            out_dim=1536,
            hidden_dims=[768 ,1536]
        )
        
    def fuse(self,text_embed,image_embed):
        projected_text=self.text_project(text_embed)
        projected_image=self.image_project(image_embed)
        return self.multimodal_fuser(torch.cat((projected_image,projected_text),-1))
        # return torch.cat((text_embed,image_embed),-1)
        
    # def forward(self, m_txt_info, m_img_info, e_txt_info, e_img_info, e_mask, att_m_txt_info=None, att_e_txt_info=None, is_train=False, is_contrastive=False,labels=None):
    def forward(self,pixel_values,input_ids,attention_mask,labels,entity_mask , is_train=None, 
                        is_contrastive=None):
        mention_text_embed,entity_text_embeds,mention_image_embed,entity_image_embeds,mention_text_attention_mask,entity_text_attention_mask,batch_size=self.encoder(pixel_values,input_ids,attention_mask)
        if self.use_text and self.use_image:
            
            mention_embed=self.fuse(mention_text_embed,mention_image_embed)
            entity_embed=self.fuse(entity_text_embeds,entity_image_embeds)
        elif self.use_text:
            mention_embed=mention_text_embed
            entity_embed=entity_text_embeds
        else:
            mention_embed=mention_image_embed
            entity_embed=entity_image_embeds
            
          # bsc * n_cand 
        score=labels
        mention_embed = mention_embed.view(batch_size, -1, 1)  # bsz * d_hid * 1
        if not (is_train and is_contrastive):
            simi = torch.bmm(entity_embed, mention_embed).view(
                batch_size, -1)  # bsz * n_cand
            score = self.masked_softmax(simi, entity_mask)
        return score, labels ,mention_embed,entity_embed
    
 
    def masked_softmax(self, tensor, mask):
        input_tensor = tensor.masked_fill((~mask), value=torch.tensor(-1e9))
        # result = nn.functional.log_softmax(input_tensor, dim=-1)
        return input_tensor
                 
                 

class V2VELCLIPAttributeFromSBERTText(nn.Module):
    def __init__(self, args, device, train_attribute,model_attribute,text_encoder,image_encoder,is_resnet=False,has_adapter=True):
        super().__init__()
        self.args = args

        self.mode = "txt_img"
        self.device = device
        self.use_image=args.use_image
        self.use_attributes=args.use_attributes
        self.use_text=args.use_text
        '''score combination'''
        # self.score_combine = nn.Linear(2, 1, bias=False)
        # custom_clip=clip_model
        self.text_encoder=text_encoder
        
        self.image_encoder=image_encoder
        # image_encoder=freeze_part_bert( image_encoder,args.freeze_text_layer_num,args.freeze_img_layer_num)
        self.encoder=ImageLevelSeparateImageAndTextEncoderText(args,   model_attribute,image_encoder,device,args.use_attributes,args.use_image,text_encoder,args.use_text)
        
    def forward(self, input_ids,attention_mask,labels,entity_mask , is_train=None, 
                        is_contrastive=None):
        mention_text_embed,entity_text_embeds, mention_text_attention_mask,entity_text_attention_mask,batch_size=self.encoder( input_ids,attention_mask)
  
        mention_embed=mention_text_embed
        entity_embed=entity_text_embeds
      
        mention_embed = mention_embed / mention_embed.norm(dim=-1, keepdim=True)
        entity_embed = entity_embed / entity_embed.norm(dim=-1, keepdim=True)
          # bsc * n_cand 
        score=labels
        mention_embed = mention_embed.view(batch_size, -1, 1)  # bsz * d_hid * 1
        if not (is_train and is_contrastive):
            simi = torch.bmm(entity_embed, mention_embed).view(
                batch_size, -1)  # bsz * n_cand
            score = self.masked_softmax(simi, entity_mask)
        return score, labels ,mention_embed,entity_embed                 

    def masked_softmax(self, tensor, mask):
        input_tensor = tensor.masked_fill((~mask), value=torch.tensor(-1e9))
        # result = nn.functional.log_softmax(input_tensor, dim=-1)
        return input_tensor