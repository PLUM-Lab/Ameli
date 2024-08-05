from enum import Enum,auto
 
import json

 
class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.name
        return json.JSONEncoder.default(self, obj)
class EncoderAttribute(Enum):
    IMAGE_ENCODER=('clip-ViT-B-32' )  
    TEXT_ENCODER_BERT=('multi-qa-MiniLM-L6-cos-v1'  )
    TEXT_ENCODER_2VEC=('multi-qa-MiniLM-L6-cos-v1' )
    
    def __init__(self,model_name):
        self.model_name=model_name
        
        
    
class WordAttribute(Enum):
    MENTION=(EncoderAttribute.IMAGE_ENCODER,EncoderAttribute.TEXT_ENCODER_BERT,"","txt_img")
    ENTITY=(EncoderAttribute.IMAGE_ENCODER,EncoderAttribute.TEXT_ENCODER_2VEC,"","txt_img")
    def __init__(self, image_embedder_enum, text_embedder_enum,model_type,  media):
        self.model_type=model_type
        self.image_embedder_enum=image_embedder_enum
        self.text_embedder_enum=text_embedder_enum 
        self.media=media 


class ModelAttribute(Enum):    
    A1=("A1","image",256,False,False,True,"" ,2,5,"encoder1",False,"")  
    A2=("A2","patch",256,False,False ,True,"",2,5,"encoder1",False,"")  
    A3=("A3","patch",224,True,False ,True ,"",2,5,"encoder1",False,"")  
    A4=("A4","patch",224,True ,True ,True,"",2,5,"encoder1",False,"")  
    A5=("A5","patch",224,False ,True ,True,"",2,5,"encoder1",False,"")  
    A6=("A6","patch",224,True ,True ,False,"",2,5,"encoder1",False,"")  
    A7=("A7","image",224,True ,True ,False,"",1,1,"encoder1",True,"")  
    A_adapt_contrast=("A_adapt_contrast","image",224,True ,True ,True,"",1,1,"encoder1",True,"")  
    resnet_adapt_contrast=("resnet_adapt_contrast","image",224,True ,True ,True,"",1,1,"encoder1",True,"")  
    A8=("A8","image",224,True ,True ,True,"",1,1,"encoder1",False,"")  
    A_contrast=("A_contrast","image",224,True ,True ,True,"",1,1,"encoder1",False,"")  
    A9=("A9","image",256,False,False,False,"" ,2,5,"encoder1",False,"")  
    A_V2TEL_contrast=("A_V2TEL_contrast","image",224,True ,True ,True,"",1,1,"encoder1",True,"")  
    A_V2TEL=("A_V2TEL","image",224,True ,True ,False,"",1,1,"encoder1",True,"")  
    B6=("B6","patch",224,False ,True ,False,"pooling1",2,5,"encoder1",False,"")  
    B7=("B7","patch",224,False ,True ,False,"",2,5,"encoder1",False,"")  
    B8=("B8","patch",224,False ,True ,False,"pooling2",2,5,"encoder1",False,"")  
    B9=("B9","patch",224,False ,True ,False,"pooling2",2,5,"encoder2",False,"")  
    B10=("B10","patch",224,False ,True ,False,"pooling3",2,5,"encoder1",False,"")  
    B11=("B11","patch",224,False ,True ,False,"pooling4",2,5,"encoder3",False,"")  
    B12=("B12","patch",224,False ,True ,False,"pooling4",2,5,"encoder4",False,"")  
    B13=("B13","patch",224,False ,True ,False,"pooling4",2,2,"encoder2",False,"")  
    B14=("B14","patch",224,False ,True ,False,"pooling4",2,5,"encoder4",False,"")  
    B15=("B15","patch",224,False ,True ,False,"pooling4",1,1,"encoder4",False,"")  
    C_joint=("C_joint","image",224,True ,True ,True,"",1,1,"encoder1",False,"")  
    C_joint_adapt=("C_joint_adapt","image",224,True ,True ,False,"",1,1,"encoder1",True,"")  
    D_MLP=("D_MLP","image",224,True ,True ,False,"",1,1,"encoder1",True,"mlp")  
    D_MLP_residual=("D_MLP_residual","image",224,True ,True ,False,"",1,1,"encoder1",True,"mlp_residual")  
    Text_cross_encoder=("Text_cross_encoder","patch",224,True ,True ,True,"",2,5,"encoder1",False,"")  
    A_FLAVA=("A_FLAVA","patch",224,True ,True ,True,"",2,5,"encoder1",False,"")  
    A_SBERT_ATTRIBUTE=("A_SBERT_ATTRIBUTE","patch",224,True ,True ,True,"",2,5,"encoder1",False,"")  
    A_SBERT_ATTRIBUTE_TEXT=("A_SBERT_ATTRIBUTE_TEXT","patch",224,True ,True ,True,"",2,5,"encoder1",False,"")  
    
    def __init__(self,inner_name,image_embed_level,image_size,is_load_clip_embedding,has_overall_cls_embed,is_contrastive,
                 match_pooling,max_mention_image_num,max_entity_image_num,encoder,has_adapter,mlp_type):
        self.inner_name=inner_name
        self.mlp_type=mlp_type
        self.image_embed_level=image_embed_level
        self.is_contrastive=is_contrastive
        self.image_size=image_size 
        self.is_load_clip_embedding=is_load_clip_embedding
        self.has_adapter=has_adapter
        self.has_overall_cls_embed=has_overall_cls_embed
        self.match_pooling=match_pooling
        self.encoder=encoder
        self.max_mention_image_num=max_mention_image_num
        self.max_entity_image_num=max_entity_image_num
class DataTrainingAttribute(Enum):
     def __init__(self):
         pass 
    
class TrainingAttribute(Enum):
    BI_ENCODER = (WordAttribute.MENTION,WordAttribute.ENTITY,77 ,20,480,"","" )
    BI_ENCODER2 = (None,None,512 ,10,35,"all-mpnet-base-v2","txt" )#80
    IMAGE_MODEL =(None,None,77 ,30,24,'clip-ViT-L-14',"img") #clip-ViT-B-32 30,480
    CROSS_ENCODER=( None,None,100,20,512,'cross-encoder/ms-marco-MiniLM-L-6-v2',"txt")
    IMAGE_CROSS_ENCODER=( None,None,77 ,20,128,'clip-ViT-L-14',"img")#512
    MULTI_MODAL=( None,None,77 ,5,15,'clip-ViT-L-14',"txt_img")#512
    
    
    
    def __init__(self, mention_attribute_enum,entity_attribute_enum,max_seq_length,epoch,batch_size ,model_name ,media):
        self.mention_attribute_enum=mention_attribute_enum
        self.entity_attribute_enum=entity_attribute_enum 
        self.max_seq_length=max_seq_length 
        self.epoch=epoch 
        self.batch_size=batch_size
        self.model_name=model_name
        self.media=media