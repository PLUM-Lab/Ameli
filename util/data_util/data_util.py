
from urllib.parse import unquote
import pandas as pd
import os 
import hashlib
import re
import json
from tqdm import tqdm
import pickle
from transformers import AutoProcessor, FlavaModel
from transformers import CLIPFeatureExtractor, CLIPProcessor
from disambiguation.data_util.inner_util import MultimodalEntityLinkingDataDealer, MultimodalEntityLinkingDataDealerSelectImage, MultimodalEntityLinkingDataDealerSurfaceForm, MultimodalEntityLinkingDataDealerSurfaceFormChooseAttribute, MultimodalEntityLinkingDataDealerSurfaceFormSelectImage, MultimodalEntityLinkingDataDealerSurfaceFormSelectImageChooseAttribute, MultimodalEntityLinkingDataDealerSurfaceFormSelectImageWithAttributeHashRichReview, MultimodalEntityLinkingDataDealerSurfaceFormSelectImageWithNLIRetrievalScore, MultimodalEntityLinkingDataDealerSurfaceFormSelectImageWithNLIRetrievalScoreRichReview
from retrieval.data_util.core.dataset import MultimodalDatasetAttenBasedMultiImage, MultimodalDatasetBERTCrossEncoder, MultimodalDatasetBERTResnet, MultimodalDatasetQueryBasedMultiImage, MultimodalDatasetQueryBasedBERT, MultimodalDatasetMultiImageCrossEncoder, MultimodalDataset_resnet, MultimodalDatasetSBERTBERTCLIPAttribute, MultimodalDatasetSelectImage, MultimodalDatasetSelectImageFLAVA, MultimodalDatasetSelectImageV2TEL, MultimodalDatasetSelectImageBERTResnet,  ReviewAttributePairDataset, ReviewAttributePairWithImageDataset, ReviewAttributePairWithImageDatasetSelectImage, ReviewAttributePairWithImageDatasetSelectImageAndRetrievalNLIScore
from retrieval.data_util.core.gen_data import WikidiverseDataDealer, MultimodalDatasetBERTCLIP, MultimodalDatasetCLIP, get_train_queries
from retrieval.eval.eval_msmarco_mocheg import load_qrels
from retrieval.data_util.datapoint import Entity, Mention, gen_entity_name 
from transformers import CLIPProcessor, CLIPModel
import torch
from torch.utils.data import Dataset  
from torch.utils.data import DataLoader 


# def change_svg_to_png():
#     image_corpus="/home/menglong/workspace/code/referred/wikiDiverse/data/wikinewsImgs"
#     removed_image_corpus="/home/menglong/workspace/code/referred/wikiDiverse/data/removed_wikinewsImgs"
#     img_names=os.listdir(image_corpus)
#     print(f"img_num {len(img_names)}")
#     for filepath in  img_names:
#         if filepath.endswith(".svg") or filepath.endswith(".SVG"):
#             prefix,suffix=filepath.split(".")
#             if suffix=="svg" or suffix == "SVG":
#                 new_file_path=prefix+".png"
#                 shutil.copyfile(os.path.join(image_corpus,filepath), os.path.join(image_corpus,new_file_path))
            # shutil.move(os.path.join(image_corpus,filepath), os.path.join(removed_image_corpus,filepath))
# convert_entity2img()                
def clean_name(text):
    text = text.strip()
    text =text.replace('_', ' ').replace('-', ' ').replace("'","").replace('"','')
    return text 
 
def search_in_entity_name_list(query,entity_dict):
    is_found=False
    for entity_name,_ in entity_dict.items():
        
        if clean_name(query)==clean_name(entity_name):
            print(entity_name )
            is_found= True
    return is_found
    
def read_example():
    entity_dict=read_entity()
    mention_dict=read_mention(entity_dict)
    matched_num=0
    
    for mention_id,mention in mention_dict.items():
        entity_name= mention.entity_wiki_name
        if entity_name in entity_dict:
            matched_num+=1
        else:
            
            # search_in_entity_name_list(mention.entity_wiki_name,entity_dict)
            # print(mention.entity_wiki_url)
            del mention_dict[mention_id]
            
    print(f"{matched_num} {len(mention_dict)}")
    # print(mention_dict[0])
    
import numpy as np  

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            # 👇️ alternatively use str()
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
        
    
def read_mention(dataset_dir,entity_dict):
   
    mention_dict={}
    
    with open(dataset_dir, 'r', encoding='utf-8') as fr:
        testData = json.load(fr)
    
        for mention_id,datapoint  in enumerate(testData):
      
            [caption, img, ment, ment_type, lctx, rctx, entity, cands, topic, start, end,img_path,is_img_downloaded]=datapoint 
            if entity!="nil" and   os.path.exists(img_path):
                entity_candidate_name_list=[gen_entity_name(entity_candidate_url) for entity_candidate_url in cands]
                mention=Mention(ment,caption,img,img_path,is_img_downloaded,entity,None,cands,entity_candidate_name_list)
                if mention.entity_wiki_name in entity_dict:
                    mention_dict[mention_id]=mention 
    return mention_dict
              
        
def get_queries(train_txt_dir,data_dealer,product_json_array=None,product_json_dict=None ,max_candidate_num=None, dataset_class=None ):
    positive_rel_docs,needed_pids,needed_qids,negative_rel_docs=data_dealer.load_qrels(product_json_dict,train_txt_dir,max_candidate_num)
    queries=data_dealer.load_queries(train_txt_dir,needed_qids)
    train_queries=get_train_queries(queries,positive_rel_docs,negative_rel_docs)  
    return train_queries    
     
from torch.utils.data.distributed import DistributedSampler
def prepare_for_ddp(rank, world_size,dataset, batch_size=32, pin_memory=False, num_workers=5):
 
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=True, shuffle=False, sampler=sampler)
    
    return dataloader

def    get_data_loader( train_txt_dir,  val_txt_dir, test_txt_dir,
                                                      tokenizer, max_len, batch_size,max_candidate_num,args,rank,world_size,mode,model_special_processor)    :
    
    if args.dataset=="entity_linking":  
        if     args.datadealer_class in["select_image_with_nli_score_and_retrieval_score_rich_review"]:
            test_data_dealer=MultimodalEntityLinkingDataDealerSurfaceFormSelectImageWithNLIRetrievalScoreRichReview(test_txt_dir,args.review_image_dir,args.candidate_base,args.test_candidate_mode,max_candidate_num,args.dataset_class ,args.entity_text_source,args.product_image_dir ,args.path_to_all_attribute_key_list,args.empty_nli_score,args.use_nli_score_num,args.allow_empty_nli)
            val_data_dealer=MultimodalEntityLinkingDataDealerSurfaceFormSelectImageWithNLIRetrievalScoreRichReview(val_txt_dir,args.review_image_dir,args.candidate_base,args.dev_candidate_mode,max_candidate_num,args.dataset_class ,args.entity_text_source,args.product_image_dir,args.path_to_all_attribute_key_list ,args.empty_nli_score,args.use_nli_score_num,args.allow_empty_nli)
            if mode not in ["test","dry_run"]:
                train_data_dealer=MultimodalEntityLinkingDataDealerSurfaceFormSelectImageWithNLIRetrievalScoreRichReview(train_txt_dir,args.review_image_dir,args.candidate_base,args.candidate_mode,args.train_max_candidate_num,args.dataset_class ,args.entity_text_source,args.product_image_dir  ,args.path_to_all_attribute_key_list,args.empty_nli_score,args.use_nli_score_num,args.allow_empty_nli)#"by_gold" 
                
            else:
                train_data_dealer=test_data_dealer
        
        elif     args.datadealer_class in["select_image_with_attribute_hash_and_retrieval_score_rich_review"]:
            test_data_dealer=MultimodalEntityLinkingDataDealerSurfaceFormSelectImageWithAttributeHashRichReview(test_txt_dir,args.review_image_dir,args.candidate_base,args.test_candidate_mode,max_candidate_num,args.dataset_class ,args.entity_text_source,args.product_image_dir ,args.path_to_all_attribute_key_list,args.empty_nli_score,args.use_nli_score_num,args.allow_empty_nli)
            val_data_dealer=MultimodalEntityLinkingDataDealerSurfaceFormSelectImageWithAttributeHashRichReview(val_txt_dir,args.review_image_dir,args.candidate_base,args.dev_candidate_mode,max_candidate_num,args.dataset_class ,args.entity_text_source,args.product_image_dir,args.path_to_all_attribute_key_list ,args.empty_nli_score,args.use_nli_score_num,args.allow_empty_nli)
            if mode not in ["test","dry_run"]:
                train_data_dealer=MultimodalEntityLinkingDataDealerSurfaceFormSelectImageWithAttributeHashRichReview(train_txt_dir,args.review_image_dir,args.candidate_base,args.candidate_mode,args.train_max_candidate_num,args.dataset_class ,args.entity_text_source,args.product_image_dir  ,args.path_to_all_attribute_key_list,args.empty_nli_score,args.use_nli_score_num,args.allow_empty_nli,args.train_max_candidate_num_in_corpus)  
                
            else:
                train_data_dealer=test_data_dealer
        elif args.datadealer_class in ["MultimodalEntityLinkingDataDealerSurfaceFormChooseAttribute"]:
            test_data_dealer=MultimodalEntityLinkingDataDealerSurfaceFormChooseAttribute(test_txt_dir,args.review_image_dir,args.candidate_base,args.test_candidate_mode,max_candidate_num,args.dataset_class ,args.entity_text_source)
            val_data_dealer=MultimodalEntityLinkingDataDealerSurfaceFormChooseAttribute(val_txt_dir,args.review_image_dir,args.candidate_base,args.dev_candidate_mode,max_candidate_num,args.dataset_class ,args.entity_text_source)
            if mode not in ["test","dry_run"]:
                train_data_dealer=MultimodalEntityLinkingDataDealerSurfaceFormChooseAttribute(train_txt_dir,args.review_image_dir,args.candidate_base,args.candidate_mode,args.train_max_candidate_num,args.dataset_class ,args.entity_text_source,is_gold_attribute=True )#"by_gold" 
                
            else:
                train_data_dealer=test_data_dealer
        elif args.datadealer_class in ["MultimodalEntityLinkingDataDealerSurfaceFormSelectImageChooseAttribute"]:
            test_data_dealer=MultimodalEntityLinkingDataDealerSurfaceFormSelectImageChooseAttribute(test_txt_dir,args.review_image_dir,args.candidate_base,args.test_candidate_mode,max_candidate_num,args.dataset_class ,args.entity_text_source,args.product_image_dir  ,review_text_mode=args.review_text_mode)
            val_data_dealer=MultimodalEntityLinkingDataDealerSurfaceFormSelectImageChooseAttribute(val_txt_dir,args.review_image_dir,args.candidate_base,args.dev_candidate_mode,max_candidate_num,args.dataset_class ,args.entity_text_source,args.product_image_dir ,review_text_mode=args.review_text_mode)
            if mode not in ["test","dry_run"]:
                train_data_dealer=MultimodalEntityLinkingDataDealerSurfaceFormSelectImageChooseAttribute(train_txt_dir,args.review_image_dir,args.candidate_base,args.candidate_mode,args.train_max_candidate_num,args.dataset_class ,args.entity_text_source,args.product_image_dir  ,review_text_mode=args.review_text_mode,is_gold_attribute=True )#"by_gold" 
                
            else:
                train_data_dealer=test_data_dealer
        
        elif args.dataset_class   in [ "v2","v3","v4","resnet","resnet_text"]:#review_text ,review_image_paths,review_attributes
            test_data_dealer=MultimodalEntityLinkingDataDealer(test_txt_dir,args.review_image_dir,args.candidate_base,args.test_candidate_mode,max_candidate_num,args.dataset_class ,args.entity_text_source)
            val_data_dealer=MultimodalEntityLinkingDataDealer(val_txt_dir,args.review_image_dir,args.candidate_base,args.dev_candidate_mode,max_candidate_num,args.dataset_class ,args.entity_text_source)
            if mode not in ["test","dry_run"]:
                train_data_dealer=MultimodalEntityLinkingDataDealer(train_txt_dir,args.review_image_dir,args.candidate_base,args.candidate_mode,args.train_max_candidate_num,args.dataset_class ,args.entity_text_source)#"by_gold" 
                
            else:
                train_data_dealer=test_data_dealer
                
        elif args.dataset_class in ["select_image","select_image_resnet_text","select_image_v2tel"]: #4 review_text ,review_image_paths,review_attributes, review_special_product_image_path
            test_data_dealer=MultimodalEntityLinkingDataDealerSelectImage(test_txt_dir,args.review_image_dir,args.candidate_base,args.test_candidate_mode,max_candidate_num,args.dataset_class ,args.entity_text_source, args.product_image_dir)
            val_data_dealer=MultimodalEntityLinkingDataDealerSelectImage(val_txt_dir,args.review_image_dir,args.candidate_base,args.dev_candidate_mode,max_candidate_num,args.dataset_class ,args.entity_text_source, args.product_image_dir)    
            if mode not in ["test","dry_run"]:
                train_data_dealer=MultimodalEntityLinkingDataDealerSelectImage(train_txt_dir,args.review_image_dir,args.candidate_base,args.candidate_mode,args.train_max_candidate_num,args.dataset_class  ,args.entity_text_source,args.product_image_dir )#"by_gold" 
                
            else:
                train_data_dealer=test_data_dealer
                
        elif args.dataset_class in ["select_image_with_nli"]:#review_text ,review_image_paths,review_attributes,surface_form,review_special_product_image_path]
            test_data_dealer=MultimodalEntityLinkingDataDealerSurfaceFormSelectImage(test_txt_dir,args.review_image_dir,args.candidate_base,args.test_candidate_mode,max_candidate_num,args.dataset_class ,args.entity_text_source,args.product_image_dir  ,review_text_mode=args.review_text_mode)
            val_data_dealer=MultimodalEntityLinkingDataDealerSurfaceFormSelectImage(val_txt_dir,args.review_image_dir,args.candidate_base,args.dev_candidate_mode,max_candidate_num,args.dataset_class ,args.entity_text_source,args.product_image_dir ,review_text_mode=args.review_text_mode)
            if mode not in ["test","dry_run"]:
                train_data_dealer=MultimodalEntityLinkingDataDealerSurfaceFormSelectImage(train_txt_dir,args.review_image_dir,args.candidate_base,args.candidate_mode,args.train_max_candidate_num,args.dataset_class ,args.entity_text_source,args.product_image_dir  ,review_text_mode=args.review_text_mode )#"by_gold" 
                
            else:
                train_data_dealer=test_data_dealer
                
        elif args.dataset_class   in ["select_image_with_nli_score_and_retrieval_score"]     :#review_text ,review_image_paths,review_attributes,surface_form,review_special_product_image_path,nli_score_vector_dict,text_image_retrieval_score_list_dict
            test_data_dealer=MultimodalEntityLinkingDataDealerSurfaceFormSelectImageWithNLIRetrievalScore(test_txt_dir,args.review_image_dir,args.candidate_base,args.test_candidate_mode,max_candidate_num,args.dataset_class ,args.entity_text_source,args.product_image_dir ,args.path_to_all_attribute_key_list,args.empty_nli_score,args.use_nli_score_num,args.allow_empty_nli)
            val_data_dealer=MultimodalEntityLinkingDataDealerSurfaceFormSelectImageWithNLIRetrievalScore(val_txt_dir,args.review_image_dir,args.candidate_base,args.dev_candidate_mode,max_candidate_num,args.dataset_class ,args.entity_text_source,args.product_image_dir,args.path_to_all_attribute_key_list ,args.empty_nli_score,args.use_nli_score_num,args.allow_empty_nli)
            if mode not in ["test","dry_run"]:
                train_data_dealer=MultimodalEntityLinkingDataDealerSurfaceFormSelectImageWithNLIRetrievalScore(train_txt_dir,args.review_image_dir,args.candidate_base,args.candidate_mode,args.train_max_candidate_num,args.dataset_class ,args.entity_text_source,args.product_image_dir  ,args.path_to_all_attribute_key_list,args.empty_nli_score,args.use_nli_score_num,args.allow_empty_nli)#"by_gold" 
                
            else:
                train_data_dealer=test_data_dealer

        else:#4 review_text ,review_image_paths,review_attributes, surface_form
            
            test_data_dealer=MultimodalEntityLinkingDataDealerSurfaceForm(test_txt_dir,args.review_image_dir,args.candidate_base,args.test_candidate_mode,max_candidate_num,args.dataset_class ,args.entity_text_source)
            val_data_dealer=MultimodalEntityLinkingDataDealerSurfaceForm(val_txt_dir,args.review_image_dir,args.candidate_base,args.dev_candidate_mode,max_candidate_num,args.dataset_class ,args.entity_text_source)
            if mode not in ["test","dry_run"]:
                train_data_dealer=MultimodalEntityLinkingDataDealerSurfaceForm(train_txt_dir,args.review_image_dir,args.candidate_base,args.candidate_mode,args.train_max_candidate_num,args.dataset_class ,args.entity_text_source )#"by_gold" 
                
            else:
                train_data_dealer=test_data_dealer
                
        corpus,product_json_array,product_json_dict=train_data_dealer.load_corpus(args.products_path_str ,args.product_image_dir)  
        test_queries=get_queries(test_txt_dir,test_data_dealer,product_json_array,product_json_dict,max_candidate_num,args.dataset_class)
        val_queries=get_queries(val_txt_dir,val_data_dealer,product_json_array,product_json_dict,max_candidate_num,args.dataset_class)
        if mode not in ["test","dry_run"]:
            train_queries=get_queries(train_txt_dir,train_data_dealer,product_json_array,product_json_dict,args.train_max_candidate_num_in_corpus,args.dataset_class)
            
        else:
            train_queries=test_queries
            
    else:
        data_dealer=WikidiverseDataDealer()
        corpus=data_dealer.load_corpus(train_txt_dir)
        train_queries=get_queries(train_txt_dir,data_dealer)
        val_queries=get_queries(val_txt_dir,data_dealer)
        test_queries=get_queries(test_txt_dir,data_dealer)
    
    # processor.size=256
    # processor.crop_size=256
    if args.dataset_class in ["v2"]:
        processor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32") 
        train_dataset = MultimodalDatasetBERTCLIP(train_queries ,corpus,tokenizer,max_len,processor,max_candidate_num,args.use_attributes,args.use_image,args.candidate_mode,args.max_attribute_num,is_train=True)
        
        val_dataset = MultimodalDatasetBERTCLIP(val_queries ,corpus,tokenizer,max_len,processor,max_candidate_num,args.use_attributes,args.use_image,args.dev_candidate_mode,args.max_attribute_num,is_train=False)
        
        test_dataset = MultimodalDatasetBERTCLIP(test_queries ,corpus,tokenizer,max_len,processor,max_candidate_num,args.use_attributes,args.use_image,args.test_candidate_mode,args.max_attribute_num,is_train=False) 
    elif args.dataset_class in ["select_image_sbert_attribute"]:
        processor=model_special_processor
        train_dataset = MultimodalDatasetSBERTBERTCLIPAttribute(train_queries ,corpus,tokenizer,max_len,processor,args.train_max_candidate_num ,args.use_attributes,args.use_image,args.candidate_mode,args.max_attribute_num,is_train=True,is_random_positive=args.is_random_positive,is_train_many_negative_per_time=args.is_train_many_negative_per_time)
        
        val_dataset = MultimodalDatasetSBERTBERTCLIPAttribute(val_queries ,corpus,tokenizer,max_len,processor,max_candidate_num,args.use_attributes,args.use_image,args.dev_candidate_mode,args.max_attribute_num,is_train=False,is_random_positive=True,is_train_many_negative_per_time=True)
        
        test_dataset = MultimodalDatasetSBERTBERTCLIPAttribute(test_queries ,corpus,tokenizer,max_len,processor,max_candidate_num,args.use_attributes,args.use_image,args.test_candidate_mode,args.max_attribute_num,is_train=False,is_random_positive=True,is_train_many_negative_per_time=True) 
    elif args.dataset_class=="v3":
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        train_dataset = MultimodalDatasetCLIP(train_queries ,corpus,tokenizer,max_len,processor,args.train_max_candidate_num,args.use_attributes,args.use_image,args.candidate_mode,args.max_attribute_num)
        
        val_dataset = MultimodalDatasetCLIP(val_queries ,corpus,tokenizer,max_len,processor,max_candidate_num,args.use_attributes,args.use_image,args.dev_candidate_mode,args.max_attribute_num)
        
        test_dataset = MultimodalDatasetCLIP(test_queries ,corpus,tokenizer,max_len,processor,max_candidate_num,args.use_attributes,args.use_image,args.test_candidate_mode,args.max_attribute_num) 
    
    elif args.dataset_class in[  "select_image"]:
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        train_dataset = MultimodalDatasetSelectImage(train_queries ,corpus,tokenizer,max_len,processor,args.train_max_candidate_num,args.use_attributes,args.use_image,args.candidate_mode,args.max_attribute_num,mode=args.mode)
        
        val_dataset = MultimodalDatasetSelectImage(val_queries ,corpus,tokenizer,max_len,processor,max_candidate_num,args.use_attributes,args.use_image,args.dev_candidate_mode,args.max_attribute_num,mode=args.mode)
        
        test_dataset = MultimodalDatasetSelectImage(test_queries ,corpus,tokenizer,max_len,processor,max_candidate_num,args.use_attributes,args.use_image,args.test_candidate_mode,args.max_attribute_num,mode=args.mode) 
    elif  args.dataset_class in ["select_image_flava"]:
        processor = AutoProcessor.from_pretrained("facebook/flava-full")
        train_dataset = MultimodalDatasetSelectImageFLAVA(train_queries ,corpus,tokenizer,max_len,processor,args.train_max_candidate_num,args.use_attributes,args.use_image,args.candidate_mode,args.max_attribute_num)
        
        val_dataset = MultimodalDatasetSelectImageFLAVA(val_queries ,corpus,tokenizer,max_len,processor,max_candidate_num,args.use_attributes,args.use_image,args.dev_candidate_mode,args.max_attribute_num)
        
        test_dataset = MultimodalDatasetSelectImageFLAVA(test_queries ,corpus,tokenizer,max_len,processor,max_candidate_num,args.use_attributes,args.use_image,args.test_candidate_mode,args.max_attribute_num) 
    elif args.dataset_class in[  "select_image_v2tel"]:
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        train_dataset = MultimodalDatasetSelectImageV2TEL(train_queries ,corpus,tokenizer,max_len,processor,args.train_max_candidate_num,args.use_attributes,args.use_image,args.candidate_mode,args.max_attribute_num)
        
        val_dataset = MultimodalDatasetSelectImageV2TEL(val_queries ,corpus,tokenizer,max_len,processor,max_candidate_num,args.use_attributes,args.use_image,args.dev_candidate_mode,args.max_attribute_num)
        
        test_dataset = MultimodalDatasetSelectImageV2TEL(test_queries ,corpus,tokenizer,max_len,processor,max_candidate_num,args.use_attributes,args.use_image,args.test_candidate_mode,args.max_attribute_num) 
    elif args.dataset_class=="v4":
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        train_dataset = MultimodalDatasetAttenBasedMultiImage(train_queries ,corpus,tokenizer,max_len,processor,args.train_max_candidate_num,args.use_attributes,args.use_image,args.candidate_mode,args.max_attribute_num)
        
        val_dataset = MultimodalDatasetAttenBasedMultiImage(val_queries ,corpus,tokenizer,max_len,processor,max_candidate_num,args.use_attributes,args.use_image,args.dev_candidate_mode,args.max_attribute_num)
        
        test_dataset = MultimodalDatasetAttenBasedMultiImage(test_queries ,corpus,tokenizer,max_len,processor,max_candidate_num,args.use_attributes,args.use_image,args.test_candidate_mode,args.max_attribute_num) 
    elif args.dataset_class in[ "v5" ]:
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        train_dataset = MultimodalDatasetQueryBasedMultiImage(train_queries ,corpus,tokenizer,max_len,processor,args.train_max_candidate_num,args.use_attributes,args.use_image,args.candidate_mode,args.max_attribute_num)
        
        val_dataset = MultimodalDatasetQueryBasedMultiImage(val_queries ,corpus,tokenizer,max_len,processor,max_candidate_num,args.use_attributes,args.use_image,args.dev_candidate_mode,args.max_attribute_num)
        
        test_dataset = MultimodalDatasetQueryBasedMultiImage(test_queries ,corpus,tokenizer,max_len,processor,max_candidate_num,args.use_attributes,args.use_image,args.test_candidate_mode,args.max_attribute_num) 
    elif args.dataset_class=="v6":
        processor=None
        train_dataset = MultimodalDatasetQueryBasedBERT(train_queries ,corpus,tokenizer,max_len,processor,args.train_max_candidate_num,args.use_attributes,args.use_image,args.candidate_mode,args,args.max_attribute_num)
        
        val_dataset = MultimodalDatasetQueryBasedBERT(val_queries ,corpus,tokenizer,max_len,processor,max_candidate_num,args.use_attributes,args.use_image,args.dev_candidate_mode,args,args.max_attribute_num)
        
        test_dataset = MultimodalDatasetQueryBasedBERT(test_queries ,corpus,tokenizer,max_len,processor,max_candidate_num,args.use_attributes,args.use_image,args.test_candidate_mode,args,args.max_attribute_num) 

    elif args.dataset_class in[  "v7"]:
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        train_dataset = MultimodalDatasetMultiImageCrossEncoder(train_queries ,corpus,tokenizer,max_len,processor,args.train_max_candidate_num,args.use_attributes,args.use_image,args.candidate_mode,args.max_attribute_num)
        
        val_dataset = MultimodalDatasetMultiImageCrossEncoder(val_queries ,corpus,tokenizer,max_len,processor,max_candidate_num,args.use_attributes,args.use_image,args.dev_candidate_mode,args.max_attribute_num)
        
        test_dataset = MultimodalDatasetMultiImageCrossEncoder(test_queries ,corpus,tokenizer,max_len,processor,max_candidate_num,args.use_attributes,args.use_image,args.test_candidate_mode,args.max_attribute_num) 
    # elif args.dataset_class in[  "v8"]:
    #     processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    #     train_dataset = MultimodalDataset8(train_queries ,corpus,tokenizer,max_len,processor,args.train_max_candidate_num,args.use_attributes,args.use_image,args.candidate_mode)
        
    #     val_dataset = MultimodalDataset8(val_queries ,corpus,tokenizer,max_len,processor,max_candidate_num,args.use_attributes,args.use_image,args.dev_candidate_mode)
        
    #     test_dataset = MultimodalDataset8(test_queries ,corpus,tokenizer,max_len,processor,max_candidate_num,args.use_attributes,args.use_image,args.test_candidate_mode) 
    elif args.dataset_class in[  "v9"]:
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        train_dataset = ReviewAttributePairDataset(train_queries ,corpus,tokenizer,max_len,processor,args.train_max_candidate_num,args.use_attributes,args.use_image,args.candidate_mode,args.max_attribute_num)
        
        val_dataset = ReviewAttributePairDataset(val_queries ,corpus,tokenizer,max_len,processor,max_candidate_num,args.use_attributes,args.use_image,args.dev_candidate_mode,args.max_attribute_num)
        
        test_dataset = ReviewAttributePairDataset(test_queries ,corpus,tokenizer,max_len,processor,max_candidate_num,args.use_attributes,args.use_image,args.test_candidate_mode,args.max_attribute_num) 
    elif args.dataset_class in[  "v10"]:
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        train_dataset = ReviewAttributePairWithImageDataset(train_queries ,corpus,tokenizer,max_len,processor,args.train_max_candidate_num,args.use_attributes,args.use_image,args.candidate_mode,args.max_attribute_num)
        
        val_dataset = ReviewAttributePairWithImageDataset(val_queries ,corpus,tokenizer,max_len,processor,max_candidate_num,args.use_attributes,args.use_image,args.dev_candidate_mode,args.max_attribute_num)
        
        test_dataset = ReviewAttributePairWithImageDataset(test_queries ,corpus,tokenizer,max_len,processor,max_candidate_num,args.use_attributes,args.use_image,args.test_candidate_mode,args.max_attribute_num) 
    elif args.dataset_class in[  "select_image_with_nli"]:
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        train_dataset = ReviewAttributePairWithImageDatasetSelectImage(train_queries ,corpus,tokenizer,max_len,processor,args.train_max_candidate_num,args.use_attributes,args.use_image,args.candidate_mode,args.max_attribute_num,mode=args.mode)
        
        val_dataset = ReviewAttributePairWithImageDatasetSelectImage(val_queries ,corpus,tokenizer,max_len,processor,max_candidate_num,args.use_attributes,args.use_image,args.dev_candidate_mode,args.max_attribute_num,mode=args.mode)
        
        test_dataset = ReviewAttributePairWithImageDatasetSelectImage(test_queries ,corpus,tokenizer,max_len,processor,max_candidate_num,args.use_attributes,args.use_image,args.test_candidate_mode,args.max_attribute_num,mode=args.mode)     
    elif args.dataset_class=="resnet":
        processor=None
        train_dataset = MultimodalDataset_resnet(train_queries ,corpus,tokenizer,max_len,processor,args.train_max_candidate_num,args.use_attributes,args.use_image,args.candidate_mode,args.max_attribute_num)
        
        val_dataset = MultimodalDataset_resnet(val_queries ,corpus,tokenizer,max_len,processor,max_candidate_num,args.use_attributes,args.use_image,args.dev_candidate_mode,args.max_attribute_num)
        
        test_dataset = MultimodalDataset_resnet(test_queries ,corpus,tokenizer,max_len,processor,max_candidate_num,args.use_attributes,args.use_image,args.test_candidate_mode,args.max_attribute_num)  
    elif args.dataset_class in ["resnet_text"]:
        processor=None
        train_dataset = MultimodalDatasetBERTResnet(train_queries ,corpus,tokenizer,max_len,processor,max_candidate_num,args.use_attributes,args.use_image,args.candidate_mode,args.max_attribute_num)
        
        val_dataset = MultimodalDatasetBERTResnet(val_queries ,corpus,tokenizer,max_len,processor,max_candidate_num,args.use_attributes,args.use_image,args.dev_candidate_mode,args.max_attribute_num)
        
        test_dataset = MultimodalDatasetBERTResnet(test_queries ,corpus,tokenizer,max_len,processor,max_candidate_num,args.use_attributes,args.use_image,args.test_candidate_mode,args.max_attribute_num) 
    elif args.dataset_class in ["select_image_resnet_text"]:
        processor=None
        train_dataset = MultimodalDatasetSelectImageBERTResnet(train_queries ,corpus,tokenizer,max_len,processor,max_candidate_num,args.use_attributes,args.use_image,args.candidate_mode,args.max_attribute_num)
        
        val_dataset = MultimodalDatasetSelectImageBERTResnet(val_queries ,corpus,tokenizer,max_len,processor,max_candidate_num,args.use_attributes,args.use_image,args.dev_candidate_mode,args.max_attribute_num)
        
        test_dataset = MultimodalDatasetSelectImageBERTResnet(test_queries ,corpus,tokenizer,max_len,processor,max_candidate_num,args.use_attributes,args.use_image,args.test_candidate_mode,args.max_attribute_num) 
    elif args.dataset_class in ["select_image_with_nli_score_and_retrieval_score"]:
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        train_dataset = ReviewAttributePairWithImageDatasetSelectImageAndRetrievalNLIScore(train_queries ,corpus,tokenizer,max_len,processor,args.train_max_candidate_num,args.use_attributes,args.use_image,args.candidate_mode,args.max_attribute_num,mode=args.mode)
        
        val_dataset = ReviewAttributePairWithImageDatasetSelectImageAndRetrievalNLIScore(val_queries ,corpus,tokenizer,max_len,processor,max_candidate_num,args.use_attributes,args.use_image,args.dev_candidate_mode,args.max_attribute_num,mode=args.mode)
        
        test_dataset = ReviewAttributePairWithImageDatasetSelectImageAndRetrievalNLIScore(test_queries ,corpus,tokenizer,max_len,processor,max_candidate_num,args.use_attributes,args.use_image,args.test_candidate_mode,args.max_attribute_num,mode=args.mode)     
    elif args.dataset_class in ["text_cross_encoder"]:
        processor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32") 
        train_dataset = MultimodalDatasetBERTCrossEncoder(train_queries ,corpus,tokenizer,max_len,processor,max_candidate_num,args.use_attributes,args.use_image,args.candidate_mode,args.max_attribute_num,is_train=True)
        
        val_dataset = MultimodalDatasetBERTCrossEncoder(val_queries ,corpus,tokenizer,max_len,processor,max_candidate_num,args.use_attributes,args.use_image,args.dev_candidate_mode,args.max_attribute_num,is_train=False)
        
        test_dataset = MultimodalDatasetBERTCrossEncoder(test_queries ,corpus,tokenizer,max_len,processor,max_candidate_num,args.use_attributes,args.use_image,args.test_candidate_mode,args.max_attribute_num,is_train=False) 
    else:
        print("wrong dataset")
        exit(0)
    if args.num_worker is not None:
        num_worker=args.num_worker
    else:
        num_worker=5
    if    args.parallel=="ddp":
        
        train_loader=prepare_for_ddp(rank, world_size,train_dataset,args.batch_size,num_workers=num_worker)
        val_loader=prepare_for_ddp(rank, world_size,val_dataset,args.batch_size,num_workers=num_worker)
        test_loader=prepare_for_ddp(rank, world_size,test_dataset,args.batch_size,num_workers=num_worker)
    else:
        ## We call the dataloader class
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            pin_memory=True,
            num_workers=num_worker,#batch_size//2
            shuffle=True,
            drop_last=True
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=num_worker,
            shuffle=False,
            drop_last=True
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=num_worker,
            shuffle=False,
            drop_last=True
        )

    dataloaders = {'Train': train_loader, 'Test': test_loader, 'Val': val_loader}
    return dataloaders 
      

def load_wikidiverse_data(dataset_dir, entity_dir, image_dir):
    entity_dict=read_entity(entity_dir)
    mention_dict=read_mention(dataset_dir,entity_dict)
    
    
    dataset=WikidiverseDataset(mention_dict)
   
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)  
    return train_dataloader,entity_dict


    
class WikidiverseDataset(Dataset):
    def __init__(self, mention_dict):
         
        self.mention_dict=mention_dict
        

    def __len__(self):
        return len(self.mention_dict)

    def __getitem__(self, idx):
        
        keys_list = list(self.mention_dict)
        key = keys_list[idx]
        news=self.mention_dict[key]
        mention=news.mention
   
        img_path=news.img_path
        entity_wiki_name=news.entity_wiki_name
        entity_candidates=news.entity_candidates
        entity_candidate_name_list=news.entity_candidate_name_list
        
 
        return mention, img_path,entity_wiki_name ,entity_candidate_name_list
 
    
def read_entity(pickle_path):  
    
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as handle:
            entity_dict = pickle.load(handle)
    else:
        entity_dict={}
        corpus_path='/home/menglong/workspace/code/referred/wikiDiverse/data/wikipedia_entity2imgs_with_path.csv' 
        df_news = pd.read_csv(corpus_path ,encoding="utf8")
        for _,row in tqdm(df_news.iterrows()):
            entity_name=row['entity']
            if  not pd.isna(entity_name):
                img_url=row[ 'wiki_img' ]
                img_path=row[ 'path' ]
                is_img_downloaded=row[ 'downloaded']
                entity=Entity(entity_name,None,None,img_url,img_path,is_img_downloaded,None)
                entity_dict[entity_name]=entity
            
    
        with open(pickle_path, 'wb') as handle:
            pickle.dump(entity_dict, handle )
    return entity_dict
        


def change_hard_code_directory_in_pickle( ):
    pickle_path="/home/menglong/workspace/code/referred/wikiDiverse/data/cleaned/wikipedia_entity.pickle"
    pickle_without_hard_code_dir_path="/home/menglong/workspace/code/referred/wikiDiverse/data/wikidiverse_cleaned/wikipedia_entity.pickle"
    new_entity_dict={}
    with open(pickle_path, 'rb') as handle:
        entity_dict = pickle.load(handle)
        for entity_name,entity in tqdm(entity_dict.items()):
            img_path_list = entity.img_path_list.split("[AND]")
            new_image_path_list_str=""
            for image_path in img_path_list:
                image_path=image_path.replace("/home/menglong/workspace/code/referred/wikiDiverse/data/","data/wikidiverse_cleaned/")
                new_image_path_list_str+="[AND]"+image_path
            entity.img_path_list=new_image_path_list_str[5:]
            new_entity_dict[entity_name]=entity
    with open(pickle_without_hard_code_dir_path, 'wb') as handle:
        pickle.dump(new_entity_dict, handle )
        
