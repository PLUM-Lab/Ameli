


from torch.utils.data import Dataset  
from torch.utils.data import DataLoader 

from urllib.parse import unquote
import pandas as pd
import os 
import hashlib
import re
import json
from tqdm import tqdm
import pickle

from retrieval.data_util.datapoint import Entity, Mention, gen_entity_name

def load_entity_linking_data(dataset_dir, entity_dir, image_dir):
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
            