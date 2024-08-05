import torch 
from torch.utils.data import Dataset 

class Entity:
    def __init__(self,id,image_path_list,text)  :
         
        self.id=id 
        self.image_path_list=image_path_list
        self.text=text 

class Query:
    def __init__(self,id,image_path_list,text,entity_candidate_name_list)  :
         
        self.id=id 
        self.image_path_list=image_path_list
        self.text=text 
        self.entity_candidate_id_list=entity_candidate_name_list

class QueryDataset(Dataset):
    def __init__(self, query_dict):
         
        self.query_dict=query_dict
        

    def __len__(self):
        return len(self.query_dict)

    def __getitem__(self, idx):
        
        keys_list = list(self.query_dict)
        key = keys_list[idx]
        news=self.query_dict[key]
        text=news.text 
        id=news.id
        image_path=news.image_path_list
         
        entity_candidate_id_list=news.entity_candidate_id_list
        
 
        return id,text, image_path, entity_candidate_id_list