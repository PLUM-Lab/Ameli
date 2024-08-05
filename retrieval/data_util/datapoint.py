"""
#In theory, it should be:
    claim
    truthfulness
    relevant_doc_list
        img_list
        text
        evidence_list
            img_list
            text_list
    claim_id
    snope_url
    ruling_article
    """
from urllib.parse import unquote
import pandas as pd 


def gen_entity_name(text):
    text = text.strip()
  
    text = text.replace('nhttps://en.wikipedia.org/wiki/', '').replace('hhttps://en.wikipedia.org/wiki/', ''). \
      replace(']https://en.wikipedia.org/wiki/', '').replace('https://en.wikipedia.org/wiki/', '').replace(
      'ttps://en.wikipedia.org/wiki/', '').replace("'","").replace('"','')
    text = unquote(text)
    return text  
    
    
import os     
class Entity:
    def __init__(self,name,entity_wiki_url,wiki_title, img_url, img_path,is_img_downloaded,text)  :
        self.entity_wiki_url=entity_wiki_url
        self.name=name 
        self.wiki_title=wiki_title
        self.img_url_list=img_url
        self.img_path_list=img_path
        self.is_img_downloaded_list=is_img_downloaded
        self.text=text 
        # image_path_list=[]
        # img_path_list_str=self.img_path
        # img_path_list=img_path_list_str.split("[AND]")
        # is_img_downloaded_list=self.is_img_downloaded.split("[AND]")
        # if is_img_downloaded=="yes" and os.path.exists(img_path_list[idx]):

class Mention:
    def __init__(self,mention,text,img_url, img_path,is_img_downloaded, entity_wiki_url ,entity_modality ,entity_candidates,entity_candidate_name_list) :
        self.mention=mention 
        self.text=text 
        self.img_path=img_path 
        self.img_url=img_url 
        self.entity_wiki_url=entity_wiki_url 
        self.entity_modality=entity_modality
        self.is_img_downloaded=is_img_downloaded
        self.entity_wiki_name=gen_entity_name(entity_wiki_url)
        self.entity_candidates=entity_candidates
        self.entity_candidate_name_list=entity_candidate_name_list
        

class News :
    """
    claim
    truthfulness
    relevant_doc_dict
        relevant_doc 
            img_list
            text
    evidence_dict
        img_list
        txt_list
    claim_id
    snope_url
    ruling_article
    """
    def __init__(self,claim_id,snopes_url,text_evidence,claim ,truthfulness,ruling_article,ruling_outline ) :
        self.claim=claim 
        self.truthfulness=truthfulness
        self.relevant_doc_dict={}
        self.evidence_dict={}
        self.evidence_dict["img_list"]=[]
        self.evidence_dict["txt_list"]=[]
        if not pd.isna(text_evidence) and len(text_evidence)>0:
            self.evidence_dict["txt_list"].append(text_evidence)
        self.claim_id=claim_id
        self.snopes_url=snopes_url
        self.ruling_article=ruling_article
        self.ruling_outline=ruling_outline
        


    def add_text_evidence(self,text_evidence):
        if not pd.isna(text_evidence) and len(text_evidence)>0:
            self.evidence_dict["txt_list"].append(text_evidence)

    def add_img_evidence(self,img_evidence):
        self.evidence_dict["img_list"].append(img_evidence)

    def add_relevant_doc(self,relevant_doc_text,relevant_doc_id):
        relevant_doc={}
        relevant_doc["text"]=relevant_doc_text
        relevant_doc["img_list"]=[]
        self.relevant_doc_dict[relevant_doc_id]=relevant_doc

    def add_relevant_doc_img(self,relevant_doc_img,relevant_doc_id):
        if relevant_doc_id in self.relevant_doc_dict:
            relevant_doc=self.relevant_doc_dict[relevant_doc_id]
            relevant_doc["img_list"].append(relevant_doc_img)
        else:
            relevant_doc={}
            relevant_doc["text"]=""
            relevant_doc["img_list"]=[relevant_doc_img]
            self.relevant_doc_dict[relevant_doc_id]=relevant_doc

    def get_text_evidence_list(self):
        return self.evidence_dict["txt_list"]

    def get_img_evidence_list(self):
        return self.evidence_dict["img_list"]