import pandas as pd  
import json

from nltk.tokenize import sent_tokenize, word_tokenize
from pathlib import Path
from time import time
from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm
import spacy 
from bestbuy.src.organize.merge_attribute import json_to_dict
from sentence_transformers.cross_encoder import CrossEncoder
import numpy as np
from spacy.matcher import PhraseMatcher
from nltk.tokenize import sent_tokenize, word_tokenize 
import nltk 
nlp = spacy.load("en_core_web_lg")
#"Quote",
    #   ":",
    #   "Originally",
#<s> <p>

#I come from [Kathmandu valley,](location) [Nepal](location)
#Refurbished Product: Refurbished products
nlp = spacy.load("en_core_web_lg")
from spacy.matcher import Matcher 
# phrase_matcher = PhraseMatcher(nlp.vocab)

# def phrase_match(term_text,review) :
    
#     terms = [term_text]
#     # Only run nlp.make_doc to speed things up
#     patterns = [nlp.make_doc(text) for text in terms]
#     phrase_matcher.add("TerminologyList", patterns)

#     review_doc = nlp(review)
#     matches = phrase_matcher(review_doc)  
#     for match_id, start, end in matches:
#         span = review_doc[start:end]
#         return span.text ,True 
#     return None ,False 

phrase_matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
matcher = Matcher(nlp.vocab)

def gen_pattern():
    pattern = [{"LOWER": "hello"}, {"IS_PUNCT": True}, {"LOWER": "world"}]

def match_spacy(product_context,review_text):
    # review_doc=nlp(review_text )
    lower_review_doc=nlp(review_text )#.lower()
    # lower_product_context_doc=nlp(product_context.lower())
    
    phrase_matcher.add("CAT", None, nlp(product_context))
    matches = phrase_matcher(lower_review_doc)
    
 
    # for token in lower_product_context_doc:
    #     # pattern.append({"NORM":product_context})
    #     # product_context_lemm= [token.lemma_.lower() ] 
    #     pattern.append({"NORM":token.norm_  })
    
    # # product_context_list=product_context.split(" ")
    # # pattern=[]
    # # for product_context in product_context_list:
        
   
    # matcher.add("HelloWorld", [pattern])
    # matches = matcher(lower_review_doc)
    
    for match_id, start, end in matches:
        string_id = nlp.vocab.strings[match_id]  # Get string representation
        span = lower_review_doc[start:end]  # The matched span
        # print(match_id, string_id, start, end, span.text)
        return span.text ,True 
    # match_lower_version_in_review(review_doc,span)
        
    return None ,False 


def match_lower_version_in_review(review_doc,span):
    pattern =[]
 
    for token in span:
        # pattern.append({"NORM":product_context})
        # product_context_lemm= [token.lemma_.lower() ] 
        pattern.append({"LOWER":token.lower_ })
    
    # product_context_list=product_context.split(" ")
    # pattern=[]
    # for product_context in product_context_list:
        
   
    matcher.add("HelloWorld", [pattern])
    matches = matcher(review_doc)
    
    for match_id, start, end in matches:
        string_id = nlp.vocab.strings[match_id]  # Get string representation
        span = review_doc[start:end]  # The matched span
        # print(match_id, string_id, start, end, span.text)
        
        return span.text,True 
    return None ,False 
  
def is_base_form_same(product_alias,gt_mention,gt_mention_doc=None,gt_mention_lemm=None,is_check_substring=True  ):
    mention_doc = nlp(product_alias)
    if gt_mention_doc is None:
        gt_mention_doc = nlp(gt_mention)
    product_alias_lemm= [token.lemma_.lower() for token in mention_doc] 
    if gt_mention_lemm is None:
        gt_mention_lemm= [token.lemma_.lower() for token in gt_mention_doc] 
    if product_alias_lemm==gt_mention_lemm:
        return True 
    elif len(product_alias_lemm)!=len(gt_mention_lemm):
        return False 
    else:
        if is_check_substring:
            return  is_words_same(product_alias_lemm,mention_doc,gt_mention_lemm,gt_mention_doc)
        else:
            return False 
        
        
def is_word_same(mention_lemm,mention_doc,gt_mention_lemm,gt_mention_doc)  :   
    if mention_doc.pos_!=gt_mention_doc.pos_:
        if not ( mention_doc.pos_ in ["PROPN","NOUN"] and  gt_mention_doc.pos_ in ["PROPN","NOUN"]):
            return False 
    is_found=mention_lemm.find(gt_mention_lemm) 
    is_reverse_found=gt_mention_lemm.find(mention_lemm) 
    if  is_found==-1 and is_reverse_found==-1:
        return False
    else:
        return True 
def is_words_same(mention_lemm,mention_doc,gt_mention_lemm,gt_mention_doc):
    for i in range(len(mention_lemm)):
        if not is_word_same(mention_lemm[i],mention_doc[i],gt_mention_lemm[i],gt_mention_doc[i]):
            return False 
    return True     
class ProductNameCandidate:
    def __init__(self,root) -> None:
        self.root=root 
        self.candidate_text_list=[ ]
        
    def add(self,candidate_text):
        self.candidate_text_list.append(candidate_text)
def compare_two_equal_len_texts_w_base_form(mention,gt_mention):
    if mention is None:
        if gt_mention==""  :
            return True 
        else:
            return False 
    else:
        if is_base_form_same(mention,gt_mention):
            return True 
        else:
            return False    
        
def match_s(lower_product_contect,lower_review,long_sentence):
    lower_product_contect_s=lower_product_contect+"s"
    find_idx=lower_review.find(lower_product_contect_s)        
    if find_idx!=-1:
        return long_sentence[find_idx:find_idx+len(lower_product_contect_s)],True   
    return None,False 
        
def search_query_in_long_sentence_w_base_form(long_sentence,short_query):
    lower_review=long_sentence.lower()
    lower_product_contect=short_query.lower()
    is_w_sub_string=False 
    is_found=False 
    span=""
    # span,is_found=match_spacy(short_query,long_sentence)
    # span,is_found=phrase_match(short_query,long_sentence)
    # span,is_found=match_s(lower_product_contect,lower_review,long_sentence)
    if is_found:
        return span ,True 
    else:
        find_idx=lower_review.find(lower_product_contect)
        if find_idx!=-1:
            return long_sentence[find_idx:find_idx+len(short_query)],True    
        else:
            if is_w_sub_string:
                matched_mention,is_found = longest_match_base_form(long_sentence,short_query) 
            else:
                matched_mention,is_found =longest_match_base_form_without_substring(long_sentence,short_query) 
            if is_found:
                return matched_mention,is_found
        return  None ,False  
        
   
def longest_match_base_form_without_substring(long_sentence,short_query)  :
    review_doc = nlp(long_sentence)
    product_context_doc = nlp(short_query)
    product_context_doc_len=len(product_context_doc)
    product_context_lemm_span=product_context_doc[:].lemma_.lower()
    # product_context_lemm_span=[token.lemma_.lower() for token in product_context_doc] 
    for i in range(0,len(review_doc)-product_context_doc_len+1):
         
        review_lemm_span=review_doc[i:product_context_doc_len+i].lemma_.lower()
        if  product_context_lemm_span==nlp(review_lemm_span)[:].lemma_.lower():
            return review_doc[i:product_context_doc_len+i].text ,True 
         
    return None ,False

def longest_match_base_form(long_sentence,short_query)  :
    review_doc = nlp(long_sentence)
    product_context_doc = nlp(short_query)
    product_context_doc_len=len(product_context_doc)
    # product_context_doc[:].lemma_.lower()
    product_context_lemm_span=[token.lemma_.lower() for token in product_context_doc] 
    for i in range(0,len(review_doc)-product_context_doc_len+1):
        if is_base_form_same(review_doc[i:product_context_doc_len+i].text,short_query,product_context_doc,product_context_lemm_span):
        # if compare_two_equal_len_texts_w_base_form(review_doc[i:product_context_doc_len+i].text, short_query): 
            return review_doc[i:product_context_doc_len+i].text,True 
        # review_lemm_span=review_doc[i:product_context_doc_len+i].lemma_.lower()
        # if  product_context_lemm_span==nlp(review_lemm_span)[:].lemma_.lower():
        #     return review_doc[i:product_context_doc_len+i].text ,True 
         
    return None ,False  
    
            
def find_longest_match( review,product_name_candidate_object,doc):
    
    
    
    for product_context in product_name_candidate_object.candidate_text_list:
        matched_mention,is_found=search_query_in_long_sentence_w_base_form(review,product_context)
         
        if is_found:
            return matched_mention,is_found
    return None ,False 

def is_nan(text):
    if text is None or text =="":
        return True 
    else:
        return False 
def is_prediction_right(mention,gt_mention,gt_mention_list):
    if is_nan(mention)  :
        if is_nan(gt_mention)   :
            return True 
        else:
            return False 
    else:
        for gt_mention in gt_mention_list:
            is_same=is_base_form_same(gt_mention,mention,is_check_substring=False)
            if is_same:
            # if    gt_mention.lower()==mention.lower():
                return True 
        return False
            
        # if    gt_mention.lower()==mention.lower():
       
        #     return True 
        # elif mention in gt_mention_list:
        #     return True 
       
        # else:
        #     return False
        
import logging        
def gen_root_by_doc(doc):
    root=""
    is_found=False 
    for token in doc:
        if token.dep_=="ROOT":
            
            if is_found:
                # logging.info(f"ERROR: the previous root: {root}, the new root:{token.text}, {doc.text}") 
                pass 
                
            else:
                is_found=True 
            root= token.text 
    return root 

def gen_root_chunk_by_doc(doc):
     
    root_chunk=""
    root=gen_root_by_doc(doc)
    
    if root !="":
        for chunk in doc.noun_chunks:
            if chunk.root.text==root:
                root_chunk=chunk.text
                
    return root_chunk    
    

    
def prepare_produt(review_json,is_product_desc_first_sent=True,is_product_category_last_term=False ,product_title_range=None ):
    product_category=review_json["product_category"]
    if is_product_category_last_term:
        product_category_list=product_category.split(" -> ")
        product_category=product_category_list[-1]
    
    else:
        product_category=". ".join(product_category.split(" -> ")[1:])
    noisy_product_name=review_json["product_name"]
    noisy_product_name_list=noisy_product_name.split(" - ")
    if len(noisy_product_name_list)>=3:
        if product_title_range=="no_brand":
            noisy_product_name=   noisy_product_name_list[1]
        else:
            noisy_product_name=" ".join(noisy_product_name_list[:-1])
    # elif len(noisy_product_name_list)==2:
    #     noisy_product_name=" ".join(noisy_product_name_list)
    # elif len(noisy_product_name_list)==2:
    #     if product_title_range=="no_brand":
    #         noisy_product_name=   noisy_product_name_list[1]
    #     else:
    #         noisy_product_name=" ".join(noisy_product_name_list)
    # else:
    #     noisy_product_name=" ".join(noisy_product_name_list)
        
    
        
    if "overview_section" in review_json and review_json["overview_section"] is not None and  "description" in review_json["overview_section"]:
        product_desc =review_json["overview_section"]["description"]
        if is_product_desc_first_sent:
            product_desc_sent_list=sent_tokenize(product_desc)
            if len(product_desc_sent_list)>0:
                
                product_desc=product_desc_sent_list[0]
                 
                    
            else:
                product_desc=""
    
            
    else:
        product_desc=""
    return product_category,noisy_product_name,product_desc
        
def convert_cprod1_to_rasa():
    #1, <s> <p> 
    #2, add annotation
    #3, join 
    df=pd.read_csv('bestbuy/data/cprod1/training-disambiguated-product-mentions.120725.csv')  

    with open('bestbuy/data/cprod1/training-annotated-text.json') as f:
        json_content= json.load(f)
        review_json_dict=json_content["TextItem"]
    review_list=[]
    for index, row in df.iterrows():
        id_range=row["id"]
        id,range=id_range.split(":")
        start,end=range.split("-")
        id_str=str(id)
        review_token_list=review_json_dict[id_str]
        review_token_list[int(start)]="["+review_token_list[int(start)]
        review_token_list[int(end)]=review_token_list[int(end)]+"](PRODUCT)"
        review=" ".join(review_token_list)
        review=review.replace("<s>","").replace("<P>","")
        review_list.append(review)
        # for token_id,review_token in enumerate(review_json_dict):
        #     if review_token not in ["<s>","<p>"]:
                
    return review_list
    
       
def filter_non_overlap():
    id_list=[438,664,747,773,776,806,835,957]
    review_json_path = Path(
        f'bestbuy/data/example/bestbuy_review_50.json'
    )        
    output_products_path = Path(
        f'bestbuy/data/example/bestbuy_review_50_v2.json'
    )  
    output_list=[] 
    
    with open(review_json_path, 'r', encoding='utf-8') as fp:
        review_json_array = json.load(fp)
        for idx,review_json   in tqdm(enumerate(review_json_array)):
            
            reviews= review_json["reviews"]
            review=reviews[0]
            review_text=review["body"]
            review_header=review["header"]
            gt_mention=review["gt_mention"]
            review_text=review_header+". "+review_text
            review_id=review["id"]
            if review_id  not in id_list:
                output_list.append(review_json)
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(output_list, fp, indent=4)
        
    
# nlp = spacy.load("en_core_web_lg")    
# is_base_form_same("Refurbished Product","Refurbished products")