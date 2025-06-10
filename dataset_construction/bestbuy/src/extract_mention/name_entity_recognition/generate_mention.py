
import spacy
from tqdm import tqdm
from bestbuy.src.extract_mention.dependency_parser.parser import gen_mention_candidates, gen_product_name_candidates 
from bestbuy.src.extract_mention.name_entity_recognition.train.inference import inference_one
from bestbuy.src.extract_mention.name_entity_recognition.train.train import gen_model
from  bestbuy.src.extract_mention.name_entity_recognition.ner_bert import ner_bert
from  bestbuy.src.extract_mention.name_entity_recognition.ner_spacy import ner_spacy
from  bestbuy.src.extract_mention.name_entity_recognition.token_similarity import find_most_similar_word
from bestbuy.src.extract_mention.name_entity_recognition.util.data_util import gen_root_chunk_by_doc
from bestbuy.src.extract_mention.name_entity_recognition.util.overlap import gen_overlap_mention, gen_overlap_mention_candidates
from  bestbuy.src.extract_mention.pos.pos_spacy import pos_spacy

from nltk.tokenize import sent_tokenize, word_tokenize
from pathlib import Path
from time import time
from pathlib import Path
import json
import pandas as pd
import logging
from bestbuy.src.organize.merge_attribute import json_to_dict
from sentence_transformers.cross_encoder import CrossEncoder
import numpy as np

from nltk.tokenize import sent_tokenize, word_tokenize
nlp = spacy.load("en_core_web_lg")

class MentionGenerator:
    def __init__(self,is_debug,similarity_threshold,similarity_target,has_model=True)  :
        
        self.is_debug=is_debug 
        self.similarity_threshold=similarity_threshold
        self.similarity_target=similarity_target
        self.has_model=has_model
        if has_model:
            self.model = CrossEncoder('cross-encoder/stsb-roberta-large') 
      
 
    def find_most_similar_word(self,query,corpus,gt_mention):
        
        # So we create the respective sentence combinations
        sentence_combinations = [[query, corpus_sentence] for corpus_sentence in corpus]

        # Compute the similarity scores for these combinations
        similarity_scores = self.model.predict(sentence_combinations)

        # Sort the scores in decreasing order
        sorted_order=np.argsort(similarity_scores)
        sim_scores_argsort = reversed(sorted_order)

        # Print the scores
        if self.is_debug   :
            # print()
            logging.debug(f"Query:{query}. Gold: {gt_mention}")
        is_first=True  
        choose_text=""
        for idx in sim_scores_argsort:
            if is_first:
                if similarity_scores[idx]>=self.similarity_threshold:
                    choose_text=corpus[idx]
                    is_first=False 
                
            if self.is_debug   :
                logging.debug("{:.2f}\t{}".format(similarity_scores[idx], corpus[idx]))
                # print()
                    
            
         
        return choose_text
    def _gen_score(self,corpus,query):
        sentence_combinations = [[query, corpus_sentence] for corpus_sentence in corpus]

        # Compute the similarity scores for these combinations
        similarity_scores = self.model.predict(sentence_combinations,show_progress_bar=False)
        return similarity_scores,sentence_combinations
    def find_most_similar_word_merge2(self,product_category,noisy_product_name,corpus,gt_mention):
        scores={i:{"mention":example} for i,example in enumerate(corpus)}
        product_title_chunk=find_chunk_product_title(noisy_product_name)
        product_category_chunk=find_chunk_product_category(product_category)
        # So we create the respective sentence combinations
        product_category_scores,product_category_query_and_value=self._gen_score(corpus,product_category)
        product_category_chunk_scores,product_category_chunk_query_and_value=self._gen_score(corpus,product_category_chunk)
        product_title_scores,product_title_query_and_value=self._gen_score(corpus,noisy_product_name)
        product_title_chunk_scores,product_title_chunk_query_and_value=self._gen_score(corpus,product_title_chunk)
        scores=self.merge(scores,product_category_scores,'product_category_score')
        scores=self.merge(scores,product_category_chunk_scores,'product_category_chunk_score')
        scores=self.merge(scores,product_title_scores,'product_title_score')
        scores=self.merge(scores,product_title_chunk_scores,'product_title_chunk_score')
         
        # Sort the scores in decreasing order
        if product_title_chunk !="":
            sorted_order=np.argsort(product_title_chunk_scores)
        else:
            sorted_order=np.argsort(product_title_scores)
        sim_scores_argsort = reversed(sorted_order)

        # Print the scores
        if self.is_debug   :
            logging.debug(f"Query:{noisy_product_name}. Gold: {gt_mention}")
            # print(f"Query:{noisy_product_name}. Gold: {gt_mention}")
        is_found=False   
        choose_text=""
        for idx in sim_scores_argsort:
            if not is_found:
                # if product_category_chunk_query_and_value[idx][0]!="":
                #     product_category_score= scores[idx]["product_category_chunk_score"]
                # else:
                #     product_category_score=scores[idx]["product_category_score"]
                product_category_score=max(scores[idx]["product_category_score"],scores[idx]["product_category_chunk_score"])
                product_title_score=max(scores[idx]["product_title_score"],scores[idx]["product_title_chunk_score"])
                if product_title_score>=self.similarity_threshold and product_category_score>=self.similarity_threshold:
                    choose_text=corpus[idx]
                    is_found=True  
                
                
            if self.is_debug   :
                logging.debug(f"{ scores[idx]}, { corpus[idx]}")
                # print(f"{ scores[idx]}, { corpus[idx]}" )
                    
            
         
        return choose_text   ,scores ,product_category_chunk,product_title_chunk
    def merge(self,base_scores,to_be_merged_scores,name):
        for idx in range(len(to_be_merged_scores)):
            base_scores[idx][name] = to_be_merged_scores[idx]
        return base_scores




class ParserProductAliasGenerator(MentionGenerator):
    def __init__(self,is_debug,similarity_threshold,similarity_target,parser_setting,parser_chunk_setting,product_title_range)  :
        super().__init__(is_debug,similarity_threshold,similarity_target,False)
        self.parser_setting=parser_setting
        self.parser_chunk_setting=parser_chunk_setting
        self.product_title_range=product_title_range
        
    def generate_mention(self,review, noisy_product_name,product_category,product_desc,gt_mention,is_debug):
        detailed_result={}
        mention_candidate_list_before_review_match= gen_product_name_candidates(noisy_product_name,product_category
                                                                           ,product_desc,self.parser_setting,
                                                                           self.parser_chunk_setting) 
        
         
        detailed_result["product_category"]=      product_category    
        detailed_result["product_title"]=      noisy_product_name          
        detailed_result["parser_product_alias"]=mention_candidate_list_before_review_match
        return None ,None,mention_candidate_list_before_review_match ,detailed_result
                  

class ParserMentionGenerator(MentionGenerator):
    def __init__(self,is_debug,similarity_threshold,similarity_target,parser_setting,parser_chunk_setting,product_title_range) -> None:
        super().__init__(is_debug,similarity_threshold,similarity_target)
        self.parser_setting=parser_setting
        self.parser_chunk_setting=parser_chunk_setting
        self.product_title_range=product_title_range
        
    def generate_mention(self,review, noisy_product_name,product_category,product_desc,gt_mention,is_debug):
        detailed_result={}
        named_entity_in_review_list,mention_candidate_list_before_review_match=gen_mention_candidates(review,noisy_product_name,
                                                                                   product_category,product_desc,
                                                                                   gt_mention,is_debug,
                                                                                   self.parser_setting,
                                                                                   self.parser_chunk_setting)
        detailed_result["product_category"]=      product_category    
        detailed_result["product_title"]=      noisy_product_name          
        detailed_result["parser_mention_candidates"]=named_entity_in_review_list
        if len(named_entity_in_review_list)>0:
            if self.similarity_target=="product_category":
                query=product_category
                return self.find_most_similar_word(query,named_entity_in_review_list,gt_mention),named_entity_in_review_list,mention_candidate_list_before_review_match ,detailed_result
            elif  self.similarity_target=="product_title":
                query=noisy_product_name
                return self.find_most_similar_word(query,named_entity_in_review_list,gt_mention),named_entity_in_review_list,mention_candidate_list_before_review_match ,detailed_result
            elif self.similarity_target=="merge2":
                mention,scores,product_category_chunk,product_title_chunk=self.find_most_similar_word_merge2(product_category,noisy_product_name,named_entity_in_review_list,gt_mention)
                mention_by_rule,is_override=override_mention_by_rule(review,named_entity_in_review_list)
                if is_override:
                    mention=mention_by_rule
                detailed_result["score"]=scores
                detailed_result["product_category_chunk"]=product_category_chunk
                detailed_result["product_title_chunk"]=product_title_chunk
                return mention,named_entity_in_review_list,mention_candidate_list_before_review_match ,detailed_result

        else:
            logging.info(f"no candidates for {noisy_product_name}")
            return None ,None,mention_candidate_list_before_review_match  ,detailed_result                    
  
     
def override_mention_by_rule(review,named_entity_in_review_list):
    for candidate in named_entity_in_review_list:
        if candidate is not None and candidate !="":
            if review.lower().find(("this "+candidate).lower()) !=-1  :
                return candidate,True 
    return None,False 

class OverlapMentionGenerator(MentionGenerator):
    def generate_mention(self,review, noisy_product_name,product_category,product_desc,gt_mention,is_debug):
        detailed_result={}
        mention_candidate_list,mention_candidate_list_before_review_match=gen_overlap_mention_candidates(noisy_product_name,product_category,product_desc,review)
        if len(mention_candidate_list)>0:
            if self.similarity_target=="product_category":
                query=product_category
            else:
                query=noisy_product_name
            return self.find_most_similar_word(query,mention_candidate_list,gt_mention),mention_candidate_list,mention_candidate_list_before_review_match ,detailed_result
        else:
            return None ,None ,mention_candidate_list_before_review_match ,detailed_result
   
def merge_list(parser_mention_candidates_before_similarity,overlap_mention_candidates_before_similarity):
    parser_mention_candidates_before_similarity.extend(overlap_mention_candidates_before_similarity)
    return parser_mention_candidates_before_similarity
def merge_object_list(parser_mention_candidates_before_review_match,overlap_mention_candidate_list_before_review_match):
    parser_mention_candidates_before_review_match.update(overlap_mention_candidate_list_before_review_match)
    return parser_mention_candidates_before_review_match

class AllMentionGenerator(MentionGenerator):    
    def generate_mention(self,review, noisy_product_name,product_category,product_desc,gt_mention,is_debug):
        parser_mention_candidates_before_similarity,parser_mention_candidates_before_review_match=gen_mention_candidates(review,noisy_product_name,product_category,product_desc,gt_mention,is_debug)
        overlap_mention_candidates_before_similarity,overlap_mention_candidate_list_before_review_match=gen_overlap_mention_candidates(noisy_product_name,product_category,product_desc,review)
        detailed_result={}
        detailed_result["parser_mention_candidates"]=parser_mention_candidates_before_similarity
        detailed_result["overlap_mention_candidates"]=overlap_mention_candidates_before_similarity
        mention_candidates_before_similarity=merge_list(parser_mention_candidates_before_similarity,overlap_mention_candidates_before_similarity)
        mention_candidates_before_review_match=merge_object_list(parser_mention_candidates_before_review_match,overlap_mention_candidate_list_before_review_match)
        
        if len(mention_candidates_before_similarity)>0:
            if self.similarity_target=="product_category":
                query=product_category
                return self.find_most_similar_word(query,mention_candidates_before_similarity,gt_mention),mention_candidates_before_similarity,mention_candidates_before_review_match ,detailed_result
            elif  self.similarity_target=="product_title":
                query=noisy_product_name
                return self.find_most_similar_word(query,mention_candidates_before_similarity,gt_mention),mention_candidates_before_similarity,mention_candidates_before_review_match ,detailed_result
            elif  self.similarity_target=="merge1":
                return self.find_most_similar_word_merge1(product_category,noisy_product_name,mention_candidates_before_similarity,gt_mention),mention_candidates_before_similarity,mention_candidates_before_review_match ,detailed_result
            elif self.similarity_target=="merge2":
                mention,scores,product_category_chunk,product_title_chunk=self.find_most_similar_word_merge2(product_category,noisy_product_name,mention_candidates_before_similarity,gt_mention)
                detailed_result["score"]=scores
                return mention,mention_candidates_before_similarity,mention_candidates_before_review_match ,detailed_result
            else:
                self.find_most_similar_word(find_root_product_category(product_category),mention_candidates_before_similarity,gt_mention),mention_candidates_before_similarity,mention_candidates_before_review_match ,detailed_result
                self.find_most_similar_word(find_root_product_title(noisy_product_name),mention_candidates_before_similarity,gt_mention),mention_candidates_before_similarity,mention_candidates_before_review_match ,detailed_result
                self.find_most_similar_word(find_chunk_product_category(product_category),mention_candidates_before_similarity,gt_mention),mention_candidates_before_similarity,mention_candidates_before_review_match ,detailed_result
                self.find_most_similar_word(find_chunk_product_title(noisy_product_name),mention_candidates_before_similarity,gt_mention),mention_candidates_before_similarity,mention_candidates_before_review_match ,detailed_result
                self.find_most_similar_word(product_category,mention_candidates_before_similarity,gt_mention),mention_candidates_before_similarity,mention_candidates_before_review_match ,detailed_result
                return self.find_most_similar_word(noisy_product_name,mention_candidates_before_similarity,gt_mention),mention_candidates_before_similarity,mention_candidates_before_review_match ,detailed_result
        else:
            return None ,None ,mention_candidates_before_review_match,detailed_result 
        
    
        
    
    
    def find_most_similar_word_merge1(self,product_category,noisy_product_name,corpus,gt_mention):
        scores={i:{} for i,example in enumerate(corpus)}
        # So we create the respective sentence combinations
        product_category_scores,product_category_query_and_value=self._gen_score(corpus,product_category)
        product_category_chunk_scores,product_category_chunk_query_and_value=self._gen_score(corpus,find_chunk_product_category(product_category))
        product_title_scores,product_title_query_and_value=self._gen_score(corpus,noisy_product_name)
        product_title_chunk_scores,product_title_chunk_query_and_value=self._gen_score(corpus,find_chunk_product_title(noisy_product_name))
        scores=self.merge(scores,product_category_scores,'product_category_score')
        scores=self.merge(scores,product_category_chunk_scores,'product_category_chunk_score')
        scores=self.merge(scores,product_title_scores,'product_title_score')
        scores=self.merge(scores,product_title_chunk_scores,'product_title_chunk_score')
         
        # Sort the scores in decreasing order
        sorted_order=np.argsort(product_category_chunk_scores)
        sim_scores_argsort = reversed(sorted_order)

        # Print the scores
        if self.is_debug   :
            print(f"Query:{noisy_product_name}. Gold: {gt_mention}")
        is_found=False   
        choose_text=""
        for idx in sim_scores_argsort:
            if not is_found:
                product_category_score=max(scores[idx]["product_category_score"],scores[idx]["product_category_chunk_score"])
                product_title_score=max(scores[idx]["product_title_score"],scores[idx]["product_title_chunk_score"])
                if product_title_score>=self.similarity_threshold and product_category_score>=self.similarity_threshold:
                    choose_text=corpus[idx]
                    is_found=True  
                
                
            if self.is_debug   :
                print(f"{ scores[idx]}, { corpus[idx]}" )
                    
            
         
        return choose_text
 
def find_root_product_category(product_category):
    product_category= product_category.split(" -> ")[-1]  
    doc = nlp(product_category)
    root=""
    for token in doc:
        if token.dep_=="ROOT":
            root= token.text 
    # for chunk in doc.noun_chunks:
    #     root=chunk.root.text
     
    return root
def find_root_product_title(noisy_product_name):
     
    doc = nlp(noisy_product_name)
     
    root=""
    for token in doc:
        if token.dep_=="ROOT":
            root= token.text 
    # for chunk in doc.noun_chunks:
    #     root=chunk.root.text
     
    return root
def find_chunk_product_category(product_category):
    product_category= product_category.split(" -> ")[-1]  
    doc = nlp(product_category)
    return gen_root_chunk_by_doc(doc)

def find_chunk_product_title(noisy_product_name):
     
    doc = nlp(noisy_product_name)
    return gen_root_chunk_by_doc(doc)
    
    


    
def  generate_mention_main():
    mention_generator=ParserMentionGenerator()
    is_debug=False 
    start_id=-1
    
    review_json_path = Path(
        f'bestbuy/data/final/v0/bestbuy_review_2.3.4_w_image.json'
    )        
    product_dataset_path = Path('bestbuy/data/final/v0/bestbuy_products_40000_3.4.2_desc_img_url.json')
    output_products_path = Path(
        f'bestbuy/data/final/v0/bestbuy_review_2.4.json'
    )  
    output_list=[]
    with open(product_dataset_path, 'r', encoding='utf-8') as fp:
        product_dataset_array = json.load(fp)
        product_json_dict=json_to_dict(product_dataset_array)
        
        with open(review_json_path, 'r', encoding='utf-8') as fp:
            review_json_array = json.load(fp)
             
            for idx,review_json   in tqdm(enumerate(review_json_array)):
                
                
                product_category=review_json["product_category"]
                product_category=product_category.split(" -> ")[-1]
                noisy_product_name=review_json["product_name"]
                noisy_product_name_list=noisy_product_name.split(" - ")
                noisy_product_name=" ".join(noisy_product_name_list[:2])
                product_json=product_json_dict[review_json["url"]]
                if "overview_section" in product_json and  "description" in product_json["overview_section"]:
                    product_desc =product_json["overview_section"]["description"]
                    product_desc_sent_list=sent_tokenize(product_desc)
                    if len(product_desc_sent_list)>0:
                        product_desc=product_desc_sent_list[0]
                    else:
                        product_desc=""
                else:
                    product_desc=""
                reviews=review_json["reviews"]
                output_review_list=[]
                gt_mention=""
                for review in reviews:
                    review_text=review["body"]
                    review_id=review["id"]
                     
                    mention=mention_generator.generate_mention(review_text,noisy_product_name,product_category,product_desc)
                    if is_debug:
                        print(f"context: {mention}, {noisy_product_name}, {product_category}, {product_desc}, {review_text} ")
                    else:
                        print(f"get {mention} for {review_id}")
                    review["mention"]=mention
                    output_review_list.append(review)
                review_json["reviews"]=output_review_list
                output_list.append(review_json)
                if idx % 1000==0:
                    with open(output_products_path, 'w', encoding='utf-8') as fp:
                        json.dump(output_list, fp, indent=4)
                
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(output_list, fp, indent=4)
                
# generate_mention_main()                    