
from tqdm import tqdm
from bestbuy.src.extract_mention.dependency_parser.parser import gen_mention_candidates
from bestbuy.src.extract_mention.name_entity_recognition.generate_mention import AllMentionGenerator, MentionGenerator, ParserMentionGenerator
from bestbuy.src.extract_mention.name_entity_recognition.train.inference import inference_one
from bestbuy.src.extract_mention.name_entity_recognition.train.train import gen_model
from  bestbuy.src.extract_mention.name_entity_recognition.ner_bert import ner_bert
from  bestbuy.src.extract_mention.name_entity_recognition.ner_spacy import ner_spacy
from  bestbuy.src.extract_mention.name_entity_recognition.token_similarity import find_most_similar_word
from bestbuy.src.extract_mention.name_entity_recognition.util.data_util import is_nan, is_prediction_right, compare_two_equal_len_texts_w_base_form, prepare_produt
from  bestbuy.src.extract_mention.pos.pos_spacy import pos_spacy

from nltk.tokenize import sent_tokenize, word_tokenize
from pathlib import Path
from time import time
from pathlib import Path
import json
import pandas as pd
import argparse
from bestbuy.src.organize.merge_attribute import json_to_dict
from sentence_transformers.cross_encoder import CrossEncoder
import numpy as np

from nltk.tokenize import sent_tokenize, word_tokenize 
import nltk 
import logging
import shutil
from bestbuy.src.util.create_example.generate_html import generate
# version="vdebug"
# version="v34"
# logging.basicConfig(filename=f'bestbuy/output/log/example_200_{version}.log', filemode='w', level=logging.DEBUG)
# nltk.download("popular")
class Evaluator:
    def __init__(self,args) :
        self.true_positive_num=0
        self.true_positive_before_review_match_num=0
        self.true_positive_before_similarity_ranking_num=0
        self.is_cur_gold_mention_in_candidate_list_before_review_match=0
        self.is_cur_gold_mention_in_candidate_list_before_similarity_ranking=0
        self.is_cur_true_positive_after_similarity_ranking=0
        self.non_empty_prediction_num=0
        self.empty_prediction_before_similarity_ranking_num=0
        self.empty_prediction_before_review_match_num=0
        self.positive_num=0 #precision
        self.true_num=0 #recall
        self.is_debug=True  
        self.similarity_threshold=0.3
        self.similarity_target="merge2" #product_category,product_title
        self.mode="parser" #"parser"bert_ner"overlap
        self.parser_chunk_setting=args.parser_chunk_setting #root_only
        self.product_title_range="no_brand"#no_brand, with_brand
        
        self.input_type="dataset" #200
        self.debug_review_id=None
        
        if self.input_type=="200":
            self.review_json_path = Path(
                f'bestbuy/data/example/bestbuy_review_200_cleaned_annotation.json' #200_cleaned_annotation
            )        
            self.output_products_path = Path(
                f'bestbuy/data/example/bestbuy_review_200_cleaned_annotation_result_{self.mode}_{version}.json'
            ) 
        elif self.input_type=="dataset":
            self.review_json_path = Path(
                f'bestbuy/data/final/v0/bestbuy_review_2.3.6_w_image_not_long.json' #200_cleaned_annotation
            )        
            self.output_products_path = Path(
                f'bestbuy/data/final/v0/bestbuy_review_2.3.6_w_image_not_long_result_{self.mode}_{version}.json'
            ) 
        else:
            self.review_json_path = Path(
                f'bestbuy/data/example/bestbuy_review_50_v2.json' #
            )        
            self.output_products_path = Path(
                f'bestbuy/data/example/bestbuy_review_50_result_{self.mode}.json'
            ) 
        
    def _init_num(self,review_json_array):
        if self.input_type !="dataset":
            self.positive_num=len(review_json_array)
            for idx,review_json   in  enumerate(review_json_array) :
                reviews= review_json["reviews"]
                review=reviews[0]
            
                gt_mention_list=review["gt_mention"]
                self.true_num+=len(gt_mention_list)
        else:
            self.true_num=0
            for idx,review_json   in  enumerate(review_json_array) :
                reviews= review_json["reviews"]
           
                self.positive_num+=len(reviews)
    
     
        
    
    
    def _check_gold_mention_in_candidate_list_before_review_match_num(self,is_debug,gt_mention_list,gt_mention,
                                                                      mention_candidate_list_before_review_match):
        
        self.is_cur_gold_mention_in_candidate_list_before_review_match=0
        for gt_mention_in_list in gt_mention_list:
            if check_is_in_candidate_list(is_debug,gt_mention_in_list,mention_candidate_list_before_review_match):# or  self.in_candidate_from_overlap(mention_candidates_by_overlap,gt_mention)
                self.true_positive_before_review_match_num+=1
                self.is_cur_gold_mention_in_candidate_list_before_review_match+=1
        if self.is_cur_gold_mention_in_candidate_list_before_review_match<len(gt_mention_list):
            pass #TODO 
       
                
                
    
    
    
    # def in_candidate_from_overlap(self,mention_candidates_by_overlap,gt_mention):
    #     is_in=False 
    #     if gt_mention=="":
    #         is_in=True 
    #     elif mention_candidates_by_overlap is not None:
    #         if gt_mention in mention_candidates_by_overlap:
    #             is_in=True 
    #     return is_in
            
    def _check_after_similarity_ranking(self,is_debug,gt_mention, gt_mention_list,mention):
        # if mention is not None:
        #     self.non_empty_prediction_num+=1
        # if is_prediction_right(mention,gt_mention,gt_mention_list):
        #     self.true_positive_num+=1
        #     self.is_cur_true_positive_after_similarity_ranking=1
        # else:
        #     self.is_cur_true_positive_after_similarity_ranking=0
 
                
      
        if mention is None:
            self.empty_prediction_before_review_match_num+=1
        elif mention =="":
            self.empty_prediction_before_similarity_ranking_num+=1
        else:
            self.non_empty_prediction_num+=1
            if self.input_type!="dataset":
                if is_prediction_right(mention,gt_mention,gt_mention_list):
                    self.true_positive_num+=1
                    self.is_cur_true_positive_after_similarity_ranking=1
                else:
                    self.is_cur_true_positive_after_similarity_ranking=0
            
    def _check_gold_mention_in_candidate_list_before_similarity_ranking_num(self,is_debug,gt_mention_list,gt_mention,
                                                                            mention_candidate_list_before_similarity_ranking):
        
        self.is_cur_gold_mention_in_candidate_list_before_similarity_ranking=0
        is_in=False 
        for gt_mention_in_list in gt_mention_list:
            
            if gt_mention_in_list=="":
                is_in=True 
            elif mention_candidate_list_before_similarity_ranking is not None:
                for candidate in mention_candidate_list_before_similarity_ranking:
                    if compare_two_equal_len_texts_w_base_form(candidate,gt_mention_in_list):
                        is_in=True 
                        break 
              
            if is_in:
                self.is_cur_gold_mention_in_candidate_list_before_similarity_ranking+=1
                self.true_positive_before_similarity_ranking_num+=1   
            
            
            # if    is_debug:
            #     print(f"{gt_mention}, predicted {mention_candidate_list_before_similarity_ranking}")
        return is_in
    def  evaluate_100(self):
        parser_setting="no_desc"
        save_step=1000
        if self.mode=="parser":
            mention_generator=ParserMentionGenerator(self.is_debug,self.similarity_threshold,self.similarity_target,
                                                     parser_setting,self.parser_chunk_setting,self.product_title_range)
        else:
            mention_generator=AllMentionGenerator(self.is_debug,self.similarity_threshold,self.similarity_target)
        
        output_list=[]
        review_num_so_far=0
        with open(self.review_json_path, 'r', encoding='utf-8') as fp:
            review_json_array = json.load(fp)
            self._init_num(review_json_array)
            is_product_desc_first_sent=True   
            is_product_category_last_term=True 
            for idx,product_json   in tqdm(enumerate(review_json_array)):
                reviews= product_json["reviews"]
                product_category,noisy_product_name,product_desc=prepare_produt(product_json,is_product_desc_first_sent,
                                                                                    is_product_category_last_term,self.product_title_range)
   
                for review_json in reviews:
                    review_num_so_far+=1
                    
                    gt_mention_list,gt_mention,review_text,review_id=prepare_review(review_json)
                    if self.debug_review_id is not None:
                        if review_id  !=self.debug_review_id:
                            print("debug")
                            continue 
                    if self.mode =="parser":
                        mention, mention_candidates,mention_candidate_list_before_review_match,detailed_result=mention_generator.generate_mention(review_text,noisy_product_name,product_category,product_desc,gt_mention,self.is_debug)
                        if self.input_type !="dataset":
                            self._check_gold_mention_in_candidate_list_before_review_match_num( self.is_debug,gt_mention_list,gt_mention,mention_candidate_list_before_review_match)
                            self._check_gold_mention_in_candidate_list_before_similarity_ranking_num( self.is_debug,gt_mention_list,gt_mention,mention_candidates)
                        self._check_after_similarity_ranking(self.is_debug,gt_mention, gt_mention_list,mention)
                    elif self.mode =="bert_ner":
                        model, tokenizer=gen_model("asahi417/tner-xlm-roberta-base-ontonotes5")  
                        mention,mention_candidates=inference_one(model,tokenizer,review_text)
                    elif self.mode =="overlap":
                        mention, mention_candidates= mention_generator.generate_mention_with_overlap(review_text,noisy_product_name,product_category,product_desc) 
                    elif self.mode == "all":
                        mention, mention_candidates,mention_candidate_list_before_review_match,detailed_result=mention_generator.generate_mention(review_text,noisy_product_name, 
                                                                                                                                                product_category,product_desc,gt_mention,self.is_debug)
                        self._check_gold_mention_in_candidate_list_before_review_match_num( self.is_debug,gt_mention_list,gt_mention,mention_candidate_list_before_review_match)
                        self._check_gold_mention_in_candidate_list_before_similarity_ranking_num( self.is_debug,gt_mention_list,gt_mention,mention_candidates)
                        self._check_after_similarity_ranking(self.is_debug,gt_mention, gt_mention_list,mention)
                    # if is_debug:
                    #     print(f"{mention}, {noisy_product_name}, {product_category}, {product_desc}, {review_text} ")
                    # else:
                    #     print(f"get {mention} for {review_id}")
                    displayed_review=self.gen_output(mention,mention_candidates,product_json,review_json,review_text,self.mode,
                                                    detailed_result,self.is_debug,mention_candidate_list_before_review_match,
                                                    gt_mention_list)
                    
                    if displayed_review is not None:
                        output_list.append(displayed_review)
                    self.log_metric(review_num_so_far)
                 
                if idx%save_step==0:
                    with open(self.output_products_path, 'w', encoding='utf-8') as fp:
                        json.dump(output_list, fp, indent=4, cls=NpEncoder)
        
        with open(self.output_products_path, 'w', encoding='utf-8') as fp:
            json.dump(output_list, fp, indent=4, cls=NpEncoder)
        self.log_metric(review_num_so_far)
        
        is_generate_html=False  
        if is_generate_html:
            html_file= f'bestbuy/output/output/bestbuy_review_200_cleaned_annotation_result_parser_{version}.json'
            generate( html_file, "bestbuy/output/html",version,  generate_claim_level_report=False ) 
            shutil.copyfile(f"/home/menglong/workspace/code/referred/product_scraper/dataset_construction/bestbuy/output/html/example0_{version}.html", "/home/menglong/workspace/code/referred/show_html/entity_linking/example0.html" )
 
 
    def gen_output(self,mention,mention_candidates,product_json,review_sub_json,review_text,mode, detailed_result,is_debug,
                   mention_candidate_list_before_review_match,
                gt_mention_list)    :
        displayed_review={}
         
         
        if "gt_mention" in review_sub_json:
            displayed_review[f"gold_mention"]=review_sub_json["gt_mention"]
        displayed_review[f"predicted_mention"]=mention 
        if mode !="all":
            displayed_review[f"mention_candidates_{mode}"]=mention_candidates
        else:
            displayed_review[f"mention_candidates_parser"]=detailed_result["parser_mention_candidates"]
            # displayed_review[f"mention_candidates_bert_ner"]=mention_candidates_by_ner
            if "overlap_mention_candidates" in detailed_result:
                displayed_review[f"mention_candidates_overlap"]=detailed_result["overlap_mention_candidates"]
        if "score" in detailed_result:
            displayed_review["similarity_score"]=detailed_result["score"]
        
        displayed_review["review_id"]=review_sub_json["id"] 
        displayed_review["product_category"]=detailed_result["product_category"] 
        displayed_review["product_title"]=detailed_result["product_title"]
        if "overview_section" in product_json and product_json["overview_section"] is not None:
            displayed_review["product_desc"]=product_json["overview_section"]["description"]
        displayed_review["review"]=review_text
        displayed_review["mention_candidate_list_before_review_match"]=gen_one_list_for(mention_candidate_list_before_review_match)
        if "product_category_chunk" in detailed_result:
            displayed_review["product_category_chunk"]=detailed_result["product_category_chunk"]
            displayed_review["product_title_chunk"]=detailed_result["product_title_chunk"]
        else:
            displayed_review["product_category_chunk"]=""
            displayed_review["product_title_chunk"]=""
        
        displayed_review[f"url"]=product_json["url"]
            
        # if self.is_cur_true_positive_after_similarity_ranking==1:
           
        #     displayed_review["is_predict_right"]=1
        # else:
        #     displayed_review["is_predict_right"]=0
        # displayed_review["review_id"]=review_sub_json["id"]
        
        
         
        
        # if  self.is_cur_gold_mention_in_candidate_list_before_review_match==0:
        # displayed_review["gold_mention_is_in_candidates_before_review_match"]=self.is_cur_gold_mention_in_candidate_list_before_review_match
        # displayed_review["gold_mention_is_in_candidates_before_similarity_ranking"]=self.is_cur_gold_mention_in_candidate_list_before_similarity_ranking
 
 
        if (self.input_type=="dataset" or self.is_cur_true_positive_after_similarity_ranking==0) and mention is not None and mention!="" :
            
            return displayed_review 
        else:
            return None  #TODO
        
          
    def log_metric(self,review_num_so_far):
        if self.input_type!="dataset":
            logging.info(f"review_num:{review_num_so_far}, precision: {gen_precision(self.true_positive_num,self.non_empty_prediction_num)}, non_empty_prediction_ratio:{self.non_empty_prediction_num/self.positive_num}, "+
                    f"non_empty_prediction_num:{self.non_empty_prediction_num},true_positive_num:{self.true_positive_num}, positive_num:{self.positive_num}"+
                            f"{gen_precision(self.true_positive_num,self.positive_num)},{gen_recall(self.true_positive_before_similarity_ranking_num,self.true_num)},"+
                            f"{gen_recall(self.true_positive_before_review_match_num,self.true_num)},{self.mode},{self.true_num},non_empty_prediction_ratio:{self.non_empty_prediction_num/self.positive_num}")        
        else:
            logging.info(f"review_num:{review_num_so_far}, non_empty_prediction_ratio:{self.non_empty_prediction_num/self.positive_num}, "+
                  f"empty_prediction_before_similarity_ranking_ratio:{self.empty_prediction_before_similarity_ranking_num/self.positive_num}, "+
                  f"empty_prediction_before_review_match_ratio:{self.empty_prediction_before_review_match_num/self.positive_num}, "+
                f"non_empty_prediction_num:{self.non_empty_prediction_num}, empty_prediction_before_similarity_ranking_num:{self.empty_prediction_before_similarity_ranking_num}, "+
                f"empty_prediction_before_similarity_ranking_num:{self.empty_prediction_before_similarity_ranking_num}, "+
                f"positive_num:{self.positive_num}"
                         )        


def gen_one_list_for(mention_candidate_list_before_review_match):
    candidate_list=[]
    for product_name_candidate_root, product_name_candidate_object in mention_candidate_list_before_review_match.items():
        candidate_list.extend( product_name_candidate_object.candidate_text_list )
    return candidate_list


def gen_one_dict_for(mention_candidate_list_before_review_match):
    candidate_list={}
    for product_name_candidate_root, product_name_candidate_object in mention_candidate_list_before_review_match.items():
        candidate_list[product_name_candidate_root]= product_name_candidate_object.candidate_text_list 
    return candidate_list


import json
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
    
    
def gen_precision(true_positive_num,positive_num):
    if positive_num>0:
        return 100*(true_positive_num/positive_num)
    else:
        return 0
    
def gen_recall(true_positive_num,true_num):
    return gen_precision(true_positive_num,true_num )


def prepare_review(review):
    
    review_text=review["body"]
    review_header=review["header"]
    if "gt_mention" in review:
        gt_mention_list=review["gt_mention"]
        gt_mention=gt_mention_list[0]
    else:
        gt_mention_list=[]
        gt_mention=""
    
    review_text=review_header+". "+review_text
    review_id=review["id"]
    return gt_mention_list,gt_mention,review_text,review_id
    
def check_is_in_candidate_list(is_debug,gt_mention,product_name_candidate_objects):
    
    # if gt_mention=="nest camera battery":#TODO
    #     print("")
    is_in=False 
    if gt_mention=="":
        is_in=True 
        return is_in 
    elif product_name_candidate_objects is not None:
        for product_name_candidate_root, product_name_candidate_object in product_name_candidate_objects.items():
            for candidate in product_name_candidate_object.candidate_text_list: 
                if compare_two_equal_len_texts_w_base_form(candidate,gt_mention): 
                    is_in=True 
                    return is_in 
    # if not is_in :
    #     if is_debug:
    #         print(f"check_is_in_candidate_list: {gt_mention}, predicted {product_name_candidate_objects}")
            
    
        

def get_args():
   


    parser = argparse.ArgumentParser() 
  
    parser.add_argument("--similarity_target",default='', type=str  ) 
    parser.add_argument("--parser_chunk_setting",default='root_only', type=str  ) 
    
    args = parser.parse_args()

    print(args)
    return args




if __name__ == "__main__":  
    args=get_args()                
    evaluator=Evaluator(args)
    evaluator.evaluate_100()                    