import os
from retrieval.utils.retrieval_util import cross_score_to_sigmoid
from util.visualize.create_example.example import *
import pandas as pd 
import nltk 
 
import json
from pathlib import Path
from random import sample 
 
 
def check_review_id_number(review_products_json_list):
    id_list=[]
    for review_product_json in review_products_json_list:
        review_json=review_product_json
        id=review_json["review_id"] 
        id_list.append(id )
        
    print(f"{len(set(id_list))}")
    

def clean_confidence_score( confidence_score_ocr )    :
    out_dict={}
    for key,score in confidence_score_ocr.items():
        if score>0.8:
            out_dict[key]=score
    return out_dict
    
    
def read_example(data_path,is_reshuffle,similar_products_path,is_need_corpus,is_add_gold,is_add_gold_at_end ):
     
    review_products_path = Path(
        data_path
    )
    review_dict={}
    product_dict={} 
    example_dict={}
    review_id_set=set()
    if is_need_corpus:
        with open(similar_products_path, 'r', encoding='utf-8') as fp_s:
            similar_products_json_list = json.load(fp_s)      
            for review_product_json in similar_products_json_list:
                if "image_path" in review_product_json:
                    product_img_path_list=review_product_json["image_path"]
                else:
                    product_img_path_list=[]
                if "overview_section" in review_product_json and review_product_json["overview_section"] is not None and "description" in review_product_json["overview_section"]:
                    product_text=review_product_json["overview_section"]["description"]
                else:
                    product_text=""
                product_id=review_product_json["id"] 
                product_name=review_product_json["product_name"]
                spec=review_product_json["Spec"]
                product=Product( product_id, product_img_path_list, product_text,product_name,spec,None)
                product_dict[product_id]=product
                
    with open(review_products_path, 'r', encoding='utf-8') as fp:
        review_products_json_list = json.load(fp)
        for review_product_json in review_products_json_list:
            review_json=review_product_json 
            review_id=review_json["review_id"]
            if review_id in review_id_set:
                print("ERROR!")
            else:
                review_id_set.add(review_id)
            if "review_image_path" in review_json:
                review_img_path_list=review_json["review_image_path"]
            else:
                review_img_path_list=[]
                            
            if "review_image_path_before_select" in review_json:
                review_image_path_before_select=review_json["review_image_path_before_select"]
            else:
                review_image_path_before_select=[]
            similar_product_id_list=review_product_json["fused_candidate_list"][:10]#similar_product_id
            review_text=review_json["header"]+". "+review_json["body"]
            mention=review_json["mention"]
            
            target_product_id=review_product_json["gold_entity_info"]["id"]#
            # target_product_name=review_product_json["gold_entity_info"]["product_name"]#["gold_entity_info"]
            
            if "is_low_quality_review" in review_json:
                is_low_quality_review=review_json["is_low_quality_review"]
            else:
                is_low_quality_review=None 
            if "attribute" in review_json:
                attribute=review_json["attribute"]
            else:
                attribute={}
            if "review_special_product_info" in review_json:
                review_special_product_info=review_json["review_special_product_info"]
            else:
                review_special_product_info={}
            if "image_similarity_score" in review_json:
                image_similarity_score=review_json["image_similarity_score"]
            else:
                image_similarity_score=-1
            if "image_similarity_score_list" in review_product_json:
                image_similarity_score_list=review_product_json["image_similarity_score_list"]
            else:
                image_similarity_score_list=[]
            if "text_similarity_score" in review_product_json:
                text_similarity_score=review_product_json["text_similarity_score"]
                product_title_similarity_score=review_product_json["product_title_similarity_score"]
                predicted_is_low_quality_review=review_product_json["predicted_is_low_quality_review"]
            else:
                text_similarity_score=1
                product_title_similarity_score=1
                predicted_is_low_quality_review="n"
            
            if "reshuffled_target_product_id_position" in review_json:
                reshuffled_target_product_id_position=review_json["reshuffled_target_product_id_position"]
            else:
                reshuffled_target_product_id_position=None 
            if reshuffled_target_product_id_position     is None:
                if "target_product_id_position" in review_json:
                    reshuffled_target_product_id_position=review_json["target_product_id_position"]
                 
            if "is_retrieval_error" in review_json:
                is_retrieval_error=review_json["is_retrieval_error"]
            else:
                is_retrieval_error=None 
            if "sample" in review_json:
                sample=review_json["sample"]
            else:
                sample=None 
            if "gold_product_index" in review_product_json:
                gold_product_index=review_product_json["gold_product_index"]
            else:
                gold_product_index=2
            if "fused_score_dict" in review_product_json:
                score_dict=review_product_json["fused_score_dict"]
            else:
                
                score_dict={}
            if "image_score_dict" in review_product_json:
                image_score_dict=review_product_json["image_score_dict"]
            else:
                
                image_score_dict={}
            if "text_score_dict" in review_product_json:
                text_score_dict=review_product_json["text_score_dict"]
            else:
                
                text_score_dict={}
            if "desc_score_dict" in review_product_json:
                desc_score_dict=review_product_json["desc_score_dict"]
            else:
                
                desc_score_dict={}
            review=Review(review_id,review_img_path_list,review_text,mention,image_similarity_score,is_low_quality_review,attribute,text_similarity_score,product_title_similarity_score,
                          predicted_is_low_quality_review,score_dict,gold_product_index,image_score_dict,text_score_dict,desc_score_dict
                           ,review_special_product_info,review_image_path_before_select)
            if   f"predicted_attribute_exact" in review_json:
                review.predicted_attribute_dict[f"Predicted Attribute exact"]=review_json[f"predicted_attribute_exact"]
            else:
                review.predicted_attribute_dict[f"Predicted Attribute exact"]="" 
            if   f"predicted_attribute" in review_json:
                review.predicted_attribute_dict[f"Predicted Attribute"]=review_json[f"predicted_attribute"]
            else:
                review.predicted_attribute_dict[f"Predicted Attribute"]="" 
            if   f"gold_attribute_for_predicted_category" in review_json:
                review.predicted_attribute_dict[f"Gold Attribute"]=review_json[f"gold_attribute_for_predicted_category"]
            else:
                review.predicted_attribute_dict[f"Gold Attribute"]="" 
            # if   f"predicted_attribute_contain_all" in review_json:
            #     review.predicted_attribute_dict[f"Predicted Attribute All"]=review_json[f"predicted_attribute_contain_all"]
            # else:
            #     review.predicted_attribute_dict[f"Predicted Attribute All"]="" 
            review=add_attribute(review,"gpt2",review_json)   
            review=add_attribute(review,"ocr",review_json) 
            # review=add_attribute(review,"chatgpt",review_json) 
            review=add_attribute(review,"model_version",review_json) 
            review=add_attribute(review,"model_version_to_product_title",review_json) 
            if   f"raw_ocr" in review_json:
                review.predicted_attribute_dict[f"Predicted Attribute raw_ocr"]=review_json[f"raw_ocr"]
            else:
                review.predicted_attribute_dict[f"Predicted Attribute raw_ocr"]="" 
            if f"confidence_score_ocr" in review_json:
                review.predicted_attribute_dict[f"Attribute Confidence ocr"]=clean_confidence_score(review_json[f"confidence_score_ocr"])
            else:
                review.predicted_attribute_dict[f"Attribute Confidence ocr"]=""  
            review_dict[review_id]=review
            
            example=Example(review_id,target_product_id,similar_product_id_list,is_reshuffle,reshuffled_target_product_id_position,is_add_gold,is_add_gold_at_end)
            example_dict[review_id]=example
             
    
            
             
    return example_dict,review_dict,product_dict 


def add_attribute(review,attribute_logic,review_json):
    # if   f"predicted_attribute_match" in review_json:
    #     review.predicted_attribute_dict[f"Predicted Attribute match"]=review_json[f"predicted_attribute_match"]
    # else:
    #     review.predicted_attribute_dict[f"Predicted Attribute match"]="" 
    
    if   f"predicted_attribute_{attribute_logic}" in review_json:
        review.predicted_attribute_dict[f"Predicted Attribute {attribute_logic}"]=review_json[f"predicted_attribute_{attribute_logic}"]
    else:
        review.predicted_attribute_dict[f"Predicted Attribute {attribute_logic}"]=""
 
    # if   "gpt" not in attribute_logic and  f"predicted_attribute_context_{attribute_logic}" in review_json:
    #     review.predicted_attribute_dict[f"Predicted Attribute Contenxt {attribute_logic}"]=review_json[f"predicted_attribute_context_{attribute_logic}"]
    # else:
    #     review.predicted_attribute_dict[f"Predicted Attribute Context {attribute_logic}"]=""
    
    return review
    
def add_score_for_product(product_dict,  review_product_json,score_dict_key):
    if score_dict_key in review_product_json: 
        for product_id, one_score_dict in review_product_json[score_dict_key].items():
            product_id=int(product_id)
            if product_id in product_dict: 
                product_object=product_dict[product_id]
                if "image_score_dict" in score_dict_key:
                    product_object.image_score=one_score_dict["score"]
                    
                elif "desc_score_dict" in score_dict_key:
                    product_object.desc_text_bi_score=one_score_dict["bi_score"]
                    product_object.desc_text_cross_score=cross_score_to_sigmoid( one_score_dict["cross_score"])
                    
                elif "text_score_dict" in score_dict_key:
                    product_object.text_bi_score=one_score_dict["bi_score"]
                    product_object.text_cross_score=cross_score_to_sigmoid( one_score_dict["cross_score"])
                elif "fused_score_dict" in score_dict_key:
                    product_object.fused_score=one_score_dict["fused_score"]
                product_dict[product_id]=product_object
    return product_dict
                
             