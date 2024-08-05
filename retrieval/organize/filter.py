from pathlib import Path
import json
import copy
import os
import pandas as pd
import shutil  
from random import sample
from tqdm import tqdm 
import random
from nltk.tokenize import word_tokenize
from disambiguation.data_util.inner_util import gen_gold_attribute, gen_review_text, json_to_dict, review_json_to_product_dict
from retrieval.utils.retrieval_util import old_spec_to_json_attribute
from util.env_config import * 
def filter_non_attribute_main(data_with_attribute_path,data_with_disambiguation_ranking_path,out_path,filter_by_attribute_field):
    
    product_path=Path("/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v5/bestbuy_products_40000_3.4.16_all_text_image_similar.json")
    with open(product_path, 'r', encoding='utf-8') as fp:
        products_url_json_array = json.load(fp)
        product_dict=json_to_dict(products_url_json_array)
    with open(data_with_disambiguation_ranking_path, 'r', encoding='utf-8') as fp:
        product_dataset_json_array = json.load(fp)
        review_product_json_with_disambiguation_ranking_dict=review_json_to_product_dict(product_dataset_json_array)
    incomplete_products_path = Path(
        data_with_attribute_path
    ) 
 
    out_list=[]
    filter_num=0
    is_only_brand=False 
    if filter_by_attribute_field !="all":
        filter_by_attribute_field=filter_by_attribute_field.split(",")
       
    # tokenizer = B1ertTokenizer.from_pretrained("bert-base-cased")
    with open(incomplete_products_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        for i,review_product_json in tqdm(enumerate(incomplete_dict_list) ):
            review_id=review_product_json["review_id"]
            # if review_id==2726:
            #     print("")
            if review_id in review_product_json_with_disambiguation_ranking_dict:
                review_product_json_with_disambiguation_ranking=review_product_json_with_disambiguation_ranking_dict[review_id]
                fused_candidate_list=review_product_json_with_disambiguation_ranking["fused_candidate_list"]
                new_fused_candidate_list=[]
                attribute_gpt_in_review=review_product_json["predicted_attribute"]#attribute
                attribute_ocr_in_review=review_product_json["predicted_attribute_ocr"]
                
                if len(attribute_gpt_in_review)>0 or len(attribute_ocr_in_review)>0:
                    for candidate_id in fused_candidate_list:
                        if candidate_id !=-1:
                            if is_filter_candidate_list():
                                new_fused_candidate_list.append(candidate_id)
                            else:
                                filter_num+=1
                        else:
                            new_fused_candidate_list=fused_candidate_list
                if len(new_fused_candidate_list)>0:
                    review_product_json["fused_candidate_list"]=new_fused_candidate_list
                else:
                    review_product_json["fused_candidate_list"]=fused_candidate_list
                
                out_list.append(review_product_json)

    with open(out_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4)   
    print(filter_num)
    
    

def method2(candidate_id,product_dict,attribute_gpt_in_review,attribute_ocr_in_review,attribute_exact,filter_by_attribute_field,review_product_json,is_only_brand):
    if is_attribute_match(product_dict[candidate_id], is_only_brand,filter_by_attribute_field,review_product_json["predicted_attribute_ocr"],is_allow_empty=False): 
        return False
    # if is_attribute_match(product_dict[candidate_id], is_only_brand,filter_by_attribute_field,review_product_json["predicted_attribute_exact"],is_allow_empty=False): 
    #     return False
    if is_attribute_match(product_dict[candidate_id], is_only_brand,filter_by_attribute_field,review_product_json["predicted_attribute"],is_allow_empty=True): 
        return False
    else:
        return True
        
     
def method4(candidate_id,product_dict,attribute_gpt_in_review,attribute_ocr_in_review,attribute_exact,filter_by_attribute_field,review_product_json,is_only_brand):
    temp_attribute_in_product=gen_gold_attribute(product_dict[candidate_id]["Spec"],product_dict[candidate_id]["product_name"],dataset_class="v6",is_list=False)
    attribute_in_product=copy.deepcopy(temp_attribute_in_product)
    for attribute_key, attribute_value in attribute_gpt_in_review.items():
        
        if attribute_key not in attribute_in_product  :
            return True
        
    if is_attribute_match(product_dict[candidate_id], is_only_brand,filter_by_attribute_field,attribute_gpt_in_review ): 
        return False
    else:
        return True
            
def method1(candidate_id,product_dict,attribute_gpt_in_review,attribute_ocr_in_review,attribute_exact,filter_by_attribute_field,review_product_json,is_only_brand):
    if is_attribute_match(product_dict[candidate_id], is_only_brand,filter_by_attribute_field,attribute_gpt_in_review): 
        return False
    else:
        return True
    # else:
    #     if is_attribute_match(product_dict[candidate_id], is_only_brand,filter_by_attribute_field,attribute_ocr_in_review,is_allow_empty=True):    
        
    #         return False
    #     else:
    #         return True    
    
def is_filter_candidate_list( candidate_id,product_dict,attribute_gpt_in_review,attribute_ocr_in_review,attribute_exact,filter_by_attribute_field
                             ,review_product_json,is_only_brand=False,filter_method="m1"):
    
    if filter_method=="m1":
        return method1(candidate_id,product_dict,attribute_gpt_in_review,attribute_ocr_in_review,attribute_exact,filter_by_attribute_field,review_product_json,is_only_brand)
    elif filter_method=="m2":
        return method2(candidate_id,product_dict,attribute_gpt_in_review,attribute_ocr_in_review,attribute_exact,filter_by_attribute_field,review_product_json,is_only_brand)
    elif filter_method=="m4":
        return method4(candidate_id,product_dict,attribute_gpt_in_review,attribute_ocr_in_review,attribute_exact,filter_by_attribute_field,review_product_json,is_only_brand)
    else:
        if candidate_id==-1:
            return True
        if is_attribute_match(product_dict[candidate_id], is_only_brand,filter_by_attribute_field,review_product_json["predicted_attribute_model_version_to_product_title"],is_allow_empty=False):    #,is_allow_empty=True
            
            return False
        elif is_attribute_match(product_dict[candidate_id], is_only_brand,filter_by_attribute_field,review_product_json["predicted_attribute_ocr"],is_allow_empty=False):    #,is_allow_empty=True
            
            return False
        elif is_attribute_match(product_dict[candidate_id], is_only_brand,filter_by_attribute_field,review_product_json["predicted_attribute_model_version"],is_allow_empty=False): 
            return False
        # else:
        #     return True
        elif is_attribute_match(product_dict[candidate_id], is_only_brand,filter_by_attribute_field,review_product_json["predicted_attribute_exact"],is_allow_empty=False):    #,is_allow_empty=True
            
            return False
        elif is_attribute_match(product_dict[candidate_id], is_only_brand,filter_by_attribute_field,review_product_json["predicted_attribute_chatgpt"] ):    #,is_allow_empty=True
            
            return False
        else:
            return True
            
                
    



def filter_by_category(review_path,out_path,k=10000):
    
    product_path=Path(products_path_str)
    with open(product_path, 'r', encoding='utf-8') as fp:
        products_url_json_array = json.load(fp)
        product_dict=json_to_dict(products_url_json_array)
    
    incomplete_products_path = Path(
        review_path
    ) 
 
    out_list=[]
    filter_num=0
    is_only_brand=False 
    filter_review_num=0
    # tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    with open(incomplete_products_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        for i,review_product_json in tqdm(enumerate(incomplete_dict_list) ):
            
            fused_candidate_list=review_product_json["fused_candidate_list"]
            new_fused_candidate_list=[]
            mention_to_category_list=review_product_json["mention_to_category_list"]
            mention_to_product_id_list=review_product_json["mention_to_product_id_list"]
            if mention_to_category_list is not None and len(mention_to_category_list)>0:
                for candidate_id in fused_candidate_list :
                    candidate_product_category=product_dict[candidate_id]["product_category"]
                    if candidate_product_category in mention_to_category_list[:k] and candidate_id in mention_to_product_id_list:
                        new_fused_candidate_list.append(candidate_id)
                    else:
                        filter_num+=1
            else:
                new_fused_candidate_list=fused_candidate_list
            if len(new_fused_candidate_list)>0:
                review_product_json["fused_candidate_list"]=new_fused_candidate_list
                out_list.append(review_product_json)
            else:
                filter_review_num+=1
                out_list.append(review_product_json)

    with open(out_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4)   
    # print(filter_num,filter_review_num)
    
  

def is_attribute_match(candidate_product_json, is_only_brand,filter_by_attribute_field,attribute_in_review,is_allow_empty=True):
    if  len(attribute_in_review )==0 :
        if not is_allow_empty:
            return False
     
            
    
    temp_attribute_in_product=gen_gold_attribute(candidate_product_json["Spec"],candidate_product_json["product_name"],dataset_class="v6",is_list=False)
    attribute_in_product=copy.deepcopy(temp_attribute_in_product)
    if filter_by_attribute_field!="all":
        for attribute_key in filter_by_attribute_field:
            if attribute_key in attribute_in_review:
                attribute_in_review={attribute_key:[one_attribute_in_review.lower() for one_attribute_in_review in attribute_in_review[attribute_key]]}
                
                if attribute_key in attribute_in_review  and   attribute_key in attribute_in_product    and attribute_in_product[attribute_key] not in attribute_in_review[attribute_key]:
                    return False 
    else:
        for attribute_key, attribute_value in attribute_in_review.items():
            # if attribute_key :#not in ["Product Title","Product Name"]
            if attribute_key in attribute_in_product  :  
                is_product_attribute_list=isinstance(attribute_in_product[attribute_key], list)
                if isinstance(attribute_in_review[attribute_key], list):
                    for attribute_value_in_review in attribute_in_review[attribute_key]:
                        if not is_product_attribute_list:
                            if attribute_in_product[attribute_key].lower() !=attribute_value_in_review.lower():
                                return   False 
                        elif attribute_value_in_review not in attribute_in_product[attribute_key]:
                            return False
                      
                else:
                    attribute_value_in_review=attribute_in_review[attribute_key]
                    if not is_product_attribute_list:
                        if attribute_in_product[attribute_key].lower() !=attribute_value_in_review.lower():
                            return   False 
                    elif attribute_value_in_review not in attribute_in_product[attribute_key]:
                        return False
                        
                # if attribute_in_product[attribute_key] !=attribute_in_review[attribute_key] and attribute_in_review[attribute_key] not in attribute_in_product[attribute_key]:
                    
                #     return False
         
    return True 



class FilterByAttribute:
    def __init__(self,test_dir,predicted_attribute_field="predicted_attribute",filter_method="m1") -> None:
        self.attribute_source="review"
        self.attribute_logic="ocr_gpt2"#numeral exact
        filter_by_attribute_field="all"#Brand,Color,Model Version"#Color  ,Model Number,Product Name "all"#
        self.attribute_field="all"
        self.mode="test" 
        self.predicted_attribute_field=predicted_attribute_field
        # data_with_attribute_path = Path(
        #     f"{data_dir}{self.mode}/attribute/bestbuy_review_2.3.16.29.6_fused_score_20_54_gpt2_brand_ocr_all.json"
        # )
        product_path=Path(products_path_str  )
        with open(product_path, 'r', encoding='utf-8') as fp:
            products_url_json_array = json.load(fp)
            self.product_dict=json_to_dict(products_url_json_array)
        is_only_brand=False 
        if filter_by_attribute_field !="all":
            self.filter_by_attribute_field=filter_by_attribute_field.split(",")
        else:
            self.filter_by_attribute_field="all"
        self.filter_method=filter_method
        with open( test_dir, 'r', encoding='utf-8') as fp:
            product_dataset_json_array = json.load(fp)
            self.review_product_json_with_attribute_dict=review_json_to_product_dict(product_dataset_json_array)
        
    def is_filtered_by_attribute(self,candidate_id ,review_id):
        review_product_json=self.review_product_json_with_attribute_dict[review_id]
        if self.predicted_attribute_field in review_product_json:
            attribute_gpt_in_review=review_product_json[self.predicted_attribute_field]#predicted_attribute
            attribute_ocr_in_review=review_product_json["predicted_attribute_ocr"]
        # attribute_exact=review_product_json["predicted_attribute_exact"]
        attribute_exact=None
        return is_filter_candidate_list( candidate_id,self.product_dict,attribute_gpt_in_review,attribute_ocr_in_review,attribute_exact,
                                        self.filter_by_attribute_field,review_product_json,filter_method=self.filter_method)

def filter_by_attribute(data_with_disambiguation_ranking_path,version="114"):
    attribute_source="review"
    attribute_logic="ocr_gpt2"#numeral exact
    filter_by_attribute_field="Brand,Color,Model Number,Product Name"#Color  ,Model Number,Product Name
    attribute_field="all"
    mode="test"
    data_with_attribute_path = Path(
        f"{data_dir}test/attribute/bestbuy_review_2.3.16.29.6_fused_score_20_54_gpt2_brand_ocr_all.json"
        # f"/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v5/test/retrieval/bestbuy_review_2.3.16.29.6_fused_score_20_54_attribute_gpt2_brand.json"
    )
 
    out_path  =  f'{data_dir}{mode}/disambiguation/bestbuy_review_2.3.16.29.12_attribute_by_{attribute_source}_{attribute_field}_{attribute_logic}_Brand,Color,Model_Number,Product_Name_{version}.json'#brand_color_
 
 
    filter_non_attribute_main( data_with_attribute_path,data_with_disambiguation_ranking_path,out_path,filter_by_attribute_field)
    return out_path

import argparse
def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',type=str,help=" ",default="mocheg2/test") #politifact_v3,mode3_latest_v5
    parser.add_argument('--out_path',type=str,help=" ",default="bestbuy/output/html")
    # parser.add_argument('--example_type',type=str,help=" ",default="verification")
    parser.add_argument('--mode',type=str,help=" ",default="test")
    
    # parser.add_argument('--generate_claim_level_report',type=str,help=" ",default="y")
    args = parser.parse_args()
    return args


  
  
  
  
if __name__ == '__main__':
    args = parser_args()
    data_path=args.data_path  
    out_path=args.out_path
    mode=args.mode
    attribute_source="review"
    attribute_logic="ocr_gpt2"#numeral exact
    filter_by_attribute_field="Brand,Color"#Color  ,Model Number,Product Name
    attribute_field="all"
    # mode
    data_with_disambiguation_ranking_path = Path(
        # f"output/disambiguation/bestbuy_100_human_performance_added.json"
        "/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v5/test/retrieval/bestbuy_review_2.3.16.29.11.3_standard_rerank_fine_tune_114.json"
    )
    data_with_attribute_path = Path(
        f"/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v5/test/attribute/bestbuy_review_2.3.16.29.6_fused_score_20_54_gpt2_brand_ocr_all.json"
        # f"/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v5/test/retrieval/bestbuy_review_2.3.16.29.6_fused_score_20_54_attribute_gpt2_brand.json"
    )
    out_path = Path(
        f'/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v5/{mode}/retrieval/bestbuy_review_2.3.16.29.12_attribute_by_{attribute_source}_{attribute_field}_{attribute_logic}_{filter_by_attribute_field}_114.json'#brand_color_
    )
    # data_with_disambiguation_ranking_path = Path(
    #     f"/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v5/test/retrieval/bestbuy_review_2.3.16.29.5_fused_score_20_54.json"
    # )
    # out_path = Path(
    #     f'/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v5/test/retrieval/bestbuy_review_2.3.16.29.6.1_out_fused_score_20_54.json'#brand_color_
    # )
    filter_non_attribute_main( data_with_attribute_path,data_with_disambiguation_ranking_path,out_path,filter_by_attribute_field)
    