import os
from bestbuy.src.util.create_example.example import *
import pandas as pd 
import nltk 
 
import json
from pathlib import Path
from random import sample 
 
def read_example(example_path_str,is_reshuffle ):
     
    review_products_path = Path(
        example_path_str
    )
     
    increased_review_id=0
    review_dict={}
    product_dict={} 
    example_list=[]
    
    with open(review_products_path, 'r', encoding='utf-8') as fp:
        review_products_json_list = json.load(fp)
        for review_product_json in review_products_json_list:
            gold_mention=review_product_json["gold_mention"]
            predicted_mention=review_product_json["predicted_mention"]
            if "similarity_score" in review_product_json:
                similarity_score=review_product_json["similarity_score"]
            else:
                similarity_score=None 
            if "mention_candidate_list_before_review_match" in review_product_json:
                mention_candidates_before_review_match=review_product_json["mention_candidate_list_before_review_match"]
            else:
                mention_candidates_before_review_match=None 
                
            
            product_category=review_product_json["product_category"]
            product_title=review_product_json["product_title"]
            review=review_product_json["review"] 
            review_id=review_product_json["review_id"]
            product_title_chunk=review_product_json["product_title_chunk"]
            product_category_chunk=review_product_json["product_category_chunk"]
            url=review_product_json["url"]
            example=Example(review_id,url,product_category_chunk,product_title_chunk,gold_mention,predicted_mention,similarity_score,mention_candidates_before_review_match,product_category,product_title,review)
            example_list.append(example)
            increased_review_id+=1 
        
     
    return example_list 

    