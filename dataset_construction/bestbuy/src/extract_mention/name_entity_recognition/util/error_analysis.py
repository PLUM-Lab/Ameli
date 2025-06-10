from matplotlib.font_manager import weight_dict
from nltk.tokenize import sent_tokenize, word_tokenize
from pathlib import Path
from time import time
from pathlib import Path
import json
import pandas as pd
import argparse 
from sentence_transformers.cross_encoder import CrossEncoder
import numpy as np

from nltk.tokenize import sent_tokenize, word_tokenize 
import nltk 
import logging
from tqdm import tqdm
def analyze():
    review_json_path = Path(
        f'bestbuy/data/example/bestbuy_review_200_cleaned_annotation_result_all_v6.json'
    )        
    no_gold_num=0
    parser_num=0
    gold_exist_one_side_num=0
    order_num=0
    weight_num=0
    score_num=0
    with open(  review_json_path, 'r', encoding='utf-8') as fp:
        review_json_array = json.load(fp)
         
        for idx,review_json   in tqdm(enumerate(review_json_array)):
            gold_mention=review_json["gold_mention"]
            gold_mention_1=gold_mention[0]
            if gold_mention_1=="":
                no_gold_num+=1
            else:
                predicted_mention_product_category_score=-1
                gold_mention_product_category_score=-1
                predicted_mention=review_json["predicted_mention"]
                mention_candidates_parser=review_json["mention_candidates_parser"]
                mention_candidates_overlap=review_json["mention_candidates_overlap"]
                if len(mention_candidates_parser)+len(mention_candidates_overlap)==0:
                    parser_num+=1
                else:
                    scores=review_json["similarity_score"]
                    for _,score_object in scores.items():
                        if score_object["mention"]==predicted_mention:
                            predicted_mention_product_category_score=score_object["product_category_score"]
                            predicted_mention_product_category_chunk_score=score_object["product_category_chunk_score"]
                            predicted_mention_product_title_score=score_object["product_title_score"]
                            predicted_mention_product_title_chunk_score=score_object["product_title_chunk_score"]
                        elif score_object["mention"]==gold_mention_1:
                            gold_mention_product_category_score=score_object["product_category_score"]
                            gold_mention_product_category_chunk_score=score_object["product_category_chunk_score"]
                            gold_mention_product_title_score=score_object["product_title_score"]
                            gold_mention_product_title_chunk_score=score_object["product_title_chunk_score"]
                    if gold_mention_product_category_score==-1:
                        parser_num+=1
                    else:
                        threshold=0.1
                        if (gold_mention_product_category_score < threshold and  gold_mention_product_category_chunk_score<threshold) or (gold_mention_product_title_score < threshold and  gold_mention_product_title_chunk_score<threshold):
                            gold_exist_one_side_num+=1
                        else:
                            if gold_mention_product_category_score>predicted_mention_product_category_score and gold_mention_product_title_score>predicted_mention_product_title_score:
                                order_num+=1
                            elif gold_mention_product_category_score <predicted_mention_product_category_score and gold_mention_product_title_score<predicted_mention_product_title_score:
                                score_num+=1
                            else:
                                weight_num+=1
        print(f"{no_gold_num}, {parser_num}, {gold_exist_one_side_num}, {order_num}, {weight_num},{score_num}, total {len(review_json_array)}")
                
                    
                    
analyze()                    