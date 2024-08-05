import json
from pathlib import Path
from time import time
from pathlib import Path
import json
import pandas as pd
import requests
import os
from tqdm import tqdm
from ast import literal_eval

def choose_no_retrieval_error():
    out_list=[]
    out_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/disambiguation/correct_retrieval_subset/bestbuy_review_2.3.17.11.19.20.1_correct_retrieval_subset_10.json"
    path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/disambiguation/bestbuy_review_2.3.17.11.19.15.1_remove_to_10_candidate.json"
    with open(path, 'r', encoding='utf-8') as fp:
        product_dataset_json_array = json.load(fp)
        for review_product_json   in tqdm(product_dataset_json_array):
            review_json=review_product_json
            review_id=review_json["review_id"]
            fused_candidate_list=review_product_json["fused_candidate_list"][:10]
            gold_product_id=review_product_json["gold_entity_info"]["id"]
            if gold_product_id in fused_candidate_list:
                out_list.append(review_product_json)
    with open(out_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4)


if __name__ == "__main__":  
    # merge_attribute()
    choose_no_retrieval_error()
    # merge_all()