

import os 
import pandas as pd 
 
from random import sample
import json
from pathlib import Path
from time import time
from pathlib import Path
import json
import pandas as pd
import requests
import os
import copy
import random 
from disambiguation.utils.post_process import post_process     
from util.env_config import * 
from tqdm import tqdm
from ast import literal_eval


def get_top_k():
    dataset_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v5/test/attribute/bestbuy_review_2.3.16.29.14_114_select_one_image_from_0_to_1000000.json"
    output_products_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v5/test/attribute/bestbuy_review_2.3.16.29.14.2_114_top5.json"
    with open( dataset_path, 'r', encoding='utf-8') as fp:
        out_list=[]
        product_dataset_json_array = json.load(fp)
        for one_example in product_dataset_json_array:
            fused_candidate_list=one_example["fused_candidate_list"]
            one_example["fused_candidate_list"]=fused_candidate_list[:5]
            out_list.append(one_example)
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4)
        
        
get_top_k()        