from numpy import product
import pandas as pd
import requests
import os
from pathlib import Path
from typing import Dict
from tqdm import tqdm
from ast import literal_eval

import requests
import json
from typing import Dict
from pathlib import Path
from time import time
from tqdm import tqdm
from lxml.html import fromstring 
import concurrent.futures
import logging
from bestbuy.src.extract_mention.name_entity_recognition.evaluate import NpEncoder, gen_one_dict_for, gen_one_list_for, prepare_review

from bestbuy.src.extract_mention.name_entity_recognition.generate_mention import ParserMentionGenerator, ParserProductAliasGenerator
from bestbuy.src.extract_mention.name_entity_recognition.util.data_util import prepare_produt  
version="v38"
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT,filename=f'bestbuy/output/log/example_200_{version}.log',   filemode='w',level=logging.INFO)

 
 
# scraper
def bs4_review_scraper(product_dict_list,mention_generator,config):
    # if 
    new_product_dict_list=[]
    for product_dict in product_dict_list:
        
        if len(product_dict["reviews"])>0:
            review_sub_json=product_dict["reviews"][0]
            gt_mention_list,gt_mention,review_text,review_id=prepare_review(review_sub_json)
        else:
            review_sub_json=None
            gt_mention_list,gt_mention,review_text,review_id=None,None,None,None 
        mode="parser"
        product_category,noisy_product_name,product_desc=prepare_produt(product_dict,config["is_product_desc_first_sent"],
                                                                                        config["is_product_category_last_term"],config["product_title_range"])
        
        mention, mention_candidates,mention_candidate_list_before_review_match,detailed_result=mention_generator.generate_mention(review_text,noisy_product_name,product_category,product_desc,gt_mention, config["is_debug"])

        displayed_review=gen_output("fake_mention_to_obtain",mention_candidates,product_dict,review_sub_json,review_text,mode, detailed_result, 
                       mention_candidate_list_before_review_match)
        if len(product_dict["reviews"])>0:
            product_dict["reviews"][0]["mention"]=mention
        product_dict["displayed_review"]=displayed_review
        new_product_dict_list.append(product_dict)

    # return the dictionary
    return new_product_dict_list
 
  
import json 
# def download_by_json():
#     headers = {"User-Agent": "Mozilla/5.0"}
#     db_path = Path('../../data/bestbuy_products_40000_0.05_desc_img_url.json')
#     complete_products_path = Path('../../data/bestbuy_products_40000_0.15_desc_img_url_img_path.json')
#     with open(db_path, 'r', encoding='utf-8') as fp:
#         incomplete_dict_list = json.load(fp)
#         download_images_by_json(incomplete_dict_list,complete_products_path, headers)    
    
# if __name__ == "__main__":
#     download_by_json()

def gen_output(mention,mention_candidates,product_json,review_sub_json,review_text,mode, detailed_result, 
                   mention_candidate_list_before_review_match)    :
        if   mention is not None and mention!="" :
            displayed_review={}
            
            if review_sub_json is not None:
                displayed_review["review_id"]=review_sub_json["id"] 
                if "gt_mention" in review_sub_json:
                    displayed_review[f"gold_mention"]=review_sub_json["gt_mention"]
            displayed_review[f"predicted_mention"]=mention 
            if mode !="all":
                displayed_review[f"mention_candidates_{mode}"]=mention_candidates
            else:
                displayed_review[f"mention_candidates_parser"]=detailed_result["parser_mention_candidates"]
                 
                if "overlap_mention_candidates" in detailed_result:
                    displayed_review[f"mention_candidates_overlap"]=detailed_result["overlap_mention_candidates"]
            if "score" in detailed_result:
                displayed_review["similarity_score"]=detailed_result["score"]
            
            displayed_review["product_category"]=detailed_result["product_category"] 
            displayed_review["product_title"]=detailed_result["product_title"]
            displayed_review["ori_product_category"]=product_json["product_category"] 
            displayed_review["ori_product_title"]=product_json["product_name"]
            if "overview_section" in product_json and product_json["overview_section"] is not None:
                displayed_review["product_desc"]=product_json["overview_section"]["description"]
            displayed_review["review"]=review_text
            displayed_review["mention_candidate_list_before_review_match"]=gen_one_dict_for(mention_candidate_list_before_review_match)
            if "product_category_chunk" in detailed_result:
                displayed_review["product_category_chunk"]=detailed_result["product_category_chunk"]
                displayed_review["product_title_chunk"]=detailed_result["product_title_chunk"]
            else:
                displayed_review["product_category_chunk"]=""
                displayed_review["product_title_chunk"]=""
            
            displayed_review[f"url"]=product_json["url"]
                
            
            
            return displayed_review 
        else:
            return None 


# config to generate the originia mention in dataset 
# config={"is_debug":True,"similarity_threshold":0.3,"similarity_target":"merge2","parser_setting":"no_desc","parser_chunk_setting":"root_only",
#         "product_title_range":"no_brand","is_product_desc_first_sent":True   ,
#             "is_product_category_last_term":True }


config={"is_debug":True,"similarity_threshold":0.3,"similarity_target":"merge2","parser_setting":"no_desc","parser_chunk_setting":"root_only",
        "product_title_range":"no_brand","is_product_desc_first_sent":True   ,
            "is_product_category_last_term":True }

class Counter:
    def __init__(self,positive_num):
        self.non_empty_prediction_num=0
        self.empty_prediction_before_similarity_ranking_num=0
        self.empty_prediction_before_review_match_num=0
        self.positive_num=positive_num #precision
        
    def check_after_similarity_ranking(self, mention):
     
        if mention is None:
            self.empty_prediction_before_review_match_num+=1
        elif mention =="":
            self.empty_prediction_before_similarity_ranking_num+=1
        else:
            self.non_empty_prediction_num+=1
            
    def log_metric(self,review_num_so_far):
        logging.info(f"review_num:{review_num_so_far}, non_empty_prediction_ratio:{self.non_empty_prediction_num/self.positive_num}, "+
                  f"empty_prediction_before_similarity_ranking_ratio:{self.empty_prediction_before_similarity_ranking_num/self.positive_num}, "+
                  f"empty_prediction_before_review_match_ratio:{self.empty_prediction_before_review_match_num/self.positive_num}, "+
                f"non_empty_prediction_num:{self.non_empty_prediction_num}, empty_prediction_before_similarity_ranking_num:{self.empty_prediction_before_similarity_ranking_num}, "+
                f"empty_prediction_before_review_match_num:{self.empty_prediction_before_review_match_num}, "+
                f"positive_num:{self.positive_num}")
            

import time
import datetime
from dateutil.relativedelta import relativedelta

class Timer(object):
    """Computes elapsed time."""
    def __init__(self, name='default'):
        self.name = name
        self.running = True
        self.total = 0
       
        self.start_time = datetime.datetime.now()
        print("<> <> <> Starting Timer [{}] at {} <> <> <>".format(self.name,self.start_time))
 

    def remains(self, total_task_num,done_task_num):
        now  = datetime.datetime.now()
        #print(now-start)  # elapsed time
        if done_task_num==0:
            done_task_num=1
        left = (total_task_num - done_task_num) * (now - self.start_time) / done_task_num
        sec = int(left.total_seconds())
        
        rt = relativedelta(seconds=sec)
     
        return "remaining time: {:02d} hours {:02d} minutes {:02d} seconds".format(int(rt.hours), int(rt.minutes), int(rt.seconds))    
from itertools import zip_longest
def group_elements(n, iterable, padvalue=""):
    return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)
        
import copy         
def main(incomplete_products_path_str,out_file,start_id,end_id,step_size_per_cpu,args):
   
    # incomplete_products_path = Path(
    #     f'bestbuy/data/bestbuy_products_40000_3.2_desc_img_url.json'
    # )
    # complete_products_path = Path(
    #     f'bestbuy/data/bestbuy_products_40000_3.5_desc_img_url_img_path_from_{start_id}_to_{end_id}.json'
    # )
    incomplete_products_path = Path(
        incomplete_products_path_str
    )
    complete_products_path = Path(
        f'{out_file}_{version}_from_{start_id}_to_{end_id}.json'
    )
    display_running_process_path = Path(
        f'bestbuy/output/output/display_{version}_from_{start_id}_to_{end_id}.json'
    )
     
    with open(incomplete_products_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
    counter=Counter(len(incomplete_dict_list))
 
    output_list = []
    displayed_review_list=[]
    print_ctr = 0
    result = []
    save_step=2000
     
    is_parallel=False 
    max_worker_num=args.max_worker_num #48
    step_size=step_size_per_cpu*max_worker_num
    if args.mode=="product_alias":
        mention_generator_list=[ ParserProductAliasGenerator( config["is_debug"], config["similarity_threshold"], config["similarity_target"],
                                                        config["parser_setting"],  config["parser_chunk_setting"], config["product_title_range"])  for i in range(max_worker_num)]
    else:
        mention_generator_list=[ ParserMentionGenerator( config["is_debug"], config["similarity_threshold"], config["similarity_target"],
                                                        config["parser_setting"],  config["parser_chunk_setting"], config["product_title_range"])  for i in range(max_worker_num)]
    config_list=[copy.deepcopy(config)  for i in range(max_worker_num) ]
    # idx_list=[i for i in range(step_size)]
    # crawl_mode_list= [args.crawl_mode for i in range(step_size)]
    timer=Timer()
    total_task=len(incomplete_dict_list)
    for i in tqdm(range(0, len(incomplete_dict_list), step_size_per_cpu*max_worker_num)):
        # if incomplete_dict_list[i]['reviews'][0]["id"]>=21195 :
        #     print(i)
        # else:
        #     continue 
        # if i>=start_id :
        #     if   i<end_id:
        if is_parallel:
            print(100 * '=')
            if len(incomplete_dict_list[i]['reviews'])>0:
                review_id=incomplete_dict_list[i]['reviews'][0]["id"]
                print(f'starting at the {print_ctr}th value, review_id:{review_id}')
                logging.info(f"starting at the {print_ctr}th value, review_id={review_id}")
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_worker_num) as executor:
                try:
                    result = list(
                        
                            executor.map(
                                bs4_review_scraper,
                                group_elements(step_size_per_cpu,incomplete_dict_list[i: i + step_size ]),
                                mention_generator_list,config_list
                            ) 
                    )
                except Exception as e :
                    logging.warning(f"{e}")
                    logging.info(f"error {e}")
                    print(e)
                    result = []
      
        else:
            result=[]
            try:
                for product_json in incomplete_dict_list[i: i + step_size]:
                    
                    one_result=bs4_review_scraper([product_json],mention_generator_list[0],config_list[0] )
                    result.append(one_result)
            except Exception as e :
                logging.warning(f"{e}")
                logging.info(f"error {e}")
                print(e,product_json["id"],product_json["url"])
                result = []
            # end = time()
        if len(result) != 0:
        
            for product_dict in result:
                # if "mention" in product_dict:
                #     mention=product_dict["mention"]
                #     counter.check_after_similarity_ranking(mention)
                # if "displayed_review" in product_dict:
                #     displayed_review=product_dict["displayed_review"]
                #     if displayed_review is not None:
                #         displayed_review_list.append(displayed_review)
                #     product_dict["displayed_review"]=None 
                
                output_list.extend(product_dict)
    
                # counter.log_metric(print_ctr)
        else:
            logging.info('something is wrong')
        
        if   i%save_step==0:
            with open(complete_products_path, 'w', encoding='utf-8') as fp:
                json.dump(output_list, fp, indent=4, cls=NpEncoder)
            with open(display_running_process_path, 'w', encoding='utf-8') as fp:
                json.dump(displayed_review_list, fp, indent=4, cls=NpEncoder)
            logging.info(timer.remains(total_task,print_ctr))
        print_ctr += step_size
    with open(complete_products_path, 'w', encoding='utf-8') as fp:
        json.dump(output_list, fp, indent=4, cls=NpEncoder)
    with open(display_running_process_path, 'w', encoding='utf-8') as fp:
        json.dump(displayed_review_list, fp, indent=4, cls=NpEncoder)
# runner
 
        
import argparse
def get_args():
   


    parser = argparse.ArgumentParser() 
    # parser.add_argument("--lr", default=1e-7, type=float)\
    parser.add_argument("--file",default='bestbuy/data/mention_to_entity_statistics/bestbuy_review_S2.3.2_statistic_remove_empty_review.json', type=str  )# #bestbuy/data/final/v0/bestbuy_review_2.3.7_separate_review_incomplete_with_mention_result_parser
    parser.add_argument("--out_file",default='bestbuy/data/mention_to_entity_statistics/bestbuy_review_statistic_S2.3.3.1_add_mention', type=str  ) 
    parser.add_argument("--step_size", default=64, type=int)
    parser.add_argument("--max_worker_num", default=24, type=int)
    parser.add_argument("--start_id", default=0, type=int)
    parser.add_argument("--end_id", default=5000000000, type=int)
    # parser.add_argument("--use_pre_trained_model", default=True, action="store_false") 
    parser.add_argument("--mode",default="product_alias", type=str  )#mention 
    args = parser.parse_args()

    print(args)
    return args    
   
# runner
if __name__ == "__main__":  
    args=get_args()
    start_id=args.start_id
    end_id=args.end_id
    
    main(args.file,args.out_file,start_id,end_id,args.step_size,args)