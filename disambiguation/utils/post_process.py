

from disambiguation.utils.disambiguation_util import calc_metric, save_prediction
from retrieval.organize.filter import FilterByAttribute, filter_by_attribute
from retrieval.scorer import product_recall_separately
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
from util.env_config import * 
from tqdm import tqdm

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report

from util.read_example import get_father_dir

def obtain_filtered_y_pred_position_list_and_y_true_position_list(disambiguation_result_path ,test_dir,max_candidate_num,disambiguation_result_postfix,is_save=True):
    filter=FilterByAttribute(test_dir)
    corpus_df = pd.read_csv(disambiguation_result_path ,encoding="utf8")  
 
    new_corpus_df=corpus_df
    
     
    total_order_list=[]
    total_entity_id_list=[]
    total_query_id_list=[]
    
    output_list=[]
    
    y_true_list=[]
    filter_y_predict_list=[]
    if "text_score" in new_corpus_df.columns:
        text_score=new_corpus_df["text_score"].tolist()
        image_score=new_corpus_df["image_score"].tolist()
    else:
        text_score=None
        image_score=None
    for index, row in new_corpus_df.iterrows():
        review_id =row["query_id"]
        y_pred=row["predict_entity_position"]
        y_true=row["gold_entity_position"]
        
        y_true_list.append(y_true)
        entity_id_list_str=row["entity_id_list"]
        order_list_str=row["order_list"]
        entity_id_list=entity_id_list_str.split("|")
        order_list=order_list_str.split("|")
        
        ordered_candidate_id_list=[]
        idx=0
        filter_y_pred_position=int(order_list[0])  #max_candidate_num+1
        if y_pred==max_candidate_num+1:
            filter_y_pred_position=y_pred
        else:
            for  idx,order in enumerate(order_list):
                entity_id= entity_id_list[int(order)]
                is_filter=filter.is_filtered_by_attribute( int(float(entity_id)),review_id)
                if not is_filter:
                    filter_y_pred_position=int(order)
                    break
            if is_filter==True:
                idx=0
        filter_y_predict_list.append(filter_y_pred_position)
        total_order_list.append( order_list[idx:]) 
        total_entity_id_list.append( entity_id_list ) 
        total_query_id_list.append(review_id)
    checkpoint_dir=get_father_dir(disambiguation_result_path)
    save_prediction(total_query_id_list,filter_y_predict_list,y_true_list,total_order_list,total_entity_id_list,"test",
                    checkpoint_dir,None,disambiguation_result_file_name=f"filter_entity_link_reproduce_{disambiguation_result_postfix}.csv",
                    text_score_np=text_score,image_score_np=image_score,is_in_list_format=False)
    
    return y_true_list,filter_y_predict_list


def  clean_entity_id_list(entity_id_list):
    clean_list=[]
    for entity_id in entity_id_list:
        cleaned_entity_id=int(float(entity_id))
        clean_list.append(cleaned_entity_id)
    return clean_list 

def obtain_filtered_y_pred_position_list_and_y_true_position_list_for_json(  test_dir,output_products_path,score_key="disambiguation_fused_score",
                                                                           is_retrieval=False,predicted_attribute_field="predicted_attribute",
                                                                           is_save=True,filter_method="m1"):
    with open( test_dir, 'r', encoding='utf-8') as fp:
        data_json_list = json.load(fp)
    filter=FilterByAttribute(test_dir,predicted_attribute_field,filter_method)
    predicted_product_position_list=[]
    gold_product_position_list=[]
    out_list=[]
    filter_gold_num=0
    filter_num=0
    for loop_id,product_dataset_json   in tqdm(enumerate(data_json_list)):
        gold_product_id=product_dataset_json["gold_entity_info"]["id"]
        score_dict=product_dataset_json["fused_score_dict"]
        
        if not is_retrieval  :
            entity_id_list=product_dataset_json["entity_id_list"]   
            entity_id_list=clean_entity_id_list(entity_id_list)
        else:
            entity_id_list=product_dataset_json["fused_candidate_list"][:10]
        score_dict_list=list(score_dict.values()) 
        sorted_score_dict_list= sorted(score_dict_list, key=lambda x: x[score_key], reverse=True)
        candidate_id_list=[]
        predicted_product_id=int(sorted_score_dict_list[0]["corpus_id"])
        review_id=product_dataset_json["review_id"]
        idx=0
        if review_id in [13218]:#4078,
            print("") 
        if predicted_product_id not in entity_id_list:
            print(f"{predicted_product_id},{entity_id_list},{review_id}")
        default_position=entity_id_list.index(predicted_product_id)
        is_filter_gold=False
        for idx,sorted_score_dict in enumerate(sorted_score_dict_list):
            corpus_id=int(sorted_score_dict["corpus_id"])
            is_filter=filter.is_filtered_by_attribute( int(float(corpus_id)),review_id)
            if not is_filter:
                predicted_product_id=corpus_id
                break
            else:
                filter_num+=1
                if corpus_id==gold_product_id:
                    is_filter_gold=True
        if predicted_product_id in entity_id_list:
            predicted_position=entity_id_list.index(predicted_product_id)
            predicted_product_position_list.append(predicted_position)
        else:
            predicted_position=default_position
            predicted_product_position_list.append(default_position)
        
        if gold_product_id in entity_id_list:
            gold_product_position=entity_id_list.index(gold_product_id)
            gold_product_position_list.append(gold_product_position)
        else:
            gold_product_position=10
            gold_product_position_list.append(gold_product_position)
        if is_filter_gold:
            if gold_product_position!=predicted_position:
                filter_gold_num+=1
            # else:
            #     if gold_product_position!=0:
            #         print("")
        if is_filter==True:
            # filter_num-=10
            idx=0
        else:
            candidate_id_list=product_dataset_json["fused_candidate_list"][idx:]
            product_dataset_json["fused_candidate_list"]=candidate_id_list
        out_list.append(product_dataset_json)
        
    if is_save:
        with open(output_products_path, 'w', encoding='utf-8') as fp:
            json.dump(out_list, fp, indent=4)          
    
    print(f"filer:{filter_num}, gold: {filter_gold_num}")
    return gold_product_position_list,predicted_product_position_list



            
   
def check_num(filter_y_predict_list,max_number):
    
    num_dict={}
    for label in range(max_number):
        num=0
        for one_item in filter_y_predict_list:
            if one_item==label:
                num+=1
        num_dict[label]=num
    label =-1
    num=0
    for one_item in filter_y_predict_list:
        if one_item==label:
            num+=1
    num_dict[label]=num
    print(num_dict)

def post_process(disambiguation_result_path,test_dir,max_candidate_num, disambiguation_result_postfix,is_save=True):
    
    y_true_list,filter_y_predict_list=obtain_filtered_y_pred_position_list_and_y_true_position_list(disambiguation_result_path ,
                                                                                                    test_dir,max_candidate_num,disambiguation_result_postfix,is_save)
    max_candidate_num_to_show=max_candidate_num+1
    f1,precision,recall=calc_metric( y_true_list,filter_y_predict_list,  max_candidate_num_to_show )
     
    
    print(f1,precision,recall)
    return f1
    # check_num(filter_y_predict_list,max_candidate_num_to_show)
     
def post_process_for_json( test_dir,max_candidate_num, output_products_path,score_key="disambiguation_fused_score",is_retrieval=False,
                          predicted_attribute_field="predicted_attribute",is_save=True,filter_method="m1"):
    
    y_true_list,filter_y_predict_list=obtain_filtered_y_pred_position_list_and_y_true_position_list_for_json( 
                                                                                                     test_dir,output_products_path,score_key=score_key,is_retrieval=is_retrieval,predicted_attribute_field=predicted_attribute_field,is_save=is_save,filter_method=filter_method)
    max_candidate_num_to_show=max_candidate_num+1
    f1,precision,recall=calc_metric( y_true_list,filter_y_predict_list,  max_candidate_num_to_show ,verbos="y")
    print(f1,precision,recall)
    return f1
  
def gen_result_for_gold_attribute(setting="end_to_end"):
    number=10
    # prediction_path= f"/home/menglong/workspace/code/multimodal/multimodal_entity_linking/multimodal_entity_linking/output/runs/00577-/entity_link_reproduce_{number}.csv"
    
    # test_dir="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/disambiguation/temp/bestbuy_review_2.3.17.13_fused_disambiguation_score048_v2.json"
    # # post_process(disambiguation_result_path,test_dir,number,"")
    # fused_score_data_path=""
    # best_retrieval_weight=None 
    # use_image=True
    if setting =="end_to_end":
        weighted_score_data_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/disambiguation/postprocess/bestbuy_review_2.3.17.11.21_fuse_disambiguation_scoretest1010_2.json"
        filtered_data_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/disambiguation/postprocess/bestbuy_review_2.3.17.11.22_filter_by_attributetest1010_2_gold_annotation.json"
    elif setting=="subset":
        weighted_score_data_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/disambiguation/postprocess/bestbuy_review_2.3.17.11.21_fuse_disambiguation_scoretest1010_2_subset.json"
        filtered_data_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/disambiguation/postprocess/bestbuy_review_2.3.17.11.22_filter_by_attribute_test1010_2_gold_annotation_subset.json"
    
    post_process_for_json(weighted_score_data_path,10, filtered_data_path,score_key="disambiguation_fused_score",is_retrieval=False,predicted_attribute_field="gold_attribute_for_predicted_category")#gold_attribute_for_predicted_category predicted_attribute
    
    
  
def gen_result_for_ghmfc(test_dir,disambiguation_result_path ):
    # run_id_str=str(run_id)
    number=10
    # if setting=="subset":
    #     disambiguation_result_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/MEL-GHMFC/code/data/Richpedia-MEL/output_data/new606_title_end_to_end/checkpoint-450/disambiguation_result.csv" 
    #     test_dir="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/disambiguation/correct_retrieval_subset/bestbuy_review_2.3.17.11.19.20.3.1_correct_retrieval_subset_10_update_gold_attribute_update_candidate.json"
    # else:
    #     disambiguation_result_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/MEL-GHMFC/code/data/Richpedia-MEL/output_data/new606_title_end_to_end/checkpoint-450/disambiguation_result.csv" 
    #     test_dir="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/disambiguation/bestbuy_review_2.3.17.11.20_update_annotated_gold_attribute.json"
    # test_dir="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/disambiguation/temp/bestbuy_review_2.3.17.13_fused_disambiguation_score048_v2.json"
    post_process(disambiguation_result_path,test_dir,number,"")
    # output_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/disambiguation/temp/bestbuy_review_2.3.17.14.1_filter_fused_disambiguation_score048_v2_reproduce.json"
    # post_process_for_json(test_dir,10, output_path,score_key="disambiguation_fused_score",is_retrieval=False,predicted_attribute_field="predicted_attribute")#gold_attribute_for_predicted_category predicted_attribute
  
  

import argparse    
def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',type=str,help=" ",default="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/disambiguation/bestbuy_review_2.3.17.11.20.1_update_annotated_gold_attribute_update_candidate.json")  
    parser.add_argument('--out_path',type=str,help=" ",default="bestbuy/output/html")
    parser.add_argument('--disambiguation_result_path',type=str,help=" ",default="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/MEL-GHMFC/code/data/Richpedia-MEL/output_data/new606_title/checkpoint-240/disambiguation_result.csv")
    parser.add_argument('--process_type',type=str,help=" ",default="ameli")
    parser.add_argument('--setting',type=str,help=" ",default="subset")
    args = parser.parse_args()
    return args


  
if __name__ == '__main__':
    args = parser_args()  
    if args.process_type=="ghmfc":
        gen_result_for_ghmfc(args.data_path,args.disambiguation_result_path )
    else:
        gen_result_for_gold_attribute(setting=args.setting)