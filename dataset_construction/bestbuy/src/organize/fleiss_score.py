import statsmodels 
import pandas as pd
import json
import pandas as pd 
import numpy as np 
import random
from  statsmodels.stats import inter_rater as inter_rater
def excel_to_dict(annotator1_answer):
    answer_dict={}
    df1 = pd.read_excel(annotator1_answer )
    for i,row in df1.iterrows():
        claim_id=row["review_id"]
        human_predict=row["human_predict"]
        gold_label=row["reshuffled_target_product_id_position"]
        if not pd.isna(claim_id):
            answer_dict[claim_id]=human_predict 
         
    return answer_dict 


def gen_disambiguation_review_id_list(new_crawled_products_url_json_array):
    disambiguation_list=[]
    end_to_end_no_retrieval_error_list=[]
    for one_example in new_crawled_products_url_json_array:
        sample=one_example["sample"]
        review_id=one_example["review_id"]
        gold_product_id=one_example["gold_entity_info"]["id"]
        candidate_product_id_list=one_example["fused_candidate_list"]
        if sample=="disambiguation":
            disambiguation_list.append(review_id)
        else:
            if gold_product_id in candidate_product_id_list:
                end_to_end_no_retrieval_error_list.append(review_id)
    end_to_end_30_no_retrieval_error_list=random.sample(end_to_end_no_retrieval_error_list,30)
    disambiguation_list.extend(end_to_end_30_no_retrieval_error_list)
    
 
    # Define the file name
    json_file = "/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/disambiguation/example/disambiguation_list.json"

    # Open the file in write ('w') mode
    with open(json_file, "w") as file:
        # Use the json.dump() method to write the list to the file in JSON format
        json.dump(disambiguation_list, file)
    return disambiguation_list
            

def get_review_id_list():
    output_products_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/disambiguation/example/temp_bestbuy_review_2.3.17.11.20.1.1_test_sample_50_v1_5.json"
    with open(output_products_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)    
    end_to_end_review_id_list=[]
    for one_example in new_crawled_products_url_json_array:
        sample=one_example["sample"]
        review_id=one_example["review_id"]
        if sample=="end_to_end":
            end_to_end_review_id_list.append(review_id)
    print(len(end_to_end_review_id_list))
    
    disambiguation_list=gen_disambiguation_review_id_list(new_crawled_products_url_json_array)
    return end_to_end_review_id_list    ,disambiguation_list
        
        
        
def inter_agreement(mode,end_to_end_review_id_list,disambiguation_list):
    print(mode) 
    annotator1_answer="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/example/human_experiment/report0.xlsx"
    annotator2_answer="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/example/human_experiment/report12.xlsx"
    
    answer_dict1=excel_to_dict(annotator1_answer)
    answer_dict2=excel_to_dict(annotator2_answer)
    data_list=[]
    correct_num=0
    
    for claim_id in answer_dict1.keys():
        if mode=="end_to_end":
            if claim_id in end_to_end_review_id_list:
                human_predict1=answer_dict1[claim_id]
                human_predict2=answer_dict2[claim_id]
                data_list.append([human_predict1,human_predict2])
        else:
            if claim_id in disambiguation_list:
                human_predict1=answer_dict1[claim_id]
                human_predict2=answer_dict2[claim_id]
                data_list.append([human_predict1,human_predict2])
            
        
    data_for_fleiss,categories=inter_rater.aggregate_raters(np.array(data_list))
    score= inter_rater.fleiss_kappa(data_for_fleiss,method="fleiss")
    print(score)
    
def accuracy(mode,end_to_end_review_id_list,disambiguation_list):
    
    annotator1_answer="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/example/human_experiment/report0.xlsx"
    annotator2_answer="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/example/human_experiment/report12.xlsx"
    
    df1 = pd.read_excel(annotator1_answer )
    df2 = pd.read_excel(annotator2_answer )
    correct_num=0
    for (idx1,row1),(idx2,row2) in zip(df1.iterrows(),df2.iterrows()):
        claim_id=row1["review_id"]
        human_predict1=row1["human_predict"]
        human_predict2=row2["human_predict"]
        gold_predict=row1["reshuffled_target_product_id_position"]
        if not pd.isna(claim_id):
            if mode=="end_to_end":
                if claim_id in end_to_end_review_id_list:
                    if gold_predict ==human_predict1  and gold_predict ==human_predict2  :
                        correct_num+=1
            else:
                if claim_id in disambiguation_list:
                    if gold_predict ==human_predict1  and gold_predict ==human_predict2  :
                        correct_num+=1
    print(correct_num,correct_num/50)
       
       
       
end_to_end_review_id_list,disambiguation_list=get_review_id_list()       
inter_agreement("end_to_end",end_to_end_review_id_list,disambiguation_list)
accuracy("end_to_end",end_to_end_review_id_list ,disambiguation_list)

inter_agreement("disambiguation",end_to_end_review_id_list,disambiguation_list)
accuracy("disambiguation",end_to_end_review_id_list ,disambiguation_list)