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
annotator_dict={
    "1":"Barry",
"2":"Zhiyang",
"3":"Minqian",
"4":"Ying",
"5":"Sijia",
"6":"Jingyuan",
"7":"Jeet",
"8":"Zoe",
"9":"Mingchen",
"10":"Pritika",
"11":"Samhita",
"12":"Sai",
"13":"Mo"
}
def unzip_file():
    from zipfile import ZipFile
  
    # loading the temp.zip and creating a zip object
    with ZipFile("bestbuy/data/example/drive-download-20230103T195001Z-001.zip", 'r') as zObject:
    
        # Extracting all the members of the zip 
        # into a specific location.
        zObject.extractall(
            path="bestbuy/data/example/annotation")


def gen_reivew_gold_product_dict():
    gold_product_dict={}
    gold_report_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/multimodal_entity_linking/output/example_answer/report0102.csv"
    # "/home/menglong/workspace/code/multimodal/multimodal_entity_linking/multimodal_entity_linking/output/report_0101.csv"
    report_df=pd.read_csv(gold_report_path)
    for idx,(_, row) in enumerate(report_df.iterrows()):
        
        review_id=row["review_id"] 
        reshuffled_target_product_id_position=row["reshuffled_target_product_id_position"]
        gold_product_dict[review_id]=reshuffled_target_product_id_position
    return gold_product_dict

def get_score():
    # unzip_file()
    gold_product_dict=gen_reivew_gold_product_dict()
    
    report_dir="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/example/annotation"
    table_output_path="bestbuy/output/human_experiment_0202.csv"
 
    total_valid_num=0
    total_right_num=0
    record_list=[]
    report_path_list=os.listdir(report_dir)
    for report_path in  report_path_list :
        report_df=pd.read_excel(os.path.join(report_dir,report_path), index_col=0,skiprows=0)
        annotator_id=1+int(report_path.split(".xlsx")[0].split("report")[1])
        annotator=annotator_dict[str(annotator_id)]
        valid_num=0
        right_num=0
        for idx,(_, row) in enumerate(report_df.iterrows()):
         
            review_id=row["Review Id"] 
            human_prediction=row["Target Product Position (1-21)"]
      
                
            if not pd.isna(human_prediction):
                gold_product=gold_product_dict[review_id]
                valid_num+=1
                if gold_product==human_prediction:
                    right_num+=1
        if valid_num>0:
            # record_list.append([annotator_id,annotator,right_num,valid_num,round(right_num/valid_num,2)])
            record_list.append([annotator_id,annotator,right_num,valid_num,round(right_num/valid_num,4)])
        else:
            record_list.append([annotator_id,annotator, right_num,valid_num,0])
        total_right_num+=right_num
        total_valid_num+=valid_num
    record_list.append(["Overall","Overall", total_right_num,total_valid_num,round(total_right_num/total_valid_num,4)])
    df = pd.DataFrame(record_list, columns =['Annotation Set' ,'Annotator','Correct Example','Finished Example','Accuracy' ])#'Annotator',
    df.to_csv(table_output_path,index=False)   
    print(total_right_num,total_valid_num,total_right_num/total_valid_num)
    



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
    return end_to_end_review_id_list    
        
        
def get_one_score():
    # unzip_file()
    # gold_product_dict=gen_reivew_gold_product_dict()
    end_to_end_review_id_list=get_review_id_list()
    
    report_dir="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/example/human_experiment"
    table_output_path="bestbuy/output/human_experiment.csv"
 
    total_valid_num=0
    total_right_num=0
    record_list=[]
    report_path_list=os.listdir(report_dir)
    for report_path in  report_path_list :
        report_df=pd.read_excel(os.path.join(report_dir,report_path), index_col=0,skiprows=0)
        annotator_id=1+int(report_path.split(".xlsx")[0].split("report")[1])
        annotator=annotator_dict[str(annotator_id)]
        valid_num=0
        right_num=0
        for idx,(_, row) in enumerate(report_df.iterrows()):
         
            review_id=row["review_id"] 
            
            human_prediction=row["human_predict"]
            gold_answer=row["reshuffled_target_product_id_position"]
            if review_id in end_to_end_review_id_list:
                if gold_answer==human_prediction:
                    right_num+=1
                valid_num+=1
                
          
        if valid_num>0:
            # record_list.append([annotator_id,annotator,right_num,valid_num,round(right_num/valid_num,2)])
            record_list.append([annotator_id,annotator,right_num,valid_num,round(right_num/valid_num,2)])
            print(annotator,round(right_num/valid_num,2))
        else:
            record_list.append([annotator_id,annotator,right_num,valid_num,0])
        
        total_right_num+=right_num
        total_valid_num+=valid_num
    df = pd.DataFrame(record_list, columns =['Annotation Set' ,'Annotator','Correct Example','Finished Example','Accuracy' ])#'Annotator',
    df.to_csv(table_output_path,index=False)   
    print(total_right_num,total_valid_num,total_right_num/total_valid_num)
        
   
if __name__ == "__main__":      
    get_score()