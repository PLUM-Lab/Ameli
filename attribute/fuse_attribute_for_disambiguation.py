from attribute.scorer import overall_metric
from disambiguation.data_util.inner_util import gen_gold_attribute_without_key_section, json_to_dict, review_json_to_product_dict
from disambiguation.utils.post_process import post_process_for_json
from retrieval.retrieval_organize_main import merge_all_for_review

from util.env_config import * 
import argparse
from collections import Counter
from tqdm import tqdm 
import json
from pathlib import Path

from retrieval.utils.retrieval_util import old_spec_to_json_attribute
def merge_attribute1(new_attribute_logic,version,original_attribute_file):
    version_str=f"{version}_add_{new_attribute_logic}"
    # new_attribute_logic="ocr"
    
    attribute_file2=f"/home/menglong/workspace/code/multimodal/multimodal_entity_linking/multimodal_entity_linking/output/error/disambiguation/50/bestbuy_50_error_with_attribute_{new_attribute_logic}.json"
    merged_attribute_file=f"/home/menglong/workspace/code/multimodal/multimodal_entity_linking/multimodal_entity_linking/output/error/disambiguation/50/bestbuy_50_error_with_attribute_gpt2_{version_str}.json"
    review_product_json_array_with_score=merge_all_for_review(attribute_file2,original_attribute_file,
                                                              merged_attribute_file,
                                                              fields=[f"predicted_attribute_{new_attribute_logic}",f"predicted_attribute_context_{new_attribute_logic}",f"is_attribute_correct_{new_attribute_logic}",f"confidence_score_{new_attribute_logic}" ],level="review")
    return merged_attribute_file
    
def merge_attribute_main():
    original_attribute_file="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/multimodal_entity_linking/output/error/disambiguation/50/bestbuy_50_error_with_attribute_gpt2.json"
    merged_attribute_file=merge_attribute("exact",1,original_attribute_file)    
    merged_attribute_file=merge_attribute("numeral",2,merged_attribute_file)    
    merged_attribute_file=merge_attribute("ocr",3,merged_attribute_file)    
    
    
# merge_attribute_main()    
def merge_attribute_main2():
    original_attribute_file="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v5/test/retrieval/bestbuy_review_2.3.16.29.6_fused_score_20_54_attribute_gpt2_brand.json"
    # merged_attribute_file=merge_attribute("ocr",1,original_attribute_file)    
    
    version_str=f"1_add_ocr"
    new_attribute_logic="ocr"
    
    attribute_file2=f"/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v5/test/retrieval/bestbuy_review_2.3.16.29.6_fused_score_20_54_ocr_all.json"
    merged_attribute_file=f"/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v5/test/retrieval/bestbuy_review_2.3.16.29.6_fused_score_20_54_gpt2_ocr_all.json"
    review_product_json_array_with_score=merge_all_for_review(attribute_file2,original_attribute_file,
                                                              merged_attribute_file,
                                                              fields=[f"predicted_attribute_{new_attribute_logic}",f"predicted_attribute_context_{new_attribute_logic}",f"is_attribute_correct_{new_attribute_logic}",f"confidence_score_{new_attribute_logic}" ],level="review")


def choose_key_k_attributes(max_attribute_num,gold_product):
    section_list=["Key Specs","General","All"]
    attribute_num=0
    merged_attribute_json=old_spec_to_json_attribute(gold_product["Spec"],False,section_list)
    new_attribute_json={}
    for key in ["Brand","Color"] :
        if   key in merged_attribute_json:
            new_attribute_json[key]=merged_attribute_json[key]
            attribute_num+=1
    for key in merged_attribute_json:
        if key not in ["Product Name","Model Number"] and key not in new_attribute_json and attribute_num<max_attribute_num:
            new_attribute_json[key]=merged_attribute_json[key]
            attribute_num+=1
        
    return new_attribute_json

class GenGoldAttribute:
    def gen_attribute(self,file_with_ocr_attribute,file_with_gpt_attribute, review_file,product_file,output_path):
        
        with open(product_file, 'r', encoding='utf-8') as fp:
            product_json_array = json.load(fp)
            product_json_dict=json_to_dict(product_json_array)
        k=10
        out_list=[]
        with open(review_file, 'r', encoding='utf-8') as fp:
            review_dataset_json_array = json.load(fp)
            for review_dataset_json   in tqdm(review_dataset_json_array): 
                product_id=review_dataset_json["gold_entity_info"]["id"]
                gold_product=product_json_dict[product_id]
                attribute_dict=choose_key_k_attributes(k,gold_product)
                review_dataset_json["attribute"]=attribute_dict
                out_list.append(review_dataset_json)
        with open(output_path , 'w', encoding='utf-8') as fp:
            json.dump(out_list, fp, indent=4)  
            
            
def gen_gpt_attribute(review_dataset_json_with_gpt_attribute,attribute_key_list=["Color","Brand","Color Category"]):
    predicted_attribute_gpt2_json=review_dataset_json_with_gpt_attribute["predicted_attribute_gpt2"]
    gpt_attribute_json={}
    for attribute_key in attribute_key_list :
        if attribute_key in predicted_attribute_gpt2_json:
            gpt_attribute_json[attribute_key]=predicted_attribute_gpt2_json[attribute_key]
    return gpt_attribute_json
def combine_gpt_with_ocr(extracted_gpt_attribute_dict,extracted_ocr_attribute_dict):
    for key,value in extracted_gpt_attribute_dict.items():
        if key not in extracted_ocr_attribute_dict:
            extracted_ocr_attribute_dict[key]    =value
    for key,value_list in extracted_ocr_attribute_dict.items():
        value_str=" ,".join(value_list)
        extracted_ocr_attribute_dict[key]=value_str
    return extracted_ocr_attribute_dict

class GenTestAttribute:
    def gen_attribute(self,file_with_ocr_attribute,file_with_gpt_attribute, review_file,product_file,output_path):
        
        with open(product_file, 'r', encoding='utf-8') as fp:
            product_json_array = json.load(fp)
            product_json_dict=json_to_dict(product_json_array)
        with open(file_with_gpt_attribute, 'r', encoding='utf-8') as fp:
            review_dataset_json_array = json.load(fp)
            review_dataset_dict_with_gpt_attribute=review_json_to_product_dict(review_dataset_json_array)
        
        k=10
        out_list=[]
        with open(file_with_ocr_attribute, 'r', encoding='utf-8') as fp:
            review_dataset_json_array = json.load(fp)
            for review_dataset_json   in tqdm(review_dataset_json_array): 
                product_id=review_dataset_json["gold_entity_info"]["id"]
                gold_product=product_json_dict[product_id]
                extracted_gpt_attribute_dict=gen_gpt_attribute(review_dataset_dict_with_gpt_attribute[review_dataset_json["review_id"]])
                extracted_ocr_attribute_dict=review_dataset_json["predicted_attribute_ocr"] 
                extracted_ocr_attribute_dict=combine_gpt_with_ocr(extracted_gpt_attribute_dict,extracted_ocr_attribute_dict)
                review_dataset_json["attribute"]=extracted_ocr_attribute_dict
                out_list.append(review_dataset_json)
        with open(output_path , 'w', encoding='utf-8') as fp:
            json.dump(out_list, fp, indent=4) 
def create_attribute_main(file_with_ocr_attribute,file_with_gpt_attribute,product_file,review_file,output_path):
    # gen_attribute_object=  GenGoldAttribute() 
    gen_attribute_object=  GenTestAttribute() 
    gen_attribute_object.gen_attribute(file_with_ocr_attribute,file_with_gpt_attribute,review_file,product_file,output_path)


def get_ocr_text(file_with_ocr_attribute,review_file,output_path):
    max_ocr_text_num=10
    with open(file_with_ocr_attribute, 'r', encoding='utf-8') as fp:
        review_dataset_json_array = json.load(fp)
        review_dataset_dict_with_gpt_attribute=review_json_to_product_dict(review_dataset_json_array)
    with open(review_file, 'r', encoding='utf-8') as fp:
        review_dataset_json_array = json.load(fp)
        for review_dataset_json in review_dataset_json_array:
            ocr_text_dict={}
            review_dataset_json_with_ocr=review_dataset_dict_with_gpt_attribute[review_dataset_json["review_id"]]
            confidence_score_ocr=review_dataset_json_with_ocr["confidence_score_ocr"]
            sorted_confidence_score_ocr = sorted(confidence_score_ocr.items(), key=lambda x:x[1],reverse=True)
            for text,score in sorted_confidence_score_ocr:
                if score>0.86 and len(ocr_text_dict)<max_ocr_text_num:
                    ocr_text_dict[text]=score
            review_dataset_json["ocr_text"]=ocr_text_dict
    with open(output_path , 'w', encoding='utf-8') as fp:
        json.dump(review_dataset_json_array, fp, indent=4) 


def merge_attribute(merged_dict,key,review_product_json):
    attribute_dict=review_product_json[key]
    for attribute_key,attribute_value_list in attribute_dict.items():
        attribute_value=attribute_value_list[0]
        if attribute_key not in merged_dict:
            merged_dict[attribute_key]=[]
        merged_dict[attribute_key].append([attribute_value,key])
    return merged_dict
         
     

def merge_attribute_all(merged_dict,key,review_product_json):
    if key in review_product_json:
        attribute_dict=review_product_json[key]
        for attribute_key,attribute_value_list in attribute_dict.items():
            attribute_value_list=set(attribute_value_list)
            for attribute_value in attribute_value_list:
                if attribute_key not in merged_dict:
                    merged_dict[attribute_key]=[]
                merged_dict[attribute_key].append([attribute_value,key])
    return merged_dict
         
    
def fuse_attribute_contain_all(data_path,output_path,mode):
    out_list=[]
    num=0
 
    with open(data_path, 'r', encoding='utf-8') as fp:
        product_dataset_json_array = json.load(fp)
        for review_product_json   in tqdm(product_dataset_json_array ):
            predicted_attribute_dict={}
            merged_dict={}
            merged_dict=merge_attribute_all(merged_dict,"predicted_attribute_exact",review_product_json)
            merged_dict=merge_attribute_all(merged_dict,"predicted_attribute_ocr",review_product_json)
            merged_dict=merge_attribute_all(merged_dict,"predicted_attribute_gpt2",review_product_json)
            merged_dict=merge_attribute_all(merged_dict,"predicted_attribute_model_version_to_product_title",review_product_json)
            merged_dict=merge_attribute_all(merged_dict,"predicted_attribute_model_version",review_product_json)
            if mode=="test":
                merged_dict=merge_attribute_all(merged_dict,"predicted_attribute_chatgpt",review_product_json)
            for attribute_key,merged_attribute_value_and_field_name_list in merged_dict.items():
                for attribute_value,field_name in merged_attribute_value_and_field_name_list:
                    if attribute_key not in predicted_attribute_dict:
                        predicted_attribute_dict[attribute_key]=[]
                    predicted_attribute_dict[attribute_key].append(attribute_value)
                    num+=1
                    
                predicted_attribute_dict[attribute_key]=list(set(predicted_attribute_dict[attribute_key]))
                
            review_product_json["predicted_attribute_contain_all"]=predicted_attribute_dict
            out_list.append(review_product_json)
    with open(output_path , 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4) 
    print(num)
    
def gen_attribute_token_dictionary(  ):
    with open(products_path_str, 'r', encoding='utf-8') as fp:
        product_json_array = json.load(fp)
        product_json_dict=json_to_dict(product_json_array)
    output_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/disambiguation/attribute_token.json"
    train_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/train/disambiguation/bestbuy_review_2.3.17.11.19.11_all_attribute.json"
    test_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/disambiguation/bestbuy_review_2.3.17.11.19.1_fuse_attribute_0.8_threshold.json"
    val_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/val/disambiguation/bestbuy_review_2.3.17.11.19.11_all_attribute.json"
    token_mapping_list={}
    token_id=0
    for data_path,mode in [[train_path,"train"],[val_path,"val"],[test_path,"test"]]:
        with open(data_path, 'r', encoding='utf-8') as fp:
            review_dataset_json_array = json.load(fp)
            
            
            for review_product_json   in tqdm(review_dataset_json_array ):
                predicted_attribute_contain_all=review_product_json["predicted_attribute_contain_all"]
                candidate_id_list=review_product_json["fused_candidate_list"][:100]
                gold_entity_id=review_product_json["gold_entity_info"]["id"]
                candidate_with_gold_id_list=[]
                candidate_with_gold_id_list.extend(candidate_id_list)
                if gold_entity_id not in candidate_id_list:
                    candidate_with_gold_id_list.append(gold_entity_id)
                total_attribute_json_in_review={"Product Title":[]}
                
                for candidate_id in candidate_with_gold_id_list:
                    product_json=product_json_dict[candidate_id]
                    total_attribute_json_in_review=gen_gold_attribute_without_key_section( product_json["Spec"],product_json["product_name"],total_attribute_json_in_review)
                for attribute_key,_ in predicted_attribute_contain_all.items():
                    if attribute_key in ["Product Title","Product Name"]:
                        continue
                    if attribute_key not in token_mapping_list:
                        token_mapping_list[attribute_key]={}
                    candidate_value_list=total_attribute_json_in_review[attribute_key]
                    for candidate_attribute_value in candidate_value_list:
                        if candidate_attribute_value not in token_mapping_list[attribute_key]:
                            token_id+=1
                            token_mapping_list[attribute_key][candidate_attribute_value]=token_id
    print(token_id)
    with open(output_path , 'w', encoding='utf-8') as fp:
        json.dump(token_mapping_list, fp, indent=4) 
    
def fuse1(merged_attribute_value_and_field_name_list,predicted_attribute_dict,attribute_key,metric_dict,num,used_rule_list,score_threshold,test_metric_dict_for_one_extract_num,test_extract_num_threshold) :   
    if test_metric_dict_for_one_extract_num is None:
        test_metric_dict_for_one_extract_num=metric_dict 
    if len(merged_attribute_value_and_field_name_list)>1:
        attribute_value_list=[inner_list[0] for inner_list in merged_attribute_value_and_field_name_list]
        test_list = Counter(attribute_value_list)
        most_common_element = test_list.most_common(1)[0][0]
        predicted_attribute_dict[attribute_key]=most_common_element
        num+=1
    else:
        attribute_value,extractor_field_name=merged_attribute_value_and_field_name_list[0]
        if attribute_key in metric_dict[extractor_field_name]:
            score=metric_dict[extractor_field_name][attribute_key]["precision"]
            if score>score_threshold and (test_extract_num_threshold==1 or attribute_key not in test_metric_dict_for_one_extract_num[extractor_field_name] or test_metric_dict_for_one_extract_num[extractor_field_name][attribute_key]["extract_num"] >test_extract_num_threshold):
                predicted_attribute_dict[attribute_key]=attribute_value
                used_rule_list.append([attribute_key,extractor_field_name,metric_dict[extractor_field_name][attribute_key]["precision"],metric_dict[extractor_field_name][attribute_key]["extract_num"]])
                num+=1
    return predicted_attribute_dict,  num,used_rule_list
def fuse2(merged_attribute_value_and_field_name_list,predicted_attribute_dict,attribute_key,metric_dict,num,used_rule_list,score_threshold,test_metric_dict_for_one_extract_num,test_extract_num_threshold) :   
    if test_metric_dict_for_one_extract_num is None:
        test_metric_dict_for_one_extract_num=metric_dict 
    for attribute_value,extractor_field_name in merged_attribute_value_and_field_name_list :
        if attribute_key in metric_dict[extractor_field_name]:
            score=metric_dict[extractor_field_name][attribute_key]["precision"]
            if score>score_threshold and (test_extract_num_threshold==1 or attribute_key not in test_metric_dict_for_one_extract_num[extractor_field_name] or test_metric_dict_for_one_extract_num[extractor_field_name][attribute_key]["extract_num"] >test_extract_num_threshold):
                predicted_attribute_dict[attribute_key]=attribute_value
                used_rule_list.append([attribute_key,extractor_field_name,metric_dict[extractor_field_name][attribute_key]["precision"],metric_dict[extractor_field_name][attribute_key]["extract_num"]])
                num+=1
    return predicted_attribute_dict,  num,used_rule_list
def fuse3(merged_attribute_value_and_field_name_list,predicted_attribute_dict,attribute_key,metric_dict,num,used_rule_list,score_threshold,test_metric_dict_for_one_extract_num,test_extract_num_threshold) :   
    if test_metric_dict_for_one_extract_num is None:
        test_metric_dict_for_one_extract_num=metric_dict 
    if len(merged_attribute_value_and_field_name_list)>1:
        attribute_value_list=[inner_list[0] for inner_list in merged_attribute_value_and_field_name_list]
        test_list = Counter(attribute_value_list)
        most_common_element = test_list.most_common(1)[0][0]
        predicted_attribute_dict[attribute_key]=most_common_element
        num+=1
        
    return predicted_attribute_dict,  num,used_rule_list
def fuse4(merged_attribute_value_and_field_name_list,predicted_attribute_dict,attribute_key,metric_dict,num,used_rule_list,score_threshold,test_metric_dict_for_one_extract_num,test_extract_num_threshold) :   
    if test_metric_dict_for_one_extract_num is None:
        test_metric_dict_for_one_extract_num=metric_dict 
    score_dict={}
    if len(merged_attribute_value_and_field_name_list)>1:
        highest_score=0
        best_attribute_value=""
        for attribute_value,extractor_field_name in merged_attribute_value_and_field_name_list:
            if attribute_key in metric_dict[extractor_field_name]:
                score=metric_dict[extractor_field_name][attribute_key]["precision"]
                if score>highest_score:
                    best_attribute_value=attribute_value
        if best_attribute_value!="":
            predicted_attribute_dict[attribute_key]=best_attribute_value
            num+=1
        
    return predicted_attribute_dict,  num,used_rule_list        
def fuse5(merged_attribute_value_and_field_name_list,predicted_attribute_dict,attribute_key,metric_dict,num,used_rule_list,score_threshold,test_metric_dict_for_one_extract_num,test_extract_num_threshold) :   
    if test_metric_dict_for_one_extract_num is None:
        test_metric_dict_for_one_extract_num=metric_dict 
    if len(merged_attribute_value_and_field_name_list)>1:
        highest_score=0
        best_attribute_value=""
        for attribute_value,extractor_field_name in merged_attribute_value_and_field_name_list:
            if attribute_key in metric_dict[extractor_field_name]:
                score=metric_dict[extractor_field_name][attribute_key]["precision"]
                if score>highest_score:
                    best_attribute_value=attribute_value
        if best_attribute_value!="":
            predicted_attribute_dict[attribute_key]=best_attribute_value
            num+=1
    else:
        attribute_value,extractor_field_name=merged_attribute_value_and_field_name_list[0]
        if attribute_key in metric_dict[extractor_field_name]:
            score=metric_dict[extractor_field_name][attribute_key]["precision"]
            if score>score_threshold and (test_extract_num_threshold==1 or attribute_key not in test_metric_dict_for_one_extract_num[extractor_field_name] or test_metric_dict_for_one_extract_num[extractor_field_name][attribute_key]["extract_num"] >test_extract_num_threshold):
                predicted_attribute_dict[attribute_key]=attribute_value
                used_rule_list.append([attribute_key,extractor_field_name,metric_dict[extractor_field_name][attribute_key]["precision"],metric_dict[extractor_field_name][attribute_key]["extract_num"]])
                num+=1
    return predicted_attribute_dict,  num,used_rule_list        
def fuse6(merged_attribute_value_and_field_name_list,predicted_attribute_dict,attribute_key,metric_dict,num,used_rule_list,score_threshold,test_metric_dict_for_one_extract_num,test_extract_num_threshold) :   
    if test_metric_dict_for_one_extract_num is None:
        test_metric_dict_for_one_extract_num=metric_dict 
    attribute_value,extractor_field_name=merged_attribute_value_and_field_name_list[0]
    if attribute_key in metric_dict[extractor_field_name]:
        score=metric_dict[extractor_field_name][attribute_key]["precision"]
        if score>score_threshold and (test_extract_num_threshold==1 or attribute_key not in test_metric_dict_for_one_extract_num[extractor_field_name] or test_metric_dict_for_one_extract_num[extractor_field_name][attribute_key]["extract_num"] >test_extract_num_threshold):
            predicted_attribute_dict[attribute_key]=attribute_value
            used_rule_list.append([attribute_key,extractor_field_name,metric_dict[extractor_field_name][attribute_key]["precision"],metric_dict[extractor_field_name][attribute_key]["extract_num"]])
            num+=1
    return predicted_attribute_dict,  num,used_rule_list        
def fuse_attribute(data_path,output_path,metric_data_path,mode,method="fuse1",is_use_chatgpt=True,score_threshold=0.7,extract_num_threshold=50,
                   is_print=False,cached_metric_dict=None,test_metric_dict_for_one_extract_num=None,test_extract_num_threshold=50,
                   attribute_logic_list="all"):
    print(attribute_logic_list)
    out_list=[]
    used_rule_list=[]
    num=0
    if cached_metric_dict is None:
        metric_dict=overall_metric(metric_data_path,"val",is_print,extract_num_threshold=extract_num_threshold)
        if mode=="test":
            metric_dict["predicted_attribute_chatgpt"]=metric_dict["predicted_attribute_gpt2"]
            metric_dict["predicted_attribute_vicuna"]=metric_dict["predicted_attribute_gpt2"]
            metric_dict["predicted_attribute_gpt2_few"]=metric_dict["predicted_attribute_gpt2"]
    else:
        metric_dict=cached_metric_dict
    with open(data_path, 'r', encoding='utf-8') as fp:
        product_dataset_json_array = json.load(fp)
        for review_product_json   in tqdm(product_dataset_json_array) :
            predicted_attribute_dict={}
            merged_dict={}
            for attribute_logic in ["exact","ocr","gpt2","model_version_to_product_title","model_version","vicuna","gpt2_few"]:
                if attribute_logic_list=="all" or attribute_logic in attribute_logic_list:
                    merged_dict=merge_attribute_all(merged_dict,f"predicted_attribute_{attribute_logic}",review_product_json)
                     
            if mode=="test" and "chatgpt" in attribute_logic_list:
                merged_dict=merge_attribute_all(merged_dict,"predicted_attribute_chatgpt",review_product_json)
            for attribute_key,merged_attribute_value_and_field_name_list in merged_dict.items():
                if method=="fuse1":
                    predicted_attribute_dict,  num,used_rule_list=fuse1(merged_attribute_value_and_field_name_list,predicted_attribute_dict,attribute_key,metric_dict,num,used_rule_list,score_threshold,test_metric_dict_for_one_extract_num,test_extract_num_threshold ) 
                elif method=="fuse2":
                    predicted_attribute_dict,  num,used_rule_list=fuse2(merged_attribute_value_and_field_name_list,predicted_attribute_dict,attribute_key,metric_dict,num,used_rule_list,score_threshold,test_metric_dict_for_one_extract_num,test_extract_num_threshold ) 
                elif method=="fuse3":
                    predicted_attribute_dict,  num,used_rule_list=fuse3(merged_attribute_value_and_field_name_list,predicted_attribute_dict,attribute_key,metric_dict,num,used_rule_list,score_threshold,test_metric_dict_for_one_extract_num,test_extract_num_threshold ) 
                elif method=="fuse4":
                    predicted_attribute_dict,  num,used_rule_list=fuse4(merged_attribute_value_and_field_name_list,predicted_attribute_dict,attribute_key,metric_dict,num,used_rule_list,score_threshold,test_metric_dict_for_one_extract_num,test_extract_num_threshold ) 
                elif method=="fuse5":
                    predicted_attribute_dict,  num,used_rule_list=fuse5(merged_attribute_value_and_field_name_list,predicted_attribute_dict,attribute_key,metric_dict,num,used_rule_list,score_threshold,test_metric_dict_for_one_extract_num,test_extract_num_threshold ) 
                elif method=="fuse6":
                    predicted_attribute_dict,  num,used_rule_list=fuse6(merged_attribute_value_and_field_name_list,predicted_attribute_dict,attribute_key,metric_dict,num,used_rule_list,score_threshold,test_metric_dict_for_one_extract_num,test_extract_num_threshold ) 
            review_product_json["predicted_attribute"]=predicted_attribute_dict
            out_list.append(review_product_json)
    
    with open(output_path , 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4) 
    print(num)
    print()
    print()
    print("##########################for test#")
    print()
    print()
    overall_metric(output_path,mode,is_print,extract_num_threshold,is_gold_attribute="y")
    used_rule_list=clean(used_rule_list)
    

def clean(used_rule_list):
    special_rule_dict={}
    for   attribute_key,extractor_field_name, precision,extract_num in used_rule_list:
        if attribute_key+"_"+extractor_field_name not in special_rule_dict:
            special_rule_dict[attribute_key+"_"+extractor_field_name]=[precision,extract_num]
    # print(special_rule_dict)
 
def gen_gold_attribute_for_predicted_category(data_path,output_path):
    out_list=[]
    num=0
    with open(products_path_str, 'r', encoding='utf-8') as fp:
        product_json_array = json.load(fp)
        product_json_dict=json_to_dict(product_json_array)
    with open(data_path, 'r', encoding='utf-8') as fp:
        product_dataset_json_array = json.load(fp)
        for review_product_json   in tqdm(product_dataset_json_array ):
            predicted_attribute_dict={}
            gold_product_id=review_product_json["gold_entity_info"]["id"]
            gold_product_json=product_json_dict[gold_product_id]
            gold_attribute_json=gen_gold_attribute_without_key_section(gold_product_json["Spec"],gold_product_json["product_name"] ,{})
            predicted_attribute_dict_for_all=review_product_json["predicted_attribute_contain_all"]
            predicted_attribute=review_product_json["predicted_attribute"]
            predicted_attribute_model_version_to_product_title=review_product_json["predicted_attribute_model_version_to_product_title"]
            predicted_attribute_exact=review_product_json["predicted_attribute_exact"]
             
            predicted_attribute_gpt2=review_product_json["predicted_attribute_gpt2"]
 
            for attribute_key,predicted_attribute_value_list in predicted_attribute_dict_for_all.items():
                if len(predicted_attribute_value_list)==1:
                    predicted_attribute_value=predicted_attribute_value_list[0]
                elif attribute_key in predicted_attribute:
                    predicted_attribute_value=predicted_attribute[attribute_key]
                else:
                    continue
                if attribute_key in ["Product Name", "Product Title"]:
                    if not ((attribute_key in predicted_attribute_gpt2 and predicted_attribute_value in predicted_attribute_gpt2[attribute_key]) or (attribute_key in predicted_attribute_exact and predicted_attribute_value in predicted_attribute_exact[attribute_key])):
                        continue
                
                if attribute_key in gold_attribute_json and len(gold_attribute_json[attribute_key])>0:
                    if predicted_attribute_value in gold_attribute_json[attribute_key]:
                        predicted_attribute_dict[attribute_key]=predicted_attribute_value
                        num+=1
            review_product_json["gold_attribute_for_predicted_category"]=predicted_attribute_dict
            out_list.append(review_product_json)
    with open(output_path , 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4) 
    print(num)
    
    
def fix_metric():
    output_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/disambiguation/attribute_accuracy.json"
    train_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/train/disambiguation/bestbuy_review_2.3.17.11.19.10_merge_model_version_to_product_title.json"
    test_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/disambiguation/bestbuy_review_2.3.17.11.19.1_fuse_attribute_0.8_threshold.json"
    val_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/val/disambiguation/bestbuy_review_2.3.17.11.19.5_similar_model_version_to_product_from_0_to_100000.json"
 
    with open(output_path, 'r', encoding='utf-8') as fp:
        metric_path_json = json.load(fp)
    for data_path,mode in [[train_path,"train"],[val_path,"val"],[test_path,"test"]]:
         
        for extract_num_threshold in [5,10,20,30,40,50,60,70,80,90,100]:
            one_metric=metric_path_json[mode][str(extract_num_threshold)]
            one_metric["predicted_attribute_chatgpt"]=one_metric["predicted_attribute_gpt2"]    
            metric_path_json[mode][str(extract_num_threshold)]=one_metric
            print(mode,extract_num_threshold)
   
            
    with open(output_path , 'w', encoding='utf-8') as fp:
        json.dump(metric_path_json, fp, indent=4) 
    
            
def compute_score():
    output_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/disambiguation/attribute_accuracy.json"
    train_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/train/disambiguation/bestbuy_review_2.3.17.11.19.10_merge_model_version_to_product_title.json"
    test_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/disambiguation/bestbuy_review_2.3.17.11.19.1_fuse_attribute_0.8_threshold.json"
    val_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/val/disambiguation/bestbuy_review_2.3.17.11.19.5_similar_model_version_to_product_from_0_to_100000.json"
    metric_dict={}
    
    for data_path,mode in [[train_path,"train"],[val_path,"val"],[test_path,"test"]]:
        metric_dict[mode]={}
        for extract_num_threshold in [5,10,20,30,40,50,60,70,80,90,100]:
            print(mode,extract_num_threshold)
            one_metric=overall_metric(data_path,mode,False,extract_num_threshold=extract_num_threshold)
            metric_dict[mode][extract_num_threshold]=one_metric
    with open(output_path , 'w', encoding='utf-8') as fp:
        json.dump(metric_dict, fp, indent=4) 
        
def search(review_file, output_path):
    metric_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/disambiguation/attribute_accuracy.json"
    with open(metric_path, 'r', encoding='utf-8') as fp:
        metric_path_json = json.load(fp)
    best_f1=0
    best_setting=[]
    test_metric_dict_for_one_extract_num=metric_path_json["test"]["5"]
    for reference_file_type in [ "val","train"]:
        for extract_num_threshold in [5,10,20,30,40,50,70]:#,80,90,100 60,
            metric_dict=metric_path_json[reference_file_type][str(extract_num_threshold)]
            for score_threshold in [0.5,0.6,0.7,0.8,0.9]:#,0.65,0.75,0.85,0.95
                for method in ["fuse4","fuse1","fuse2","fuse3","fuse5","fuse6"]:
                    for is_use_chatgpt in [True]:#,False
                        for test_extract_num_threshold in [10]:#,1,20,50,60,70,80,90,100
                            f1=test_one_setting(review_file, output_path, None, "test",method,metric_dict,score_threshold,is_use_chatgpt,test_metric_dict_for_one_extract_num,test_extract_num_threshold)
                            print(f1,test_extract_num_threshold,is_use_chatgpt,method,score_threshold,extract_num_threshold,reference_file_type)
                            if f1>best_f1:
                                best_f1=f1
                                best_setting=[f1,test_extract_num_threshold,is_use_chatgpt,method,score_threshold,extract_num_threshold,reference_file_type]
    print(best_setting)
                        
            
            
def test_one_setting(review_file, output_path, metric_data_path, mode,method,cached_metric_dict,score_threshold,is_use_chatgpt,test_metric_dict_for_one_extract_num,test_extract_num_threshold):
    fuse_attribute(review_file,output_path, metric_data_path, mode,method=method,is_use_chatgpt=is_use_chatgpt,score_threshold=score_threshold,extract_num_threshold=50,is_print=False,cached_metric_dict=cached_metric_dict,test_metric_dict_for_one_extract_num=test_metric_dict_for_one_extract_num,test_extract_num_threshold=test_extract_num_threshold)
    temp_output_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/temp/test.json"
    predicted_attribute_field="predicted_attribute"#gold_attribute_for_predicted_category
    f1=post_process_for_json( output_path,10, temp_output_path,score_key="fused_score",is_retrieval=True,predicted_attribute_field=predicted_attribute_field,is_save=False)       
    return f1   
def product_attribute_dict_to_token_id(attribute_dict,product_attribute_dict_list,token_mapping_list):
    token_id_list=[]
    for attribute_key,_ in attribute_dict.items():
        if attribute_key in product_attribute_dict_list:
            if attribute_key not in ["Product Title","Product Name"]:
                attribute_value_list=product_attribute_dict_list[attribute_key]
                if len(attribute_value_list)>0:
                    attribute_value=attribute_value_list[0]
                    token_id=token_mapping_list[attribute_key][attribute_value]
                    token_id_list.append(token_id)
    max_num=10
    pad_id=0
    for i in range(len(token_id_list),max_num):
        token_id_list.append(pad_id)
    return token_id_list

def attribute_dict_to_token_id(attribute_dict,token_mapping_list):
    token_id_list=[]
    for attribute_key,attribute_value in attribute_dict.items():
        if attribute_key not in ["Product Title","Product Name"]:
            token_id=token_mapping_list[attribute_key][attribute_value]
            token_id_list.append(token_id)
    max_num=10
    pad_id=0
    for i in range(len(token_id_list),max_num):
        token_id_list.append(pad_id)
    return token_id_list

def add_attribute_token_id_list(data_path,output_path,token_id_mapping_path):
    with open(products_path_str, 'r', encoding='utf-8') as fp:
        product_json_array = json.load(fp)
        product_json_dict=json_to_dict(product_json_array)
    out_list=[]
    with open(token_id_mapping_path, 'r', encoding='utf-8') as fp:
        token_id_mapping_dict = json.load(fp)
    with open(data_path, 'r', encoding='utf-8') as fp:
        product_dataset_json_array = json.load(fp)
        for review_product_json   in tqdm(product_dataset_json_array ):
            predicted_attribute=review_product_json["predicted_attribute"]
            token_id_list=attribute_dict_to_token_id(predicted_attribute,token_id_mapping_dict)    
            review_product_json["token_id_list"]=token_id_list
            fused_candidate_list=review_product_json["fused_candidate_list"][:100]
            review_special_product_token_id_dict={}
            for candidate_id in fused_candidate_list:
                product_json=product_json_dict[candidate_id]
                product_attribute_dict_list=gen_gold_attribute_without_key_section(product_json["Spec"],product_json["product_name"] ,{})
                product_token_id_list=product_attribute_dict_to_token_id(predicted_attribute,product_attribute_dict_list,token_id_mapping_dict)
                review_special_product_token_id_dict[candidate_id]=product_token_id_list
            review_product_json["review_special_product_token_id_dict"]=review_special_product_token_id_dict
            out_list.append(review_product_json)
    with open(output_path , 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4)
            
import argparse
def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--review_file',type=str,help=" ",default="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/disambiguation/bestbuy_review_2.3.17.11.21_add_gpt2_vicuna.json") 
    parser.add_argument('--output_path',type=str,help=" ",default="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/attribute/bestbuy_review_2.3.17.11.21_fuse1.json")
    parser.add_argument('--metric_data_path',type=str,help=" ",default="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/train/disambiguation/bestbuy_review_2.3.17.11.19.16_update_gold_attribute.json") #/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/disambiguation/attribute_token.json
    parser.add_argument('--file_with_ocr_attribute',type=str,help=" ",default="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v5/test/attribute/ocr/bestbuy_review_2.3.16.29.6_fused_score_20_54_ocr_all.json")
    parser.add_argument('--file_with_gpt_attribute',type=str,help=" ",default="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v5/test/attribute/gpt/bestbuy_review_2.3.16.29.6_fused_score_20_54_attribute_gpt2_all.json")
    parser.add_argument('--product_file',type=str,help=" ",default=products_path_str)
    parser.add_argument('--mode',type=str,help=" ",default="test")#exact numeral
    parser.add_argument('--attribute_logic',type=str,help=" ",default="gpt2")#exact numeral
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=50000)
    parser.add_argument("--is_mp", type=str, default="y", help="Text added prior to input.")
    parser.add_argument("--fuse_method", type=str, default="fuse1", help="Text added prior to input.")
    args = parser.parse_args()
    return args




   

if __name__ == '__main__':
    args = parser_args()
    # create_attribute_main(args.file_with_ocr_attribute,args.file_with_gpt_attribute,args.product_file,args.review_file,args.output_path)    
    # get_ocr_text(args.file_with_ocr_attribute,args.review_file,args.output_path)
    
    # fuse_attribute_contain_all(args.review_file,args.output_path,args.mode)
    fuse_attribute(args.review_file,args.output_path,args.metric_data_path,args.mode,method=args.fuse_method,
                   is_use_chatgpt=True,score_threshold=0.5,extract_num_threshold=5,is_print=True,cached_metric_dict=None,
                   test_metric_dict_for_one_extract_num=None,test_extract_num_threshold=10,
                   attribute_logic_list=["ocr","exact","model_version_to_product_title","model_version","gpt2","chatgpt"]) #,","gpt2_few","vicuna" ,"gpt2_few","vicuna"
    #["ocr","exact","model_version_to_product_title","model_version","gpt2","chatgpt"]  ,"vicuna"
    # gen_attribute_token_dictionary()
    # test_one_setting(args.review_file,args.output_path,args.metric_data_path,args.mode)
    # add_attribute_token_id_list(args.review_file,args.output_path,args.metric_data_path)
    # fix_metric()
    # search(args.review_file,args.output_path)
    # compute_score()
    
    # overall_metric(args.output_path,"val")
    # gen_gold_attribute_for_predicted_category(args.review_file,args.output_path )