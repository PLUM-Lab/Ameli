
import json
from attribute.util.util import compare_with_gold
from util.env_config import * 
from tqdm import tqdm 
from attribute.attribution_extraction import gen_candidate_attribute, init 
def clean_predicted_attribute():
    fused_score_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v5/test/retrieval/bestbuy_review_2.3.16.29.5_fused_score_20_54.json" 
    
    review_product_json_array_with_fused_score_dict,product_dict=init(fused_score_path,products_path_str,{})
    # output_products_path=""
    output_list=[]
    data_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/multimodal_entity_linking/output/error/disambiguation/50/bestbuy_50_error_with_attribute_gpt2.json"
    with open(data_path, 'r', encoding='utf-8') as fp:
        review_dataset_json_array = json.load(fp)
        for review_dataset_json in review_dataset_json_array:
            predicted_attribute_gpt2=review_dataset_json["predicted_attribute_gpt2"]
            review_id=review_dataset_json["review_id"]
            candidate_attribute_json=gen_candidate_attribute(review_id,review_product_json_array_with_fused_score_dict,product_dict)
            new_predicted_attribute_gpt2={}
            for attribute_key, predicted_attribute_value_list in predicted_attribute_gpt2.items():
                if attribute_key in candidate_attribute_json:
                    candidate_attribute_value_list=candidate_attribute_json[attribute_key]
                    extracted_attribute_value_list=[]
                    for predicted_attribute_value in predicted_attribute_value_list:
                        extracted_attribute_value_list.extend(compare_with_gold(predicted_attribute_value,candidate_attribute_value_list))
                    if len(extracted_attribute_value_list)>0:
                        new_predicted_attribute_gpt2[attribute_key]=extracted_attribute_value_list
            
            review_dataset_json["predicted_attribute_gpt2"]=new_predicted_attribute_gpt2
            output_list.append(review_dataset_json)
    with open(data_path, 'w', encoding='utf-8') as fp:
        json.dump(output_list, fp, indent=4)  
import copy         
        
def clean_each_predicted_attribute_approach():
    # fused_score_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v5/test/retrieval/bestbuy_review_2.3.16.29.5_fused_score_20_54.json" 
    data_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/disambiguation/bestbuy_review_2.3.17.11.21_add_gpt2_vicuna.json"
    out_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/temp/bestbuy_review_2.3.17.11.21_remove_diff.json"
    review_product_json_array_with_fused_score_dict,product_dict=init(data_path,products_path_str,{})
    # output_products_path=""
    output_list=[]
    diff_num=0
    # data_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/disambiguation/bestbuy_review_2.3.17.11.21_add_gpt2_vicuna.json"
    with open(data_path, 'r', encoding='utf-8') as fp:
        review_dataset_json_array = json.load(fp)
        for review_dataset_json in tqdm(review_dataset_json_array):
            review_id=review_dataset_json["review_id"]
            candidate_attribute_json=gen_candidate_attribute(review_id,product_dict,review_dataset_json,candidate_num=10)
            for attribute_logic in ["gpt2","gpt2_few","chatgpt","ocr","exact","vicuna"]:
                attribute_logic_key=f"predicted_attribute_{attribute_logic}"
            
                if attribute_logic_key in review_dataset_json:
                    predicted_attributes=review_dataset_json[attribute_logic_key]
                    
                    new_predicted_attributes={}
                    for attribute_key, predicted_attribute_value_list in predicted_attributes.items():
                        if attribute_key in candidate_attribute_json:
                            candidate_attribute_value_list=candidate_attribute_json[attribute_key]
                            extracted_attribute_value_list=[]
                            if isinstance(predicted_attribute_value_list,list):
                                predicted_attribute_value_set=set(predicted_attribute_value_list)
                                if len(predicted_attribute_value_set)==1:
                                    new_predicted_attributes[attribute_key]=list(predicted_attribute_value_set)
                                else:
                                    # new_predicted_attributes[attribute_key]=predicted_attribute_value_list[:1]
                                    diff_num+=1
                    if attribute_logic in [ "gpt2_few",  "vicuna"]:
                        for  attribute_key, candidate_attribute_value_list in candidate_attribute_json.items():       
                            if len(candidate_attribute_value_list)==1:
                                new_predicted_attributes[attribute_key]=candidate_attribute_value_list
                    review_dataset_json[attribute_logic_key]=copy.deepcopy(new_predicted_attributes)
                else:
                    review_dataset_json[attribute_logic_key]={}
            output_list.append(review_dataset_json)
    with open(out_path, 'w', encoding='utf-8') as fp:
        json.dump(output_list, fp, indent=4)        
    print(diff_num)
        
# clean_predicted_attribute()        


clean_each_predicted_attribute_approach()