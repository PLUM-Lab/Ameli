import pandas as pd
import gzip
import json 
from tqdm import tqdm 


def load_attribute_key_value(review_path):
    output_list={}
 
    with open(review_path, 'r', encoding='utf-8') as fp:
        review_dataset_json_array = json.load(fp) 
        for entity_id,attributes_str in  review_dataset_json_array.items() :
            output_list[entity_id]={}
            attribute_key_value_list=attributes_str.split(".")
            for attribute_key_value in attribute_key_value_list:
                attribute_key_value=attribute_key_value.strip()
                if attribute_key_value!="" and ":" in attribute_key_value:
                    attribute_key_attribute_multi_value=attribute_key_value.split(":")
                    attribute_key =attribute_key_attribute_multi_value[0]
                    attribute_multi_value=":".join(attribute_key_attribute_multi_value[1:])
                    
                    
                    if attribute_key not in [ "Birth","Death"]:
                        output_list[entity_id][attribute_key]=[]
                        attribute_value_list=attribute_multi_value.split(",")
                        for i,attribute_value in enumerate(attribute_value_list):
                            attribute_value=attribute_value.strip()
                            output_list[entity_id][attribute_key ].append(attribute_value)
                    else:
                        the_date_the_place_list=attribute_multi_value.split(",")
                        the_date=the_date_the_place_list[0]
                        the_place=",".join(the_date_the_place_list[1:])
                        # if len(the_date_the_place_list)>2:
                            
                        # else:
                        #     the_place=the_date_the_place_list[0]
                            
                        output_list[entity_id][attribute_key+" date"]=[the_date.strip()]
                        output_list[entity_id][attribute_key+" place"]=[the_place.strip()]
    return output_list
                    



 
# infile_path="data/amazon/All_Amazon_Meta.json.gz"
# product_w_attribute_file_path="data/amazon/All_Amazon_Product_w_Attribute.json.gz"
# # get_product_w_attribute(infile_path,product_w_attribute_file_path)
# # df = getProductDF(file_path)
review_path="data/melbench/Richpedia-MEL.json"
# sampled_review_path="data/amazon/sampled_100000_Amazon_Review.json.gz"
product_path="data/melbench/qid2abs_long.json"
# # sample_review_subset(review_path,sampled_review_path,product_path)
# sampled_100000_review_path="data/amazon/sampled_100000_Amazon_Review_w_attribute_from_0_to_10000000.json"
# sampled_10000_review_path="data/amazon/sampled_10000_Amazon_Review.json"
# sample_10000_review(sampled_100000_review_path,sampled_10000_review_path)


def merge_attribute(review_path,output_products_path):
    output_list=[]
    no_attribute_review_list=[]
    
    with open(review_path, 'r', encoding='utf-8') as fp:
        review_dataset_json_array = json.load(fp) 
        for idx,review_dataset_json in  enumerate(review_dataset_json_array) :   
            predicted_attribute_vicuna_complete=review_dataset_json["predicted_attribute_melbench_vicuna_complete"]      
            predicted_attribute_amazon_exact=review_dataset_json["predicted_attribute_amazon_exact"]
            predicted_attribute={}
            predicted_attribute.update(predicted_attribute_vicuna_complete)
            predicted_attribute.update(predicted_attribute_amazon_exact)
            review_dataset_json["predicted_attribute"]=predicted_attribute
            output_list.append(review_dataset_json)
            # if idx>=9999:
            #     break
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(output_list, fp, indent=4)
 
def count_number(review_path,key="predicted_attribute"):
    total_number=0
    with open(review_path, 'r', encoding='utf-8') as fp:
        review_dataset_json_array = json.load(fp) 
        for review_dataset_json in  review_dataset_json_array :   
            predicted_attribute=review_dataset_json[key]     
            cur_number=len(predicted_attribute)
            total_number+=cur_number
            
        print(f"{total_number},{len(review_dataset_json_array)}")            
            
infile_path="data/amazon/All_Amazon_Meta.json.gz"
product_w_attribute_file_path="data/amazon/All_Amazon_Product_w_Attribute.json.gz"
# get_product_w_attribute(infile_path,product_w_attribute_file_path)
# df = getProductDF(file_path)
review_path="data/melbench/Richpedia-MEL_w_attribute_vicuna_from_0_to_10000000.json"
merged_review_path="data/melbench/merged_Richpedia-MEL_w_attribute_vicuna.json"
# review_path="data/amazon/merged_10000_Amazon_Review_w_attribute_all.json"
sampled_review_path="data/amazon/sampled_100000_Amazon_Review.json.gz"
product_path="data/amazon/All_Amazon_Product_w_Attribute.json.gz"
# sample_review_subset(review_path,sampled_review_path,product_path)
sampled_100000_review_path="data/amazon/sampled_100000_Amazon_Review_w_attribute_from_0_to_10000000.json"
sampled_10000_review_path="data/amazon/sampled_10000_Amazon_Review.json"
# sample_10000_review(sampled_100000_review_path,sampled_10000_review_path)
merge_attribute(review_path,merged_review_path)

# count_number(review_path,key="predicted_attribute_amazon_exact")
count_number(merged_review_path )
# with gzip.open(file_path, "rt") as f:
#     expected_dict = json.load(f)
#     print("e")
#     print(expected_dict[0])