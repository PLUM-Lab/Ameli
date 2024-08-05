import pandas as pd
import gzip
import json 
from tqdm import tqdm 

def parse(path):
    g = gzip.open(path, 'rb')
 
    for l in g:
        yield json.loads(l)

def getProductDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        if d["tech1"]!="" or d["tech2"] !="":
            print(d["tech1"],d["tech2"])
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

# Function to filter items based on a condition
def filter_items(item):
    if "tech1" not in item or "tech2" not in item:
        return True
    elif item["tech1"]== "" and item["tech2"] ==""  :
        return True 
    elif item["tech1"] is None  and item["tech2"] is None  :
        return True 
    else:
        return False
        

def get_gz_file_length(file_path):
    with gzip.open(file_path, 'rt') as gz_file:
        line_count = gz_file.readlines().count('\n')
    return line_count

def get_product_w_attribute(input_file_path,output_file_path):
    # total_num=get_gz_file_length(input_file_path)
    # Load the gzipped file and filter items
    with gzip.open(input_file_path, "rt") as input_file, gzip.open(output_file_path, "wt") as output_file:
       
        valid_num=0
        for i,line in tqdm(enumerate(input_file )):
            try:
                # Assuming each line is a JSON object
                item = json.loads(line.strip())
                 
                # Check the condition and write to the output file if it's met
                if not filter_items(item):
                    valid_num+=1
                    output_file.write(json.dumps(item) + "\n")
                    
                    # if valid_num>100000:#100000
                    #     break 
                if i%10000==0:
                    print(f"valid_num:{valid_num},total_num:{i}")            
            except json.JSONDecodeError:
                print("Error decoding JSON from line:", line)

    print("Filtering and saving completed.")





def gen_valid_product_id_list(product_path):
    valid_product_id_list=[]
    with gzip.open(product_path, "rt") as input_file :
       
        for i,line in tqdm(enumerate(input_file )) :
            try:
                # Assuming each line is a JSON object
                item = json.loads(line.strip())
                valid_product_id_list.append(item["asin"])
            except json.JSONDecodeError:
                print("Error decoding JSON from line:", line)
            
    return valid_product_id_list


# Function to filter items based on a condition
def filter_review_items(item,valid_product_id_list):
    
    if item["asin"] not in  valid_product_id_list :
        return True  
    else:
        return False

def sample_review_subset(review_path,sampled_review_path,product_path):
    valid_product_id_list=gen_valid_product_id_list(product_path)
    # total_num=get_gz_file_length(input_file_path)
    # Load the gzipped file and filter items
    previous_failed_product_id=-1
    with gzip.open(review_path, "rt") as input_file, gzip.open(sampled_review_path, "wt") as output_file:
       
        valid_num=0
        for i,line in tqdm(enumerate(input_file )):
            
            if i%100000==0:
                print(f"valid_num:{valid_num},total_num:{i}") 
            try:
                # Assuming each line is a JSON object
                item = json.loads(line.strip())
                if previous_failed_product_id==item["asin"]:
                    continue 
                # Check the condition and write to the output file if it's met
                if not filter_review_items(item,valid_product_id_list):
                    valid_num+=1
                    output_file.write(json.dumps(item) + "\n")
                    
                    if valid_num>100000:#100000
                        break 
                else:
                    previous_failed_product_id=item["asin"]
                           
            except json.JSONDecodeError:
                print("Error decoding JSON from line:", line)

import random 
def sample_10000_review(review_path,output_products_path,is_has_attribute_first=True ):
    output_list=[]
    no_attribute_review_list=[]
    with open(review_path, 'r', encoding='utf-8') as fp:
        review_dataset_json_array = json.load(fp) 
        if is_has_attribute_first:
            for review_dataset_json in  review_dataset_json_array :
                predicted_attribute=review_dataset_json["predicted_attribute_amazon_exact"]
                if len(predicted_attribute)>0:
                    output_list.append(review_dataset_json)
                else:
                    no_attribute_review_list.append(review_dataset_json)
            number=len(output_list)
            if number<10000:
                remaining_number=10000-number
                remaining_review_list=random.sample(no_attribute_review_list,remaining_number)
                output_list.extend(remaining_review_list)
        else:
            output_list=random.sample(review_dataset_json_array,10000)
        with open(output_products_path, 'w', encoding='utf-8') as fp:
            json.dump(output_list, fp, indent=4)
 
 
def merge_attribute(review_path,output_products_path):
    output_list=[]
    no_attribute_review_list=[]
    with open(review_path, 'r', encoding='utf-8') as fp:
        review_dataset_json_array = json.load(fp) 
        for review_dataset_json in  review_dataset_json_array :   
            predicted_attribute_vicuna_complete=review_dataset_json["predicted_attribute_vicuna_complete"]      
            predicted_attribute_amazon_exact=review_dataset_json["predicted_attribute_amazon_exact"]
            predicted_attribute={}
            predicted_attribute.update(predicted_attribute_vicuna_complete)
            predicted_attribute.update(predicted_attribute_amazon_exact)
            review_dataset_json["predicted_attribute"]=predicted_attribute
            output_list.append(review_dataset_json)
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
            
        print(f"{total_number},{len(review_dataset_json_array)},{total_number/len(review_dataset_json_array)}")            
            
infile_path="data/amazon/All_Amazon_Meta.json.gz"
product_w_attribute_file_path="data/amazon/All_Amazon_Product_w_Attribute.json.gz"
# get_product_w_attribute(infile_path,product_w_attribute_file_path)
# df = getProductDF(file_path)
review_path="data/amazon/sampled_10000_Amazon_Review_w_attribute_vicuna_from_0_to_10000000.json"
# review_path="data/amazon/merged_10000_Amazon_Review_w_attribute_all.json"
sampled_review_path="data/amazon/sampled_100000_Amazon_Review.json.gz"
product_path="data/amazon/All_Amazon_Product_w_Attribute.json.gz"
# sample_review_subset(review_path,sampled_review_path,product_path)
sampled_100000_review_path="data/amazon/sampled_100000_Amazon_Review_w_attribute_from_0_to_10000000.json"
sampled_10000_review_path="data/amazon/sampled_10000_Amazon_Review.json"
sampled_10000_review_path_no_bias="data/amazon/allow_no_attribute/sampled_10000_Amazon_Review_allow_no_attribute.json"
# sample_10000_review(sampled_100000_review_path,sampled_10000_review_path)
# merge_attribute(review_path,merged_review_path)
merged_review_path="data/amazon/merged_10000_Amazon_Review_w_attribute_all.json"

sampled_10000_review_path_no_bias_w_vicuna="data/amazon/allow_no_attribute/sampled_10000_Amazon_Review_allow_no_attribute_w_attribute_vicuna_from_0_to_10000000.json"
merged_review_no_bias_path="data/amazon/allow_no_attribute/merged_10000_Amazon_Review_w_attribute_all.json"
merge_attribute(sampled_10000_review_path_no_bias_w_vicuna,merged_review_no_bias_path)
# sample_10000_review(sampled_100000_review_path,sampled_10000_review_path_no_bias,is_has_attribute_first=False)
count_number(merged_review_no_bias_path )
count_number(merged_review_no_bias_path,key="predicted_attribute_amazon_exact")
count_number(merged_review_no_bias_path,key="predicted_attribute_vicuna_complete")
# with gzip.open(file_path, "rt") as f:
#     expected_dict = json.load(f)
#     print("e")
#     print(expected_dict[0])