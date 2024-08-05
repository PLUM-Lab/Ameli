import re
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm 
import random
from nltk.tokenize import word_tokenize

def is_numeral_match(attribute_value,numeral_in_review,pattern):
    numeral_in_product = re.findall(pattern, attribute_value)
    if len(numeral_in_product)>0:
        for numeral in numeral_in_product:
            if numeral not in numeral_in_review:
                return False 
        return True 
    else:
        return False 

def is_sublist_in_list(sublist,test_list):
    return any(test_list[idx: idx + len(sublist)] == sublist
          for idx in range(len(test_list) - len(sublist) + 1))
    
def filter_extracted_attribute_value(attribute_value):
    if len(attribute_value)==1 or   attribute_value   in stopwords.words('english') or attribute_value.isdigit():
        return None 
    else:
        return attribute_value
    
def compare_with_gold(predicted_attribute_value,candidate_product_attribute_value_list,is_allow_numeral_match=False):
    extracted_attribute_value_list=[]
    if filter_extracted_attribute_value(predicted_attribute_value) is not None :
        review_tokens=word_tokenize(predicted_attribute_value.lower())
        pattern='[a-zA-Z]+[0-9]+\w*|[0-9]+\.[0-9]+'
        # numeral_in_review = re.findall(pattern, predicted_attribute_value)
        
        for candidate_product_attribute_value in candidate_product_attribute_value_list:
            
            attribute_value_tokens=word_tokenize(candidate_product_attribute_value.lower())
            if is_sublist_in_list(attribute_value_tokens,review_tokens ):
                extracted_attribute_value_list.append(candidate_product_attribute_value)  
            # elif is_allow_numeral_match and  is_numeral_match(candidate_product_attribute_value,numeral_in_review,pattern) :
            #     extracted_attribute_value_list.append(candidate_product_attribute_value)  
    return extracted_attribute_value_list



         
def extract_model_version(candidate_attribute):
    model_version,is_found="",False
    pattern='\s[a-zA-Z]+[0-9]+\w*|\s[a-zA-Z]+-+[0-9]+\w*|\s[a-zA-Z]+-[a-zA-Z]+[0-9]+\w*|\s[a-zA-Z]+\s[0-9]+\s'
    model_version_list = re.findall(pattern, candidate_attribute)
    output_model_version_list=[]
    filter_key_list=["ps3","ps4","ps5","PS4","PS5","PS3","playstation 4","playstation 3","playstation 5","PlayStation 4","PlayStation 3","PlayStation 5","v1","v2","v3","v4","v5","v6","v7","v8"]
    for model_version in model_version_list:
        if model_version.strip() not in filter_key_list:
            output_model_version_list.append(model_version.strip())
    return output_model_version_list
        
    
    