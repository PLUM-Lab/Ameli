# import cv2
# import easyocr
import os

from attribute.extractor.base import Extractor 
# import imutils
from nltk.tokenize import word_tokenize
import re

from attribute.extractor.amazon_review_gen_review_attribute_by_rule import is_numeral_match, is_sublist_in_list
from attribute.util.util import compare_with_gold, extract_model_version 
class ModelVersionToProductTitleExtractor(Extractor):
    def __init__(self) -> None:
        pass
 
    
    def generate_per_review_attribute(self,args, prefix,review,attribute_key,candidate_attribute_list,is_constrained_beam_search,
                                      review_dataset_json,review_image_dir,total_attribute_value_candidate_list,
                                      total_confidence_score_list,mention,gold_attribute_value,ocr_raw,gpt_context,chatgpt_context):
        extracted_attribute_value_list=[]
        context_list=[]
        review_source_list=[review]
        review_source_list.extend(ocr_raw)
        
        # review_source_list.extend(gpt_context)
        # review_source_list.extend(chatgpt_context)
        if attribute_key in[ "Product Title","Product Name"]:
            for candidate_attribute_value in candidate_attribute_list:
                model_version_list=extract_model_version(candidate_attribute_value)
                
                if len(model_version_list)>0:
                    is_found_in_loop=False
                    model_version = model_version_list[0]
                    model_version=model_version.strip()
                    # if model_version   in filter_key_list:
                    #     continue
                    
                    # print(model_version_list,candidate_attribute_value)
                    for review_source in review_source_list:
                        if model_version.lower() in review_source.lower():
                            if candidate_attribute_value not in gold_attribute_value:
                                print(f"version:{model_version}, find:{candidate_attribute_value}")
                                print(f"gold {gold_attribute_value}")
                            extracted_attribute_value_list.append(candidate_attribute_value)
                            context_list.append(model_version)
                            is_found_in_loop=True
                            break
                    # if is_found_in_loop:
                    #     break
                            
        return None,None, extracted_attribute_value_list,context_list,[],[]
        

class ModelVersionExtractor(Extractor):
    def __init__(self) -> None:
        pass
 
    
    def generate_per_review_attribute(self,args, prefix,review,attribute_key,candidate_attribute_list,is_constrained_beam_search,
                                      review_dataset_json,review_image_dir,total_attribute_value_candidate_list,
                                      total_confidence_score_list,mention,gold_attribute_value,ocr_raw,gpt_context,chatgpt_context):
        extracted_attribute_value_list=[]
        context_list=[]
        review_source_list=[review]
        review_source_list.extend(ocr_raw)
        
        # review_source_list.extend(gpt_context)
        # review_source_list.extend(chatgpt_context)
        if attribute_key in[ "Model Version"]:
            for model_version in candidate_attribute_list:
                 
           
                    
                    # print(model_version_list,candidate_attribute_value)
                for review_source in review_source_list:
                    if model_version.lower() in review_source.lower():
                         
                        extracted_attribute_value_list.append(model_version)
                         
                        break
            
        extracted_attribute_value_list=list(set(extracted_attribute_value_list))                            
        # if len(extracted_attribute_value_list)>1:
        #     print("")
        return None,None, extracted_attribute_value_list,context_list,[],[]        
        
        
 