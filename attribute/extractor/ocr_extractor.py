import cv2
import easyocr
import os

from attribute.extractor.base import Extractor 
# import imutils
from nltk.tokenize import word_tokenize
import re

from attribute.extractor.amazon_review_gen_review_attribute_by_rule import is_numeral_match, is_sublist_in_list
from attribute.util.util import compare_with_gold, filter_extracted_attribute_value 
class OCRExtractor(Extractor):
    def __init__(self,device) -> None:
        self.reader = easyocr.Reader([ 'en']) # this needs to run only once to load the model into memory
        
    def generate_per_review(self,args, prefix,prompt_text,attribute_key,candidate_attribute_list,is_constrained_beam_search,review_dataset_json,
                            review_image_dir):
        threshold=0.5
        extracted_attribute_value_list=[]
        confidence_score_dict={}
        is_check_rotated_image=True
        for review_image_name in review_dataset_json["review_image_path"]:
            # review_image_name="0009077-b950014ca6ebb8ad7e0bbaea4a1ff069.jpg"
            review_image_path = os.path.join(review_image_dir,review_image_name )  
            extracted_attribute_value_list,confidence_score_dict=self.ocr_for_one_image( review_image_path,threshold,extracted_attribute_value_list,confidence_score_dict)
            # if is_check_rotated_image:
            #     image = cv2.imread(review_image_path)
            #     for rotation in [90,180,270]:
            #         rotated = imutils.rotate_bound(image, angle=rotation)
            #         # show the original image and output image after orientation
            #         # correction
            #         rotate_path="output/temp/rotate.jpg"
            #         cv2.imwrite(rotate_path, rotated)
            #         extracted_attribute_value_list,confidence_score_dict=self.ocr_for_one_image( rotated,threshold,extracted_attribute_value_list,confidence_score_dict)
                
                

            # print(result)
            
        return None,None,extracted_attribute_value_list,[],confidence_score_dict
    
    def generate_per_review_attribute(self,args, prefix,prompt_text,attribute_key,candidate_attribute_list,is_constrained_beam_search,
                                      review_dataset_json,review_image_dir,total_attribute_value_candidate_list,
                                      total_confidence_score_list,mention,attribute_value):
        extracted_attribute_value_list=[]
        for predicted_attribute_value in total_attribute_value_candidate_list:
            extracted_attribute_value_list.extend(compare_with_gold(predicted_attribute_value,candidate_attribute_list))
        return None,None, extracted_attribute_value_list,[],[]
        
         
    
    


    def ocr_for_one_image(self,review_image_path,threshold,extracted_attribute_value_list,confidence_score_dict):
        result = self.reader.readtext(review_image_path,batch_size=64,workers=32,rotation_info=[90, 180 ,270] )
        for one_item in result:
            bounding_box,word_text,confidence_score=one_item
            if confidence_score>threshold and len(word_text)>1:
                if word_text in confidence_score_dict and confidence_score_dict[word_text]>=confidence_score:
                    continue 
                extracted_attribute_value_list.append(word_text)
                confidence_score_dict[word_text]= confidence_score
        return extracted_attribute_value_list,confidence_score_dict
    
    
    
    

class OCRResultParser(Extractor):
    def __init__(self,device) -> None:
        pass 
    
    def generate_per_review(self,args, prefix,prompt_text,attribute_key,candidate_attribute_list,is_constrained_beam_search,review_dataset_json,review_image_dir):
        confidence_score_dict=review_dataset_json["confidence_score_ocr"]
            
        return None,None,list(confidence_score_dict.keys()),[],confidence_score_dict
    
    def generate_per_review_attribute(self,args, prefix,prompt_text,attribute_key,candidate_attribute_list,is_constrained_beam_search,
                                      review_dataset_json,review_image_dir,total_attribute_value_candidate_list,
                                      total_confidence_score_list,mention,gold_attribute_value,ocr_raw,gpt_context,chatgpt_context):
        extracted_attribute_value_list=[]
        for predicted_attribute_value in total_attribute_value_candidate_list:
            # if predicted_attribute_value=="jet" and "Color" in attribute_key :
            #     print()
            extracted_attribute_value_list.extend(compare_with_gold(predicted_attribute_value,candidate_attribute_list))
        return None,None, extracted_attribute_value_list,[],[],[]
        
         
    
     