import nltk 

class Review():
    def __init__(self,review_id,img_list,text,mention,image_similarity_score,is_low_quality_review,attribute,
                 text_similarity_score,product_title_similarity_score,predicted_is_low_quality_review,score_dict,
                 gold_product_index,image_score_dict,text_score_dict,desc_score_dict,review_special_product_info,review_image_path_before_select ):
        self.text=text 
        self.img_list=img_list 
        self.review_id=review_id
        self.mention=mention
        self.image_similarity_score=image_similarity_score
        self.is_low_quality_review=is_low_quality_review
        self.attribute=attribute
        self.text_similarity_score=text_similarity_score
        self.product_title_similarity_score=product_title_similarity_score
        self.predicted_is_low_quality_review=predicted_is_low_quality_review
        new_score_dict={}
        self.gold_product_index=gold_product_index
        for product_id_str,one_score_item in score_dict.items():
            new_score_dict[int(product_id_str)]=one_score_item
        self.score_dict=new_score_dict
        self.image_score_dict=image_score_dict
        self.text_score_dict=text_score_dict
        self.desc_score_dict=desc_score_dict
        self.predicted_attribute_dict={}
        self.review_special_product_info=review_special_product_info
        self.review_image_path_before_select=review_image_path_before_select
       
        # self.predicted_attribute_ocr,self.confidence_score_ocr=predicted_attribute_ocr,confidence_score_ocr
        
class Product():
    def __init__(self,product_id,img_list,text,name,spec_top_list,image_similarity_score_list):
        self.product_id=product_id
        self.text=text 
        self.name=name
        self.img_list=img_list   
        spec_dict={}
        self.image_similarity_score_list=image_similarity_score_list
        self.image_score=-100
        self.text_bi_score=-100
        self.text_cross_score=-100
        self.desc_text_cross_score=-100
        self.desc_text_bi_score=-100
        self.fused_score=-100
        for spec_section in spec_top_list:
             
            spec_item_list=spec_section["text"]
           
            for spec_item in spec_item_list:
             
                subheader=spec_item["specification"]
                subtext=spec_item["value"]
                spec_dict[subheader]=subtext
                 
       
        self.spec_dict=   spec_dict 
        
import random          

class Example():
    """
    review_id, product_id, similar_product_id_list
    """
    def __init__(self,review_id, product_id, similar_product_id_list,is_reshuffle,reshuffled_target_product_id_position,is_add_gold,is_add_gold_at_end) :
        self.review_id=review_id 
        self.target_product_id=product_id
        self.similar_product_id_list=similar_product_id_list
        self.candidate_product_id_list=similar_product_id_list
        candidate_len=len(similar_product_id_list)
        
        if is_reshuffle   :
            if reshuffled_target_product_id_position is None :
                self.reshuffled_target_product_id_position=random.randint(0,candidate_len)
            else:
                self.reshuffled_target_product_id_position  =reshuffled_target_product_id_position  
        
            
        elif is_add_gold_at_end:
            self.reshuffled_target_product_id_position=candidate_len
        else:
            self.reshuffled_target_product_id_position=reshuffled_target_product_id_position
        if is_add_gold:
            self.candidate_product_id_list.insert(self.reshuffled_target_product_id_position,product_id)
        
           
import pandas as pd 
 