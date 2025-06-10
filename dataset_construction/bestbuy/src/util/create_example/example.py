import nltk 

 
import random          

class Example():
    """
    review_id, product_id, similar_product_id_list
    """
    def __init__(self,review_id,url,product_category_chunk,product_title_chunk,gold_mention,predicted_mention,similarity_score,mention_candidates_before_review_match,product_category,product_title,review) :
        self.gold_mention=gold_mention
        self.review_id=review_id 
        self.product_category_chunk=product_category_chunk
        self.product_title_chunk=product_title_chunk
        self.predicted_mention=predicted_mention
        self.similarity_score=similarity_score
        self.mention_candidates_before_review_match=mention_candidates_before_review_match
        self.product_category=product_category
        self.product_title=product_title
        self.review=review
        self.url=url
        
       
        
           
import pandas as pd 
 