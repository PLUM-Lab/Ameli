from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
from ast import literal_eval
import pandas as pd
import spacy
import pickle
import torch
import pandas as pd
import pickle
import json
import logging 
from PIL import Image
import os 
from sentence_transformers.cross_encoder import CrossEncoder

from bestbuy.src.extract_mention.name_entity_recognition.evaluate import NpEncoder
from bestbuy.src.organize.merge_attribute import json_to_product_id_dict
def main():
    # product_json_path_str=f'bestbuy/data/example/bestbuy_100_human_performance_w_similar_score.json'
    # output_products_path_str=f'bestbuy/data/example/bestbuy_100_human_performance_w_similar_score_v2.json'
    product_json_path_str=f'bestbuy/data/final/v2_course/bestbuy_products_40000_3.4.10_remove_wrong_image.json'
    review_json_path_str='bestbuy/data/final/v2_course/bestbuy_review_2.3.16.10_remove_error_image.json'
    output_products_path_str=f'bestbuy/data/final/v2_course/bestbuy_review_2.3.16.11_similarity_score.json'
    review_image_dir="bestbuy/data/final/v2_course/review_images"
    product_image_dir="bestbuy/data/final/v2_course/product_images"
    # mode="text"
    with open(Path(product_json_path_str), 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        product_json_dict=json_to_product_id_dict(new_crawled_products_url_json_array)
    
    image_model = SentenceTransformer('clip-ViT-L-14')#all-mpnet-base-v2
    review_json_path = Path(
        review_json_path_str
    )    
    output_products_path = Path(
        output_products_path_str
    ) 
    text_model_name="cross-encoder/ms-marco-MiniLM-L-12-v2"  
    text_model = CrossEncoder(text_model_name)
    output_list=[]    
    save_step=3000
    with open(review_json_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        # check_review_id_number(new_crawled_products_url_json_array)
        for idx,review_dataset_json   in tqdm(enumerate(new_crawled_products_url_json_array)):
            product_id=review_dataset_json["id"]
            product_json=product_json_dict[product_id]
            
            product_text=product_json["overview_section"]["description"]
            product_title=product_json["product_name"]
            product_images=product_json["image_path"]
            
            review_json=review_dataset_json["reviews"][0]
            review_text=review_json["header"]+". "+review_json["body"]
            review_images=review_json["image_path"]
            # product_text_similarity_score=gen_text_similarity_score(review_text,product_text)
            product_image_similarity_score,cosine_score_list=gen_image_similarity_score(review_images,product_images,image_model,product_image_dir,review_image_dir)  
            text_similarity_score=gen_text_similarity(review_text,product_text,text_model)
            # product_title_similarity_score=gen_text_similarity(review_text,product_title,text_model)
            review_dataset_json["reviews"][0]["image_similarity_score"]= product_image_similarity_score 
            # review_dataset_json["similar_product_id"]=[]
            # review_dataset_json["product_id_with_similar_image"]=[]
            review_dataset_json["text_similarity_score"]=text_similarity_score
            # review_dataset_json["product_title_similarity_score"]=product_title_similarity_score
            review_dataset_json["image_similarity_score_list"]=cosine_score_list
            output_list.append(review_dataset_json)
            if idx%save_step==0:
                with open(output_products_path, 'w', encoding='utf-8') as fp:
                    json.dump(output_list, fp, indent=4, cls=NpEncoder)  
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(output_list, fp, indent=4, cls=NpEncoder)  
           
           

def gen_real_image_path(image_dir,relative_image_paths):
    image_path_list=[os.path.join(image_dir,image_name) for image_name in relative_image_paths ]
    return image_path_list
           
def open_images(image_dir,relative_image_paths):
    image_path_list=gen_real_image_path(image_dir,relative_image_paths)
    image_list=[]
    for image_path in image_path_list:
        image=Image.open(image_path)
        image_list.append(image)
    return image_list
                
           
           
def gen_image_similarity_score(review_images,product_images,model,product_image_dir,review_image_dir)            :
    #Compute embedding for both lists
    review_images=open_images(review_image_dir,review_images)
    product_images=open_images(product_image_dir,product_images)
    
    
    review_embeddings = model.encode(review_images, convert_to_tensor=True,show_progress_bar=False)
    product_embeddings  = model.encode(product_images, convert_to_tensor=True,show_progress_bar=False)
    

    #Compute cosine-similarities
    cosine_scores = util.cos_sim(product_embeddings, review_embeddings) 
    max_similarity_score=0
    for i in range(len(cosine_scores) ):
        for j in range( len(cosine_scores[0])):
     
            if max_similarity_score<cosine_scores[i][j]:
                max_similarity_score=cosine_scores[i][j]
    return round(max_similarity_score.item(),2)     ,    round_2(cosine_scores.tolist())        
    
    
def round_2(cosine_score_list_list):
    list_list=[]
    for cosine_score_list in cosine_score_list_list:
        one_list=[]
        for cosine_score in cosine_score_list:
        
            new_cosine_score=round(cosine_score,2)
            one_list.append(new_cosine_score)
        list_list.append(one_list)
    return list_list
    
def gen_text_similarity(review_text,product_text,model):
    scores=model.predict([[review_text,product_text]])
    return scores[0]
    
    
def main2():
    # product_json_path_str=f'bestbuy/data/example/bestbuy_100_human_performance_w_similar_score.json'
    # output_products_path_str=f'bestbuy/data/example/bestbuy_100_human_performance_w_similar_score_v2.json'
    product_json_path_str=f'bestbuy/data/final/v2_course/bestbuy_products_40000_3.4.10_remove_wrong_image.json'
    review_json_path_str='bestbuy/data/final/v2_course/bestbuy_review_2.3.16.11_similarity_score.json'
    output_products_path_str=f'bestbuy/data/final/v2_course/bestbuy_review_2.3.16.12_title_similarity_score.json'
    review_image_dir="bestbuy/data/final/v2_course/review_images"
    product_image_dir="bestbuy/data/final/v2_course/product_images"
    # mode="text"
    with open(Path(product_json_path_str), 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        product_json_dict=json_to_product_id_dict(new_crawled_products_url_json_array)
    
    # image_model = SentenceTransformer('clip-ViT-L-14')#all-mpnet-base-v2
    review_json_path = Path(
        review_json_path_str
    )    
    output_products_path = Path(
        output_products_path_str
    ) 
    text_model_name="cross-encoder/ms-marco-MiniLM-L-12-v2"  
    text_model = CrossEncoder(text_model_name)
    output_list=[]    
    save_step=3000
    cross_inp=[]
    with open(review_json_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        # check_review_id_number(new_crawled_products_url_json_array)
        for idx,review_dataset_json   in tqdm(enumerate(new_crawled_products_url_json_array)):
            product_id=review_dataset_json["id"]
            product_json=product_json_dict[product_id]
            
            product_text=product_json["overview_section"]["description"]
            product_title=product_json["product_name"]
            product_images=product_json["image_path"]
            
            review_json=review_dataset_json["reviews"][0]
            review_text=review_json["header"]+". "+review_json["body"]
            review_images=review_json["image_path"]
            # product_text_similarity_score=gen_text_similarity_score(review_text,product_text)
            cross_inp.append([review_text,product_title])
        
            # product_title_similarity_score=gen_text_similarity(review_text,product_title,text_model)
           
            # review_dataset_json["product_title_similarity_score"]=product_title_similarity_score
       
             
        
        scores=text_model.predict(cross_inp,batch_size=1024)
        
        for idx,review_dataset_json   in tqdm(enumerate(new_crawled_products_url_json_array)):
            review_dataset_json["product_title_similarity_score"]=scores[idx]
            output_list.append(review_dataset_json)
            if idx%save_step==0:
                with open(output_products_path, 'w', encoding='utf-8') as fp:
                    json.dump(output_list, fp, indent=4, cls=NpEncoder) 
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(output_list, fp, indent=4, cls=NpEncoder)      

if __name__ == "__main__":
    main2()