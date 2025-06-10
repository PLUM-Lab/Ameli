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
logging.basicConfig( filename="bestbuy/output/review_image_check.txt",filemode="w",   level=logging.INFO)
from bestbuy.src.util.embed_prod_names import gen_embed_product_names_for_json

"""
this script takes all of the product names and embeds them
"""


 
def check_id(review_id_list):
    logging.info(review_id_list)
    review_id_set=set()
    for review_id in review_id_list:  
        if review_id in review_id_set:
            print("ERROR!")
        else:
            review_id_set.add(review_id)     

def check_review_id_number(review_products_json_list):
 
    id_list=[]
    for review_product_json in review_products_json_list:
        review_json=review_product_json["reviews"][0]
        id=review_json["id"] 
        id_list.append(id )
    check_id(id_list )
    print(f"{len(set(id_list))}")
 
def find_similar_products_for_json(top_k,product_json_path_str,output_products_path_str,is_only_in_category):
    """
    :param samples_df:
    :return:
    """
    
    product_json_path = Path(
        product_json_path_str
    )    
    output_products_path = Path(
        output_products_path_str
    ) 
    output_list=[]    
    with open(product_json_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        # check_review_id_number(new_crawled_products_url_json_array)
        for review_dataset_json   in tqdm(new_crawled_products_url_json_array):
            product_path=review_dataset_json["product_category"]
            product_name=review_dataset_json["product_name"]
            product_id=review_dataset_json["id"]
            similar_product_id_list=review_dataset_json["similar_product_id"] 
    
            similar_ids = _find_similar_products(product_path,product_name,top_k,is_only_in_category,similar_product_id_list)
            similar_ids_without_gold=[]
            for retrieved_product_id in similar_ids:
                if retrieved_product_id!=product_id:
                    similar_ids_without_gold.append(retrieved_product_id)
                if len(similar_ids_without_gold)==top_k:
                    break 
            review_dataset_json["similar_product_id"]=similar_ids_without_gold
            output_list.append(review_dataset_json)

    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(output_list, fp, indent=4)   
        
        

def _find_similar_products(product_path,product_name,top_k,is_only_in_category,similar_prods_ids_from_input):
    """
    :param row:
    :return:
    """
    if not is_only_in_category and len(similar_prods_ids_from_input)!=0:
        return similar_prods_ids_from_input
        
   
    relevant_path = product_path
    if is_only_in_category:
        path_embeddings = [
            embedding for embedding in embeddings
            if relevant_path == embedding['product_path']
        ]
    else:
        path_embeddings=embeddings
    query =product_name
    query_encoded = model.encode(query,show_progress_bar=False )
    path_embeddings_list = [pe['embedding'] for pe in path_embeddings]
    
    hits = util.semantic_search(query_encoded,
                                        torch.tensor(path_embeddings_list),
                                        top_k=top_k+1)
    hits = hits[0] 
    hits_before_cross_encoder = sorted(hits, key=lambda x: x['score'], reverse=True) 
    similar_prods_ids=[]
    for semantic_dict in hits_before_cross_encoder:
        similar_prods_ids.append(path_embeddings[semantic_dict['corpus_id']]['id'])    
    # subset = db_df[db_df.apply(lambda row: row['id'] in similar_prods_ids, axis=1)]
    return similar_prods_ids 
   


if __name__ == "__main__":
    # dataset="entity_linking"
    dataset="entity_linking"
    if dataset=="entity_linking":
        corpus_path=f'bestbuy/data/final/v3/bestbuy_products_40000_3.4.13_sensitive.json'
        embeddings_path_str='bestbuy/data/final/v3/embed/corpus_text_embeddings.pickle'
        product_json_path=f'bestbuy/data/final/v3/bestbuy_products_40000_3.4.13_sensitive.json'
        output_json_path=f'bestbuy/data/final/v3/bestbuy_products_40000_3.4.14_text_similar.json'
    else:
        corpus_path=f'bestbuy/data/final/v2_course/bestbuy_products_40000_3.4.13_sensitive.json'
        embeddings_path_str='bestbuy/data/final/v2_course/embed/corpus_text_embeddings.pickle'
        # product_json_path=f'bestbuy/data/example/bestbuy_100_human_performance.json'
        product_json_path=f'bestbuy/data/example/bestbuy_100_human_performance_no_similarity_product.json'
        output_json_path=f'bestbuy/data/example/bestbuy_100_human_performance_w_fix_text_similar_products.json'
    top_k=10
    embeddings_path = Path(embeddings_path_str)    
    # gen_embed_product_names_for_json(corpus_path,embeddings_path_str)
    mode="text"
    
    model = SentenceTransformer("all-mpnet-base-v2")
    
    with open(embeddings_path, 'rb') as fp:
        embeddings = pickle.load(fp)
    # embeddings_list = [d['embedding'] for d in embeddings]
    nlp = spacy.load("en_core_web_lg")
    is_only_in_category=True 
    find_similar_products_for_json(top_k,product_json_path,output_json_path,is_only_in_category)
    find_similar_products_for_json(top_k,output_json_path,output_json_path,False)
