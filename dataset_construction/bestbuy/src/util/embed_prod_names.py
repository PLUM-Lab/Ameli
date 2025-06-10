from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import pandas as pd
import pickle
import json
from pathlib import Path
from time import time
from pathlib import Path
import json
import pandas as pd
import requests
import os
from tqdm import tqdm
from ast import literal_eval
"""
this script takes all of the product names and embeds them
"""
model = SentenceTransformer("all-mpnet-base-v2")


def embed_product_names(df: pd.DataFrame) -> List[Dict]:
    """
    :param df:
    :return:
    """
    smaller_df = df[['id', 'product_name', 'product_path']]
    embedding_dict_list = []
    for row_id, row in tqdm(smaller_df.iterrows()):
        id = row['id']
        product_name = row['product_name']
        product_path = row['product_path']
        embedding = model.encode(product_name)
        embedding_dict_list.append({
            "id": id,
            "product_name": product_name,
            "embedding": embedding,
            "product_category": product_path
        })
    return embedding_dict_list




def embed_product_names_for_json(product_json_path_str) -> List[Dict]:
    product_json_path = Path(
        product_json_path_str
    )        
    # review_dataset_path = Path('bestbuy/data/final/v1/bestbuy_review_2.3.9_w_mention.json')
    # output_products_path = Path(
    #     f'bestbuy/data/example/bestbuy_100_human_performance.json'
    # )  
    # number=0
    
    # fields=["reviews","overview_section","product_images_fnames","product_images","Spec"]
    
    # fields=["thumbnails", "thumbnail_paths","reviews", "Spec"]
    output_list=[]
    embedding_dict_list = []
    with open(product_json_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        for review_dataset_json   in tqdm(new_crawled_products_url_json_array):
        
            id  = review_dataset_json['id']
            product_name = review_dataset_json['product_name']
            product_path = review_dataset_json['product_path']
            embedding = model.encode(product_name,show_progress_bar=False )
            embedding_dict_list.append({
                "id": id,
                "product_name": product_name,
                "embedding": embedding,
                "product_category": product_path
            })
    return embedding_dict_list


def write_embeddings(embeddings_list: List[Dict], outpath: str):
    """
    :param embeddings_list:
    :param outpath:
    :return:
    """
    with open(outpath, 'wb') as fp:
        pickle.dump(embeddings_list, fp)


def main_for_csv():
    db_path = Path('../data/bestbuy_data_with_ids_and_specs.csv')
    outpath = Path('../data/product_name_embeddings.pickle')
    db = pd.read_csv(db_path)
    prod_names = embed_product_names(db)
    write_embeddings(prod_names, outpath)
    
def gen_embed_product_names_for_json(corpus_path,embeddings_path_str):
  
    prod_names = embed_product_names_for_json(corpus_path )
    write_embeddings(prod_names, embeddings_path_str)

if __name__ == "__main__":
    gen_embed_product_names_for_json()
