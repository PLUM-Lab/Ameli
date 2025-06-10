
from tqdm import tqdm
from bestbuy.src.extract_mention.dependency_parser.parser import gen_mention_candidates
from bestbuy.src.extract_mention.name_entity_recognition.train.inference import inference_one
from bestbuy.src.extract_mention.name_entity_recognition.train.train import gen_model
from  bestbuy.src.extract_mention.name_entity_recognition.ner_bert import ner_bert
from  bestbuy.src.extract_mention.name_entity_recognition.ner_spacy import ner_spacy
from  bestbuy.src.extract_mention.name_entity_recognition.token_similarity import find_most_similar_word
from  bestbuy.src.extract_mention.pos.pos_spacy import pos_spacy

from nltk.tokenize import sent_tokenize, word_tokenize
from pathlib import Path
from time import time
from pathlib import Path
import json
import pandas as pd
import requests
import json
from typing import Dict
from pathlib import Path
from time import time
from tqdm import tqdm
from lxml.html import fromstring
from bestbuy.src.crawler.image_downloader import download_row_images_for_json
from bestbuy.src.crawler.old.scraper_specs_main import gen_wait
from bestbuy.src.crawler.old.specs_scraper import SpecScraper
from bestbuy.src.organize.checker import is_nan, is_nan_or_miss
from bestbuy.src.scraper_classes import OverviewScraper, ReviewScraper, ThumbnailScraper
import concurrent.futures

# headers
headers = {'User-Agent': 'Mozilla/5.0'}
import logging  
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"

from bestbuy.src.organize.merge_attribute import json_to_dict
from sentence_transformers.cross_encoder import CrossEncoder
import numpy as np

from nltk.tokenize import sent_tokenize, word_tokenize
class MentionGenerator:
    def __init__(self) -> None:
        self.model = CrossEncoder('cross-encoder/stsb-roberta-large') 
    

    def generate_mention(self,review, noisy_product_name,product_category,product_desc):
        named_entity_in_review_list=gen_mention_candidates(review,noisy_product_name,product_category,product_desc)
        if len(named_entity_in_review_list)>0:
            return self.find_most_similar_word(noisy_product_name,named_entity_in_review_list)
        else:
            return None 
 
 
    def find_most_similar_word(self,query,corpus):
        # Pre-trained cross encoder
        

        

        # So we create the respective sentence combinations
        sentence_combinations = [[query, corpus_sentence] for corpus_sentence in corpus]

        # Compute the similarity scores for these combinations
        similarity_scores = self.model.predict(sentence_combinations)

        # Sort the scores in decreasing order
        sorted_order=np.argsort(similarity_scores)
        sim_scores_argsort = reversed(sorted_order)

        # Print the scores
        # print("Query:", query)
        is_first=True  
        for idx in sim_scores_argsort:
            if is_first:
                choose_text=corpus[idx]
                is_first=False 
                return choose_text
            # print("{:.2f}\t{}".format(similarity_scores[idx], corpus[idx]))
         
        return choose_text
        
        
  
def bs4_review_scraper(review_json: Dict,product_json_dict,mention_generator):
    is_debug=False 
    product_category=review_json["product_category"]
    product_category=product_category.split(" -> ")[-1]
    noisy_product_name=review_json["product_name"]
    noisy_product_name_list=noisy_product_name.split(" - ")
    noisy_product_name=" ".join(noisy_product_name_list[:2])
    product_json=product_json_dict[review_json["url"]]
    if "overview_section" in product_json and  "description" in product_json["overview_section"]:
        product_desc =product_json["overview_section"]["description"]
        product_desc_sent_list=sent_tokenize(product_desc)
        if len(product_desc_sent_list)>0:
            product_desc=product_desc_sent_list[0]
        else:
            product_desc=""
    else:
        product_desc=""
    reviews=review_json["reviews"]
    output_review_list=[]
    for review in reviews:
        review_text=review["body"]
        review_id=review["id"]
        # if review_id <start_id:
        #     continue 
        mention=mention_generator.generate_mention(review_text,noisy_product_name,product_category,product_desc)
        if is_debug:
            print(f"{mention}, {noisy_product_name}, {product_category}, {product_desc}, {review_text} ")
        else:
            print(f"get {mention} for {review_id}")
        review["mention"]=mention
        output_review_list.append(review)
    review_json["reviews"]=output_review_list
    return review_json
                    
def main():
    step_size = 4
 
    print_ctr = 0
    result = []
    save_step=1000
      
    
    review_json_path = Path(
        f'bestbuy/data/final/v0/bestbuy_review_2.3.4_w_image.json'
    )        
    product_dataset_path = Path('bestbuy/data/final/v0/bestbuy_products_40000_3.4.2_desc_img_url.json')
    output_products_path = Path(
        f'bestbuy/data/final/v0/bestbuy_review_2.4.1.json'
    )  
    output_list=[]
    
    mention_generator_list= [MentionGenerator() for i in range(step_size)]
    with open(product_dataset_path, 'r', encoding='utf-8') as fp:
        product_dataset_array = json.load(fp)
        product_json_dict=json_to_dict(product_dataset_array)
        product_json_dict_list= [product_json_dict for i in range(step_size)]
        with open(review_json_path, 'r', encoding='utf-8') as fp:
            review_json_array = json.load(fp)
            for i in tqdm(range(0, len(review_json_array), step_size)):
                print(100 * '=')
                print(f'starting at the {print_ctr}th value')
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    try:
                        result = list(
                            tqdm(
                                executor.map(
                                    bs4_review_scraper,
                                    review_json_array[i: i + step_size],
                                    product_json_dict_list,mention_generator_list
                                ),
                                total=step_size)
                        )
                    except Exception as e:
                        logging.warning(f"{e}")
                        print(f"__main__ error {e}")
                        result = []
                end = time()
                if len(result) != 0:
                    output_list.extend(result)
                else:
                    print('something is wrong')
                if i%save_step==0:
                    with open(output_products_path, 'w', encoding='utf-8') as fp:
                        json.dump(output_list, fp, indent=4)
            print_ctr += step_size
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(output_list, fp, indent=4)  
    
    
    
     

if __name__ == "__main__":
    main()
    