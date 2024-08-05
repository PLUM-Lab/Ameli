from util.visualize.create_example.generate_html import generate 
from util.visualize.create_example.read_example import read_example
from pathlib import Path
import webbrowser
from util.env_config import * 
import argparse
def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',type=str,help=" ",default="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/multimodal_entity_linking/output/disambiguation/temp_bestbuy_review_2.3.17.11.21.1_50_error.json") 
    parser.add_argument('--out_path',type=str,help=" ",default="disambiguation_error")
    parser.add_argument('--example_type',type=str,help=" ",default="verification")
    parser.add_argument('--generate_claim_level_report',type=str,help=" ",default="y")
    args = parser.parse_args()
    return args


  
  
  
  
if __name__ == '__main__':
    args = parser_args()
    data_path=args.data_path 
    # similar_products_path = Path(
    #     f'/home/menglong/workspace/code/referred/product_scraper/dataset_construction/bestbuy/data/final/v2_course/bestbuy_products_40000_3.4.10_remove_wrong_image.json'
    # )
    # review_image_dir="/home/menglong/workspace/code/referred/product_scraper/dataset_construction/bestbuy/data/final/v2_course/review_images"
    # product_image_dir="/home/menglong/workspace/code/referred/product_scraper/dataset_construction/bestbuy/data/final/v2_course/product_images"
    similar_products_path = Path(
        products_path_str
    )
    review_image_dir="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/review_images"
    product_image_dir="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/product_images"

    generate(  args.out_path, args.data_path, similar_products_path,product_image_dir,review_image_dir,
             generate_claim_level_report=True ,is_reshuffle=False,is_need_image=False ,is_tar=False,is_need_corpus=True,
             is_generate_report_for_human_evaluation=False,is_add_gold=False, is_add_gold_at_end=False ,
             html_template="human_experiment",is_generate_attribute_annotation_json=False ,is_add_gold_in_html=False) #attribute_annotation
  
   
