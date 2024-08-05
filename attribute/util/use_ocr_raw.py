 
 
import argparse
from collections import Counter
from tqdm import tqdm 
import json
from pathlib import Path



def add_ocr_raw(data_path,output_path):
    out_list=[]
    with open(data_path, 'r', encoding='utf-8') as fp:
        product_dataset_json_array = json.load(fp)
        max_ocr_num=10
        min_ocr_num=5
        for review_product_json   in product_dataset_json_array :
            ocr_raw_list=[]
            ocr_raw_confidence_dict=review_product_json["confidence_score_ocr"]
            sorted_ocr_raw_confidence_dict= sorted(ocr_raw_confidence_dict.items(), key=lambda x:x[1],reverse=True)
            for idx,(sorted_ocr_value,confidence_score) in enumerate(sorted_ocr_raw_confidence_dict ):
                if idx<min_ocr_num:
                    ocr_raw_list.append(sorted_ocr_value)
                elif confidence_score>0.8 and idx<max_ocr_num:
                    ocr_raw_list.append(sorted_ocr_value)
            review_product_json["raw_ocr"]=ocr_raw_list
            out_list.append(review_product_json)
    with open(output_path , 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4) 
        
        
        
        


import argparse
def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--review_file',type=str,help=" ",default="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/val/disambiguation/bestbuy_review_2.3.17.11.11_fused_attribute.json") 
    parser.add_argument('--output_path',type=str,help=" ",default="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/val/disambiguation/bestbuy_review_2.3.17.11.12_add_raw_ocr.json")
    parser.add_argument('--metric_data_path',type=str,help=" ",default="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/val/disambiguation/bestbuy_review_2.3.17.11.10.3_similar_exact_from_0_to_50000.json") 
    parser.add_argument('--file_with_ocr_attribute',type=str,help=" ",default="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v5/test/attribute/ocr/bestbuy_review_2.3.16.29.6_fused_score_20_54_ocr_all.json")
    parser.add_argument('--file_with_gpt_attribute',type=str,help=" ",default="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v5/test/attribute/gpt/bestbuy_review_2.3.16.29.6_fused_score_20_54_attribute_gpt2_all.json")
 
    parser.add_argument('--mode',type=str,help=" ",default="test")#exact numeral
    parser.add_argument('--attribute_logic',type=str,help=" ",default="gpt2")#exact numeral
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=50000)
    parser.add_argument("--is_mp", type=str, default="y", help="Text added prior to input.")
 
    args = parser.parse_args()
    return args


   

if __name__ == '__main__':
    args = parser_args()
     
    add_ocr_raw(args.review_file,args.output_path)