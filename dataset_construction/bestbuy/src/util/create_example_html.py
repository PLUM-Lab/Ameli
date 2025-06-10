from bestbuy.src.util.create_example.generate_html import generate 
from bestbuy.src.util.create_example.read_example import read_example
 
import webbrowser
import argparse
def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',type=str,help=" ",default="mocheg2/test") #politifact_v3,mode3_latest_v5
    parser.add_argument('--out_path',type=str,help=" ",default="bestbuy/output/html")
    parser.add_argument('--example_type',type=str,help=" ",default="verification")
    parser.add_argument('--generate_claim_level_report',type=str,help=" ",default="y")
    args = parser.parse_args()
    return args

    
  
  
  
  
if __name__ == '__main__':
    args = parser_args()
    data_path=args.data_path 
    # example_num=50
    generate('bestbuy/data/example/bestbuy_review_200_cleaned_annotation_result_parser_v26_fix_no_brand_bug.json',  args.out_path,"v26",  generate_claim_level_report=False ,is_reshuffle=False) 
  
   
