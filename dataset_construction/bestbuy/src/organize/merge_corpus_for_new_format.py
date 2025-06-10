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
from bestbuy.src.organize.checker import is_nan, is_nan_or_miss

from bestbuy.src.organize.merge_attribute import json_to_dict,json_to_review_dict, new_review_json_to_product_dict, review_json_to_product_dict
def merge():
    incomplete_products_path_1 = Path(
        f'../../data/bestbuy_scraped_from_10000_to_15000.json'
    )
    incomplete_products_path_2 = Path(
        f'../../data/bestbuy_scraped_from_0_to_10240.json'
    )
    incomplete_products_path_3 = Path(
        f'../../data/bestbuy_scraped_from_15000.json'
    )
    merged_products_path = Path(
        f'../../data/bestbuy_products_1_w_img_url.json'
    )
    output_list=[]
    with open(incomplete_products_path_1, 'r', encoding='utf-8') as fp:
        incomplete_dict_list_1 = json.load(fp)
        with open(incomplete_products_path_2, 'r', encoding='utf-8') as fp:
            incomplete_dict_list_2 = json.load(fp)
            with open(incomplete_products_path_3, 'r', encoding='utf-8') as fp:
                incomplete_dict_list_3 = json.load(fp)
        
                output_list.extend(incomplete_dict_list_1)
                output_list.extend(incomplete_dict_list_2)
                output_list.extend(incomplete_dict_list_3)
                print(len(output_list))
                with open(merged_products_path, 'w', encoding='utf-8') as fp:
                    json.dump(output_list, fp, indent=4)
   
"""
csv 
x(['product_name', 'url', 'product_path', 'thumbnails', 'overview_section',
        'reviews', 'id', 'specs'],    
json. product_name url product_path thumbnails overview_section.description id Spec
"""   
#    
# def merge_csv_json_img_url_to_v1():
#     products_url_json_path = Path(
#         f'bestbuy/data/bestbuy_products_40000_0.1_desc_img_url.json'
#     )        
#     product_url_csv_path = Path('bestbuy/data/bestbuy_data_with_ids_and_specs.csv')
#     output_products_path = Path(
#         f'bestbuy/data/bestbuy_products_40000_1_desc_img_url.json'
#     )    
#     df = pd.read_csv(product_url_csv_path)
#     output_list=[]
#     with open(products_url_json_path, 'r', encoding='utf-8') as fp:
#         products_url_json_array = json.load(fp)
        
#         for products_url_json  in products_url_json_array:
#             if   is_nan(products_url_json["thumbnails"]) :
#                 url=products_url_json["url"]
#                 df_row=df[df['url']==url]
#                 if len(df_row)>0:
#                     value_in_csv=df_row["thumbnails"].values[0]
#                     if not is_csv_nan(value_in_csv,"[]"):
#                         img_urls=literal_eval(value_in_csv)
#                         products_url_json["thumbnails"]=img_urls
#                         output_list.append(products_url_json)
                    
#     with open(output_products_path, 'w', encoding='utf-8') as fp:
#         json.dump(output_list, fp, indent=4)
        
def convert_csv_to_json():
    product_url_csv_path = Path('bestbuy/data/bestbuy_data_with_ids_and_specs.csv')
    output_products_path = Path(
        f'bestbuy/data/bestbuy_data_with_ids_and_specs.json'
    )   
    output_list=[]
    df = pd.read_csv(product_url_csv_path)   
    for idx, db_row in tqdm(df.iterrows()):
        thumbnails = literal_eval(db_row['thumbnails'])
        reviews = literal_eval(db_row['reviews'])
        if not pd.isna(db_row['specs']) :
            specs = literal_eval(db_row['specs'])
        else:
            specs=[]
        overview_section = literal_eval(db_row['overview_section'])
        id=db_row['id']
        product_name=db_row['product_name']
        url=db_row['url']
        product_path=db_row['product_path']
        json_object={
            "product_name":  product_name,
            "url": url,
            "product_category": product_path,
            "id":id,
            "thumbnails": thumbnails,
            "overview_section":overview_section,
            "reviews_urls": None,
            "reviews":reviews,
            "Spec":specs
        }
        output_list.append(json_object)
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(output_list, fp)
        
#TODO if remove products, please remove their reviews        
def separate_review_product():
    product_dataset_path = Path('bestbuy/data/to_be_combined/bestbuy_data_with_ids_and_specs_40000_review_img_url.json')
    output_products_path = Path(
        f'bestbuy/data/intermediate/bestbuy_data_with_ids_and_specs_wo_review.json'
    )   
    output_list=[] 
        
    with open(product_dataset_path, 'r', encoding='utf-8') as fp:
        product_dataset_json_array = json.load(fp)
        for product_dataset_json   in product_dataset_json_array:
            product_dataset_json["reviews"]=[]
            output_list.append(product_dataset_json)
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(output_list, fp, indent=4) 


    """
    1, merge img_url
 
    """
def merge_img_url_to_get_v1():
    new_crawled_img_url_json_path = Path(
        f'bestbuy/data/bestbuy_products_40000_0.1_desc_img_url.json'
    )        
    product_dataset_path = Path('bestbuy/data/bestbuy_data_with_ids_and_specs_wo_review.json')
    output_products_path = Path(
        f'bestbuy/data/bestbuy_products_40000_1_desc_img_url.json'
    )    
    output_list=[]
    with open(new_crawled_img_url_json_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        new_crawled_products_url_json_dict=json_to_dict(new_crawled_products_url_json_array)
        
        with open(product_dataset_path, 'r', encoding='utf-8') as fp:
            product_dataset_json_array = json.load(fp)
            for product_dataset_json   in tqdm(product_dataset_json_array):
                if   is_nan(product_dataset_json["thumbnails"]) :
                    url=product_dataset_json["url"]
                    if url in new_crawled_products_url_json_dict:
                        new_crawled_products_url_json=new_crawled_products_url_json_dict[url]
                        if not is_nan(new_crawled_products_url_json["thumbnails"]):
                            product_dataset_json["thumbnails"]=new_crawled_products_url_json["thumbnails"]
                    
                output_list.append(product_dataset_json)
                    
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(output_list, fp, indent=4)         
        
        
def is_csv_nan(value,pad):
    if value is None or   value==pad:          
        return True 
    else:
        return False 
    
    #TODO remove corresponding reviews, if remove products in the future
    """
    TODO
    2, if exist in new_crawled_img_url_json_path, not in product_dataset_path, then insert
        3, unify the format of Specs item. He has "Key Attribute, General"  prefix while I do not have.
        4, handle product_id 
    """    
def merge_spec_to_get_v2() :
    product_dataset_path = Path('bestbuy/data/bestbuy_products_40000_2_desc_img_url.json')
    output_products_path = Path(
        f'bestbuy/data/bestbuy_products_40000_2_desc_img_url.json'
    ) 
    with open(product_dataset_path, 'r', encoding='utf-8') as fp:
        product_dataset_json_array = json.load(fp)
        product_dataset_json_object_dict=json_to_dict(product_dataset_json_array)
        
    dir_path="bestbuy/data/spec"
    spec_file_list=os.listdir(dir_path)
    for path in spec_file_list:
        spec_file=os.path.join(dir_path, path)
        new_crawled_img_url_json_path = Path(
            spec_file
            ) 
        product_dataset_json_object_dict=update_dict_spec(new_crawled_img_url_json_path,product_dataset_json_object_dict)
              
    output_list=list(product_dataset_json_object_dict.values())
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(output_list, fp, indent=4)         
              
              
def update_dict_spec(new_crawled_img_url_json_path,product_dataset_json_object_dict):
    with open(new_crawled_img_url_json_path, 'r', encoding='utf-8') as fp:
        new_crawled_img_url_array = json.load(fp)
        for new_crawled_dataset_json   in tqdm(new_crawled_img_url_array):
            if not is_nan(new_crawled_dataset_json["Spec"]) :
                url=new_crawled_dataset_json["url"]
                if url in product_dataset_json_object_dict:
                    product_dataset_json_object =product_dataset_json_object_dict[url]
                    if   is_nan(product_dataset_json_object["Spec"]):
                        product_dataset_json_object["Spec"]=[{"subsection": "All", "text": new_crawled_dataset_json["Spec"]  }]
                        product_dataset_json_object_dict[url]=product_dataset_json_object
    return product_dataset_json_object_dict

          
def update_dict_img_path(new_crawled_img_url_json_path,product_dataset_json_object_dict):
    
    
    
    field="thumbnail_paths"
    with open(new_crawled_img_url_json_path, 'r', encoding='utf-8') as fp:
        new_crawled_img_url_array = json.load(fp)
        for new_crawled_dataset_json   in tqdm(new_crawled_img_url_array):
            if not is_nan_or_miss(new_crawled_dataset_json,field) :
                url=new_crawled_dataset_json["url"]
                if url in product_dataset_json_object_dict:
                    product_dataset_json_object =product_dataset_json_object_dict[url]
                    if  is_nan_or_miss(product_dataset_json_object,field)  :
                        product_dataset_json_object[field]= new_crawled_dataset_json[field]  
                        product_dataset_json_object_dict[url]=product_dataset_json_object
    return product_dataset_json_object_dict       

def update_review_dict_img_path(new_crawled_img_url_json_path,product_dataset_json_array):
    
    
    out_list=[]
    
    with open(new_crawled_img_url_json_path, 'r', encoding='utf-8') as fp:
        new_crawled_img_url_array = json.load(fp)
        new_crawled_review_list=json_to_review_dict(new_crawled_img_url_array)
        
        for product_dataset_json   in tqdm(product_dataset_json_array):
            if not is_nan_or_miss(product_dataset_json,"reviews"):
                reviews=product_dataset_json["reviews"]
                out_review_list=[]
                for review in reviews:
                    if is_nan_or_miss(review,"thumbnail_paths"):
                        review_id=review["id"]
                        if review_id in new_crawled_review_list:
                            new_crawled_review=new_crawled_review_list[review_id]
                            if not is_nan_or_miss(new_crawled_review,"thumbnail_paths"):
                                review["thumbnail_paths"]=new_crawled_review["thumbnail_paths"]
                    out_review_list.append(review)
                product_dataset_json["reviews"]=out_review_list
            out_list.append(product_dataset_json)
                
         
    return out_list   


def change_spec_name():
    out_list=[]
    product_dataset_path = Path('bestbuy/data/bestbuy_products_40000_1_desc_img_url.json')
    output_products_path = Path(
        f'bestbuy/data/bestbuy_products_40000_2_desc_img_url.json'
    ) 
    with open(product_dataset_path, 'r', encoding='utf-8') as fp:
        product_dataset_json_array = json.load(fp)
        for dataset_json   in tqdm(product_dataset_json_array):
            if not is_nan(dataset_json["Spec"]) :
                spec_top_list=dataset_json["Spec"]
                new_spec_top_list=[]
                for spec_section in spec_top_list:
                    new_spec_section={} 
                    spec_item_list=spec_section["text"]
                    new_spec_item_list=[]
                    for spec_item in spec_item_list:
                        new_spec_item={}
                        subheader=spec_item["subheader"]
                        subtext=spec_item["subtext"]
                        new_spec_item["specification"]=subheader
                        new_spec_item["value"]=subtext
                        new_spec_item_list.append(new_spec_item)
                    new_spec_section["subsection"]=spec_section["subsection"]
                    new_spec_section["text"]=new_spec_item_list
                    
                    new_spec_top_list.append(new_spec_section)
                dataset_json["Spec"]=new_spec_top_list
            out_list.append(dataset_json)
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4)    
            
def merge_img_path_to_get_v3():
    new_crawled_img_url_json_path = Path(
        f'bestbuy/data/bestbuy_products_40000_0.15_desc_img_url_img_path.json'
    )        
    product_dataset_path = Path('bestbuy/data/bestbuy_products_40000_2_desc_img_url.json')
    output_products_path = Path(
        f'bestbuy/data/bestbuy_products_40000_2.5_desc_img_url.json'
    )    
    output_list=[]
    field="thumbnail_paths"
    with open(new_crawled_img_url_json_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        new_crawled_products_url_json_dict=json_to_dict(new_crawled_products_url_json_array)
        
        with open(product_dataset_path, 'r', encoding='utf-8') as fp:
            product_dataset_json_array = json.load(fp)
            for product_dataset_json   in tqdm(product_dataset_json_array):
                if field not in product_dataset_json  or is_nan(product_dataset_json[field]) :
                    url=product_dataset_json["url"]
                    if url in new_crawled_products_url_json_dict:
                        new_crawled_products_url_json=new_crawled_products_url_json_dict[url]
                        if not is_nan(new_crawled_products_url_json[field]):
                            product_dataset_json[field]=new_crawled_products_url_json[field]
                    
                output_list.append(product_dataset_json)
                    
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(output_list, fp, indent=4)                  
            

    
def merge_review_to_get_v4():
    new_crawled_img_url_json_path = Path(
        f'bestbuy/data/to_be_combined/bestbuy_products_0.5_w_img_url_20000_review_img_url.json'
    )        
    product_dataset_path = Path('bestbuy/data/to_be_combined/bestbuy_data_with_ids_and_specs_40000_review_img_url.json')
    output_products_path = Path(
        f'bestbuy/data/to_be_combined/bestbuy_products_40000_3.5_review.json'
    )    
    output_list=[]
    field="reviews"
    with open(new_crawled_img_url_json_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        new_crawled_products_url_json_dict=json_to_dict(new_crawled_products_url_json_array)
        
        with open(product_dataset_path, 'r', encoding='utf-8') as fp:
            product_dataset_json_array = json.load(fp)
            for product_dataset_json   in tqdm(product_dataset_json_array):
                if field not in product_dataset_json  or is_nan(product_dataset_json[field]) :
                    url=product_dataset_json["url"]
                    if url in new_crawled_products_url_json_dict:
                        new_crawled_products_url_json=new_crawled_products_url_json_dict[url]
                        if not is_nan(new_crawled_products_url_json[field]):
                            product_dataset_json[field]=new_crawled_products_url_json[field]
                    
                output_list.append(product_dataset_json)
                    
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(output_list, fp, indent=4)     


def merge_image_path_from_multi_server_to_get_v3() :
    product_dataset_path = Path('bestbuy/data/bestbuy_products_40000_2.5_desc_img_url.json')
    output_products_path = Path(
        f'bestbuy/data/bestbuy_products_40000_3_desc_img_url.json'
    ) 
    with open(product_dataset_path, 'r', encoding='utf-8') as fp:
        product_dataset_json_array = json.load(fp)
        product_dataset_json_object_dict=json_to_dict(product_dataset_json_array)
        
    dir_path="bestbuy/data/img_path"
    spec_file_list=os.listdir(dir_path)
    for path in spec_file_list:
        spec_file=os.path.join(dir_path, path)
        new_crawled_img_url_json_path = Path(
            spec_file
            ) 
        product_dataset_json_object_dict=update_dict_img_path(new_crawled_img_url_json_path,product_dataset_json_object_dict)
              
    output_list=list(product_dataset_json_object_dict.values())
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(output_list, fp, indent=4)      

def gen_overlap_img_list(overlap_products_path):
    img_list=[]
    with open(overlap_products_path, 'r', encoding='utf-8') as fp:
        overlap_product_dataset_json_array = json.load(fp)
        for overlap_product_dataset_json in overlap_product_dataset_json_array:
            if "thumbnail_paths" in overlap_product_dataset_json:
                thumbnail_paths=overlap_product_dataset_json["thumbnail_paths"]
                for img_path in thumbnail_paths:

                    img_list.append(img_path.split("/")[1])
    return img_list 
import shutil
def merge_image():
    # overlap_products_path = Path(
    #     f'bestbuy/data/intermediate/bestbuy_products_40000_3_desc_img_url_img_path_from_16000_to_22425.json'
    # ) 
    # overlap_img_list=gen_overlap_img_list(overlap_products_path)
    current_image_dir="bestbuy/data/product_images"
    current_image_list=os.listdir(current_image_dir)
    new_dir_path="bestbuy/data/img_path/0/bestbuy/data/images"
    new_image_list=os.listdir(new_dir_path)
    for new_image_path in tqdm(new_image_list):
        # if new_image_path not in current_image_list:

        new_image_real_path=os.path.join(new_dir_path, new_image_path)
        shutil.move(new_image_real_path,os.path.join(current_image_dir, new_image_path))

def merge_review_image():
    # overlap_products_path = Path(
    #     f'bestbuy/data/intermediate/bestbuy_products_40000_3_desc_img_url_img_path_from_16000_to_22425.json'
    # ) 
    # overlap_img_list=gen_overlap_img_list(overlap_products_path)
    current_image_dir="bestbuy/data/review_images"
    
    new_dir_path="bestbuy/data/final/v1/review_image_separate"
    new_image_dir_list=os.listdir(new_dir_path)
    for new_image_dir in  new_image_dir_list :
        # if new_image_path not in current_image_list:

        new_image_real_dir=os.path.join(new_dir_path, new_image_dir,"bestbuy/data/review_images")
        new_image_list=os.listdir(new_image_real_dir)
        for new_image in tqdm(new_image_list):
            new_image_real_path=os.path.join(new_image_real_dir,new_image)
            shutil.move(new_image_real_path,os.path.join(current_image_dir, new_image))


def merge_review_image_path_from_multi_server_to_get_v3() :
    product_dataset_path = Path('bestbuy/data/bestbuy_review_2.3.2_w_image_incomplete.json')
    output_products_path = Path(
        f'bestbuy/data/bestbuy_review_2.3.4_w_image.json'
    ) 
    with open(product_dataset_path, 'r', encoding='utf-8') as fp:
        review_dataset_json_array = json.load(fp)
        
        
    dir_path="bestbuy/data/review_image_separate/json"
    spec_file_list=os.listdir(dir_path)
    for path in spec_file_list:
        spec_dir=os.path.join(dir_path, path)
        new_crawled_img_url_json_path = Path(
            spec_dir
            ) 
        review_dataset_json_array=update_review_dict_img_path(new_crawled_img_url_json_path,review_dataset_json_array)
        print(f"finish {path}")
              
     
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(review_dataset_json_array, fp, indent=4)     



def merge_all(mode):
    # new_crawled_img_url_json_path = Path(
    #     f'bestbuy/data/bestbuy_products_40000_3.1_img_path_img_url_incomplete.json'
    # )        
    # product_dataset_path = Path('bestbuy/data/bestbuy_products_40000_3_desc_img_url.json')
    # output_products_path = Path(
    #     f'bestbuy/data/bestbuy_products_40000_3.2_desc_img_url.json'
    # )  
    # new_crawled_img_url_json_path = Path(
    #     f'bestbuy/data/final/v1/bestbuy_review_2.3.12.4_fix_real_wrong_review_from_0_to_150000.json'
    # )        
    # product_dataset_path = Path('bestbuy/data/final/v1/bestbuy_review_2.3.12_remove_image_path_prefix.json')
    # output_products_path = Path(
    #     f'bestbuy/data/final/v1/bestbuy_review_2.3.13_fix_image_url_bug.json'
    # )  
    new_crawled_img_url_json_path = Path(
        f'bestbuy/data/final/v6/{mode}/retrieval/bestbuy_review_2.3.17.7.1_top10_fused_score_fine_tune_27_145.json'
    )        
    product_dataset_path = Path(f'bestbuy/data/final/v6/{mode}/disambiguation/bestbuy_review_2.3.17.11.20_update_annotated_gold_attribute.json')
    output_products_path = Path(
        f'bestbuy/data/final/v6/{mode}/disambiguation/bestbuy_review_2.3.17.11.20.1_update_annotated_gold_attribute_update_candidate.json'
    )
 
    
    # fields=[ "overview_section","product_images_fnames","product_images","Spec"]
    
    fields=["fused_candidate_list","fused_score_dict"]#,"thumbnails", "thumbnail_paths",  "Spec"
    output_list=[]
    with open(new_crawled_img_url_json_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        new_crawled_products_url_json_dict=json_to_dict(new_crawled_products_url_json_array)
        
        with open(product_dataset_path, 'r', encoding='utf-8') as fp:
            product_dataset_json_array = json.load(fp)
            for product_dataset_json   in tqdm(product_dataset_json_array):
                url=product_dataset_json["url"]
                if url in new_crawled_products_url_json_dict:
                    new_crawled_products_url_json=new_crawled_products_url_json_dict[url]
                    for field in fields:
                        if field not in product_dataset_json  or is_nan(product_dataset_json[field]) :
                            if not is_nan_or_miss(new_crawled_products_url_json, field ):
                                product_dataset_json[field]=new_crawled_products_url_json[field]
                            
                    
                output_list.append(product_dataset_json)
                    
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(output_list, fp, indent=4)     





def parser_json_to_dict(to_merge_img_url_dict_list):
    output_dict={}
    for to_merge_img_url_dict in to_merge_img_url_dict_list:
        review_id=to_merge_img_url_dict["review_id"]
        output_dict[review_id]=to_merge_img_url_dict
    return output_dict

def merge_bestbuy_review_234_w_image_result_parser_v33_to_review_with_mention():
    to_be_merge_path="bestbuy/data/final/v0/bestbuy_review_2.3.4_w_image_result_parser_v33.json"
    base_path="bestbuy/data/final/v0/bestbuy_review_2.3.7_separate_review.json"
    new_crawled_img_url_json_path = Path(
        to_be_merge_path
    )        
    product_dataset_path = Path(base_path)
    output_products_path = Path(
        f'bestbuy/data/final/v0/bestbuy_review_2.3.7_separate_review_incomplete_with_mention_result_parser_from_0_to_19020.json'
    )  
    output_list=[]
    with open(new_crawled_img_url_json_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        new_crawled_products_url_json_dict=parser_json_to_dict(new_crawled_products_url_json_array)
        
        with open(product_dataset_path, 'r', encoding='utf-8') as fp:
            product_dataset_json_array = json.load(fp)
            for product_dataset_json   in tqdm(product_dataset_json_array):
                review_id=product_dataset_json["reviews"][0]["id"]
                if review_id in new_crawled_products_url_json_dict:
                    new_crawled_products_url_json=new_crawled_products_url_json_dict[review_id]
                    mention=new_crawled_products_url_json["predicted_mention"]
                    product_dataset_json["mention"]=mention
                    # if field not in product_dataset_json  or is_nan(product_dataset_json[field]) :

                    output_list.append(product_dataset_json)
                    
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(output_list, fp, indent=4)     
        
        
        

def merge_mentions() :
     
    output_products_path = Path(
        f'bestbuy/data/final/v1/bestbuy_review_2.3.8_w_mention.json'
    ) 
    product_json_dict={}
    output_list=[]
    dir_path="bestbuy/data/final/v0/separate"
    spec_file_list=os.listdir(dir_path)
    for path in spec_file_list:
        spec_file=os.path.join(dir_path, path)
        new_crawled_img_url_json_path = Path(
            spec_file
            ) 
        with open(new_crawled_img_url_json_path, 'r', encoding='utf-8') as fp:
            product_dataset_json_array = json.load(fp)
            for product_dataset_json   in tqdm(product_dataset_json_array):
                review_id=product_dataset_json["reviews"][0]["id"] 
                product_json_dict[review_id]=product_dataset_json

    for i in sorted(product_json_dict.keys()):
        output_list.append(product_json_dict[i])
    # output_list=list(product_json_dict.values())
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(output_list, fp, indent=4)      
        
def gen_review_product_list(spec_file):
    new_crawled_img_url_json_path = Path(
        spec_file
        ) 
    with open(new_crawled_img_url_json_path, 'r', encoding='utf-8') as fp:
        product_dataset_json_array = json.load(fp)
    return product_dataset_json_array
        
def merge_train_dev_test():
    output_products_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v4/bestbuy_review_2.3.16.27_train_dev_test.json"
    train_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v4/train/bestbuy_review_2.3.16.27.1_similar_split.json"
    review_product_list_train=gen_review_product_list(train_path)
    test_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v4/test/bestbuy_review_2.3.16.27.1_similar_split.json"
    review_product_list_test=gen_review_product_list(test_path)
    val_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v4/val/bestbuy_review_2.3.16.27.1_similar_split.json"
    review_product_list_val=gen_review_product_list(val_path)
    review_product_list_train.extend(review_product_list_test)
    review_product_list_train.extend(review_product_list_val)
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(review_product_list_train, fp, indent=4)
                    

def merge_all_for_review(mode ):
     
      
    # new_crawled_img_url_json_path = Path(
    #     f'bestbuy/data/final/v6/{mode}/retrieval/bestbuy_review_2.3.17.7.1_top10_fused_score_fine_tune_27_145.json'
    # )        
    # review_dataset_path = Path(f'bestbuy/data/final/v6/{mode}/disambiguation/bestbuy_review_2.3.17.11.19.16_update_gold_attribute.json')
    # output_products_path = Path(
    #     f'bestbuy/data/final/v6/{mode}/disambiguation/bestbuy_review_2.3.17.11.19.16.1_update_candidate.json'
    # )
 
     
    
    new_crawled_img_url_json_path = Path(
        f'bestbuy/data/final/v6/{mode}/temp/bestbuy_review_2.3.17.11.21_similar_vicuna_from_0_to_100000.json'
    )        
    review_dataset_path = Path(f'/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/{mode}/temp/bestbuy_review_2.3.17.11.20.1.1_add_gpt2_4.json')
    output_products_path = Path(
        f'/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/{mode}/disambiguation/bestbuy_review_2.3.17.11.21_add_gpt2_vicuna.json'
    )  
    
    fields=["predicted_attribute_vicuna","predicted_attribute_context_vicuna","is_attribute_correct_vicuna"]#,
    in_num=0
    output_list=[]
    with open(new_crawled_img_url_json_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        new_crawled_products_url_json_dict=new_review_json_to_product_dict(new_crawled_products_url_json_array)
        
        with open(review_dataset_path, 'r', encoding='utf-8') as fp:
            product_dataset_json_array = json.load(fp)
            for review   in tqdm(product_dataset_json_array):
                # review=product_dataset_json["reviews"][0]
                # review=review_json["reviews"]
                review_id=review["review_id"]
                 
                if review_id in new_crawled_products_url_json_dict:
                    in_num+=1
                    new_crawled_products_url_json=new_crawled_products_url_json_dict[review_id]
                    for field in fields:
                        # if field not in product_dataset_json  or is_nan(product_dataset_json[field]) :
                        if field in new_crawled_products_url_json:
                            review[field]=new_crawled_products_url_json[field]
                        elif field=="fused_candidate_list":
                            review[field]=new_crawled_products_url_json[field]
                # review_json["reviews"]=review
                output_list.append(review) 
                    
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(output_list, fp, indent=4)                      
    print(in_num)
    
def merge_attribute():
     
     
    output_products_path = Path(
        f'/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v5/test/retrieval/bestbuy_review_2.3.16.29.6_fused_score_20_54_ocr_all.json'
    ) 
    product_json_dict={}
    output_list=[]
    dir_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v5/test/retrieval/attribute"
    spec_file_list=os.listdir(dir_path)
    for path in spec_file_list:
        spec_file=os.path.join(dir_path, path)
        new_crawled_img_url_json_path = Path(
            spec_file
            ) 
        with open(new_crawled_img_url_json_path, 'r', encoding='utf-8') as fp:
            product_dataset_json_array = json.load(fp)
            for product_dataset_json   in tqdm(product_dataset_json_array):
                review_id=product_dataset_json["reviews"][0]["id"] 
                product_json_dict[review_id]=product_dataset_json

    for i in sorted(product_json_dict.keys()):
        output_list.append(product_json_dict[i])
    # output_list=list(product_json_dict.values())
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(output_list, fp, indent=4)                          
# merge_all()        


def merge_gold_attribute(mode,setting="end_to_end" ):
    print(setting)
    # new_crawled_img_url_json_path = Path(
    #     f'bestbuy/data/final/v1/bestbuy_review_2.3.14.2_fix_review_to_check_image_url_from_0_to_150000.json'
    # )        
    # review_dataset_path = Path('bestbuy/data/final/v1/bestbuy_review_2.3.15.1_download_image_path_nan_from_0_to_150000.json')
    # output_products_path = Path(
    #     f'bestbuy/data/final/v1/bestbuy_review_2.3.16_fix_all_duplicate_image_url.json'
    # )  
    # new_crawled_img_url_json_path = Path(
    #     f'bestbuy/data/final/v3/bestbuy_review_train_val_2.3.16.22.2_scraper_fix_from_0_to_150000.json'
    # )        
    # review_dataset_path = Path('bestbuy/data/final/v3/bestbuy_review_2.3.16.22_train_val_remove_duplicate_review.json')
    # output_products_path = Path(
    #     f'bestbuy/data/final/v3/bestbuy_review_train_val_2.3.16.23_merge_scraper_fix.json'
    # )  
    new_crawled_img_url_json_path = Path(
            f'/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/annotation.json'
        )    
    if setting =="end_to_end":
            
        review_dataset_path = Path(f'/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/{mode}/disambiguation/bestbuy_review_2.3.17.11.19.16_update_gold_attribute.json')
        output_products_path = Path(
            f'/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/{mode}/disambiguation/bestbuy_review_2.3.17.11.20_update_annotated_gold_attribute.json'
        )  
    elif setting =="subset":
        review_dataset_path = Path(f'/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/disambiguation/correct_retrieval_subset/bestbuy_review_2.3.17.11.19.20.2_correct_retrieval_subset_10_update_gold_attribute.json')
        output_products_path = Path(
            f'/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/disambiguation/correct_retrieval_subset/bestbuy_review_2.3.17.11.19.20.3_correct_retrieval_subset_10_update_gold_attribute.json'
        )
    elif setting =="end_to_end_post_process":
        review_dataset_path = Path(f'/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/disambiguation/postprocess/bestbuy_review_2.3.17.11.21_fuse_disambiguation_score_test1008.json')
        output_products_path = Path(
            f'/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/disambiguation/postprocess/bestbuy_review_2.3.17.11.21_fuse_disambiguation_score_test1008_update_annotation.json'
        )
    elif setting =="subset_post_process":
        review_dataset_path = Path(f'/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/disambiguation/postprocess/bestbuy_review_2.3.17.11.21_fuse_disambiguation_score_test1008_subset.json')
        output_products_path = Path(
            f'/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/test/disambiguation/postprocess/bestbuy_review_2.3.17.11.21_fuse_disambiguation_score_test1008_subset_update_annotation.json'
        )
    
    fields=[ "gold_attribute_for_predicted_category" ]#,"thumbnails", "thumbnail_paths",  "Spec"
    
    output_list=[]
    with open(new_crawled_img_url_json_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_dict = json.load(fp)
        # new_crawled_products_url_json_dict=new_review_json_to_product_dict(new_crawled_products_url_json_array)
        
        with open(review_dataset_path, 'r', encoding='utf-8') as fp:
            product_dataset_json_array = json.load(fp)
            for review   in tqdm(product_dataset_json_array):
                
                review_id=review["review_id"]
                # if review_id==89208:
                #     print()
                if str(review_id) in new_crawled_products_url_json_dict:
                    new_crawled_products_url_json=new_crawled_products_url_json_dict[str(review_id)][0]
                    field=new_crawled_products_url_json[ "gold_attribute_for_predicted_category" ] 
                    cleaned_field={}
                    for key,value in field.items():
                        if key not in [ "Product Title","Model Version"]:
                            value=value.lower() 
                        cleaned_field[key]=value
                    review[ "gold_attribute_for_predicted_category" ]=cleaned_field
                     
                 
                    output_list.append(review) 
                else:
                    print(f"no {review_id}")
                    output_list.append(review) 
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(output_list, fp, indent=4)  



if __name__ == "__main__":  
    # merge_attribute()
    merge_all_for_review("test")
    
    # merge_gold_attribute("test","subset_post_process")
    # merge_all()
    # merge_train_dev_test()
        
# merge_bestbuy_review_234_w_image_result_parser_v33_to_review_with_mention()
# merge_review_image()            
# merge_spec_to_get_v2()
# merge_spec_to_get_v2()