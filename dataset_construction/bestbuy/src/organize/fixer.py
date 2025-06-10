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

from bestbuy.src.organize.merge_attribute import json_to_dict, json_to_product_id_dict, review_json_to_product_dict
import copy

from bestbuy.src.organize.sampler import filter_similar_product_id_list, gen_brand_from_spec, get_brand_from_review
from bestbuy.src.organize.score import gen_reivew_gold_product_dict   
def query_target_product_info( ):
    product_corpus_path = Path(
        f'bestbuy/data/bestbuy_products_40000_3_desc_img_url.json'
    ) 
    with open(product_corpus_path, 'r', encoding='utf-8') as fp:
        product_dataset_json_array = json.load(fp)
        product_dataset_json_object_dict={}
        for to_merge_img_url_dict in product_dataset_json_array:
            product_dataset_json_object_dict[to_merge_img_url_dict["product_name"]]=to_merge_img_url_dict
 
    out_list=[]
    target_product_example_path = Path(
        f'bestbuy/data/example/samples_100_directories.json'
    )
    with open(target_product_example_path, 'r', encoding='utf-8') as fp:
        review_products_json_list = json.load(fp)
        for review_product_json in review_products_json_list:
            target_product_name=review_product_json["product_name"]
            if target_product_name in product_dataset_json_object_dict:
                product_dataset_json_object=product_dataset_json_object_dict[target_product_name]
                review_product_json["url"]=product_dataset_json_object["url"]
                review_product_json["product_category"]=product_dataset_json_object["product_category"]
                review_product_json["overview_section"]=product_dataset_json_object["overview_section"]
                review_product_json["Spec"]=product_dataset_json_object["Spec"]
            else:
                review_product_json["url"]=""
            out_list.append(review_product_json)
    with open("bestbuy/data/example/samples_100_directories_2.json", 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4)    

def generate_one_example_to_json(product_file,review_file):
    incomplete_products_path = Path(
        product_file
    )
    with open(incomplete_products_path, 'r', encoding='utf-8') as fp:
        product_dict_list = json.load(fp)
        with open("bestbuy/data/example/product.json", 'w', encoding='utf-8') as fp:
            json.dump(product_dict_list[0], fp, indent=4) 

    review_path = Path(
        review_file
    )
    with open(review_path, 'r', encoding='utf-8') as fp:
        review_dict_list = json.load(fp)
        with open("bestbuy/data/example/review.json", 'w', encoding='utf-8') as fp:
            json.dump(review_dict_list[0], fp, indent=4)    


def query_similar_products_info( ):
    product_corpus_path = Path(
        f'bestbuy/data/bestbuy_products_40000_3.2_desc_img_url.json'
    ) 
    with open(product_corpus_path, 'r', encoding='utf-8') as fp:
        product_dataset_json_array = json.load(fp)
        product_dataset_json_object_dict={}
        for to_merge_img_url_dict in product_dataset_json_array:
            product_dataset_json_object_dict[to_merge_img_url_dict["url"]]=to_merge_img_url_dict
 
    out_list=[]
    target_product_example_path = Path(
        f'bestbuy/data/example/samples_100_similar_prods_with_paths.json'
    )
    with open(target_product_example_path, 'r', encoding='utf-8') as fp:
        review_products_json_list = json.load(fp)
        for review_product_json in review_products_json_list:
            target_product_name=review_product_json["url"]
            if target_product_name in product_dataset_json_object_dict:
                product_dataset_json_object=product_dataset_json_object_dict[target_product_name]
                review_product_json["url"]=product_dataset_json_object["url"]
                review_product_json["product_name"]=product_dataset_json_object["product_name"]
                review_product_json["product_category"]=product_dataset_json_object["product_category"]
                review_product_json["overview_section"]=product_dataset_json_object["overview_section"]
                review_product_json["Spec"]=product_dataset_json_object["Spec"]
            else:
                review_product_json["Spec"]=[]
                review_product_json["product_name"]=""
                review_product_json["product_category"]=""
            review_product_json["reviews"]=[]

             
            
            out_list.append(review_product_json)
    with open("bestbuy/data/example/samples_100_similar_prods_with_paths_3.json", 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4)    




        


def fix_spec_format():
    out_list=[]
    product_dataset_path = Path('bestbuy/data/final/v1/bestbuy_products_40000_3.4.4_fix_spec.json')
    output_products_path = Path(
        f'bestbuy/data/final/v1/bestbuy_products_40000_3.4.5_fix_spec_format.json'
    ) 
    with open(product_dataset_path, 'r', encoding='utf-8') as fp:
        product_dataset_json_array = json.load(fp)
        for dataset_json   in tqdm(product_dataset_json_array):
            if not is_nan(dataset_json["Spec"]) :
                spec_top_list=dataset_json["Spec"]
                if "subsection" not in spec_top_list[0]:
                     
                    dataset_json["Spec"]=[{"subsection": "All", "text":spec_top_list }]      
            out_list.append(dataset_json)
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4)    
            
def choose_wrong_review_json():
     
 
    path="bestbuy/data/final/v1/bestbuy_review_2.3.12_remove_image_path_prefix.json"
    incomplete_products_path = Path(
        path
    )
    output_products_path = Path(
        f'bestbuy/data/final/v1/bestbuy_review_2.3.12.1_wrong_review.json'
    ) 
    wrong_num=0
    wrong_distribution={}
    wrong_review_id_list=[]
    repeated_header_bigger_than_2_num=0
    out_list=[]
    product_review_header_review_id_dict={}
    with open(incomplete_products_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        review_id_review_dict=review_json_to_product_dict(incomplete_dict_list)
        for idx,product_json in enumerate(incomplete_dict_list):
            review=product_json["reviews"][0]
            product_id=product_json["id"]
            review_id=review["id"]
            header=review["header"]
            if product_id in product_review_header_review_id_dict:
                review_header_review_id_dict=product_review_header_review_id_dict[product_id]
                if header in review_header_review_id_dict:
                    review_header_review_id_dict[header].append(review_id)
                else:
                    review_header_review_id_dict[header]=[review_id]
                    
            else:
                review_header_review_id_dict={}
                review_header_review_id_dict[header]=[review_id]
                product_review_header_review_id_dict[product_id]=review_header_review_id_dict
                 
    for product_id,review_header_review_id_dict in product_review_header_review_id_dict.items():
        for header,review_id_list in review_header_review_id_dict.items():
            if len(review_id_list)>1:
                for review_id in review_id_list:
                    out_list.append(review_id_review_dict[review_id])
                wrong_review_id_list.extend(review_id_list)
                wrong_num_for_cur_header=len(review_id_list)
                wrong_num+=wrong_num_for_cur_header
                if wrong_num_for_cur_header in wrong_distribution:
                    wrong_distribution[wrong_num_for_cur_header]+=1
                else:
                    wrong_distribution[wrong_num_for_cur_header]=1
    print(f"review wrong num: { wrong_num }, {wrong_distribution} ")
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4)  
# def fix_review_issue():

# generate_one_example_to_json("bestbuy/data/bestbuy_products_40000_3_desc_img_url.json","bestbuy/data/bestbuy_review_2.json")
# query_similar_products_info()
# fix_spec_format()

def remove_maxHeight_postfix(image_url_list):
    new_image_url_list=[]
    for image_url in image_url_list:
        image_url=image_url.split(";maxHeight=")[0]
        new_image_url_list.append(image_url)
        
    return new_image_url_list

def is_list_same(list1,list2):
    set1=set(list1)
    set2=set(list2)
    if len(set1.difference(set2))>0 or len(set2.difference(set1))>0:
        return False  
    else:
        return True   
    
def choose_real_wrong_review_json():
    # path="bestbuy/data/final/v1/bestbuy_review_2.3.16.5_download_image_from_0_to_150000.json"
    # out_path=f'bestbuy/data/final/v2/bestbuy_review_2.3.16.6_change_image_path_nan.json'
    # path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/example/bestbuy_100_human_performance.json"
    # out_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/example/bestbuy_100_human_performance_mismatch_image.json"
    path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v3/bestbuy_review_train_val_2.3.16.24_remove_empty.json"
    out_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v3/bestbuy_review_train_val_2.3.16.25_wrong_image.json"
    incomplete_products_path = Path(
        path
    )
    output_products_path = Path(
        out_path
    )
    is_remove=True 
    out_list=[]
    filter_num=0
    product_review_header_review_id_dict={}
    with open(incomplete_products_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        
        for idx,product_json in enumerate(incomplete_dict_list):
            review=product_json["reviews"][0]
            review_id=review["id"]
            # if review["id"]==511:
            #     print("") 
            
            new_image_path_list=gen_new_iamge_path_list(review["image_url"],review_id)
            if "wrong_image_url" in review:
                if not is_list_same(remove_maxHeight_postfix(review["image_url"]),remove_maxHeight_postfix(review["wrong_image_url"])):

                    if not is_list_same(new_image_path_list, review["image_path"] ):
                        # review["image_path"]=None 
                        if is_remove:
                            filter_num+=1
                            continue 
                        review["is_image_wrong"]="y"
                        product_json["reviews"][0]=review
            out_list.append(product_json)
 
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4)    
    print(filter_num)



def gen_new_iamge_path_list( image_url_list,review_id ):
    image_path_list=[]
    for image_url in image_url_list:
        image_path=image_url.split('.jpg')[0].split('/')[-1] + '.jpg'
        image_path= str(review_id).zfill(7)+"-"+image_path
        image_path_list.append(image_path )
    
    return image_path_list


def gen_duplicate_review_body():
    products_path = Path(
        "bestbuy/data/final/v1/bestbuy_products_40000_3.4.7_remove_image_path_prefix.json"
    )
    review_path="bestbuy/data/final/v1/bestbuy_review_2.3.13.3_remove_duplicate_review_same_product.json"
    incomplete_review_path = Path(
        review_path
    )
    output_products_path = Path(
        "bestbuy/data/final/v1/bestbuy_review_2.3.13.4_duplicate_review.json"
    )
    product_id_list=[]
    # with open(products_path, 'r', encoding='utf-8') as fp:
    #     product_dict_list = json.load(fp)
    #     for i,product_json in  enumerate(product_dict_list)  :
    #         product_id_list.append(product_json["id"])
            
    out_list=[]
    filter_num=0
    review_body_product_json_dict={}
    # tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    with open(incomplete_review_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        for i,cur_product_json in tqdm(enumerate(incomplete_dict_list) ):
            review=cur_product_json["reviews"][0]
            review_body=review["body"]
            # if review["id"]==2186:
            #     print("")
            is_found=False 
            if  review_body   in review_body_product_json_dict:
                # product_json_list=review_body_product_json_dict[review_body]
                
                review_body_product_json_dict[review_body].append(cur_product_json)
            else:
                review_body_product_json_dict[review_body]=[cur_product_json]
            # if not is_found:        
            #     out_list.append(cur_product_json)
    wrong_distribution={}
    wrong_num=0
    wrong_review_id_list=[]
    print(f"review num {len(review_body_product_json_dict)}")
    for review_body,product_json_list in review_body_product_json_dict.items():
        if len(product_json_list)>1:
            for product_json in product_json_list:
                out_list.append(product_json)
            wrong_num_for_cur_header=len(product_json_list)
            if wrong_num_for_cur_header in wrong_distribution:
                wrong_distribution[wrong_num_for_cur_header]+=1
            else:
                wrong_distribution[wrong_num_for_cur_header]=1
            wrong_num+=wrong_num_for_cur_header-1
        
    print(f"review wrong num: { wrong_num }, {wrong_distribution} ")
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4)   

def choose_wrong_review_json():
     
 
    path="bestbuy/data/final/v1/bestbuy_review_2.3.12_remove_image_path_prefix.json"
    incomplete_products_path = Path(
        path
    )
    output_products_path = Path(
        f'bestbuy/data/final/v1/bestbuy_review_2.3.12.1_wrong_review.json'
    ) 
    wrong_num=0
    wrong_distribution={}
    wrong_review_id_list=[]
    repeated_header_bigger_than_2_num=0
    out_list=[]
    product_review_header_review_id_dict={}
    with open(incomplete_products_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        review_id_review_dict=review_json_to_product_dict(incomplete_dict_list)
        for idx,product_json in enumerate(incomplete_dict_list):
            review=product_json["reviews"][0]
            product_id=product_json["id"]
            review_id=review["id"]
            header=review["header"]
            if product_id in product_review_header_review_id_dict:
                review_header_review_id_dict=product_review_header_review_id_dict[product_id]
                if header in review_header_review_id_dict:
                    review_header_review_id_dict[header].append(review_id)
                else:
                    review_header_review_id_dict[header]=[review_id]
                    
            else:
                review_header_review_id_dict={}
                review_header_review_id_dict[header]=[review_id]
                product_review_header_review_id_dict[product_id]=review_header_review_id_dict
                 
    for product_id,review_header_review_id_dict in product_review_header_review_id_dict.items():
        for header,review_id_list in review_header_review_id_dict.items():
            if len(review_id_list)>1:
                for review_id in review_id_list:
                    out_list.append(review_id_review_dict[review_id])
                wrong_review_id_list.extend(review_id_list)
                wrong_num_for_cur_header=len(review_id_list)
                wrong_num+=wrong_num_for_cur_header
                if wrong_num_for_cur_header in wrong_distribution:
                    wrong_distribution[wrong_num_for_cur_header]+=1
                else:
                    wrong_distribution[wrong_num_for_cur_header]=1
    print(f"review wrong num: { wrong_num }, {wrong_distribution} ")
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4)  

def gen_review_not_check_image_url():
    # path="bestbuy/data/final/v1/bestbuy_review_2.3.14_clean_duplicate_review.json"
    # path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v2_course/test/bestbuy_review_2.3.16.20.1_split.json"
    # path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/example/bestbuy_100_human_performance_added.json"
    path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v3/bestbuy_review_2.3.16.22_train_val_remove_duplicate_review.json"
    incomplete_products_path = Path(
        path
    )
    output_products_path = Path(
        # f'bestbuy/data/final/v1/bestbuy_review_2.3.14.1_review_to_check_image_url.json'
        # f'/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v2_course/test/bestbuy_review_2.3.16.20.2_no_search.json'
        # f'/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/example/bestbuy_100_human_performance_added_wrong_review.json'
        f'/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v3/bestbuy_review_train_val_2.3.16.22.1_to_scrawl_fix.json'
    )
    out_list=[]
    product_review_header_review_id_dict={}
    with open(incomplete_products_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        
        for idx,product_json in enumerate(incomplete_dict_list):
            review=product_json["reviews"][0]
            if "has_check_image_url" not in review and "corrected_body" not in review :
                out_list.append(product_json)
 
                
 
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4)  

def add_id_into_image_path():
    path="bestbuy/data/final/v1/bestbuy_review_2.3.16.4_remove_image_url_null.json"
    incomplete_products_path = Path(
        path
    )
    # output_products_path = Path(
    #     f'bestbuy/data/final/v1/bestbuy_review_2.3.16.5_change_image_path_nan.json'
    # )
    out_list=[]
    product_review_header_review_id_dict={}
    target_image_dir="/home/menglong/workspace/code/referred/product_scraper/dataset_construction/bestbuy/data/final/v2/review_images"
    source_image_dir="/home/menglong/workspace/code/referred/product_scraper/dataset_construction/bestbuy/data/final/v1/review_images"
    with open(incomplete_products_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        
        for idx,product_json in enumerate(incomplete_dict_list):
            review=product_json["reviews"][0]
            # if review["id"]==511:
            #     print("") 
            
            # new_image_path_list=gen_new_iamge_path_list(review["image_url"])
            image_path_list=review["image_path"]
            new_image_path_list=[]
            review_id=review["id"]
            if image_path_list is not None:
                for image_path in image_path_list:
                    new_image_path=str(review_id).zfill(7)+"-"+image_path
                    if os.path.exists(os.path.join(source_image_dir,image_path)):
                        os.rename(os.path.join(source_image_dir,image_path),os.path.join(target_image_dir,new_image_path))
                         
 


def only_keep_right_images():
    path="bestbuy/data/final/v2/bestbuy_review_2.3.16.8_add_review_id_into_image_path.json"
    incomplete_products_path = Path(
        path
    )
    # output_products_path = Path(
    #     f'bestbuy/data/final/v1/bestbuy_review_2.3.16.5_change_image_path_nan.json'
    # )
    out_list=[]
    product_review_header_review_id_dict={}
    target_image_dir="/home/menglong/workspace/code/referred/product_scraper/dataset_construction/bestbuy/data/final/v2/review_images"
    source_image_dir="/home/menglong/workspace/code/referred/product_scraper/dataset_construction/bestbuy/data/final/v2/review_images_wrong"
    with open(incomplete_products_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        
        for idx,product_json in tqdm(enumerate(incomplete_dict_list)):
            review=product_json["reviews"][0]
            # if review["id"]==511:
            #     print("") 
            
            # new_image_path_list=gen_new_iamge_path_list(review["image_url"])
            image_path_list=review["image_path"]
            
            review_id=review["id"]
            if image_path_list is not None:
                for image_path in image_path_list:
                    # new_image_path=str(review_id).zfill(7)+"-"+image_path
                    if os.path.exists(os.path.join(source_image_dir,image_path)):
                        os.rename(os.path.join(source_image_dir,image_path),os.path.join(target_image_dir,image_path))
                    else:
                        print(image_path)


def only_keep_right_product_images():
    path="bestbuy/data/final/v2/bestbuy_products_40000_3.4.7_remove_image_path_prefix.json"
    incomplete_products_path = Path(
        path
    )
    # output_products_path = Path(
    #     f'bestbuy/data/final/v1/bestbuy_review_2.3.16.5_change_image_path_nan.json'
    # )
    out_list=[]
    product_review_header_review_id_dict={}
    target_image_dir="bestbuy/data/final/v2/product_images"
    source_image_dir="bestbuy/data/final/v1/product_images"
    with open(incomplete_products_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        
        for idx,product_json in tqdm(enumerate(incomplete_dict_list)):
            
            # if review["id"]==511:
            #     print("") 
            
            # new_image_path_list=gen_new_iamge_path_list(review["image_url"])
            image_path_list=product_json["image_path"]
             
            if image_path_list is not None:
                for image_path in image_path_list:
                    # new_image_path=str(review_id).zfill(7)+"-"+image_path
                    if os.path.exists(os.path.join(source_image_dir,image_path)):
                        os.rename(os.path.join(source_image_dir,image_path),os.path.join(target_image_dir,image_path))
                    else:
                        print(image_path)



def update_image_path():
    path="bestbuy/data/final/v2/bestbuy_review_2.3.16.7_download_image_from_0_to_150000.json"
    incomplete_products_path = Path(
        path
    )
    output_products_path = Path(
        f'bestbuy/data/final/v2/bestbuy_review_2.3.16.8_add_review_id_into_image_path.json'
    )
    out_list=[]
    product_review_header_review_id_dict={}
    target_image_dir="bestbuy/data/final/v2/review_images"
    source_image_dir="bestbuy/data/final/v2/review_images_wrong"
    with open(incomplete_products_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        
        for idx,product_json in enumerate(incomplete_dict_list):
            review=product_json["reviews"][0]
            # if review["id"]==511:
            #     print("") 
            
            # new_image_path_list=gen_new_iamge_path_list(review["image_url"])
            image_path_list=review["image_path"]
            new_image_path_list=[]
            review_id=review["id"]
            if image_path_list is not None:
                for image_path in image_path_list:
                    new_image_path=str(review_id).zfill(7)+"-"+image_path
                    if os.path.exists(os.path.join(source_image_dir,new_image_path)):
                        # os.rename(os.path.join(source_image_dir,new_image_path),os.path.join(target_image_dir,new_image_path))
                        new_image_path_list.append(new_image_path)
                    elif image_path[0:8]==str(review_id).zfill(7)+"-":
                        new_image_path_list.append(image_path)
                    else:
                        print(f"wrong {image_path}")
                review["image_path"]=new_image_path_list     
            product_json["reviews"]=[review]
            out_list.append(product_json)
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4)  


def extract_example_from_report( end_idx,report_path):
    
    product_json_path = Path(
        f'bestbuy/data/final/v2_course/bestbuy_products_40000_3.4.10_remove_wrong_image.json'
    )        
    review_dataset_path = Path('bestbuy/data/final/v2_course/bestbuy_review_2.3.16.15_filter_low_information.json')
    output_products_path = Path(
        f'bestbuy/data/example/bestbuy_100_human_performance_550-750.json'
    )  
    number=0
    
    # fields=["reviews","overview_section","product_images_fnames","product_images","Spec"]
    
    # fields=["thumbnails", "thumbnail_paths","reviews", "Spec"]
    output_list=[]
    with open(product_json_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        product_json_dict=json_to_product_id_dict(new_crawled_products_url_json_array)
        
    with open(review_dataset_path, 'r', encoding='utf-8') as fp:
        review_dataset_json_array = json.load(fp) 
        review_product_json_dict=review_json_to_product_dict(review_dataset_json_array)
    
    report_df=pd.read_excel(report_path, index_col=0,skiprows=0)
    for idx,(_, row) in enumerate(report_df.iterrows()):
        if idx >end_idx:
            break  
        review_id=row["review_id"] 
        if review_id in review_product_json_dict:
            review_product_json=review_product_json_dict[review_id]
            product_id=review_product_json["id"]
            product_json=product_json_dict[product_id]
            human_predict_product_id=row["human_predict"]
            if human_predict_product_id=="x":
                is_low_quality_review=True 
            else:
                is_low_quality_review=False 

            new_product_json=copy.deepcopy(product_json)
            new_product_json["reviews"]=review_product_json["reviews"]
            new_product_json["reviews"][0]["is_low_quality_review"]=is_low_quality_review
            new_product_json["text_similarity_score"]=review_product_json["text_similarity_score"]
            new_product_json["image_similarity_score_list"]=review_product_json["image_similarity_score_list"]
            new_product_json["product_title_similarity_score"]=review_product_json["product_title_similarity_score"]
            new_product_json["predicted_is_low_quality_review"]=review_product_json["predicted_is_low_quality_review"]

            output_list.append(new_product_json)
                                  
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(output_list, fp, indent=4)        
    print(len(output_list))
    
    
def fix_for_low_information():
    # path="bestbuy/data/example/bestbuy_100_human_performance_550.json"
    path="bestbuy/data/final/v2_course/bestbuy_review_2.3.16.13_filter_low_information.json"
    incomplete_products_path = Path(
        path
    )
    output_products_path = Path(
        # f'bestbuy/data/example/bestbuy_100_human_performance_550_clean.json'
        f'bestbuy/data/final/v2_course/bestbuy_review_2.3.16.14_round2.json'
    )
    out_list=[]
 
   
    with open(incomplete_products_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        
        for idx,product_json in enumerate(incomplete_dict_list):
            review=product_json["reviews"][0]
            product_title_similarity_score=product_json["product_title_similarity_score"]
            text_similarity_score=product_json["text_similarity_score"]
            product_json["product_title_similarity_score"]=round(product_title_similarity_score,2)
            product_json["text_similarity_score"]=round(text_similarity_score,2)
            attribute_dict=review["attribute"]
            new_attribute_dict={}
            for key,value in attribute_dict.items():
                if value.lower()=="other" or value.lower()=="none":
                    continue 
                # elif   len(value)==1:
                #     continue 
                else:
                    new_attribute_dict[key]=value 
            review["attribute"]=new_attribute_dict
            product_json["reviews"][0]=review
            out_list.append(product_json)
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4)  



def fix_reshuffled_target_product_id_position_23_for_error_review():
    error_list=["21237",
"22976",
"23310",
"24513",
"25032",
"25034",
"52488",
"52494",
"52498",
"52507",
"52523",
"75191",
"75196",
"75198",
"75201",
"83247",
"83258",
"83262",
"83274",
"83275"]
    testset_example_file_path=Path("/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/example/bestbuy_100_human_performance.json")
    out_list=[]
    update_num=0
    with open(testset_example_file_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        for idx,review_dataset_json in tqdm(enumerate(new_crawled_products_url_json_array)):
            if not is_nan_or_miss(review_dataset_json,"reviews"):
                review=review_dataset_json["reviews"][0]
                review_id=review["id"]
                if str(review_id) in error_list:
                    update_num+=1
                    review_dataset_json["reviews"][0]["reshuffled_target_product_id_position"]=23
                 
            out_list.append(review_dataset_json)
        print(update_num)

    with open(testset_example_file_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4) 

def record_replace_table():
    testset_example_file_path=Path("/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/example/bestbuy_100_human_performance.json")
    replace_example_path=Path("/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/example/bestbuy_100_human_performance_added.json")
    error_data_path=Path("/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v2_course/test/bestbuy_review_2.3.16.20.2.1_error_example.json")
    table_output_path=Path("bestbuy/output/replace_testset_error_example116.csv")
    with open(replace_example_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        replace_example_list =[]
        for review_dataset_json in new_crawled_products_url_json_array:
            if not is_nan_or_miss(review_dataset_json,"reviews"):
                replace_example_list.append(review_dataset_json)
    
    error_review_id_list=[]
    with open(error_data_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        for review_dataset_json in new_crawled_products_url_json_array:
            if not is_nan_or_miss(review_dataset_json,"reviews"):
                review=review_dataset_json["reviews"][0]
                review_id=review["id"]
                error_review_id_list.append(review_id) 
                
    out_list=[]
    replace_idx=0
    record_list=[]
    
    with open(testset_example_file_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        for idx,review_dataset_json in tqdm(enumerate(new_crawled_products_url_json_array)):
            if not is_nan_or_miss(review_dataset_json,"reviews"):
                review=review_dataset_json["reviews"][0]
                review_id=review["id"]
                if review_id not in error_review_id_list:
                    out_list.append(review_dataset_json)
                else:
                    replace_example=replace_example_list[replace_idx]
                    out_list.append(replace_example)
                    replace_idx+=1
                    record_list=record(replace_example,review_dataset_json,record_list,idx)
                    
    with open(testset_example_file_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4)   
             

    df = pd.DataFrame(record_list, columns =['original_review_id', 'replace_review_id','annotator_id (1-12)' ])
    df.to_csv(table_output_path,index=False)   
    
def record(replace_example ,error_review_dataset_json,record_list,example_position):
    annotator_id=example_position//330+1
   
    record_list.append([ error_review_dataset_json["reviews"][0]["id"],replace_example["reviews"][0]["id"],annotator_id])
    return record_list

def gen_brand_product_num_dict():
    example_path=Path("/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v2_course/bestbuy_products_40000_3.4.13_sensitive.json")
 
    brand_product_num_dict={}
    with open( example_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        for review_dataset_json in new_crawled_products_url_json_array:
            brand=gen_brand_from_spec(review_dataset_json)
            if brand in brand_product_num_dict:
                brand_product_num_dict[brand]+=1
            else:
                brand_product_num_dict[brand]=1
    return brand_product_num_dict

def find_no_similarity_and_not_extract_brand_attribute_from_review():
    
    
    
    example_path=Path("/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/example/bestbuy_100_human_performance.json")
    output_path=Path("/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/example/bestbuy_100_human_performance_no_similarity_product.json")
    out_list=[]
    brand_product_num_dict=gen_brand_product_num_dict()
    with open( example_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
    
        for review_dataset_json in new_crawled_products_url_json_array:
            if not is_nan_or_miss(review_dataset_json,"reviews"):
                text_similar_num=len(review_dataset_json["similar_product_id"])
 
                merged_similar_num=len(set(review_dataset_json["similar_product_id"]).union(set(review_dataset_json["product_id_with_similar_image"])))
                if text_similar_num==0 or len(review_dataset_json["product_id_with_similar_image"])==0:
                    brand=get_brand_from_review(review_dataset_json)
                    brand_in_product= gen_brand_from_spec(review_dataset_json)
                    if brand is None:
                        out_list.append(review_dataset_json)
                    elif brand !=brand_in_product:
                        out_list.append(review_dataset_json)
                    else:
                        product_in_brand_num=brand_product_num_dict[brand]
                        if product_in_brand_num-1>merged_similar_num:
                            out_list.append(review_dataset_json)
    with open(output_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4)   


def clean_removed_product_for_similar_product_list():
    product_data_path=Path("/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v2_course/bestbuy_products_40000_3.4.13_sensitive.json")
    product_id_list=[]
    with open(product_data_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        for review_dataset_json in new_crawled_products_url_json_array:
             
               
            review_id=review_dataset_json["id"]
            product_id_list.append(review_id) 
    all_product_id_set=set(product_id_list)
    example_path=Path("/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/example/bestbuy_100_human_performance_w_fix_image_similar_product.json")
    output_path=Path("/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/example/bestbuy_100_human_performance_w_v3_filter_similar_product.json")
    out_list=[]
    
    with open( example_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
    
        for review_dataset_json in new_crawled_products_url_json_array:
            review_dataset_json=clean_removed_product_for_similar_product_list_for_one_example(review_dataset_json,all_product_id_set)
            out_list.append(review_dataset_json)
    with open(output_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4)  


def clean_removed_product_for_similar_product_list_for_one_example(product_json,all_product_id_set):
    
 
    similar_product_id_set=set(product_json["similar_product_id"])
    new_similar_product_id_set=similar_product_id_set.intersection(all_product_id_set)
    product_json["similar_product_id"]=list(new_similar_product_id_set)
    
    similar_product_id_image_set=set(product_json["product_id_with_similar_image"])
    new_similar_product_image_id_set=similar_product_id_image_set.intersection(all_product_id_set)
    product_json["product_id_with_similar_image"]=list(new_similar_product_image_id_set)
    return product_json

    
def clean_similar_product_id_list_by_brand():
    
    product_json_path=Path("/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v2_course/bestbuy_products_40000_3.4.13_sensitive.json")
   
    with open(product_json_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        product_json_dict=json_to_product_id_dict(new_crawled_products_url_json_array)
         
    example_path=Path("/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/example/bestbuy_100_human_performance_w_v3_filter_similar_product.json")
    output_path=Path("/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/example/bestbuy_100_human_performance_w_v4_filter_similar_product.json")
    out_list=[]
    filter_num=0
    with open( example_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
    
        for review_dataset_json in new_crawled_products_url_json_array:
            brand=get_brand_from_review(review_dataset_json)
            brand_in_product=  gen_brand_from_spec(review_dataset_json)
            if   brand_in_product ==brand:
                review_dataset_json["similar_product_id"],filter_num=filter_similar_product_id_list(review_dataset_json["similar_product_id"],brand,product_json_dict,filter_num)
                review_dataset_json["product_id_with_similar_image"],filter_num=filter_similar_product_id_list(review_dataset_json["product_id_with_similar_image"],brand,product_json_dict,filter_num)
            out_list.append(review_dataset_json)
    with open(output_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4)  
    
import numpy as np 
def record_similar_product_num_1_error_case():
    original_example_path=Path("/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/example/bestbuy_100_human_performance.json")
   
    with open(original_example_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        original_example_json_dict=review_json_to_product_dict(new_crawled_products_url_json_array)
        
        
    new_example_path=Path("/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/example/bestbuy_100_human_performance_v2.json")
   
    with open(new_example_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        new_example_json_dict=review_json_to_product_dict(new_crawled_products_url_json_array)
        
    
    
    product_data_path=Path("/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/example/bestbuy_100_human_performance_w_v4_filter_similar_product.json")
    report_dir="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/example/annotation"
    table_output_path="bestbuy/output/no_similar_product_error_example246.csv"
    error_review_id_list=[]
    with open(product_data_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        for review_dataset_json in new_crawled_products_url_json_array:
            review_id=review_dataset_json["reviews"][0]["id"]
            error_review_id_list.append(review_id) 
            
    record_list=[]
    report_path_list=os.listdir(report_dir)
    for report_path in report_path_list:
        report_df=pd.read_excel(os.path.join(report_dir,report_path), index_col=0,skiprows=0)
        for idx,(_, row) in enumerate(report_df.iterrows()):
         
            review_id=row["Review Id"] 
            human_prediction=row["Target Product Position (1-21)"]
            if review_id in error_review_id_list   and human_prediction is not None and human_prediction!="" and not pd.isna(human_prediction)  :
                if not is_list_same(original_example_json_dict[review_id]["similar_product_id"],new_example_json_dict[review_id]["similar_product_id"]) or not is_list_same(original_example_json_dict[review_id]["product_id_with_similar_image"],new_example_json_dict[review_id]["product_id_with_similar_image"]):
                    record_list.append([review_id])
    df = pd.DataFrame(record_list, columns =['review_id'  ])
    df.to_csv(table_output_path,index=False)   

import random 
def read_review_id_and_shuffle_by_report():
    review_id_list=[]
    
    report_dir="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/example/annotation"
 
    for i in range(0,12):
        report_path=f"report{i}.xlsx" 
        report_df=pd.read_excel(os.path.join(report_dir,report_path), index_col=0,skiprows=0)
        review_id_to_shuffle_list=[]
        if i in [8,7,2]:
            is_start=False
        else:
            is_start=True 
        for idx,(_, row) in enumerate(report_df.iterrows()):
            review_id=row["Review Id"] 
            human_prediction=row["Target Product Position (1-21)"]
            if not pd.isna(human_prediction):
                review_id_list.append(review_id)
            else:
                if is_start:
                    review_id_to_shuffle_list.append(review_id)
                else:
                    review_id_list.append(review_id)
                    if i==8 :
                        if review_id==65147 and idx==179:
                            is_start=True 
                   
                            
                    elif i==7:
                        if review_id==55927 and idx==209:
                            is_start=True 
                        
                    elif i==2:
                        if review_id==18095  and idx==259:
                            is_start=True 
                        
                            
                         
        random.shuffle(review_id_to_shuffle_list)
        review_id_list.extend(review_id_to_shuffle_list)
    return review_id_list

def get_remaining_review_id_from_json():
    previous_review_id=92019
    original_example_path=Path("bestbuy/data/example/bestbuy_100_human_performance_v2.json")
    review_id_list=[]
    with open(original_example_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        is_start=False 
        for review_dataset_json in new_crawled_products_url_json_array:
            
            review=review_dataset_json["reviews"][0]
            review_id=review["id"]
            if is_start:
                review_id_list.append(review_id)
            if review_id==previous_review_id:
                is_start=True 
    random.shuffle(review_id_list)
    return  review_id_list


def gen_example(review_id_list):
    original_example_path=Path("bestbuy/data/example/bestbuy_100_human_performance_v2.json")   
    out_example_path=Path("bestbuy/data/example/bestbuy_100_human_performance_v3.json")
    with open(original_example_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        original_example_json_dict=review_json_to_product_dict(new_crawled_products_url_json_array)
    out_list=[]
    for review_id in review_id_list:
        review_product_json=original_example_json_dict[review_id]
        out_list.append(review_product_json)
    with open(out_example_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4)      
    
    
def check_order():
    report_dir="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/example/annotation"
    review_id_list=[]
    for i in range(0,12):
        report_path=f"report{i}.xlsx" 
        report_df=pd.read_excel(os.path.join(report_dir,report_path), index_col=0,skiprows=0)
 
        for idx,(_, row) in enumerate(report_df.iterrows()):
            review_id=row["Review Id"] 
            review_id_list.append(review_id)
    new_example_path=Path("bestbuy/data/example/bestbuy_100_human_performance_v4.json")
    with open(new_example_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        is_start=False 
        for idx,review_dataset_json in enumerate(new_crawled_products_url_json_array):
            
            review=review_dataset_json["reviews"][0]
            review_id=review["id"]
            if is_start:
                review_id_list.append(review_id)
            if idx==3959:
                is_start=True 
    
    new_review_id_list=[] 
     
    with open(new_example_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        is_start=False 
        for idx,review_dataset_json in enumerate(new_crawled_products_url_json_array):
            
            review=review_dataset_json["reviews"][0]
            review_id=review["id"]
            new_review_id_list.append(review_id)
            
    if len(new_review_id_list) != len(review_id_list):
        print("error!")
    else:
        is_right=True  
        for i in range(len(new_review_id_list)):
            if new_review_id_list[i]!=review_id_list[i]:
                print(f"{i}, {new_review_id_list[i]},{review_id_list[i]}")
                is_right=False 
        if is_right:
            print("correct")
     
                     
    review_id_list_in_answer_csv=[]      
    gold_report_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/multimodal_entity_linking/output/example_answer/report0102.csv"
    report_df=pd.read_csv(gold_report_path)
    for idx,(_, row) in enumerate(report_df.iterrows()):
        
        review_id=row["review_id"] 
        review_id_list_in_answer_csv.append(review_id)
    if new_review_id_list == review_id_list_in_answer_csv:
        print("report.csv correct")
    else:
        print("report.csv wrong")
    
"""
report 8: 149-179
    65147 -> 
report 7: -209
report 2: -259
report 10: -90
"""
def shuffle_example():
    
    review_id_list=read_review_id_and_shuffle_by_report()
    remaining_review_id_list=get_remaining_review_id_from_json()
    review_id_list.extend(remaining_review_id_list)

    gen_example(review_id_list)
    
def gen_review_id_list(json_path):
    new_review_id_list=[] 
    new_example_path=Path(json_path)#
    with open(new_example_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
 
        for idx,review_dataset_json in enumerate(new_crawled_products_url_json_array):
            
            review=review_dataset_json["reviews"][0]
            review_id=review["id"]
            new_review_id_list.append(review_id)
    return new_review_id_list
    
def gen_duplicate_review_id_list():
    review_path="bestbuy/data/final/v1/bestbuy_review_2.3.13.6_clean_error_Y.json"
    incomplete_review_path = Path(
        review_path
    )
    output_products_path = Path(
        "bestbuy/data/final/v1/bestbuy_review_2.3.13.7.1_remove_all_duplicate.json"
    )
 
    out_list=[]
    review_body_product_json_dict={}
    with open(incomplete_review_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        for i,cur_product_json in tqdm(enumerate(incomplete_dict_list) ):
            review=cur_product_json["reviews"][0]
            review_body=review["body"]
            if review_body!="" :
                if  review_body   in review_body_product_json_dict:
                    review_body_product_json_dict[review_body].append(cur_product_json)
                else:
                    review_body_product_json_dict[review_body]=[cur_product_json]
    wrong_distribution={}
    wrong_num=0
    wrong_review_id_list=[]
    print(f"review num {len(review_body_product_json_dict)}")
    for review_body,product_json_list in review_body_product_json_dict.items():
        if len(product_json_list)>1:
            for product_json in product_json_list:
                out_list.append(product_json)
                wrong_review_id_list.append(product_json["reviews"][0]["id"])
            wrong_num_for_cur_header=len(product_json_list)
            if wrong_num_for_cur_header in wrong_distribution:
                wrong_distribution[wrong_num_for_cur_header]+=1
            else:
                wrong_distribution[wrong_num_for_cur_header]=1
            wrong_num+=wrong_num_for_cur_header 
        
    print(f"review wrong num: { wrong_num }, {wrong_distribution} ")
    
    with open(output_products_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4)
    return wrong_review_id_list 


annotated_error_review_list=[42784,
56501,
74458,
74467,
74507,
74505,
74475,
74521,
74526,
74469,
74519,
74492,
74495,
74509,
74528,
74480,
74496,
74479,
74487]
def remove_duplicate_review():
    wrong_review_id_list=gen_duplicate_review_id_list()
    review_json_path="bestbuy/data/example/bestbuy_100_human_performance_v3.json"
    current_review_id_list=gen_review_id_list(review_json_path)
    review_to_remove_id_set=set(current_review_id_list).intersection(set(wrong_review_id_list))
    review_to_remove_id_set=review_to_remove_id_set-set(annotated_error_review_list)
    print(len(review_to_remove_id_set))
    
    original_review_to_replace_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/example/previous/bestbuy_100_human_performance_added.json"
    original_review_id_to_replace_list=gen_review_id_list(original_review_to_replace_path)
    current_review_id_to_replace_set=set(original_review_id_to_replace_list)-set(current_review_id_list)
    
    with open(original_review_to_replace_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        review_id_product_dict_to_replace=review_json_to_product_dict(new_crawled_products_url_json_array)
        
    output_review_json_path=Path("/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/example/bestbuy_100_human_performance_v4.json")
    table_output_path=Path("bestbuy/output/replace_testset_error_example_duplicate_review_23.csv")

    out_list=[]
    replace_idx=0
    record_list=[]
    current_review_id_to_replace_list=list(current_review_id_to_replace_set)
    with open(review_json_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        for idx,review_dataset_json in tqdm(enumerate(new_crawled_products_url_json_array)):
            if not is_nan_or_miss(review_dataset_json,"reviews"):
                review=review_dataset_json["reviews"][0]
                review_id=review["id"]
                if review_id not in review_to_remove_id_set:
                    out_list.append(review_dataset_json)
                else:
                    replace_example=review_id_product_dict_to_replace[current_review_id_to_replace_list[replace_idx]]
                    out_list.append(replace_example)
                    replace_idx+=1
                    record_list=record(replace_example,review_dataset_json,record_list,idx)
                    
    with open(output_review_json_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4)   
             

    df = pd.DataFrame(record_list, columns =['original_review_id', 'replace_review_id','annotator_id (1-12)' ])
    df.to_csv(table_output_path,index=False)  
    
def gen_wrong_annotation_mingchen():
    gold_product_dict=gen_reivew_gold_product_dict()
    table_output_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/output/report8_fixed_by_Sai.xlsx"
    report8_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/example/annotation/report8.xlsx"    
    report_df=pd.read_excel(report8_path, index_col=0,skiprows=0)
    record_list=[]
    for idx,(_, row) in enumerate(report_df.iterrows()):
        review_id=row["Review Id"] 
        review_text=row["Review Text"]
        human_prediction=row["Target Product Position (1-21)"]
        gold=gold_product_dict[review_id]
        if gold!=human_prediction:
            webpage_num=264 + idx//10
            record_list.append([ review_text,review_id,"","",f"Annotation Webpage {webpage_num}",f"http://nlplab1.cs.vt.edu/~menglong/for_inner_usage/example/example{webpage_num}.html"])
    df = pd.DataFrame(record_list, columns =['Review Text','Review Id','Target Product Position (1-21)','Confident? (y/n) (If you are not confident with the choice, label n)','Annotation Webpage Name', 'Annotation Webpage Url'  ])
 
    with pd.ExcelWriter( table_output_path ) as writer:  
        df.to_excel(writer,sheet_name= "entity_linking" )      
         
def check_to_be_added_reviews():
    
  
    review_json_path="bestbuy/data/example/bestbuy_100_human_performance_v4.json"
    current_review_id_list=gen_review_id_list(review_json_path)
 
    original_review_to_replace_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/example/previous/bestbuy_100_human_performance_added.json"
    original_review_id_to_replace_list=gen_review_id_list(original_review_to_replace_path)
    current_review_id_to_replace_set=set(original_review_id_to_replace_list)-set(current_review_id_list)         
    print(current_review_id_to_replace_set)
    
    
    
def update_mingchen_report():
    table_output_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/example/annotation/report8.xlsx"
    fixed_sub_report1_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/example/annotation/report8_fixed_by_Sai.xlsx"
    report_df=pd.read_excel(fixed_sub_report1_path, index_col=0,skiprows=0)
    review_id_row_dict={}
    for idx,(_, row) in enumerate(report_df.iterrows()):
        review_id=row["Review Id"] 
        review_text=row["Review Text"]
        human_prediction=row["Target Product Position (1-21)"]
        confident=row["Confident? (y/n) (If you are not confident with the choice, label n)"]
        review_id_row_dict[ review_id ]=[ review_text,review_id,human_prediction,confident ]
         
    record_list=[]
    fixed_report_path="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/example/annotation/report8_fixed_by_Barry.xlsx"
    report_df=pd.read_excel(fixed_report_path, index_col=0,skiprows=0)
    for idx,(_, row) in enumerate(report_df.iterrows()):
        review_id=row["Review Id"] 
        review_text=row["Review Text"]
        human_prediction=row["Target Product Position (1-21)"]
        confident=row["Confident? (y/n) (If you are not confident with the choice, label n)"]
        if review_id not in review_id_row_dict:
            record_list.append([ review_text,review_id,human_prediction,confident ])
        else:
            record_list.append(review_id_row_dict[review_id])
    
    
 
    
    
    df = pd.DataFrame(record_list, columns =['Review Text','Review Id','Target Product Position (1-21)','Confident? (y/n) (If you are not confident with the choice, label n)'   ])
 
    with pd.ExcelWriter( table_output_path ) as writer:  
        df.to_excel(writer,sheet_name= "entity_linking" )    
    
def remove_duplicate_review_for_dataset():
    wrong_review_id_list=gen_duplicate_review_id_list()
    review_json_path="bestbuy/data/final/v3/bestbuy_review_2.3.16.21_train_val.json"
    output_review_json_path=Path("bestbuy/data/final/v3/bestbuy_review_2.3.16.22_train_val_remove_duplicate_review.json")
    out_list=[]
    filter_num=0
 
    with open(review_json_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        for idx,review_dataset_json in tqdm(enumerate(new_crawled_products_url_json_array)):
            if not is_nan_or_miss(review_dataset_json,"reviews"):
                review=review_dataset_json["reviews"][0]
                review_id=review["id"]
                if review_id not  in wrong_review_id_list:
                    out_list.append(review_dataset_json)
                else:              
                    filter_num+=1      
    with open(output_review_json_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4)  
    print(f"{filter_num}")
    
def gen_empty_image_similar():
    review_json_path=Path("bestbuy/data/final/v3/bestbuy_products_40000_3.4.14_text_similar.json")
    output_review_json_path=Path("bestbuy/data/final/v3/bestbuy_products_40000_3.4.14.1_miss_image_similar.json")
    out_list=[]
    filter_num=0
 
    with open(review_json_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        for idx,review_dataset_json in tqdm(enumerate(new_crawled_products_url_json_array)):
            if len(review_dataset_json["product_id_with_similar_image"])==0:
                
            
                out_list.append(review_dataset_json)
                       
    with open(output_review_json_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4)  
    print(f"{filter_num}")
    
from PIL import Image


from pixelmatch.contrib.PIL import pixelmatch    
def remove_empty_image():
    empty_img = Image.open("bestbuy/data/final/v2_course/product_images/6425410_rd.jpg")#"bestbuy/data/final/v2_course/product_images/6425410ld.jpg"
    review_json_path=Path("bestbuy/data/final/v3/bestbuy_products_40000_3.4.14_text_similar.json")
    output_review_json_path=Path("bestbuy/data/final/v3/bestbuy_products_40000_3.4.14.a_remove_empty_image.json")
    image_dir="bestbuy/data/final/v2_course/product_images/"
    out_list=[]
    filter_num=0
    
    with open(review_json_path, 'r', encoding='utf-8') as fp:
        new_crawled_products_url_json_array = json.load(fp)
        for idx,review_dataset_json in tqdm(enumerate(new_crawled_products_url_json_array)):
            image_path_list=review_dataset_json["image_path"]
            new_image_path_list=[]
            for image_path in image_path_list:
                real_image_dir=os.path.join(image_dir,image_path)
                if not check_same_image(real_image_dir,empty_img):
                    new_image_path_list.append(image_path)
                else:
                    filter_num+=1
            review_dataset_json["image_path"]=new_image_path_list
            
         
            
            out_list.append(review_dataset_json)
                       
    with open(output_review_json_path, 'w', encoding='utf-8') as fp:
        json.dump(out_list, fp, indent=4)  
    print(f"{filter_num}")
    
def merge_product_without_empty_image():
    output_review_json_path="bestbuy/data/final/v6/bestbuy_products_40000_3.4.17_remove_empty_image.json"
    product_without_empty_image_str=Path("bestbuy/data/final/v3/bestbuy_products_40000_3.4.14.a_remove_empty_image.json")
    with open(product_without_empty_image_str, 'r', encoding='utf-8') as fp:
        product_without_empty_image = json.load(fp)
        product_without_empty_image_dict=json_to_product_id_dict(product_without_empty_image)
    output_list=[]
    filter_num=0
    lastest_product_json_path=Path("bestbuy/data/final/v5/bestbuy_products_40000_3.4.16_all_text_image_similar.json")
    with open(lastest_product_json_path, 'r', encoding='utf-8') as fp:
        lastest_product_json = json.load(fp)
        for product_json in tqdm(lastest_product_json):
            product_id=product_json["id"]
            product_image_path_without_empty_image=product_without_empty_image_dict[product_id]["image_path"]
            if len(product_image_path_without_empty_image)>0:
                product_json["image_path"]=product_image_path_without_empty_image
                output_list.append(product_json)
            else:
                filter_num+=1
    with open(output_review_json_path, 'w', encoding='utf-8') as fp:
        json.dump(output_list, fp, indent=4)  
    print(f"{filter_num}")

def check_same_image(img_b_path,empty_img):
    

    
    img_b = Image.open(img_b_path)
    # img_b=Image.open("bestbuy/data/final/v2_course/product_images/6425410cv11d.jpg")
    img_diff = Image.new("RGBA", empty_img.size)
    if empty_img.size!=img_b.size :
        return False 
    else:
        # note how there is no need to specify dimensions
        mismatch = pixelmatch(empty_img, img_b, img_diff, includeAA=True)
        if mismatch<10:
            return True 
        else:
            return False 
        # img_diff.save("diff.png")
    
# def move_mingqian_reannotation_images():
        
    
    
if __name__ == "__main__":  
    # report_path="/home/menglong/workspace/code/referred/product_scraper/dataset_construction/bestbuy/data/example/report.xlsx"
    # 
    report_path="bestbuy/data/example/report650.xlsx"
    # gen_review_not_check_image_url()
    # choose_real_wrong_review_json()
    # find_no_similarity_and_not_extract_brand_attribute_from_review()
    # clean_removed_product_for_similar_product_list()
    # clean_similar_product_id_list_by_brand()
    # record_similar_product_num_1_error_case()
    # shuffle_example()
    # check_order()
    # check_to_be_added_reviews()
    # update_mingchen_report()
    # gen_review_not_check_image_url()
    # choose_real_wrong_review_json()
    # gen_empty_image_similar()
    # check_same_image()
    # remove_empty_image()
    merge_product_without_empty_image()
    # remove_duplicate_review_for_dataset()
    # gen_wrong_annotation_mingchen()
    # remove_duplicate_review()
    # extract_example_from_report(10000,report_path)
    # gen_review_not_check_image_url()
    # record_replace_table()
    # fix_1()
    # fix_for_low_information()
