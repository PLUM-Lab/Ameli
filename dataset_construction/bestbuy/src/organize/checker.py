from pathlib import Path
import json
import pandas as pd
import requests
import os
from tqdm import tqdm
from ast import literal_eval
def check_review_nan(review_path):
    incomplete_products_path = Path(
        review_path
    )
    review_err=0
    review_img_url_err=0
    review_img_path_err=0
    with open(incomplete_products_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        for product_json in tqdm(incomplete_dict_list):
 
            if  (is_nan_or_miss(product_json,"reviews") and is_nan_or_miss(product_json,"review")) :
                review_err+=1
            else:
                reviews=product_json["reviews"]
                for review in reviews:
                    if is_nan_or_miss(review,"image_url"):
                        review_img_url_err+=1
                    if is_nan_or_miss(review,"image_path"):
                        review_img_path_err+=1 
    return review_err, review_img_url_err,review_img_path_err,len(incomplete_dict_list)
                
       

def check_nan(file_path,review_path):
    incomplete_products_path = Path(
        file_path
    )
    if review_path!=None:
        review_err, review_img_url_err,review_img_path_err,review_num=check_review_nan(review_path)
    else:
        review_err, review_img_url_err,review_img_path_err,review_num=0,0,0,0
     
    output_list=[]
    error_num=0
    url_err=0
    path_err=0
    name_err=0
    overview_section_err=0
    img_url_err=0
    img_path_err=0
    spec_err=0
    
    with open(incomplete_products_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        for product_json in incomplete_dict_list:
            
            is_error=False 
            if  (is_nan_or_miss(product_json,"product_name")) :
         
                is_error=True 
                name_err+=1
            if  (is_nan_or_miss(product_json,"product_category")) :
          
             
                path_err+=1
                is_error=True 
            
        
            if  (is_nan_or_miss(product_json,"url")) :
                
                print(product_json["product_name"])
                url_err+=1
                is_error=True 
             
            if  (is_nan_or_miss(product_json,"image_url") ) :#and is_nan_or_miss(product_json,"product_images")
                img_url_err+=1
                is_error=True 
            # url=product_json["reviews"]
            # if  (is_nan(url)) :
            #     review_err+=1
            #     is_error=True 
                
   
            if  (is_nan_or_miss(product_json,"Spec")) :
           
                spec_err+=1
                is_error=True   
            if  (is_nan_or_miss(product_json,"image_path")):#and is_nan_or_miss(product_json,"product_images_fnames")) 
                img_path_err+=1
                # print(product_json["url"])
                is_error=True 
            if is_miss_desc(product_json):
                overview_section_err+=1
                is_error=True 
             
            if is_error:
                error_num+=1
                # print(product_json)
        print(f"all err:{error_num},url_err:{url_err}, path_err:{path_err}, name_err:{name_err},overview_section_err:{overview_section_err}, img_url_err:{img_url_err}, " 
        f"img_path_err:{img_path_err}, spec_err:{spec_err}, review_err:{review_err},  review_img_url_err:{review_img_url_err}, review_img_path_err:{review_img_path_err}, "
        +f"product:{len(incomplete_dict_list)}, review:{review_num}")



def is_miss_desc(product_json):
    if  (is_nan_or_miss(product_json,"overview_section")) :
               
        return True 
    else:
        overview_section=product_json["overview_section"]
        description=overview_section["description"]
        if  (is_nan(description)) :
            return True 
    return False 
def is_nan_or_miss(json,key):
    if key  not in json:
        return True
    elif json[key] is None or len( json[key]) == 0:          
        return True 
    else:
        return False             
def is_nan(value):
    if value is None or len( value) == 0:          
        return True 
    else:
        return False 



def check_current_csv():
    db_path = Path('bestbuy/data/bestbuy_data_with_ids_and_specs.csv')
    df = pd.read_csv(db_path)
    image_row_list = []
    # print(df.isna())
    print(df.isna().sum())
    df_without_review=df[df["reviews"]=="[]"]
    df_without_desc=df[df["overview_section"]=="{}"]
    df_without_img=df[df["thumbnails"]=="[]"]
    df_without_specs=df[df["specs"].isna()]
    print(f"{len(df_without_review)},{len(df_without_desc)},{len(df_without_img)},{len(df_without_specs)}")
    # for idx, db_row in tqdm(df.iterrows()):
        
    #     thumbnails = literal_eval(db_row['thumbnails'])
    #     if len(thumbnails) != 0:
    #         for thumbnail in tqdm(thumbnails):
    #             thumbnail_cleaned = thumbnail.split(';')[0]
            
    #     reviews = literal_eval(db_row['reviews'])
    #     if len(reviews) != 0:
    #         for review in tqdm(reviews):
    #             review_fnames = []
    #             review_images = review['product_images']
    #             for review_image in review_images:
    #                 review_image_cleaned = review_image.split(';')[0]

def check_whether_duplicate_product_image_name(path):
    image_name_image_url_dict={}
    image_name_set=set()
    image_name_list=[]
    incomplete_products_path = Path(
        path
    )
    
    with open(incomplete_products_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        for idx,product_json in enumerate(incomplete_dict_list):
            thumbnail_paths=product_json["image_path"]
            for image_path in thumbnail_paths:
                if image_path in image_name_set:
                    print(f"ERROR! {image_path}, the first url:{incomplete_dict_list[image_name_image_url_dict[image_path]]['image_url']}"
                          +f" , the second url:{product_json['image_url']}")
                image_name_set.add(image_path)
                image_name_image_url_dict[image_path]=idx
                image_name_list.append(image_path)
    
    print(f"num: {len(image_name_set)}, {len(image_name_list)}")
 
    

def check_whether_duplicate_review_image(review_path):
    image_name_image_url_dict={}
    image_name_idx_dict={}
    image_name_set=set()
    image_list=[]
   
    incomplete_products_path = Path(
        review_path
    )
    wrong_num=0
    wrong_distribution={}
    wrong_review_id_list=[]
    image_path_review_id_dict={}
    with open(incomplete_products_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        for idx,product_json in enumerate(incomplete_dict_list):
            review=product_json["reviews"][0]
            review_id=review["id"]
            
            thumbnail_paths=review["image_path"]
            if thumbnail_paths is not None:
                for image_path_idx,image_path in enumerate(thumbnail_paths):
                    # if image_path in image_name_set:
                        # print(f"ERROR! {image_path}, the first url:{image_name_image_url_dict[image_path]}"
                        #     +f" , the second url:{gen_corresponding_image_url(image_path_idx,review['image_url'])}"
                        #     +f"the first review id :{incomplete_dict_list[image_name_idx_dict[image_path]]['reviews'][0]['id']}"
                        #     +f" , the second url:{review['id']}")
                    
                    if image_path in image_path_review_id_dict:
                        image_path_review_id_dict[image_path].append(review_id)
                    else:
                        image_path_review_id_dict[image_path]=[review_id]
                    image_name_set.add(image_path)
                    image_list.append(image_path)
                    image_name_image_url_dict[image_path]=gen_corresponding_image_url(image_path_idx,review['image_url'])
                    image_name_idx_dict[image_path]=idx 
    
    wrong_product_json_distribution={}
    wrong_review_body_distribution={}
    print(f"image_list_num:{len(image_list)}, review num: {len(image_name_set)}, {len(image_name_image_url_dict)}")
    for image_path,review_id_list in image_path_review_id_dict.items():
        if len(review_id_list)>1:
            wrong_review_id_list.extend(review_id_list)
            wrong_num_for_cur_header=len(review_id_list)
            wrong_num+=wrong_num_for_cur_header-1
            if wrong_num_for_cur_header in wrong_distribution:
                wrong_distribution[wrong_num_for_cur_header]+=1
            else:
                wrong_distribution[wrong_num_for_cur_header]=1
            if wrong_num_for_cur_header in wrong_product_json_distribution:
                wrong_product_json_distribution[wrong_num_for_cur_header].extend(review_id_list)
            else:
            
                wrong_product_json_distribution[wrong_num_for_cur_header]=review_id_list
                
            if wrong_num_for_cur_header in wrong_review_body_distribution:
                wrong_review_body_distribution[wrong_num_for_cur_header].append(image_path)
            else:
                wrong_review_body_distribution[wrong_num_for_cur_header]=[image_path]
    print(f"review wrong num: { wrong_num }, {wrong_distribution} ")
                    

def check_whether_duplicate_review_body(review_path):
    # products_path = Path(
    #     "bestbuy/data/final/v1/bestbuy_products_40000_3.4.7_remove_image_path_prefix.json"
    # )
    
    incomplete_review_path = Path(
        review_path
    )
    # output_products_path = Path(
    #     "bestbuy/data/final/v1/bestbuy_review_2.3.13.4_remove_duplicate_review_same_product.json"
    # )
    product_id_list=[]
    # with open(products_path, 'r', encoding='utf-8') as fp:
    #     product_dict_list = json.load(fp)
    #     for i,product_json in  enumerate(product_dict_list)  :
    #         product_id_list.append(product_json["id"])
            
    out_list=[]
    filter_num=0
    
    error_num=0
    review_body_product_json_dict={}
    # tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    with open(incomplete_review_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        for i,cur_product_json in tqdm(enumerate(incomplete_dict_list) ):
            review=cur_product_json["reviews"][0]
            review_body=review["body"]
            if "error" in cur_product_json and cur_product_json["error"]=="Y":
                error_num+=1
                continue 
            # if review["id"]==2186:
            #     print("")
            is_found=False 
            if  review_body   in review_body_product_json_dict:
                product_json_list=review_body_product_json_dict[review_body]
                
                review_body_product_json_dict[review_body].append(cur_product_json)
            else:
                review_body_product_json_dict[review_body]=[cur_product_json]
            # if not is_found:        
            #     out_list.append(cur_product_json)
    wrong_distribution={}
    wrong_product_json_distribution={}
    wrong_review_body_distribution={}
    wrong_num=0
    print(f"review num {len(review_body_product_json_dict)}")
    for review_body,product_json_list in review_body_product_json_dict.items():
        if len(product_json_list)>1:
             
            wrong_num_for_cur_header=len(product_json_list)
            if wrong_num_for_cur_header in wrong_distribution:
                wrong_distribution[wrong_num_for_cur_header]+=1
                
            else:
                wrong_distribution[wrong_num_for_cur_header]=1
            if wrong_num_for_cur_header in wrong_product_json_distribution:
                wrong_product_json_distribution[wrong_num_for_cur_header].extend(product_json_list)
            else:
            
                wrong_product_json_distribution[wrong_num_for_cur_header]=product_json_list
                
            if wrong_num_for_cur_header in wrong_review_body_distribution:
                wrong_review_body_distribution[wrong_num_for_cur_header].append(review_body)
            else:
                wrong_review_body_distribution[wrong_num_for_cur_header]=[review_body]
            wrong_num+=wrong_num_for_cur_header-1
        
    print(f"review wrong num: { wrong_num }, {wrong_distribution} ,error_num:{error_num}")
    
                   
def check_image_folder(current_image_dir):
    image_num=0
    image_list=[]
    current_image_list=os.listdir(current_image_dir)        
    for image in current_image_list:
        if ".jpg" in image:
            image_num+=1
            image_list.append(image)
        else:
            print(image )
    print(f"{image_num}")  
    return image_list
                    
def gen_corresponding_image_url(image_path_idx,image_url_list):
    return image_url_list[image_path_idx]
  


def check_current_csv_has_wrong_review():
    db_path = Path('bestbuy/data/old/bestbuy_data_with_ids_and_specs.csv')
    df = pd.read_csv(db_path)
    image_wrong_dict = {}
    wrong_num=0
    # print(df.isna())
    print(df.isna().sum())
    wrong_url="https://www.bestbuy.com/site/rca-3-2-cu-ft-upright-freezer-white/6346491.p?skuId=6346491"
    wrong_body="This is such a great size freezer with 3 shelves. Fits perfect in small space and perfect for extra food that won\u2019t fit in freezer in fridge. I\u2019m thrilled with the front open design. I did not want a small chest freezer to hard to rearrange frozen food. This is GREAT."
    wrong_df=df[df["url"]==wrong_url]
    if len(wrong_df )>0:

        for idx, row in tqdm(wrong_df.iterrows()):
            reviews = literal_eval(row['reviews'])
            for review in reviews:
                if wrong_body in review["body"]:
                    # print("csv has wrong review image url")
                    # if image_wrong_dict
                    pass 
                    
    print('end')
    
def gen_review_with_wrong_image_url():
 
    path="bestbuy/data/final/v1/bestbuy_review_2.3.12_remove_image_path_prefix.json"
    incomplete_products_path = Path(
        path
    )
    wrong_num=0
    wrong_distribution={}
    wrong_review_id_list=[]
    repeated_header_bigger_than_2_num=0
    product_review_header_review_id_dict={}
    with open(incomplete_products_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
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
                wrong_review_id_list.extend(review_id_list)
                wrong_num_for_cur_header=len(review_id_list)
                wrong_num+=wrong_num_for_cur_header
                if wrong_num_for_cur_header in wrong_distribution:
                    wrong_distribution[wrong_num_for_cur_header]+=1
                else:
                    wrong_distribution[wrong_num_for_cur_header]=1
    print(f"review wrong num: { wrong_num }, {wrong_distribution} ")
    return wrong_review_id_list
    
    
# def check_dif_before_and_after_fixing_the_image_url_bug():

import argparse
def get_args():
   


    parser = argparse.ArgumentParser() 
    # parser.add_argument("--lr", default=1e-7, type=float)
    # parser.add_argument("--step_size", default=10, type=int)
    # parser.add_argument("--start_id", default=0, type=int)
    # parser.add_argument("--end_id", default=6000, type=int)
    # parser.add_argument("--use_pre_trained_model", default=True, action="store_false") 
    parser.add_argument("--file",default='bestbuy/data/final/v2/bestbuy_products_40000_3.4.7_remove_image_path_prefix.json', type=str  ) 
    parser.add_argument("--review_file",default='bestbuy/data/final/v2/bestbuy_review_2.3.16.8_add_review_id_into_image_path.json', type=str  ) #default='bestbuy/data/bestbuy_review_2.3_incomplete_before_remove_reviews_wo_image.json',
    args = parser.parse_args()

    print(args)
    return args        
    
# runner
if __name__ == "__main__":  
    args=get_args()
    # check_nan(args.file,args.review_file)
    check_image_folder("bestbuy/data/final/v2/product_images")
    check_whether_duplicate_product_image_name(args.file )
    
    # check_whether_duplicate_review_image(args.review_file)
    # check_image_folder("bestbuy/data/final/v2/review_images")
    
    # check_whether_duplicate_review_image(args.review_file)
    # check_whether_duplicate_product_image_name()
    # print(check_review_nan(args.review_file))
    # check_current_csv()             
    # check_current_csv_has_wrong_review()
    # check_whether_duplicate_review_image_name()
    