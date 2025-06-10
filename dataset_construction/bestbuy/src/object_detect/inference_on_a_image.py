import argparse
import os
import sys

import sys
import json
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from bestbuy.src.object_detect.detect.detect import ObjectDetector 
from  bestbuy.src.object_detect.groundingdino.datasets import transforms as T
from  bestbuy.src.object_detect.groundingdino.models import build_model
from bestbuy.src.object_detect.groundingdino.util import box_ops
from  bestbuy.src.object_detect.groundingdino.util.slconfig import SLConfig
from bestbuy.src.object_detect.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

import os 
import pandas as pd 
 
from random import sample
import json
from pathlib import Path
from time import time
from pathlib import Path
import json
import pandas as pd
import requests
import os
import copy 
import concurrent.futures
import random 
 

import shutil
def cp_original_image(image_path,output_dir):
    shutil.copy(image_path, output_dir)



def clean_text(text_prompt):
    text_prompt=text_prompt.replace(" ","")
    text_prompt=text_prompt.replace("\\","")
    text_prompt=text_prompt.replace("/","")
    return text_prompt
    
    
def gen_text_prompt( product_name,mention,is_test ):
    if is_test:
        return mention
    else:
        product_name_split=product_name.split("-")
        if len(product_name_split)!=3:
            return product_name
        else:
            return product_name_split[2].strip()+product_name_split[1].strip() #color + name
        
def try_by_category( product_path ,object_detector,image_path):
    product_path_list=product_path.split(" -> ")
    text_prompt=product_path_list[-1]
    is_detected,detected_image=object_detector.detecte_image(image_path, text_prompt)
    return is_detected,detected_image
    
def detect_one_image(args,is_copy_original_image,review_image_name,object_detector,review_product_json,detected_dir,is_test,
                     detect_error_dir, coped_image_dir,is_product):
    image_path=os.path.join(args.image_dir,review_image_name)
    coped_image_path=os.path.join( coped_image_dir,review_image_name)
    if is_copy_original_image:
        shutil.copy(image_path,coped_image_path)
    if not is_product:
        mention=review_product_json["reviews"][0]["mention"]
    else:
        mention=""
    text_prompt=gen_text_prompt(review_product_json["product_name"],mention, is_test)
    is_detected,detected_image=object_detector.detecte_image(image_path, text_prompt)
        
    if is_detected:
        image_name_split=review_image_name.split(".")
        text_prompt_for_name=clean_text(text_prompt)
        detected_image.save(os.path.join(detected_dir,review_image_name ))
    else:
        if not is_test:
            is_detected,detected_image=try_by_category(review_product_json["product_category"],object_detector,image_path)
            if is_detected:
         
                detected_image.save(os.path.join(detected_dir,review_image_name ))
            else:
                print(f"can not find {review_image_name} by prompt {text_prompt}")
                detect_error_path= os.path.join( detect_error_dir,review_image_name)
                cp_original_image(image_path,detect_error_path)
        else:
            print(f"can not find {review_image_name} by prompt {text_prompt}")
            detect_error_path= os.path.join( detect_error_dir,review_image_name)
            cp_original_image(image_path,detect_error_path)
    return is_detected

    # import ipdb; ipdb.set_trace()
    
def detect_one_image_func(review_product_json,object_detector,args):
    is_copy_original_image=False
    is_sample=False
    if args.is_product:
        review_image_path_list=review_product_json["image_path"]
    else:
        review_image_path_list=review_product_json["reviews"][0]["image_path"]
    new_image_path_list=[]
    for review_image_path in review_image_path_list:
        # if image_num>=25:
        #     return 
        # image_num+=1
        is_detected=detect_one_image(args,is_copy_original_image,review_image_path,object_detector,review_product_json,args.detected_dir,
                            args.is_test,args.detect_error_dir,args.coped_image_dir,args.is_product)
        if is_detected:
            new_image_path_list.append(review_image_path)
            
    if len(new_image_path_list)>0:
        if args.is_product:
            review_product_json["image_path"]=new_image_path_list
        else:
            review_product_json["reviews"][0]["image_path"]=new_image_path_list
        return review_product_json,True
    else:
        return {},False
        
        
def main(args):
    start_id=args.start_idx
    end_id=args.end_idx
    save_step=100
    # cfg
    config_file = args.config_file  # change the path of the model config file
    checkpoint_path = args.checkpoint_path  # change the path of the model
    detected_dir = args.detected_dir
    box_threshold = args.box_threshold
    complete_products_path = Path(
        f"{args.out_path}_from_{start_id}_to_{end_id}.json"
    )
    text_threshold = args.text_threshold
    device=torch.device("cuda")
    # load model
    object_detector=ObjectDetector(config_file, checkpoint_path,args.cpu_only,box_threshold, text_threshold,device)
    # make dir
    os.makedirs(detected_dir, exist_ok=True)
    os.makedirs(args.coped_image_dir, exist_ok=True)
    os.makedirs(args.detect_error_dir, exist_ok=True)
    
    start_review_num=0
    image_num=0
    is_copy_original_image=False
    is_sample=False
    output_list=[]
    with open(args.review_path, 'r', encoding='utf-8') as fp:
        review_product_json_array = json.load(fp)
        if is_sample:
            review_product_json_array=sample(review_product_json_array,100)
        for i,review_product_json in tqdm(enumerate(review_product_json_array), total=end_id-start_id):
            if i>=start_id :
                if   i<end_id:
                    review_product_json,is_found=detect_one_image_func(review_product_json,object_detector,args)
                    if is_found:
                        output_list.append(review_product_json)
                    if i%save_step==0:
                        with open(complete_products_path, 'w', encoding='utf-8') as fp:
                            json.dump(output_list, fp, indent=4)
    with open(complete_products_path, 'w', encoding='utf-8') as fp:
        json.dump(output_list, fp, indent=4)

def mp_main(args):
    start_id=args.start_idx
    end_id=args.end_idx
    incomplete_products_path = Path(
        args.review_path
    )
    complete_products_path = Path(
        f"{args.out_path}_from_{start_id}_to_{end_id}.json"
    )
    config_file = args.config_file  # change the path of the model config file
    checkpoint_path = args.checkpoint_path  # change the path of the model
    detected_dir = args.detected_dir
    box_threshold = args.box_threshold
    
    text_threshold = args.text_threshold
    # make dir
    os.makedirs(detected_dir, exist_ok=True)
    os.makedirs(args.coped_image_dir, exist_ok=True)
    os.makedirs(args.detect_error_dir, exist_ok=True)
  
     
     
    step_size =  args.step_size
    output_list = []
    print_ctr = 0
    result = []
    save_step=100
    world_size=torch.cuda.device_count()
    gpu_list=[i for i in range(world_size)]
    
    args_list=[args for i in range(step_size)]
    model_list=[  ObjectDetector(config_file, checkpoint_path,args.cpu_only,box_threshold, text_threshold, device=torch.device('cuda',gpu_list[i%world_size]))  for i in range(step_size)]
    with open(incomplete_products_path, 'r', encoding='utf-8') as fp:
        product_json_list = json.load(fp)
    for i in tqdm( range(0, len(product_json_list), step_size)):
        if i>=start_id :
            if   i<end_id:
                print(100 * '=')
                print(f'starting at the {print_ctr}th value')
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    try:
                        result = list(
                            tqdm( 
                                executor.map(
                                    detect_one_image_func,
                                    product_json_list[i: i + step_size],
                                    model_list ,
                                    args_list
                                ),
                                total=step_size)
                        )
                    except Exception as e:
                        print(f"{e}")
                        print(f"__main__ error {e}")
                        result = []
             
                if len(result) != 0:
                    output_list.extend(result)
                else:
                    print('something is wrong')
                if i%save_step==0:
                    with open(complete_products_path, 'w', encoding='utf-8') as fp:
                        json.dump(output_list, fp, indent=4)
        print_ctr += step_size
    with open(complete_products_path, 'w', encoding='utf-8') as fp:
        json.dump(output_list, fp, indent=4)
    
   

def parser_args():
    parser = argparse.ArgumentParser("Grounding DINO example", add_help=True)
    parser.add_argument('--mode',type=str,help=" ",default="mp")
    parser.add_argument('--step_size',type=int,help=" ",default=2)
    parser.add_argument('--start_idx',type=int,help=" ",default=0)
    parser.add_argument('--end_idx',type=int,help=" ",default=1000000)
    
    parser.add_argument("--image_dir",  type=str, help="path to config file",default="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v5/product_images")
    parser.add_argument("--coped_image_dir",   type=str, help="path to config file",default="outputs/original")
    parser.add_argument("--review_path",  type=str, help="path to config file",default="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/bestbuy_products_40000_3.4.17_remove_empty_image.json")
    parser.add_argument('--out_path',type=str,help=" ",default= "/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v6/bestbuy_products_40000_3.4.18_clean_by_object_detect")
    parser.add_argument(
        "--detected_dir", "-o", type=str, default="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v5/cleaned_product_images",  help="output directory"
    )
    parser.add_argument(
        "--detect_error_dir" , type=str, default="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v5/error_product_images",  help="output directory"
    )
    parser.add_argument("--config_file", "-c", type=str, help="path to config file",default="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/GroundingDINO/groundingdino/config/GroundingDINO_SwinB.cfg.py")
    parser.add_argument(
        "--checkpoint_path", "-p", type=str, help="path to checkpoint file",default="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth"
    )
    parser.add_argument("--image_path", "-i", type=str,  help="path to image file",default="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v5/review_images/0001321-1bb55859e324ba245b8b32e2bdfb5ffc.jpg")
    parser.add_argument("--text_prompt", "-t", type=str,  help="text prompt",default="fridge")
    

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")#3
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")

    parser.add_argument("--cpu-only", action="store_true", help="running on cpu only!, default=False")
    parser.add_argument("--is_test", action="store_true", help="running on cpu only!, default=False")
    parser.add_argument("--is_product", action="store_true", help="running on cpu only!, default=False")
    args = parser.parse_args()
    return args



  
  
  
  
if __name__ == '__main__':
    args = parser_args()     
    
    if args.mode=="mp":
        mp_main(args)
    else:
        main(args )