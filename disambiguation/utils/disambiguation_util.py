 
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
from util.env_config import * 
from tqdm import tqdm

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report


def myPrinter_without_args( str_to_be_printed,device="cuda",rank=0):
    if  device == 'cuda' or  device == 'cpu' or  rank == 0:
        print(str_to_be_printed)
        


def freeze_part_bert(model,freeze_text_layer_num,freeze_img_layer_num):
    count = 0
 
    for p in model[0].model.text_model.named_parameters():
        
        if (count<=freeze_text_layer_num):
            p[1].requires_grad=False    
        else:
            break
              
        count=count+1
        print(p[0], p[1].requires_grad)
        
    count=0
    for p in model[0].model.vision_model.named_parameters():
        
        if (count<=freeze_img_layer_num):
            p[1].requires_grad=False    
        else:
            break
              
        count=count+1
        print(p[0], p[1].requires_grad)
        
    return model 


def save_prediction(total_query_id_list,y_pred,y_true,total_order_list,total_entity_id_list,mode,checkpoint_dir,run_dir,
                  disambiguation_result_file_name,text_score_np=None,image_score_np=None,text_score_before_softmax_np=None,
                  image_score_before_softmax_np=None,is_in_list_format=True ):
    if disambiguation_result_file_name is None:
        disambiguation_result_file_name="entity_link_reproduce.csv"
    if text_score_np is not None  :
        if is_in_list_format:
            
            print("is_in_list_format")
            if  len(total_query_id_list)==len(list_for_csv(text_score_np.tolist())):
                prediction={"query_id":total_query_id_list,"predict_entity_position":y_pred,"gold_entity_position":y_true,
                            "entity_id_list":list_for_csv(total_entity_id_list),"order_list":list_for_csv(total_order_list),
                            "text_score" :list_for_csv(text_score_np.tolist()),"image_score" :list_for_csv(image_score_np.tolist()),
                            "text_score_before_softmax" :list_for_csv(text_score_before_softmax_np.tolist()),
                            "image_score_before_softmax" :list_for_csv(image_score_before_softmax_np.tolist())}
            else:
                prediction={"query_id":total_query_id_list,"predict_entity_position":y_pred,"gold_entity_position":y_true,
                    "entity_id_list":list_for_csv(total_entity_id_list),"order_list":list_for_csv(total_order_list) }
                
        else:
            print("not is_in_list_format")
            prediction={"query_id":total_query_id_list,"predict_entity_position":y_pred,"gold_entity_position":y_true,
                        "entity_id_list":list_for_csv(total_entity_id_list),"order_list":list_for_csv(total_order_list),
                        "text_score" : text_score_np ,"image_score" : image_score_np }
    else:
        print("text_score_np is None")
        prediction={"query_id":total_query_id_list,"predict_entity_position":y_pred,"gold_entity_position":y_true,
                    "entity_id_list":list_for_csv(total_entity_id_list),"order_list":list_for_csv(total_order_list) }
    for key,one_list in prediction.items():
        print(f"{key},{len(one_list)}")
    df = pd.DataFrame(prediction)
    if mode in["mp_test","test","dry_run"]:
        cur_run_dir= checkpoint_dir
        if checkpoint_dir is None:
            cur_run_dir=run_dir
    else:
        cur_run_dir=run_dir
    
    # if "retrieval" not in args.evidence_file_name :
         
    #     prediction_path=os.path.join(run_dir,"verification_result.csv" )
        
    # else:
    prediction_path=os.path.join(cur_run_dir,disambiguation_result_file_name)
    # myPrinter(args,prediction_path)
    df.to_csv(prediction_path,index=False)
    return prediction_path
    


def list_for_csv(a_list):
    str_array_list=[]
    for one_list in a_list:
        str_array=""
        for one_item in one_list:
            str_array+=str(one_item)+"|"
        str_array=str_array[:-1]        
        str_array_list.append(str_array)
    return str_array_list

def calc_metric( y_true, y_pred, max_candidate_num=None ,verbos="y",rank=0 ,device="cuda" ,is_print=True):
    
    # pre = precision_score(y_true, y_pred, average='micro')
    # recall = recall_score(y_true, y_pred, average='micro')
    check_same(y_true, y_pred)
    label_range=list(range(0,max_candidate_num))
    f1 = f1_score(y_true, y_pred,labels=label_range, average='micro')
    precision = precision_score(y_true, y_pred,labels=label_range, average='micro')
    recall = recall_score(y_true, y_pred,labels=label_range, average='micro')
    # pre=f1
    # recall=f1
    # myPrinter(args,f"f1 {f1}")
    
    if  verbos=="y" and is_print and   rank==0:
 
        confusion_matrix_result=confusion_matrix(y_true, y_pred,labels=label_range)
        myPrinter_without_args(  confusion_matrix_result,device,rank)
    
        if max_candidate_num<=50:
            myPrinter_without_args(  classification_report(y_true, y_pred,labels=label_range),device,rank)
       
    
        
        
    
    
    return f1,precision,recall


def check_same(list1,list2):
    match_num=0
    for item1,item2 in zip(list1,list2):
        if item1==item2:
            match_num+=1
    print(match_num,len(list1))