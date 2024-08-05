import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp


from torch.utils.data.sampler import Sampler
from torchvision import datasets, transforms
import os 
from disambiguation.train_verify import   train_loop  


def gen_checkpoint_dir_list_from_one_ray_folder(ray_folder):
    from os import listdir
    from os.path import isfile, join
    trial_folders=[]
    # search_folders =  listdir(ray_folder)    
    search_folders=["train_loop_2023-10-13_22-15-00","train_loop_2023-10-14_01-08-08","train_loop_2023-10-14_01-10-35","train_loop_2023-10-14_01-11-25",
                    "train_loop_2023-10-14_01-13-22","train_loop_2023-10-14_06-47-19"]
    # search_folders=[]
    # except_trail_dir_list=["DEFAULT_6a1e2_00000_0_batch_size=256,lr=0.001_2022-08-08_11-11-44","DEFAULT_6a1e2_00001_1_batch_size=16,lr=0.001_2022-08-08_11-11-44"
    #                        ,"DEFAULT_6a1e2_00002_2_batch_size=32,lr=1e-05_2022-08-08_11-11-44","DEFAULT_6a1e2_00003_3_batch_size=32,lr=0.0001_2022-08-08_11-11-44"]
    except_trail_dir_list=[]
    for search_folder in search_folders:
        search_folder=join(ray_folder,search_folder)
        for trial_folder in listdir(search_folder):
            if   trial_folder not in except_trail_dir_list:
                trial_folder=join(search_folder, trial_folder)
                if not isfile(trial_folder) :
                    checkpoint_father_dir=os.path.join(trial_folder,"output/runs")
                    if  os.path.isdir(checkpoint_father_dir) :
                        for checkpoint_dir_name in listdir(checkpoint_father_dir):
                            checkpoint_dir=os.path.join(checkpoint_father_dir,checkpoint_dir_name)
                            if  os.path.isdir(checkpoint_dir) :
                                if os.path.exists(os.path.join(checkpoint_dir,"base.pt")):
                                    trial_folders.append(checkpoint_dir )
    
    # trial_folders=trial_folders[51:] # 
    return trial_folders
        
def group_elements(n, iterable):
    from itertools import zip_longest
    return zip_longest(*[iter(iterable)]*n)
    

def load_args(checkpoint_dir) :
    import json
    f = open(checkpoint_dir+'/training_options.json')
    data=json.load(f)
    f.close()
    return data
    
def gen_checkpoint_dir_list(ray_folder,num_processes_per_gpu):
    # checkpoint_dir_list=[]
    checkpoint_dir_list=gen_checkpoint_dir_list_from_one_ray_folder(ray_folder)
    # checkpoint_dir_list.append("/home/menglong/ray_results/DEFAULT_2022-08-13_23-14-12/DEFAULT_2bdc5_00006_6_batch_size=32,lr=0.001_2022-08-14_02-23-00/verification/output/runs/00000-")
    # checkpoint_dir_list.append("/home/menglong/ray_results/DEFAULT_2022-08-13_23-14-12/DEFAULT_2bdc5_00001_1_batch_size=64,lr=0.0001_2022-08-13_23-14-13/verification/output/runs/00000-")
    # checkpoint_dir_list.append("/home/menglong/ray_results/DEFAULT_2022-08-13_23-05-40/DEFAULT_fa962_00002_2_batch_size=256,lr=0.001_2022-08-13_23-05-41/verification/output/runs/00000-")
    # checkpoint_dir_list.append("/home/menglong/ray_results/DEFAULT_2022-08-13_23-05-40/DEFAULT_fa962_00003_3_batch_size=128,lr=0.0001_2022-08-13_23-05-41/verification/output/runs/00000-")
    # checkpoint_dir_list.append("/home/menglong/ray_results/DEFAULT_2022-08-13_23-05-40/DEFAULT_fa962_00005_5_batch_size=64,lr=0.01_2022-08-14_02-00-25/verification/output/runs/00000-")
    # checkpoint_dir_list.append("/home/menglong/ray_results/DEFAULT_2022-08-13_23-05-40/DEFAULT_fa962_00006_6_batch_size=256,lr=0.001_2022-08-14_02-00-55/verification/output/runs/00000-")
    # checkpoint_dir_list.append("verification/output/runs/00267-")
    # checkpoint_dir_list=checkpoint_dir_list[:2] # 
    checkpoint_dir_num=len(checkpoint_dir_list)
    print(checkpoint_dir_num)
    
    checkpoint_dir_list_group=[]
    for checkpoint_dir_list_one_group in group_elements(num_processes_per_gpu,checkpoint_dir_list):
        checkpoint_dir_list_group.append(checkpoint_dir_list_one_group)
    return checkpoint_dir_list_group,checkpoint_dir_num
 
 
import os
import time
     
def mp_inference(config_kwargs):
    num_processes_per_gpu=config_kwargs["num_processes_per_gpu"]
    mp.set_start_method('spawn')
    checkpoint_dir_list_group,checkpoint_dir_num=gen_checkpoint_dir_list("/home/menglong/ray_results",num_processes_per_gpu)


    for group_id,checkpoint_dir_list_one_group in enumerate(checkpoint_dir_list_group):
        processes = []
        for checkpoint_id,checkpoint_dir   in    enumerate(checkpoint_dir_list_one_group) :
            if checkpoint_dir!=None :
                if checkpoint_id%num_processes_per_gpu==0:
                    gpu="0"
                elif checkpoint_id%num_processes_per_gpu==1:
                    gpu="1"
                elif checkpoint_id%num_processes_per_gpu==2:
                    gpu="2"
                elif checkpoint_id%num_processes_per_gpu==3:
                    gpu="3"
                elif checkpoint_id%num_processes_per_gpu==4:
                    gpu="4"
                else:
                    gpu="5"
                     
                os.environ['CUDA_VISIBLE_DEVICES']=gpu 
                saved_config_kwargs=load_args(checkpoint_dir)
                # config_kwargs["batch_size"]=saved_config_kwargs["batch_size"]
                config_kwargs["lr"]=saved_config_kwargs["lr"]
                config_kwargs["accum_iter"]=saved_config_kwargs["accum_iter"]
                config_kwargs["entity_text_source"]=saved_config_kwargs["entity_text_source"]
                config_kwargs["checkpoint_dir"]=checkpoint_dir
                p = mp.Process(target=train_loop, args=( None, config_kwargs,))
                # We first train the model across `num_processes` processes
                p.start()
                processes.append(p)
                time.sleep(2)
        for p in processes:
            p.join()
        print(f"finish {num_processes_per_gpu*(group_id+1)}/{checkpoint_dir_num}")
