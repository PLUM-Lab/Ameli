from collections import OrderedDict
from email.policy import strict
from typing import Union
from PIL import Image
import os
import numpy as np
from ray import tune
import pickle
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from time import sleep
import numpy.ma as ma
import warnings
import wandb
from torch.distributed import ReduceOp
import torch.nn.functional as F
from disambiguation.data_util.inner_util import AverageMeter
from disambiguation.disambiguation_organize_main import apply_best_retrieval_weight, apply_best_retrieval_weight_without_filter, gen_retrieval_field, search_best_retrieval_weight_by_loading_file
from disambiguation.model.model_contrastive import myPrinter

from disambiguation.model.model_util import gen_model, gen_output, gen_tokenizer, load_specific_checkpoint_to_AMELIJoint
from disambiguation.utils.disambiguation_util import calc_metric, save_prediction
from disambiguation.utils.post_process import post_process, post_process_for_json
from retrieval.utils.enums import ModelAttribute, TrainingAttribute
from retrieval.utils.retrieval_util import setup_config 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertModel, AdamW
from transformers import AdamW, get_linear_schedule_with_warmup
import math
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy.ma as ma
from util.data_util.data_util import get_data_loader

import retrieval.utils as utils
from util.read_example import get_father_dir
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader 
import re 
# wandb.login(key=   )
CLIP_GRAD_NORM_VALUE=1



def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    
def load_ddp_to_one(state_dict,model):
    # in case we load a DDP model checkpoint to a non-DDP model
    model_dict = OrderedDict()
    pattern = re.compile('module.')
    for k,v in state_dict.items():
        if re.search("module", k):
            model_dict[re.sub(pattern, '', k)] = v
        else:
            model_dict = state_dict
    model.load_state_dict(model_dict,strict=False)
    return model 

def verify_init(rank,config_kwargs):
    warnings.filterwarnings("ignore") 

    # Check GPU
    if config_kwargs["parallel"]=="ddp":
        device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    else:
        device=torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled   = True
    return device

    # for param in model.parameters():
    #     param.requires_grad = False         
def   update_config(config_kwargs,config):
    if config!=None:
        for key in config.keys():
            config_kwargs[key]=config[key]
        # if "batch_size" in config.keys():
        #     config_kwargs['batch_size']=config["batch_size"]
        # if "lr" in config.keys():
        #     config_kwargs['lr']=config["lr"]
    return config_kwargs

def setup_args_for_verify(config_kwargs,device,rank):
    args,logger = setup_config(config_kwargs,rank)   
    # args.train_txt_dir=os.path.join(args.train_dir,args.evidence_file_name)
    # args.val_txt_dir=os.path.join(args.val_dir,args.evidence_file_name)
    # args.test_txt_dir=os.path.join(args.test_dir,args.evidence_file_name )
    args.whole_dataset_dir=get_father_dir(args.test_dir) 
    args.device=device
    args.rank=rank    
    return args,logger


def classifier_predict(tgt_text_list,max_len,device,classifier_tokenizer,classifier):
    tgt_idx=classifier_tokenizer(tgt_text_list,add_special_tokens=True,  
                                               max_length=max_len, 
                                               truncation=True, 
                                               return_tensors='pt',
                                               padding="max_length").to(device)
    # tgt_idx = collate_fn(tgt).to(device)
    tgt_cls = F.softmax(classifier(**tgt_idx).detach(),-1)
    return tgt_cls


# def load_classifier(checkpoint_dir ):
#     device=torch.device("cuda")
#     bert_model = BertModel.from_pretrained("bert-base-uncased")
#     model = TextModel(bert_model,  128, 64, 3,20).to(device)
#     path=os.path.join(checkpoint_dir,"base.pt")
#     model.load_state_dict(torch.load(path)  )#
#     myPrinter(args,f"resume from {checkpoint_dir}")
#     model.requires_grad_(False)
#     classifier_tokenizer=  BertTokenizer.from_pretrained("bert-base-uncased")
#     classifier_tokenizer.truncation_side="left"
#     return model,classifier_tokenizer


def init(config_kwargs,tuner_config,rank):
    config_kwargs=update_config(config_kwargs,tuner_config)
    device=verify_init(rank,config_kwargs)
    args,logger=setup_args_for_verify(config_kwargs,device,rank) 
    args.weighted_score_data_path=args.weighted_score_data_path+args.disambiguation_result_postfix+".json"
    train_attribute=TrainingAttribute[args.train_config]
    model_attribute=ModelAttribute[args.model_attribute]
    if args.model_attribute=="CJoint":
        model_attribute=ModelAttribute.CJoint
   
    return args,model_attribute, train_attribute,device,logger

def ddp_train_loop(rank, world_size,config_kwargs):
    # setup the process groups
    setup(rank, world_size)
    train_loop( None,config_kwargs,rank,world_size)
    cleanup()

def train_loop( tuner_config,config_kwargs,rank=0,world_size=0):
    args,model_attribute,train_attribute,device,logger=init(config_kwargs,tuner_config,rank)
    model,tokenizer,model_special_processor=gen_model(args,device,model_attribute,train_attribute)
    dataloaders=get_data_loader(args.train_dir, args.val_dir,args.test_dir, 
                                                      tokenizer,args.max_len,args.batch_size,args.max_candidate_num,args,rank,world_size,args.mode,model_special_processor)
    if args.is_gpu:
        model=model.to(device)
        if args.parallel=="data_parallel":
            if args.gpu_list is not None and args.gpu_list!="":
                model = torch.nn.DataParallel(model, device_ids=args.gpu_list)
            else:
                model = torch.nn.DataParallel(model)# , device_ids=[0,1,2,3,4,5,6,7]
        elif args.parallel=="ddp":
            model = DDP(model, device_ids=[rank], output_device=rank) #, find_unused_parameters=True
    model=resume_model(args.checkpoint_dir,model,args)
    # print(model.module.encoder.text_model.pooler.dense.weight.requires_grad)
    #optimizer
    optimizer = Adam(model.parameters(), lr = args.lr, eps=1e-8 )
    print(f"warm up:{len(dataloaders['Train'])*args.epoch* 0.1}")
    warm_up_step=math.ceil(len(dataloaders["Train"]) * args.epoch * 0.1) 
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warm_up_step, 
        num_training_steps=len(dataloaders["Train"])*args.epoch
    )
    train(model,dataloaders,device,optimizer, args,logger,args.batch_size,scheduler,args.accum_iter,model_attribute)

def resume_model(checkpoint_dir,model,args):
    if checkpoint_dir!=None:
        path=os.path.join(checkpoint_dir,"base.pt")
        if args.load_mode=="ddp_to_ddp":
            map_location = {'cuda:%d' % 0: 'cuda:%d' % args.rank}
            model.load_state_dict(torch.load(path, map_location=map_location),strict=False  )#
        elif args.load_mode=="ddp_to_one":
            model=load_ddp_to_one(torch.load(path),model)
        else:
            if args.model_attribute in["C_joint","C_joint_adapt"]:
                model=load_specific_checkpoint_to_AMELIJoint(model,path,args.model_group)
            else:
                model.load_state_dict(torch.load(path))#,strict=False  
          
        myPrinter(args,f"resume from {checkpoint_dir}")
    
    return model   




def sync_acc(acc,y_pred,f1,device,rank,args):
    torch.cuda.set_device(rank)
    # correct_num,total_num=compute_correct_num(y_pred, y_true)
    acc.update(f1, len(y_pred))
    acc.all_reduce(device)
    myPrinter(args,f"after sync, acc support: {acc.count}")
    return acc.avg

def sync_list_across_gpus(sync_list,device,rank):
    torch.cuda.set_device(rank)
    if sync_list is None:
        return None
    total = torch.tensor(sync_list, dtype=torch.float32, device= device)
    total_group=sync_tensor_across_gpus(total)
 
    flat_total=torch.flatten(total_group)
    gathered_list=flat_total.tolist()
    return gathered_list
    
def sync_tensor_across_gpus(t: Union[torch.Tensor, None]
                            ) -> Union[torch.Tensor, None]:
    # t needs to have dim 0 for troch.cat below. 
    # if not, you need to prepare it.
    if t is None:
        return None
    group = dist.group.WORLD
    group_size = torch.distributed.get_world_size(group)
    gather_t_tensor = [torch.zeros_like(t) for _ in
                       range(group_size)]
    dist.all_gather(gather_t_tensor, t)  # this works with nccl backend when tensors need to be on gpu. 
   # for gloo and mpi backends, tensors need to be on cpu. also this works single machine with 
   # multiple   gpus. for multiple nodes, you should use dist.all_gather_multigpu. both have the 
   # same definition... see [here](https://pytorch.org/docs/stable/distributed.html). 
   #  somewhere in the same page, it was mentioned that dist.all_gather_multigpu is more for 
   # multi-nodes. still dont see the benefit of all_gather_multigpu. the provided working case in 
   # the doc is  vague... 
    return torch.cat(gather_t_tensor, dim=0)

def train_phase(accum_iter,model,dataloader,device,optimizer ,criterion,scheduler,args,epoch,model_attribute,is_contrastive):
    myPrinter(args,"Train" + ":")
    live_loss = 0
    y_true = []
    y_pred = []
    model.train()
    acc = AverageMeter('Acc', ':6.2f')
    # iterator_stop = torch.tensor(0).to("cuda"  )
    # wandb.watch(model.verifier.stance_detect_layer, log='all')
    with tqdm(dataloader, unit="batch") as tepoch:
        for batch_idx, batch in enumerate(tepoch):
            # if args.parallel=="ddp":
            #     torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
            #     if iterator_stop > 0:
            #         break
                
            output,labels,entity_id_list,query_id,loss,text_score,image_score,_,_=gen_output(batch,model,args.dataset_class,device,args, is_train=True,
                                                             is_contrastive=is_contrastive,model_attribute=model_attribute)
            labels_numpy=labels.detach().cpu().numpy()
            
            _, preds = output.data.max(1)
            y_pred.extend(preds.tolist())
            y_true.extend(labels_numpy.tolist())
            if    args.is_outer_loss or loss is  None  :
                loss = criterion(output, labels)
            live_loss+=loss
            # normalize loss to account for batch accumulation
            loss = loss / accum_iter
            # Backward pass  (calculates the gradients)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            # weights update
            if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(dataloader)):
                # gradient clipping
                # nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM_VALUE)
                # wandb.log({"live_loss":loss})
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()     
                
            avg_loss= live_loss.item()/(batch_idx+1)
            tepoch.set_postfix(loss=avg_loss)
        # else:
        #     if args.parallel=="ddp":
        #         iterator_stop.fill_(1)
        #         torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)   
        
    if args.rank==0 and  args.is_wandb=="y":
        wandb.log({"loss": avg_loss,"live_loss":loss})
    if model_attribute.is_contrastive:
        label_num=args.batch_size*args.max_candidate_num
    else:
        label_num= args.max_candidate_num
    

    
    f1,pre,recall=calc_metric( y_true, y_pred, label_num ,args.verbos ,args.rank  ,args.device)
    if args.parallel=="ddp":
        f1=sync_acc(acc,y_pred,f1,device,args.rank,args)
        # y_pred=sync_acc(y_pred,device,args.rank)
    myPrinter(args,f"f1 {f1}")
    # 
    if args.rank==0 and  args.is_wandb=="y":
        wandb.log({"train_f1": f1,"epoch":epoch})
    return avg_loss


def compute_correct_num(y_pred, y_true):
    correct_num=0
    total_num=0
    for one_pred,one_true in zip(y_pred,y_true):
        if one_pred==one_true:
            correct_num+=1
        total_num+=1
    return correct_num,total_num

def gen_label_num(args,candidate_mode):
    if args.subset_to_check=="Train":
        is_train=True  
        label_num=args.batch_size*args.max_candidate_num 
    else:
        is_train=False   
        if candidate_mode=="end2end":
            label_num= args.max_candidate_num  +1
        else:
            label_num= args.max_candidate_num
    return label_num,is_train
def  search_threshold(pred_logits_list,y_pred,y_true , label_num ,args):
    best_threshold=0.5
    best_f1=-np.inf
    best_y_pred=None
    for threshold in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        new_y_pred=np.where(np.array(pred_logits_list)<threshold,args.max_candidate_num +1,y_pred)
        
        f1,pre,recall=calc_metric( y_true, new_y_pred.tolist(), label_num ,args.verbos ,args.rank  ,args.device)
        if f1>best_f1:
            best_f1=f1
            best_threshold=threshold
            best_y_pred=new_y_pred
    print(f"best threshold:{best_threshold}")
    return f1,pre,recall, best_threshold,best_y_pred


def  search_retrieval_weight_with_logits( y_pred,y_true ,  args,text_score_np,image_score_np,disambiguation_result_file_name,
                            total_query_id_list,total_order_list,total_entity_id_list,checkpoint_dir,run_dir,
                             data_path , fused_score_data_path,text_score_before_softmax_np,image_score_before_softmax_np):
    
    prediction_path=save_prediction(total_query_id_list,y_pred,y_true,total_order_list,total_entity_id_list,args.mode,
                             checkpoint_dir, run_dir,"val_"+disambiguation_result_file_name,text_score_np,image_score_np,text_score_before_softmax_np,image_score_before_softmax_np)
    # third_score_key =None
    
        # search_retrieval_score_field="bi_score"#TODO
        # search_disambiguation_score_field="disambiguation_text_score_before_softmax"
        # third_score_key="image_score"
    search_retrieval_score_field,search_disambiguation_score_field,third_score_key=gen_retrieval_field( args.use_image,args.use_text,args.search_retrieval_score_field )
    best_text_weight ,image_ratio=search_best_retrieval_weight_by_loading_file(prediction_path, data_path , fused_score_data_path,search_retrieval_score_field
                                                                   ,search_disambiguation_score_field,third_score_key  )
    
    return best_text_weight,image_ratio

def  search_modality_weight(pred_logits_list,y_pred,y_true , label_num ,args,text_score_np,image_score_np):
    best_weight=0.5
    best_f1=-np.inf
    best_y_pred=None
    for  text_weight in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
        
        score=text_score_np*text_weight+image_score_np*(1-text_weight)
        pred=np.argmax(score, axis=1)
         
        
        f1,pre,recall=calc_metric( y_true, pred.tolist(), label_num ,args.verbos ,args.rank  ,args.device)
        print(f"use text_weight: {text_weight}, f1:{f1}")
        if f1>best_f1:
            best_f1=f1
            best_y_pred=pred
            best_weight=text_weight
    print(f"  best_weight:{best_weight}")
    return best_f1,pre,recall, best_weight,best_y_pred

def gen_y_pred(accum_iter,model,dataloader,device, best_valid_f1,args,is_save_predict,epoch,best_epoch ,is_test,is_search_threshold,model_attribute,candidate_mode):
    y_true = []
    y_pred = []
    pred_logits_list=[]
    total_order_list=[]
    total_entity_id_list=[]
    total_query_id_list=[]
    text_score_np_list=[]
    image_score_np_list=[]
    text_score_before_softmax_np_list=[]
    image_score_before_softmax_np_list=[]
    activation= nn.Sigmoid()
    with tqdm(dataloader, unit="batch") as tepoch:
        label_num,is_train=gen_label_num(args, candidate_mode)
        for batch_idx, batch in enumerate(tepoch):
            output,labels,entity_id_list,query_id,loss,text_score,image_score,text_score_before_softmax,image_score_before_softmax=gen_output(batch,model,args.dataset_class, device,args,
                                                                                         is_train=is_train,model_attribute=model_attribute) 
            labels_numpy=labels.detach().cpu().numpy()
            if is_search_threshold:
                output=activation(output)
            pred_logits, preds = output.data.max(1)
            
            pred_logits_list.extend(pred_logits.tolist())
            y_pred.extend(preds.tolist())
            y_true.extend(labels_numpy.tolist())
            if text_score is not None:
                text_score_np_list.extend(text_score.detach().cpu().numpy())
                image_score_np_list.extend(image_score.detach().cpu().numpy())
                text_score_before_softmax_np_list.extend(text_score_before_softmax.detach().cpu().numpy())
                image_score_before_softmax_np_list.extend(image_score_before_softmax.detach().cpu().numpy())
            # live_acc = get_accuracy(y_pred, y_true)
            tepoch.set_postfix( )
            if args.rank==0:
                if is_save_predict=="y":
                    order_list=torch.argsort(output , dim=-1,descending=True).detach().cpu().numpy().tolist()
                    total_order_list.extend(order_list)
                    total_entity_id_list.extend(entity_id_list.detach().cpu().numpy().tolist())
                    total_query_id_list.extend(query_id.detach().cpu().numpy().tolist())
            # if batch_idx>10:
            # break  
    text_score_np=np.array(text_score_np_list)
    image_score_np=np.array(image_score_np_list)
    text_score_before_softmax_np =np.array(text_score_before_softmax_np_list)
    image_score_before_softmax_np =np.array(image_score_before_softmax_np_list)
    return y_true,y_pred,total_order_list,total_entity_id_list,total_query_id_list,label_num,pred_logits_list,text_score_np,image_score_np,text_score_before_softmax_np,image_score_before_softmax_np


def update_by_best_text_weight(text_score_np,image_score_np,best_text_weight):
    score=text_score_np*best_text_weight+image_score_np*(1-best_text_weight)
    y_pred=np.argmax(score, axis=1)
    sorted_indices = np.argsort(score)  # ascending order
    total_order_list = np.flip(sorted_indices,axis=1)  # descending order
     
    return y_pred,total_order_list.tolist()



def val_phase(accum_iter,model,dataloader,device, best_valid_f1,args,is_save_predict,epoch,best_epoch ,is_test=False,
              is_search_threshold=False ,best_threshold=None,model_attribute=None,is_search_modality_weight=False,best_text_weight=None,
              candidate_mode=None,disambiguation_result_file_name=None ,is_print=True ):
    myPrinter(args,"Val" + ":")
    model.eval()
    acc = AverageMeter('Acc', ':6.2f')
    
    y_true,y_pred,total_order_list,total_entity_id_list,total_query_id_list,label_num,pred_logits_list,text_score_np,image_score_np,text_score_before_softmax_np,image_score_before_softmax_np=gen_y_pred(accum_iter,model,dataloader,
                                                                                                                  device, best_valid_f1,args,
                                                                                                                  is_save_predict,epoch,best_epoch 
                                                                                                                  ,is_test,is_search_threshold,model_attribute,candidate_mode)
    
    if is_search_modality_weight :
        f1,pre,recall,best_text_weight,y_pred=search_modality_weight(pred_logits_list,y_pred,y_true , label_num ,args,text_score_np,image_score_np)
    elif is_search_threshold:
        f1,pre,recall,best_threshold,y_pred=search_threshold(pred_logits_list,y_pred,y_true , label_num ,args)
    
    else:
        if best_threshold is not None:
            y_pred= np.where(np.array(pred_logits_list)<best_threshold,args.max_candidate_num +1,y_pred).tolist()
        if best_text_weight is not None:
            y_pred,total_order_list=update_by_best_text_weight(text_score_np,image_score_np,best_text_weight)
        f1,pre,recall=calc_metric( y_true, y_pred, label_num ,args.verbos ,args.rank  ,args.device,is_print=is_print)

    if args.parallel=="ddp":
        f1=sync_acc(acc,y_pred,f1,device,args.rank,args)
    if args.rank==0:
        if is_save_predict=="y":
            save_prediction(total_query_id_list,y_pred,y_true,total_order_list,total_entity_id_list,args.mode,
                            args.checkpoint_dir,args.run_dir,disambiguation_result_file_name,text_score_np,image_score_np,text_score_before_softmax_np,image_score_before_softmax_np)
        if  args.mode=="hyper_search": 
            tune.report(val_f1=f1)
        if not is_test :
            if  args.is_wandb=="y":
                wandb.log({"dev_f1": f1,"epoch":epoch})
            if not is_search_threshold:
                best_valid_f1,best_epoch=save_model(best_valid_f1,f1,epoch,best_epoch,model,args)

    
    if args.early_stop and (epoch - best_epoch) >= args.early_stop:
        myPrinter(args,'early stop at epc {}'.format(epoch))
        is_early_stop=True
    else:
        is_early_stop=False
    return f1,f1,pre,recall,best_valid_f1,best_epoch,is_early_stop,best_threshold,best_text_weight


def search_best_weight(accum_iter,model,dataloaders,device, args, logger,epoch,best_epoch,model_attribute):
    f1,f1,pre,recall,best_valid_f1,best_epoch,is_early_stop,best_threshold,best_text_weight=val_phase( accum_iter,model,dataloaders["Val"],device,np.inf ,
                                                                                     args,'n' ,epoch,best_epoch,is_test=True,
                                                                                     is_search_threshold=False,is_search_modality_weight=True,
                                                                                     model_attribute=model_attribute,candidate_mode=args.dev_candidate_mode)
    return best_text_weight
        
        

def search_best_retrieval_weight(accum_iter,model,dataloaders,device, args, logger,epoch,best_epoch,model_attribute,disambiguation_result_file_name):
     
    y_true,y_pred,total_order_list,total_entity_id_list,total_query_id_list,label_num,pred_logits_list,text_score_np,image_score_np,text_score_before_softmax_np,image_score_before_softmax_np=gen_y_pred(accum_iter,
                                                                                                                                               model,
                                                                                                                                               dataloaders["Val"],
                                                                                                                  device, np.inf ,args,
                                                                                                                  "y",epoch,best_epoch 
                                                                                                                  ,True,False,model_attribute,args.dev_candidate_mode)
    best_retrieval_weight,image_ratio=search_retrieval_weight_with_logits( y_pred,y_true ,  args,text_score_np,image_score_np,disambiguation_result_file_name,
                            total_query_id_list,total_order_list,total_entity_id_list,args.checkpoint_dir,args.run_dir,
                             args.val_dir , args.fused_score_data_path,text_score_before_softmax_np,image_score_before_softmax_np)
        
    return best_retrieval_weight ,image_ratio    

def search_best_threshold(accum_iter,model,dataloaders,device, args, logger,epoch,best_epoch,model_attribute):
    f1,f1,pre,recall,best_valid_f1,best_epoch,is_early_stop,best_threshold,_=val_phase( accum_iter,model,dataloaders["Val"],device,np.inf ,
                                                                                     args,'n' ,epoch,best_epoch,is_test=True,
                                                                                     is_search_threshold=True,model_attribute=model_attribute,candidate_mode=args.dev_candidate_mode)
    return best_threshold
        
def test_phase(accum_iter,model,dataloaders,device, args, logger,epoch,best_epoch,model_attribute ,best_text_weight ):
    
    disambiguation_result_file_name=f"entity_link_reproduce_{args.disambiguation_result_postfix}.csv"
    myPrinter(args,'Final test ' )
    
    if args.mode in ["train","hyper_search"]:
        if args.parallel=="ddp":
            dist.barrier()
        model=resume_model( args.run_dir ,model,args) 
    best_threshold=None
     
    # best_threshold=search_best_threshold(accum_iter,model,dataloaders,device, args, logger,epoch,best_epoch,model_attribute)
    if   args.model_attribute in ["C_joint_adapt","C_joint"] and best_text_weight is None:
        best_text_weight=search_best_weight(accum_iter,model,dataloaders,device, args, logger,epoch,best_epoch,model_attribute)
    if   args.model_attribute in ["C_joint_adapt","C_joint","A_adapt_contrast" ,"B15"] and args.best_retrieval_weight is None:
        best_retrieval_weight,image_ratio=search_best_retrieval_weight(accum_iter,model,dataloaders,device, args, logger,epoch,best_epoch,
                                                           model_attribute,disambiguation_result_file_name)
    else:
        best_retrieval_weight=args.best_retrieval_weight
    # best_text_weight=0.8 
    if  args.model_attribute in ["C_joint_adapt","C_joint","A_adapt_contrast" ,"B15"] and best_retrieval_weight!=-1 :
        is_fuse_retrieval_score=True
        is_print_in_val_phase=False
    else:
        is_fuse_retrieval_score=False
        is_print_in_val_phase=True
    
    epoch_avg_acc,f1_score,pre,recall,best_valid_f1,best_epoch,is_early_stop,_,_=val_phase( accum_iter,model,dataloaders[args.subset_to_check],device,np.inf ,
                                                                                         args,args.save_predict ,epoch,best_epoch,is_test=True,
                                                                                         best_threshold=best_threshold,
                                                                                         model_attribute=model_attribute,best_text_weight=best_text_weight
                                                                                         ,candidate_mode=args.test_candidate_mode,
                                                                                         disambiguation_result_file_name=disambiguation_result_file_name ,is_print =is_print_in_val_phase)
    if args.mode in["mp_test","test","dry_run"]:
        run_dir=args.checkpoint_dir
        if args.checkpoint_dir is None:
            run_dir=args.run_dir
    else:
        run_dir=args.run_dir
    prediction_path=os.path.join(run_dir,disambiguation_result_file_name)
    if is_fuse_retrieval_score:
        f1_score,pre, recall=apply_best_retrieval_weight_without_filter(prediction_path,args.test_dir,args.fused_score_data_path,best_retrieval_weight, 
                                             args.weighted_score_data_path,args.filtered_data_path,args.use_image,
                                             args.use_text,args.search_retrieval_score_field,image_ratio) 
    # print(epoch_avg_acc)
    # print("#")
    print("f1 before filter: "+str(f1_score))
    try:
        logger.write("F1: {:.4f}, Precision: {:.4f}, Recall : {:.4f}".format(f1_score, pre, recall))
        if args.rank==0 and args.is_wandb=="y":
            wandb.log({"test_f1": f1_score})
    except Exception as e:
        print("error for logger.write")
        logger.write("error for logger.write")
        print(e)
        logger.write(e)
        
    
 
    check_dir=args.test_dir if args.subset_to_check=="Test" else args.val_dir
    if args.is_filter_by_attribute:
        if is_fuse_retrieval_score:
            f1_score=post_process_for_json( args.weighted_score_data_path,10,args.filtered_data_path,filter_method="m1")
        else:
            f1_score=post_process(prediction_path,check_dir,args.max_candidate_num, args.disambiguation_result_postfix)
  
    if args.rank==0 and args.is_wandb=="y":
        wandb.log({"filtered_test_f1": f1_score})  
    # print(f1_score)
    
    
        

def train(model,dataloaders,device,optimizer,args,logger,batch_size,scheduler,accum_iter,model_attribute):
    
    best_valid_f1 = 0
    best_epoch=0
    #Loss function
    criterion = nn.CrossEntropyLoss( )
    if args.rank==0:
        if args.is_wandb=="y":
            wandb.init(project=f'entity-linking-disambiguation-{args.mode}',config=args)
            wandb.run.name=f"{args.cur_run_id}-{wandb.run.name}"
     
    epoch=-1
    if args.mode in ["train","hyper_search"]:
        myPrinter(args,'Epoch {}/{}'.format(epoch, args.epoch))
        if args.parallel=="ddp":
            dataloaders["Val"].sampler.set_epoch(epoch)
        # epoch_avg_acc,f1,pre,recall,best_valid_f1,best_epoch,is_early_stop=val_phase(accum_iter, model,dataloaders["Val"],device,best_valid_f1 ,args,"n",epoch ,best_epoch )
        # myPrinter(args,"F1: {:.4f},   Accuracy: {:.4f}".format(f1,  epoch_avg_acc)) 
        for epoch in range(0, args.epoch):
            myPrinter(args,'-'*50)
            myPrinter(args,'Epoch {}/{}'.format(epoch, args.epoch))   
            if args.parallel=="ddp":
                dataloaders["Train"].sampler.set_epoch(epoch)     
                dataloaders["Val"].sampler.set_epoch(epoch)  
            epoch_avg_loss=train_phase(accum_iter,model,dataloaders["Train"],device,optimizer ,criterion,scheduler,args,epoch,model_attribute,model_attribute.is_contrastive)  
            epoch_avg_acc,f1,pre,recall,best_valid_f1,best_epoch,is_early_stop,_,_=val_phase(accum_iter, model,dataloaders["Val"],device,
                                                                                           best_valid_f1 ,args,"n",epoch ,best_epoch 
                                                                                           ,model_attribute=model_attribute,candidate_mode=args.dev_candidate_mode)
            c="F1: {:.4f},   Accuracy: {:.4f}, Loss: {:.4f}.".format(f1, epoch_avg_acc, epoch_avg_loss)
            myPrinter(args,c)
            # # logger.write(c)
            if is_early_stop:
                break
        inner_save_model(0,1,epoch,best_epoch,model,args,"last.pt")
    if args.parallel=="ddp":
        dataloaders["Test"].sampler.set_epoch(epoch)   
    test_phase(accum_iter,model,dataloaders,device, args, logger,epoch+1,epoch+1,model_attribute,args.best_text_weight)
     
    
    
def save_model(best_valid_f1,f1,epoch,best_epoch,model,args):
    if f1 > best_valid_f1:
        best_valid_f1,best_epoch=inner_save_model(best_valid_f1,f1,epoch,best_epoch,model,args,"base.pt")
    return best_valid_f1,best_epoch



def inner_save_model(best_valid_f1,f1,epoch,best_epoch,model,args,model_name):
     
    best_valid_f1 = f1
    best_epoch=epoch
    torch.save(model.state_dict(), os.path.join(args.run_dir,model_name) )
    myPrinter(args,'Model Saved!')
    return best_valid_f1,best_epoch