import json
 
import gzip
import os
import torch
from tqdm import tqdm
from retrieval.search.image_search import ImageSearcher
from retrieval.search.image_search_for_object_with_multiple_image import ImageSearcherForObjectWithMultipleImage
 
from retrieval.search.semantic_search import SemanticSearcher
from retrieval.utils.config import config
 
from nltk import tokenize
from itertools import zip_longest
from retrieval.utils.metrics import ImageScorer, TextScorer
from retrieval.utils.saver import Saver
from util.data_util.data_util import load_wikidiverse_data
from util.data_util.entity_linking_data_util import load_entity_linking_data,EntityLinkingSaver
from util.entity_linking_ground_truth_data_util import EntityLinkingGroundTruthSaver, load_entity_linking_ground_truth_data
 


def training_loop(args,logger,rank=0):
                  #Number of passages we want to retrieve with the bi-encoder
    saver=Saver()
    if args.dataset=="wikidiverse":
        dataloader,corpus_dict=load_wikidiverse_data(args.dataset_dir,args.entity_dir,args.image_dir)
    elif args.dataset=="entity_linking":
        if args.mode=="retrieve_by_gold":
            dataloader,corpus_dict=load_entity_linking_ground_truth_data(args.dataset_dir, args.corpus_dir,args.corpus_pickle_dir  ,args.corpus_image_dir)
            saver=EntityLinkingGroundTruthSaver(args.dataset_dir,args.top_k)
        else:
            dataloader,corpus_dict=load_entity_linking_data(args.dataset_dir, args.corpus_dir,args.corpus_pickle_dir  ,args.corpus_image_dir,args.dataset_image_dir,args.text_base)
            saver=EntityLinkingSaver(args.dataset_dir,args.top_k,args.text_base)
     
    if args.media=="txt":
        text_retrieve(args,corpus_dict,dataloader,saver)
    elif   args.media=="img":
        image_retrieve(args,logger,corpus_dict,dataloader,saver)
    # elif args.media=="img_txt":
    #     text_retrieve(args,relevant_document_text_list,dataloader,saver)
    #     image_retrieve(args,relevant_document_img_list,dataloader,saver)
 
        
    

 
def group_elements(n, iterable, padvalue=""):
    return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)



def gen_relevant_document_text_paragraph_list(sent_num,relevant_document_text_list ):
    relevant_document_text_paragraph_list =[]
    i=0  
    for relevant_document_text in relevant_document_text_list:
        i+=1
        # print(i)
        relevant_document_text_sent_list=tokenize.sent_tokenize(relevant_document_text)
        for output in group_elements(sent_num,relevant_document_text_sent_list):
            relevant_document_text_paragraph_list.append(" ".join(output))
        
    return relevant_document_text_paragraph_list
    
    
def gen_document_text_list(corpus_dict):
    relevant_document_text_list=[]
    corpus_text_corpus_id_dict={}
    for corpus_id,entity_object in corpus_dict.items():
        relevant_document_text_list.append(entity_object.text)
        corpus_text_corpus_id_dict[entity_object.text]=corpus_id
    return relevant_document_text_list,corpus_text_corpus_id_dict,list(corpus_dict.keys())

def text_retrieve(args,corpus_dict,dataloader,saver ):
    corpus_text_list,corpus_text_corpus_id_dict,corpus_id_list=gen_document_text_list(corpus_dict )
    # relevant_document_text_paragraph_list=gen_relevant_document_text_paragraph_list(args.sent_num,relevant_document_text_list)
    searcher=SemanticSearcher(args.bi_encoder_checkpoint,args.cross_encoder_checkpoint,args.no_rerank)
    searcher.encode_corpus(corpus_text_list)
    scorer=TextScorer(args.bi_encoder_checkpoint,args.score_by_fine_tuned)
    valid_num=1
    precision,recall=0,0
    save_step=500
    length=len(dataloader)
    for iters in tqdm(range(length)):
        query_id,query_text,query_image_path_list,gold_candidates ,gold_entity_category =dataloader.dataset[iters]
        
        semantic_results=searcher.search(query_text,args.top_k   )
        
        if len(gold_candidates)>0  :
            valid_num+=1
            # cur_precision,cur_recall=scorer.precision_recall_by_similarity(semantic_results,corpus_text_list,gold_candidates)
            # precision+=cur_precision
            # recall+=cur_recall
        if  iters%100==0:
            print(f"{iters}/{length}: {precision/valid_num}, {recall/valid_num}")
        if config.verbose==True:
            print(f"claim:{query_text},semantic_results:{semantic_results}")
            
        saver.add_retrieved_text(query_id,query_text,semantic_results,corpus_text_list,corpus_id_list,corpus_dict,corpus_text_corpus_id_dict)
        if iters % save_step==0:
            saver.save(args.csv_out_dir)
            print(f"just save file after {iters} iters")
    if valid_num>1:
        precision/=(valid_num-1)
        recall/=(valid_num-1)
    saver.save(args.csv_out_dir)
    print(f"{precision}, {recall},{compute_f1(precision, recall)}")
 
 
def compute_f1(precision, recall):
    if precision+recall !=0:
        f1=2*precision*recall/(precision+recall)
    else:
        f1=0
    return f1 

def gen_claim(claim):
    claim_token_list=tokenize.word_tokenize(claim)
    safe_clip_text_len=77 
    if len(claim_token_list)>safe_clip_text_len:
        claim_token_list=claim_token_list[0:safe_clip_text_len]
        # claim=" ".join(claim_token_list)
    return claim 
    
def gen_retrieved_imgs( hits,relevant_document_list):
    retrieved_document_list=[]
    for hit in hits:
        retrieved_document_list.append(relevant_document_list[hit['corpus_id']])
    
    return ";".join(retrieved_document_list)


def gen_metrics(top_k):
    pass 
def gen_acc_recall(top_k,labels,nns):    
    accuracy = -1
    recall_at = -1
 
    # get recall values
    
    x = []
    y = []
    for i in range(1, top_k):
        temp_y = 0.0
        for label, top in zip(labels, nns):
            if label in top[:i]:
                temp_y += 1
        if len(labels) > 0:
            temp_y /= len(labels)
        x.append(i)
        y.append(temp_y)
    # plt.plot(x, y)
    accuracy = y[0]
    recall_at = y[-1]
    print("biencoder accuracy: %.4f" % accuracy)
    print("biencoder recall@%d: %.4f" % (top_k, y[-1]))
    
 
def image_retrieve(args,logger,entity_dict,dataloader ,saver):
    if args.dataset!="entity_linking":
        image_searcher=ImageSearcher(args.image_encoder_checkpoint,logger)
    else:
        image_searcher=ImageSearcherForObjectWithMultipleImage(args.image_encoder_checkpoint,logger,args.corpus_pickle_dir,
                                                               args.is_only_in_category,args.img_fuse_mode)
    if args.use_precomputed_embeddings=="y":
        use_precomputed_embeddings_flag=True 
    else:
        use_precomputed_embeddings_flag=False 
    image_searcher.encode_corpus(entity_dict ,args.corpus_image_dir,use_precomputed_embeddings_flag)
    # scorer=ImageScorer(args.image_encoder_checkpoint,args.score_by_fine_tuned)
    precision,recall=0,0
    right_num_at_k_dict={1:0,10:0,20:0,50:0,100:0}
    retrieved_imgs_list=[]
    valid_num=1
    save_step=500
    start_id=-1
    # start_id= 10000  #10000 
    length=len(dataloader)
    for iters in tqdm(range(length)):
        if iters<=start_id:
            continue 
        id,text, img_path_list, entity_candidate_name_list,gold_entity_category  =dataloader.dataset[iters]
        if args.dataset!="entity_linking":
            img_path=img_path_list[0]
            if os.path.exists(img_path):
                semantic_results,retrieved_entity_name_list=image_searcher.search(img_path,args.top_k   )
        else:
            
            semantic_results,retrieved_entity_name_list,score_dict=image_searcher.search(img_path_list,gold_entity_category,args.top_k   )
        for k,num in right_num_at_k_dict.items():
            for entity_candidate_name  in entity_candidate_name_list:
                if entity_candidate_name in retrieved_entity_name_list[:k]:
                    right_num_at_k_dict[k]+=1
        saver.add_retrieved_image(id,img_path_list,semantic_results,retrieved_entity_name_list, entity_dict,score_dict )  
            
        if iters%100==0:
            print(f"{iters}: {right_num_at_k_dict} ")
        if iters % save_step==0:
            saver.save(args.csv_out_dir)
            print(f"just save file after {iters} iters")
    saver.save(args.csv_out_dir)
    recall_at_k_dict={}
    for k,num in right_num_at_k_dict.items():
        
        recall_at_k_dict[k]=num/length
    acc=right_num_at_k_dict[1]/length
    # saver.insert_and_save(args.csv_out_dir,"img_evidences",retrieved_imgs_list)
    print(f"{acc}, {recall_at_k_dict} ")

   
    

    
    
    
    
    

    
