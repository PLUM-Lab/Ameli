import requests
from torch import Tensor, device
from typing import List, Callable
from tqdm.autonotebook import tqdm
import sys
import importlib
import os
import torch
import numpy as np
import queue
import logging
from typing import Dict, Optional, Union
from pathlib import Path
from sentence_transformers.util import cos_sim


def gen_averaged_score_for_one_query(entity_image_num_list,cos_scores,mode):
    start_idx=0
    entity_score_list=[]
    for entity_image_num in entity_image_num_list:
        end_idx=start_idx+entity_image_num
        one_result=torch.max(cos_scores[:,start_idx:end_idx],dim=1)[0]
        if mode!="average":
            score_for_this_entity=torch.max(one_result)
        else:
            score_for_this_entity=torch.mean(one_result)
        entity_score_list.append(score_for_this_entity)
        start_idx=end_idx 
    return torch.unsqueeze(torch.Tensor(entity_score_list),0)

        
def semantic_search_by_list_attribute_of_one_query(query_embeddings: Tensor,
                    corpus_embeddings: Tensor,
                    query_chunk_size: int = 100,
                    corpus_chunk_size: int = 500000,
                    entity_image_num_list = [],
                    top_k: int = 10,
                    score_function: Callable[[Tensor, Tensor], Tensor] = cos_sim,mode="average"):
    """
    This function performs a cosine similarity search between a list of query embeddings  and a list of corpus embeddings.
    It can be used for Information Retrieval / Semantic Search for corpora up to about 1 Million entries.

    :param query_embeddings: A 2 dimensional tensor with the query embeddings.
    :param corpus_embeddings: A 2 dimensional tensor with the corpus embeddings.
    :param query_chunk_size: Process 100 queries simultaneously. Increasing that value increases the speed, but requires more memory.
    :param corpus_chunk_size: Scans the corpus 100k entries at a time. Increasing that value increases the speed, but requires more memory.
    :param top_k: Retrieve top k matching entries.
    :param score_function: Function for computing scores. By default, cosine similarity.
    :return: Returns a list with one entry for each query. Each entry is a list of dictionaries with the keys 'corpus_id' and 'score', sorted by decreasing cosine similarity scores.
    """

    if isinstance(query_embeddings, (np.ndarray, np.generic)):
        query_embeddings = torch.from_numpy(query_embeddings)
    elif isinstance(query_embeddings, list):
        query_embeddings = torch.stack(query_embeddings)

    if len(query_embeddings.shape) == 1:
        query_embeddings = query_embeddings.unsqueeze(0)

    if isinstance(corpus_embeddings, (np.ndarray, np.generic)):
        corpus_embeddings = torch.from_numpy(corpus_embeddings)
    elif isinstance(corpus_embeddings, list):
        corpus_embeddings = torch.stack(corpus_embeddings)


    #Check that corpus and queries are on the same device
    if corpus_embeddings.device != query_embeddings.device:
        query_embeddings = query_embeddings.to(corpus_embeddings.device)

    queries_result_list = [[] ]

    for query_start_idx in range(0, len(query_embeddings), query_chunk_size):
        # Iterate over chunks of the corpus
        for corpus_start_idx in range(0, len(corpus_embeddings), corpus_chunk_size):
            # Compute cosine similarities
            cos_scores = score_function(query_embeddings[query_start_idx:query_start_idx+query_chunk_size], corpus_embeddings[corpus_start_idx:corpus_start_idx+corpus_chunk_size])
            
            merged_scores=gen_averaged_score_for_one_query(entity_image_num_list,cos_scores, mode)
            
            
            # Get top-k scores
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(merged_scores, min(top_k, len(merged_scores[0])), dim=1, largest=True, sorted=False)
            cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
            cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()

            for query_itr in range(len(merged_scores)):
                for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
                    corpus_id = corpus_start_idx + sub_corpus_id
                    query_id = query_start_idx + query_itr
                    queries_result_list[query_id].append({'corpus_id': corpus_id, 'score': score})

    #Sort and strip to top_k results
    for idx in range(len(queries_result_list)):
        queries_result_list[idx] = sorted(queries_result_list[idx], key=lambda x: x['score'], reverse=True)
        queries_result_list[idx] = queries_result_list[idx][0:top_k]

    return queries_result_list