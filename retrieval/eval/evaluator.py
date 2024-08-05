from sentence_transformers import SentenceTransformer, LoggingHandler, util, models, evaluation
from typing import List, Tuple, Dict, Set, Callable
import torch
from torch import Tensor
from PIL import Image
import logging
import numpy as np
import os
import csv

logger = logging.getLogger(__name__)
class MultiMediaInformationRetrievalEvaluator(evaluation.InformationRetrievalEvaluator):
    def __init__(self,
                 queries: Dict[str, str],  #qid => query
                 corpus: Dict[str, str],  #cid => doc
                 relevant_docs: Dict[str, Set[str]],  #qid => Set[cid]
                 corpus_chunk_size: int = 50000,
                 mrr_at_k: List[int] = [10],
                 ndcg_at_k: List[int] = [10],
                 accuracy_at_k: List[int] = [1, 3, 5, 10],
                 precision_recall_at_k: List[int] = [1, 3, 5, 10],
                 map_at_k: List[int] = [100],
                 show_progress_bar: bool = False,
                 batch_size: int = 32,
                 name: str = '',
                 write_csv: bool = True,
                 score_functions: List[Callable[[Tensor, Tensor], Tensor] ] = {'cos_sim': util.cos_sim, 'dot_score': util.dot_score},       #Score function, higher=more similar
                 main_score_function: str = None,
                 corpus_embed=None,
                 save_query_result_path=None
                 ,media=None
                 ):
        super().__init__(queries,corpus,
                 relevant_docs,
                 corpus_chunk_size,
                 mrr_at_k,
                 ndcg_at_k,
                 accuracy_at_k,
                 precision_recall_at_k,
                 map_at_k,
                 show_progress_bar,
                 batch_size,
                 name,
                 write_csv,
                 score_functions,
                 main_score_function 
                 )
        if media=="img":
            query_image_list=[]
            for query_path in self.queries:
                query_image=Image.open(query_path)
                query_image_list.append(query_image)
            self.queries=query_image_list
        
            
        self.corpus_embed=corpus_embed
        self.save_query_result_path=save_query_result_path
        
    def compute_metrices(self, model, corpus_model = None, corpus_embeddings: Tensor = None):
        if corpus_embeddings==None:
            corpus_embeddings=self.corpus_embed
        return super().compute_metrices(model,corpus_model,corpus_embeddings)
    
    
    def compute_metrics(self, queries_result_list: List[object]):
        if self.save_query_result_path !=None:
            save(queries_result_list,self.save_query_result_path)
        return super().compute_metrics(queries_result_list)
    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1, *args, **kwargs) -> float:
        if epoch != -1:
            out_txt = " after epoch {}:".format(epoch) if steps == -1 else " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("Information Retrieval Evaluation on " + self.name + " dataset" + out_txt)

        scores = self.compute_metrices(model, *args, **kwargs)

        # Write results to disc
        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                fOut = open(csv_path, mode="w", encoding="utf-8")
                fOut.write(",".join(self.csv_headers))
                fOut.write("\n")

            else:
                fOut = open(csv_path, mode="a", encoding="utf-8")

            output_data = [epoch, steps]
            for name in self.score_function_names:
                for k in self.accuracy_at_k:
                    output_data.append(scores[name]['accuracy@k'][k])

                for k in self.precision_recall_at_k:
                    output_data.append(scores[name]['precision@k'][k])
                    output_data.append(scores[name]['recall@k'][k])

                for k in self.mrr_at_k:
                    output_data.append(scores[name]['mrr@k'][k])

                for k in self.ndcg_at_k:
                    output_data.append(scores[name]['ndcg@k'][k])

                for k in self.map_at_k:
                    output_data.append(scores[name]['map@k'][k])

            fOut.write(",".join(map(str, output_data)))
            fOut.write("\n")
            fOut.close()

        if self.main_score_function is None:
            return max([scores[name]['recall@k'][1] for name in self.score_function_names])
        else:
            return scores[self.main_score_function]['recall@k'][1]
    
def save(queries_result_list,save_query_result_path):
    torch.save( queries_result_list ,save_query_result_path)    
    
    
     