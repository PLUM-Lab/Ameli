"""
This examples show how to train a Cross-Encoder for the MS Marco dataset (https://github.com/microsoft/MSMARCO-Passage-Ranking).
The query and the passage are passed simoultanously to a Transformer network. The network then returns
a score between 0 and 1 how relevant the passage is for a given query.
The resulting Cross-Encoder can then be used for passage re-ranking: You retrieve for example 100 passages
for a given query, for example with ElasticSearch, and pass the query+retrieved_passage to the CrossEncoder
for scoring. You sort the results then according to the output of the CrossEncoder.
This gives a significant boost compared to out-of-the-box ElasticSearch / BM25 ranking.
Running this script:
python train_cross-encoder.py
"""
import torch
from torch.utils.data import DataLoader
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from sentence_transformers import InputExample
import logging
from torch import nn
from datetime import datetime
import gzip
import os
import tarfile
import tqdm
from retrieval.data_util.core.gen_data import load_ameli_data_to_train_cross_encoder
from retrieval.eval.eval_cross_encoder_mocheg import test_cross_encoder 
from retrieval.model.matcher import AmeliCrossEncoder

from retrieval.utils.enums import TrainingAttribute

from util.common_util import setup_with_args
# from verification.util.data_util import get_loss_weight_fun1, get_loss_weights


def load_model(args,model_name,max_seq_length,train_attribute):
    # Load our embedding model
    # model =  CrossEncoder(model_name, num_labels=1, max_length=train_attribute.max_seq_length)
    model = AmeliCrossEncoder(model_name, num_labels=1, max_length=train_attribute.max_seq_length)
    
    # if args.use_pre_trained_model:
    #     logging.info("use pretrained SBERT model")
    #     model = MultiMediaSentenceTransformer(model_name)
    #     model.max_seq_length = max_seq_length
    #     # if args.media=="txt":
    #     #     model.max_seq_length = max_seq_length
    # else:
    #     logging.info("Create new SBERT model")
    #     word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
    #     pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), args.pooling)
    #     model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return model 

def set_up():
    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    #### /print debug information to stdout
    
    
def train_cross_encoder(args):
    #First, we define the transformer model we want to fine-tune
    train_attribute=TrainingAttribute[args.train_config]
    if args.model_name is None:
        args.model_name =train_attribute.model_name
    if args.train_batch_size is None:
        args.train_batch_size = train_attribute.batch_size
    if args.epochs is None:
        args.epochs  =train_attribute.epoch
    if args.media is None:
        args.media=train_attribute.media 
    if args.max_seq_length is None:
        args.max_seq_length=train_attribute.max_seq_length
    
    # The  model we want to fine-tune
    model_name =args.model_name #'distilbert-base-uncased'
    train_batch_size = args.train_batch_size           #Increasing the train batch size improves the model performance, but requires more GPU memory
    max_seq_length = args.max_seq_length            #Max length for passages. Increasing it, requires more GPU memory
    ce_score_margin = args.ce_score_margin             #Margin for the CrossEncoder score between negative and positive passages
    num_negs_per_system = args.num_negs_per_system         # We used different systems to mine hard negatives. Number of hard negatives to add from each system
    num_epochs = args.epochs     
    
      
    model_save_path,args=setup_with_args(args,'retrieval/output/runs_3','train_cross-encoder-{}-{}'.format(model_name.replace("/", "-"), datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    # model_save_path = 'output/training_ms-marco_cross-encoder-'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # We train the network with as a binary label task
    # Given [query, passage] is the label 0 = irrelevant or 1 = relevant?
    # We use a positive-to-negative ratio: For 1 positive sample (label 1) we include 4 negative samples (label 0)
    # in our training setup. For the negative samples, we use the triplets provided by MS Marco that
    # specify (query, positive sample, negative sample).
    pos_neg_ration = 4
    # Maximal number of training samples we want to use
    max_train_samples = 2e7
    ### Now we read the MS Marco dataset
    data_folder = args.train_data_folder
    # os.makedirs(data_folder, exist_ok=True)
    model=load_model(args,model_name,max_seq_length,train_attribute)
    
    #We set num_labels=1, which predicts a continous score between 0 and 1
    # train_dataset,corpus =load_data_for_ameli( args,ce_score_margin,num_negs_per_system)
    train_dataset,val_dict_dataset,pos_weight=load_ameli_data_to_train_cross_encoder(data_folder,args.val_data_folder,pos_neg_ration,
                                                                           max_train_samples,args,args.media,args.mode)

    # We create a DataLoader to load our train samples
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)

    # We add an evaluator, which evaluates the performance during training
    # It performs a classification task and measures scores like F1 (finding relevant passages) and Average Precision
    evaluator = CERerankingEvaluator(val_dict_dataset, name='train-eval')

    # Configure the training
    warmup_steps = 5000
    logging.info("Warmup-steps: {}".format(warmup_steps))
    device=torch.device("cuda")
     
    logging.info(f"normed:{pos_weight}")
    loss_fct= nn.BCEWithLogitsLoss(pos_weight= torch.tensor(pos_weight) )
    # Train the model
    model.fit(train_dataloader=train_dataloader,
            evaluator=evaluator,
            epochs=num_epochs,
            evaluation_steps=10000,
            warmup_steps=warmup_steps,
            output_path=model_save_path,
            optimizer_params = {'lr': args.lr},
            use_amp=True,
            loss_fct=loss_fct)

    #Save latest model
    postfix="-latest"
    model.save(model_save_path,'-latest')
    
    
    match_path=os.path.join(model_save_path,f"match{postfix}.pt")
    feature_extractor_path=os.path.join(model_save_path,f"feature_extractor{postfix}.pt")
    print(f"begin to test {feature_extractor_path}")
    test_cross_encoder(args,match_path,feature_extractor_path)