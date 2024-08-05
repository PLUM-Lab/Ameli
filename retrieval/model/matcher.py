import torch
import torch.nn as nn
import torch.nn.functional as F

from retrieval.model.model import MultiMediaSentenceTransformer

from .position_encoding import PositionEmbeddingSine, PositionEmbeddingLearned
from .transformer import TransformerEncoder, TransformerEncoderLayer
from sentence_transformers.cross_encoder import CrossEncoder
from typing import Dict, Tuple, Type, Callable, List, Union
import transformers
from retrieval.utils.retrieval_util import EasyDict
from torchvision import transforms
from sentence_transformers import    util,SentenceTransformer
from sentence_transformers.util import batch_to_device
import os 
from tqdm.autonotebook import tqdm, trange
import numpy as np 
import logging
logger = logging.getLogger(__name__)
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel

def get_transforms(crop_size=224):
    train_transform, test_transform = [], []
    train_transform.extend([
        transforms.RandomResizedCrop(size=crop_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()])
    test_transform.append(transforms.Resize((256, 256)))
    test_transform.append(transforms.CenterCrop(size=224))
    test_transform.append(transforms.ToTensor())
    return transforms.Compose(train_transform), transforms.Compose(test_transform)
 
        
        


class MatchERT(nn.Module):
    def __init__(self, d_global=128, d_model=768, nhead=4, num_encoder_layers=6, dim_feedforward=1024, output_classification=True,dropout=0, activation="relu", normalize_before=False):
        super(MatchERT, self).__init__()
        assert (d_model % 2 == 0)
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.pos_encoder = PositionEmbeddingSine(d_model//2, normalize=True, scale=2.0)
        self.seg_encoder = nn.Embedding(4, d_model)
        if output_classification:
            self.classifier = nn.Linear(d_model, 1)
        self.output_classification=output_classification
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def change_shape(self,src_local):
        bsize,seq_len,dim=src_local.size()
        src_local=src_local.permute(0, 2, 1)
        if src_local.shape[-1]==50: 
            src_local=src_local[:,:,1:]
        src_local=torch.reshape(src_local,(bsize,dim,7,-1))
        return src_local
    
    def forward(self, src_global, src_local, tgt_global, tgt_local, normalize=False):
        ##########################################################
        # Global features are not used in the final model
        # Keep the API here for future study
        ##########################################################
        # src_global = self.remap(src_global)
        # tgt_global = self.remap(tgt_global)    
        # if normalize:
        #     src_global = F.normalize(src_global, p=2, dim=-1)
        #     tgt_global = F.normalize(tgt_global, p=2, dim=-1)
        src_local=self.change_shape(src_local)
        tgt_local=self.change_shape(tgt_local)
        
        bsize, fsize, h, w = src_local.size()
        pos_embed  = self.pos_encoder(src_local.new_ones((1, h, w))).expand(bsize, fsize, h, w)
        cls_embed  = self.seg_encoder(src_local.new_zeros((bsize, 1), dtype=torch.long)).permute(0, 2, 1)
        sep_embed  = self.seg_encoder(src_local.new_ones((bsize, 1),  dtype=torch.long)).permute(0, 2, 1)
        src_local  = src_local.flatten(2)    + self.seg_encoder(2 * src_local.new_ones((bsize, 1),  dtype=torch.long)).permute(0, 2, 1) + pos_embed.flatten(2)
        tgt_local  = tgt_local.flatten(2)    + self.seg_encoder(3 * src_local.new_ones((bsize, 1),  dtype=torch.long)).permute(0, 2, 1) + pos_embed.flatten(2)
        # src_global = src_global.unsqueeze(1) + self.seg_encoder(4 * src_local.new_ones((bsize, 1),  dtype=torch.long)).permute(0, 2, 1)
        # tgt_global = tgt_global.unsqueeze(1) + self.seg_encoder(5 * src_local.new_ones((bsize, 1),  dtype=torch.long)).permute(0, 2, 1)
        
        # global features were not used in the final model
        input_feats = torch.cat([cls_embed, src_local, sep_embed, tgt_local], -1).permute(2, 0, 1)
        logits = self.encoder(input_feats)[0]
        if self.output_classification:
            return self.classifier(logits).view(-1)
        else:
            return logits
    
    
    
    
    
class AmeliCrossEncoder(CrossEncoder):
    def __init__(self, extractor_model_name:str, num_labels:int = None, max_length:int = None, device:str = None, tokenizer_args:Dict = {},
                  automodel_args:Dict = {}, default_activation_function = None):
        self.config=EasyDict()
        self.config.num_labels = 1
    
        self.model=MatchERT()
        # self.feature_extractor= MultiMediaSentenceTransformer(extractor_model_name)
        self.feature_extractor= CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32") 
 
        self.max_length = max_length
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print("Use pytorch device: {}".format(device))
        self._target_device = torch.device(device)
        # self.default_activation_function = nn.Sigmoid()
        self.train_transform, self.test_transform = get_transforms()
        self.default_activation_function = util.import_from_string("torch.nn.modules.linear.Identity")()
    
    def tokenize(self,img_list):
        inputs = self.processor(text=None, images=img_list, return_tensors="pt", padding=True,truncation=True)
        return inputs
  
    def smart_batching_collate(self, batch):
        """
        Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model
        Here, batch is a list of tuples: [(tokens, label), ...]

        :param batch:
            a batch from a SmartBatchingDataset
        :return:
            a batch of tensors for the model
        """
        num_texts = len(batch[0].texts)
        texts = [[] for _ in range(num_texts)]
        labels = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text)

            labels.append(example.label)

        labels = torch.tensor(labels, dtype=torch.float if self.config.num_labels == 1 else torch.long).to(self._target_device)

        sentence_features = []
        for idx in range(num_texts):
            tokenized = self.tokenize(texts[idx])
            sentence_features.append(tokenized)

        return sentence_features, labels
 

    def _first_module(self):
        """Returns the first module of this sequential embedder"""
        return self._modules[next(iter(self._modules))]
    
    
    def smart_batching_collate_text_only(self, batch):
        """
        Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model
        Here, batch is a list of tuples: [(tokens, label), ...]

        :param batch:
            a batch from a SmartBatchingDataset
        :return:
            a batch of tensors for the model
        """
        num_texts = len(batch[0])
        texts = [[] for _ in range(num_texts)]
        labels = []

        for example in batch:
            for idx, text in enumerate(example):
                texts[idx].append(text)

           

        

        sentence_features = []
        for idx in range(num_texts):
            tokenized = self.tokenize(texts[idx])
            sentence_features.append(tokenized)

        return sentence_features 
    
    def fit(self,
            train_dataloader,
            evaluator= None,
            epochs: int = 1,
            loss_fct = None,
            activation_fct = nn.Identity(),
            scheduler: str = 'WarmupLinear',
            warmup_steps: int = 10000,
            optimizer_class = torch.optim.AdamW,
            optimizer_params: Dict[str, object] = {'lr': 2e-5},
            weight_decay: float = 0.01,
            evaluation_steps: int = 0,
            output_path: str = None,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            use_amp: bool = False,
            callback: Callable[[float, int, int], None] = None,
            show_progress_bar: bool = True
            ):
        """
        Train the model with the given training objective
        Each training objective is sampled in turn for one batch.
        We sample only as many batches from each objective as there are in the smallest one
        to make sure of equal training with each dataset.

        :param train_dataloader: DataLoader with training InputExamples
        :param evaluator: An evaluator (sentence_transformers.evaluation) evaluates the model performance during training on held-out dev data. It is used to determine the best model that is saved to disc.
        :param epochs: Number of epochs for training
        :param loss_fct: Which loss function to use for training. If None, will use nn.BCEWithLogitsLoss() if self.config.num_labels == 1 else nn.CrossEntropyLoss()
        :param activation_fct: Activation function applied on top of logits output of model.
        :param scheduler: Learning rate scheduler. Available schedulers: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        :param warmup_steps: Behavior depends on the scheduler. For WarmupLinear (default), the learning rate is increased from o up to the maximal learning rate. After these many training steps, the learning rate is decreased linearly back to zero.
        :param optimizer_class: Optimizer
        :param optimizer_params: Optimizer parameters
        :param weight_decay: Weight decay for model parameters
        :param evaluation_steps: If > 0, evaluate the model using evaluator after each number of training steps
        :param output_path: Storage path for the model and evaluation files
        :param save_best_model: If true, the best model (according to evaluator) is stored at output_path
        :param max_grad_norm: Used for gradient normalization.
        :param use_amp: Use Automatic Mixed Precision (AMP). Only for Pytorch >= 1.6.0
        :param callback: Callback function that is invoked after each evaluation.
                It must accept the following three parameters in this order:
                `score`, `epoch`, `steps`
        :param show_progress_bar: If True, output a tqdm progress bar
        """
        train_dataloader.collate_fn = self.smart_batching_collate

        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        self.model.to(self._target_device)
        self.feature_extractor.to(self._target_device)

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)

        self.best_score = -9999999
        num_train_steps = int(len(train_dataloader) * epochs)

        # Prepare optimizers
        param_optimizer = list(self.model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

        if isinstance(scheduler, str):
            scheduler = SentenceTransformer._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)

        if loss_fct is None:
            loss_fct = nn.BCEWithLogitsLoss() if self.config.num_labels == 1 else nn.CrossEntropyLoss()


        skip_scheduler = False
        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            training_steps = 0
            self.model.zero_grad()
            self.model.train()

            for features, labels in tqdm(train_dataloader, desc="Iteration", smoothing=0.05, disable=not show_progress_bar):
                labels = labels.to(self._target_device)
                features = list(map(lambda batch: batch_to_device(batch, self._target_device), features))
                if use_amp:
                    with autocast():
                        model_feature =[ self.feature_extractor.vision_model( **feature  ).last_hidden_state  for feature in features] 
                        model_predictions=self.model(src_global=None, src_local=model_feature[0],  tgt_global=None, tgt_local=model_feature[1])
                        logits = activation_fct(model_predictions)
                        if self.config.num_labels == 1:
                            logits = logits.view(-1)
                        loss_value = loss_fct(logits, labels)

                    scale_before_step = scaler.get_scale()
                    scaler.scale(loss_value).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()

                    skip_scheduler = scaler.get_scale() != scale_before_step
                else:
                    model_predictions = self.model(**features, return_dict=True)
                    logits = activation_fct(model_predictions.logits)
                    if self.config.num_labels == 1:
                        logits = logits.view(-1)
                    loss_value = loss_fct(logits, labels)
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()

                optimizer.zero_grad()

                if not skip_scheduler:
                    scheduler.step()

                training_steps += 1

                if evaluator is not None and evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    self._eval_during_training(evaluator, output_path, save_best_model, epoch, training_steps, callback)

                    self.model.zero_grad()
                    self.model.train()

            if evaluator is not None:
                self._eval_during_training(evaluator, output_path, save_best_model, epoch, -1, callback)
    def predict(self, sentences: List[List[str]],
               batch_size: int = 32,
               show_progress_bar: bool = None,
               num_workers: int = 0,
               activation_fct = None,
               apply_softmax = False,
               convert_to_numpy: bool = True,
               convert_to_tensor: bool = False
               ):
        """
        Performs predicts with the CrossEncoder on the given sentence pairs.

        :param sentences: A list of sentence pairs [[Sent1, Sent2], [Sent3, Sent4]]
        :param batch_size: Batch size for encoding
        :param show_progress_bar: Output progress bar
        :param num_workers: Number of workers for tokenization
        :param activation_fct: Activation function applied on the logits output of the CrossEncoder. If None, nn.Sigmoid() will be used if num_labels=1, else nn.Identity
        :param convert_to_numpy: Convert the output to a numpy matrix.
        :param apply_softmax: If there are more than 2 dimensions and apply_softmax=True, applies softmax on the logits output
        :param convert_to_tensor:  Conver the output to a tensor.
        :return: Predictions for the passed sentence pairs
        """
        input_was_string = False
        if isinstance(sentences[0], str):  # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        inp_dataloader = DataLoader(sentences, batch_size=batch_size, collate_fn=self.smart_batching_collate_text_only, num_workers=num_workers, shuffle=False)

        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)

        iterator = inp_dataloader
        if show_progress_bar:
            iterator = tqdm(inp_dataloader, desc="Batches")

        if activation_fct is None:
            activation_fct = self.default_activation_function

        pred_scores = []
        self.model.eval()
        self.model.to(self._target_device)
        with torch.no_grad():
            for features in iterator:
               
                features = list(map(lambda batch: batch_to_device(batch, self._target_device), features))
                model_feature =[ self.feature_extractor.vision_model( **feature  ).last_hidden_state  for feature in features] 
                model_predictions=self.model(src_global=None, src_local=model_feature[0],  tgt_global=None, tgt_local=model_feature[1])
                logits = activation_fct(model_predictions)

                if apply_softmax and len(logits[0]) > 1:
                    logits = torch.nn.functional.softmax(logits, dim=1)
                pred_scores.extend(logits)

        if self.config.num_labels == 1:
            pred_scores = [score for score in pred_scores]

        if convert_to_tensor:
            pred_scores = torch.stack(pred_scores)
        elif convert_to_numpy:
            pred_scores = np.asarray([score.cpu().detach().numpy() for score in pred_scores])

        if input_was_string:
            pred_scores = pred_scores[0]

        return pred_scores
    
    
    def save(self, path,postfix=""):
        """
        Saves all model and tokenizer to path
        """
        if path is None:
            return

        torch.save(self.model,os.path.join(path,f"match{postfix}.pt"))
        torch.save(self.feature_extractor,os.path.join(path,f"feature_extractor{postfix}.pt"))
 
    def load(self,model_path,feature_extractor_path):
         
        self.model.load_state_dict(torch.load(model_path)) 
        self.feature_extractor.load_state_dict(torch.load(feature_extractor_path)) 
       