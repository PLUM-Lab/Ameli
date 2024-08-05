from pathlib import Path
import json
from tqdm import tqdm
import os
import pandas as pd
from nltk.tokenize import word_tokenize
import argparse
import logging

import numpy as np
import torch
from nltk.tokenize import sent_tokenize
from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
)

from attribute.extractor.base import Extractor
from attribute.util.util import compare_with_gold


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer),
}

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PREFIX = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


#
# Functions to prepare models' input
#


def prepare_ctrl_input(args, _, tokenizer, prompt_text):
    if args.temperature > 0.7:
        logger.info("CTRL typically works better with lower temperatures (and lower top_k).")

    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
    if not any(encoded_prompt[0] == x for x in tokenizer.control_codes.values()):
        logger.info("WARNING! You are not starting your generation from a control code so you won't get good results")
    return prompt_text


def prepare_xlm_input(args, model, tokenizer, prompt_text):
    # kwargs = {"language": None, "mask_token_id": None}

    # Set the language
    use_lang_emb = hasattr(model.config, "use_lang_emb") and model.config.use_lang_emb
    if hasattr(model.config, "lang2id") and use_lang_emb:
        available_languages = model.config.lang2id.keys()
        if args.xlm_language in available_languages:
            language = args.xlm_language
        else:
            language = None
            while language not in available_languages:
                language = input("Using XLM. Select language in " + str(list(available_languages)) + " >>> ")

        model.config.lang_id = model.config.lang2id[language]
        # kwargs["language"] = tokenizer.lang2id[language]

    #   fix mask_token_id setup when configurations will be synchronized between models and tokenizers
    # XLM masked-language modeling (MLM) models need masked token
    # is_xlm_mlm = "mlm" in args.model_name_or_path
    # if is_xlm_mlm:
    #     kwargs["mask_token_id"] = tokenizer.mask_token_id

    return prompt_text


def prepare_xlnet_input(args, _, tokenizer, prompt_text):
    prefix = args.prefix if args.prefix else args.padding_text if args.padding_text else PREFIX
    prompt_text = prefix + prompt_text
    return prompt_text


def prepare_transfoxl_input(args, _, tokenizer, prompt_text):
    prefix = args.prefix if args.prefix else args.padding_text if args.padding_text else PREFIX
    prompt_text = prefix + prompt_text
    return prompt_text


PREPROCESSING_FUNCTIONS = {
    "ctrl": prepare_ctrl_input,
    "xlm": prepare_xlm_input,
    "xlnet": prepare_xlnet_input,
    "transfo-xl": prepare_transfoxl_input,
}


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length

def gen_model_gpt2(args,device):
    try:
        args.model_type = args.model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    except KeyError:
        raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    model.to(device)
    
    if args.fp16:
        model.half()
    return model,tokenizer



def gen_model(args,device):
    return gen_model_gpt2(args,device)

class GPT2Extractor(Extractor):
    def __init__(self,device,args) -> None:
        self.model,self.tokenizer=gen_model(args,device)
        self.device=device
        args.length = adjust_length_to_model(args.length, max_sequence_length=self.model.config.max_position_embeddings)
        
    def generate_per_review_attribute(self,args, prefix,prompt_text,attribute_key,candidate_attribute_list,is_constrained_beam_search,
                                      review_dataset_json,review_image_dir,total_attribute_value_candidate_list,total_confidence_score_list,mention,
                                      gold_attribute_value,all_ocr_raw, gpt_context, chatgpt_context):
        
        example=prefix +"Review: Sharp!. Extremely satisfied with microwave drawer. The silver microwave looks and feels great" + ".\n Color Finish: Silver.\n"
        # example2=prefix +"Splattoon 2. We love Splatoon 2, the only downfall is that it is one player, we would absolutely love if you could play multiplayer on one console. Very fun, very colorful, we are loving this game.\n Model Version: Splatoon 2.\n "
        
        postfix="\n"+attribute_key+":"
        model=self.model
        tokenizer=self.tokenizer
        # Different models need different input formatting and/or extra arguments
        requires_preprocessing = args.model_type in PREPROCESSING_FUNCTIONS.keys()
        if requires_preprocessing:
            prepare_input = PREPROCESSING_FUNCTIONS.get(args.model_type)
            preprocessed_prompt_text = prepare_input(args, model, tokenizer, prompt_text)

            if model.__class__.__name__ in ["TransfoXLLMHeadModel"]:
                tokenizer_kwargs = {"add_space_before_punct_symbol": True}
            else:
                tokenizer_kwargs = {}

            encoded_prompt = tokenizer.encode(
                preprocessed_prompt_text, add_special_tokens=False, return_tensors="pt", **tokenizer_kwargs
            )
        else:
             
            input_text=example+ "Review: "+ prompt_text+postfix 
            encoded_prompt = tokenizer.encode(input_text, add_special_tokens=False, return_tensors="pt")
        encoded_prompt = encoded_prompt.to(self.device)

        if encoded_prompt.size()[-1] == 0:
            input_ids = None
        else:
            input_ids = encoded_prompt

        if is_constrained_beam_search and len(candidate_attribute_list)>0:
            force_flexible = candidate_attribute_list  
            force_words_ids =  tokenizer(force_flexible, add_prefix_space=True, add_special_tokens=False).input_ids
        else:
            force_words_ids=None
        output_sequences = model.generate(
            input_ids=input_ids,
            max_length=args.length + len(encoded_prompt[0]),
            temperature=args.temperature,
            top_k=args.k,
            top_p=args.p,
            repetition_penalty=args.repetition_penalty,
            do_sample=True if not is_constrained_beam_search else False ,
            num_return_sequences=args.num_return_sequences,
            force_words_ids=  force_words_ids,
            num_beams=2,
            pad_token_id=tokenizer.eos_token_id, 
            early_stopping=True
            # no_repeat_ngram_size=1,
            # remove_invalid_values=True,
        )

        # Remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        generated_sequences = []
        attribute_value_candidate_list=[]
        for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
            # print(f"=== GENERATED SEQUENCE {generated_sequence_idx + 1} ===")
            generated_sequence = generated_sequence.tolist()

            # Decode text
            text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

            # Remove all text after the stop token
            text = text[: text.find(args.stop_token) if args.stop_token else None]

            # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
            total_sequence = (
                text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]#prompt_text +
            )

            generated_sequences.append(total_sequence)
            # print(total_sequence) #TODO may be check \n is better?
            total_sequence_words=word_tokenize(total_sequence)
            if len(total_sequence_words)>0:
                attribute_value=total_sequence_words[0] if len(total_sequence_words[0])>0 else total_sequence_words[0]
                attribute_value_candidate_list.extend(compare_with_gold(attribute_value,candidate_attribute_list))
                
            else:
                attribute_value=""
                total_sequence=""
            
        return attribute_value,total_sequence,attribute_value_candidate_list,generated_sequences,[],[]




class GPT2ResultParser(Extractor):
    def __init__(self,device,args) -> None:
        
        self.device=device
        
        
    def generate_per_review_attribute(self,args, prefix,prompt_text,attribute_key,candidate_attribute_list,is_constrained_beam_search,
                                      review_dataset_json,review_image_dir,total_attribute_value_candidate_list,
                                      total_confidence_score_list,mention,attribute_value,ocr_raw,gpt_context,chatgpt_context):
        attribute_value_candidate_list=[]
        generated_sequences=review_dataset_json["predicted_attribute_context_gpt2"]
        if attribute_key in generated_sequences:
            predicted_attribute_context_gpt2_for_this_attribute=generated_sequences[attribute_key]
            for one_trial in predicted_attribute_context_gpt2_for_this_attribute:
                if "\n" in one_trial:
                    phrase_list=one_trial.split("\n")
                elif "<|endoftext|>" in one_trial:
                    phrase_list=one_trial.split("<|endoftext|>")
                else:
                    phrase_list=[one_trial]
                if len(phrase_list)>0:
                    is_found=False
                    for i in range(len(phrase_list)):
                        if len(phrase_list[i])>0:
                            attribute_value=phrase_list[i]
                            attribute_value_candidate_list.extend(compare_with_gold(attribute_value,candidate_attribute_list))
                            is_found=True
                            # break
                    
                    if not is_found:
                        attribute_value=""
                        total_sequence="" 
         
        
        return None,None,attribute_value_candidate_list,generated_sequences,[],[]

