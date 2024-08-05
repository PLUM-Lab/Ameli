import os
import openai
from nltk.tokenize import word_tokenize
import random
import time
from attribute.extractor.base import Extractor
import argparse
import torch
import os
import json
from tqdm import tqdm
# import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from util.env_config import review_image_dir
from PIL import Image
import math

"""
Review: Splattoon 2. We love Splatoon 2, the only downfall is that it is one player, we would absolutely loveeeeeeee if you could play multiplayer on one console. Very fun, very colorful, we are loving this game.
Question: What is the product title of the game based on this review?
A. Splatoon 3 - Nintendo Switch (OLED Model), Nintendo Switch, Nintendo Switch Lite
B. Belkin - USB-C 11-in-1 Multiport Dock - Gray
C. Splatoon 5 - Nintendo Switch [Digital]
D. Plants vs Zombies Battle for Neighborville - Nintendo Switch, Nintendo Switch Lite
E. Mario Party Superstars - Nintendo Switch, Nintendo Switch Lite
F. Hyper - Viper 10-Port USB-C Hub Dock for Apple MacBook Pro & MacBook Air
G. Satechi USB-C Pro Hub Max Adapter - Space Gray - Space Gray
H. Satechi - USB-C On-The-Go 9-in-1 Multiport Adapter - Matte Black
I. Splatoon 2 Standard Edition - Nintendo Switch
J. Satechi - USB-C Monitor Stand Hub XL - Space Gray
Answer:
"""
is_debug=False

 

def compare_with_gold(attribute_value,candidate_attribute_list):
    if attribute_value in candidate_attribute_list:
        return [attribute_value]
    elif attribute_value.lower() in candidate_attribute_list:
        return [attribute_value.lower()]
    else:
        return []
  
class LLaVAExtractor(Extractor):
    def __init__(self,args) -> None:
        super().__init__()
        # to get proper authentication, make sure to use a valid key that's listed in
        # the --api-keys flag. if no flag value is provided, the `api_key` will be ignored.
        openai.api_key = "EMPTY"
        openai.api_base = "http://localhost:8000/v1"
        disable_torch_init()
        model_path = os.path.expanduser(args.model_name_or_path)
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
        self.args=args
        self.image_folder=review_image_dir
        # self.model,self.tokenizer=gen_model(args,device)
        # self.device=device
        # self.model = "vicuna-13b-v1.5"
 
    def build_message(self,mention,attribute_key,candidate_attribute_value_list,review,review_image_path):
        # mention="game"
        # attribute_key="product title"
        # candidate_attribute_value_list=["1","2"]
        # review="Splattoon 2. We love Splatoon 2, the only downfall is that it is one player, we would absolutely loveeeeeeee if you could play multiplayer on one console. Very fun, very colorful, we are loving this game."
        random.shuffle(candidate_attribute_value_list)
        
        self.option_id_list_without_blank=["A.","B.","C.","D.","E.","F.","G.","H.","I.","J."]
        self.option_id_list_pure=["A","B","C","D","E","F","G","H","I","J"]
        self.option_id_list=["A. ","B. ","C. ","D. ","E. ","F. ","G. ","H. ","I. ","J. "]
        example="Review: Splattoon 2. We love Splatoon 2, the only downfall is that it is one player, we would absolutely love if you could play multiplayer on one console. Very fun, very colorful, we are loving this game. \
            Question: What is the product title of the game based on this review?  \
            A. Splatoon 3 - Nintendo Switch (OLED Model), Nintendo Switch, Nintendo Switch Lite   \
            B. Belkin - USB-C 11-in-1 Multiport Dock - Gray   \
            C. Splatoon 2 Standard Edition - Nintendo Switch \
            Answer: C. Splatoon 2 Standard Edition - Nintendo Switch"
        example2="Review: Sharp!. Extremely satisfied with microwave drawer. The silver microwave looks and feels great. I made the purchase right before the price increased which was even better. My cabinet was wider than the microwave so adjustments need to be be made other than that no issues.  \
            Question: What is the Color Finish of microware based on this review? \
            E. Stainless Steel With Black Glass \
            F. Silver\
            G. Black stainless steel \
            Answer: F. Silver"
        question=f"Question: What is the {attribute_key} of the {mention} based on this review?"
    
        content=example+"<s>"+example2+"<s>"+"Review: "+review+"\n"+question+"\n"
        for candidate_attribute_value,option_id in zip(candidate_attribute_value_list,self.option_id_list):
            content+=option_id+candidate_attribute_value+"\n"
        content+="Answer:"
        cur_prompt=content
        qs=content
        if review_image_path is not None:
            image_file =review_image_path[0]
            image = Image.open(os.path.join(self.image_folder, image_file))
            image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            images = image_tensor.unsqueeze(0).half().cuda()
            if getattr(self.model.config, 'mm_use_im_start_end', False):
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            cur_prompt = '<image>' + '\n' + cur_prompt
        else:
            images = None

        # if self.args.single_pred_prompt:
        #     qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
        #     cur_prompt = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."
                
        conv = conv_templates[self.args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()    
            
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        
        # messages=[
        #     {"role": "system", "content": f"You choose the {attribute_key} of the {mention} from ten candidates based on the user review."},
        #     {"role": "user", "content":content},
        # ]
        # if is_debug:
        #     print(messages)
        return prompt,image_tensor,images,input_ids,conv,candidate_attribute_value_list
  
    def decode_multi_choice(self,total_sequence,candidate_attribute_list,gold_attribute_value):
        attribute_value_list=[]
        is_found=False
        has_index=False
        if "\n" in total_sequence:
            phrase_list=total_sequence.split("\n")
        elif "<|endoftext|>" in total_sequence:
            phrase_list=total_sequence.split("<|endoftext|>")
        else:
            phrase_list=[total_sequence]
        if len(phrase_list)>0:
            
            for i in range(len(phrase_list)):
                if len(phrase_list[i])>0:
                    attribute_value=phrase_list[i]
                    for option_position,option_id in enumerate(self.option_id_list):
                        if attribute_value.startswith(option_id):
                            attribute_value=attribute_value[len(option_id):]
                            has_index=True
                            break
                    if not has_index:
                        for option_position,option_id in enumerate(self.option_id_list_without_blank):
                            if attribute_value.startswith(option_id):
                                attribute_value=attribute_value[len(option_id):]
                                has_index=True
                                break
                    if not has_index and len(attribute_value)==1:
                        for option_position,option_id in enumerate(self.option_id_list_pure):
                            if attribute_value.startswith(option_id):
                                attribute_value=attribute_value[len(option_id):]
                                has_index=True
                                break
                    attribute_value_list=compare_with_gold(attribute_value,candidate_attribute_list)
                    if len(attribute_value_list)==0 and has_index and option_position < len(candidate_attribute_list):
                        attribute_value_list=[candidate_attribute_list[option_position]]
                    if len(attribute_value_list)>0:
                        print(f"is_correct:{attribute_value_list==gold_attribute_value}, {attribute_value_list},{gold_attribute_value}")
                    is_found=True
                    break
        return attribute_value_list,is_found
    def _call_model(self, images,input_ids,conv):
        
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = [KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)] if conv.version == "v0" else None

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images,
                do_sample=True if self.args.temperature > 0 else False,
                temperature=self.args.temperature,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=stopping_criteria,
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        return outputs

    def generate_per_review_attribute(self,args, prefix,review,attribute_key,candidate_attribute_list,is_constrained_beam_search,review_dataset_json,review_image_dir,
                                      total_attribute_value_candidate_list,total_confidence_score_list,mention
                                      ,attribute_value,ocr_raw,gpt_context,chatgpt_context):
        postfix="\n"+attribute_key+":"
        review_image_path=review_dataset_json["review_image_path"]
        messages,image_tensor,images,input_ids,conv,shuffled_candidate_attribute_value_list=self.build_message(mention,attribute_key,candidate_attribute_list,review,review_image_path)
        total_sequence=self._call_model(images,input_ids,conv)
        
        # response = openai.ChatCompletion.create(
        # model=self.model,
        
        # # temperature=0,
        # # max_tokens=512
        # # n=1
        # )
        # response= completion.choices[0].text
        # if attribute_key=="Product Title":
        #     print()
        # print(response)
        generated_sequences = []
        attribute_value_candidate_list=[]
        # for one_response in  response["choices"]:
        #     total_sequence=one_response["message"]["content"]
        generated_sequences.append(total_sequence)
        # print(total_sequence)
        attribute_value_list,is_found=self.decode_multi_choice( total_sequence,shuffled_candidate_attribute_value_list,attribute_value)
        
        attribute_value_candidate_list.extend(attribute_value_list)
        return None,None,attribute_value_candidate_list,generated_sequences,[],[]
    
    
    #The brand of the camera is Arlo.
def   extract_by_chat(context,candidate_attribute_list):
    is_found=False
    extracted_attribute_list=[]
    for candidate_attribute in candidate_attribute_list:
        if candidate_attribute in context.lower() or candidate_attribute in context:
            is_found=True
            extracted_attribute_list.append(  candidate_attribute)
    return extracted_attribute_list,is_found
    
def extract_by_no_option_id(context,candidate_attribute_list):
    is_found=False
    extracted_attribute_list=[]
    for candidate_attribute in candidate_attribute_list:
        if candidate_attribute in context or context in candidate_attribute:
            is_found=True
            extracted_attribute_list.append(  candidate_attribute)
 
 
    return extracted_attribute_list,is_found

def extract_by_template(context,mention,attribute_key,option_id_list):
    is_found_out_of_top10=False
    extracted_attribute_list=[]
    template1=f"The {attribute_key} of the {mention} based on this review is "
    template2=f"The {attribute_key} of the {mention} is "
    template3=f"The {attribute_key} of the {mention} mentioned in the review is "
    for template in [template1,template2,template3]:
        if template.lower() in context.lower():
            attribute_value=context[len(template):]
            if attribute_value.endswith("."):
                attribute_value=attribute_value[:-1]
            for option_position,option_id in enumerate( option_id_list):
                if attribute_value.startswith(option_id):
                    attribute_value=attribute_value[len(option_id):]
                     
            extracted_attribute_list.append(  attribute_value)
            is_found_out_of_top10=True
    return extracted_attribute_list,is_found_out_of_top10
 