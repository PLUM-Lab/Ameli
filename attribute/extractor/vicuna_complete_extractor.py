import os
import openai
from nltk.tokenize import word_tokenize
import random
import time
from attribute.extractor.base import Extractor
openai.api_key = os.getenv("OPENAI_API_KEY") 
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


# define a retry decorator
def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (openai.error.RateLimitError,openai.error.OpenAIError,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


@retry_with_exponential_backoff
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def compare_with_gold(attribute_value,candidate_attribute_list):
    if attribute_value.lower() in candidate_attribute_list:
        return [attribute_value]
    else:
        return []


class VicunaCompleteExtractor(Extractor):
    def __init__(self) -> None:
        super().__init__()
        # to get proper authentication, make sure to use a valid key that's listed in
        # the --api-keys flag. if no flag value is provided, the `api_key` will be ignored.
        openai.api_key = "EMPTY"
        openai.api_base = "http://localhost:8000/v1"

        self.model = "vicuna-13b-v1.5"
 
    def build_message(self,mention,attribute_key,candidate_attribute_value_list,review,prefix):
        # mention="game"
        # attribute_key="product title"
        # candidate_attribute_value_list=["1","2"]
        # review="Splattoon 2. We love Splatoon 2, the only downfall is that it is one player, we would absolutely loveeeeeeee if you could play multiplayer on one console. Very fun, very colorful, we are loving this game."
        
        postfix="\n"+attribute_key+":"
        
        example=prefix +"Review: Sharp!. Extremely satisfied with microwave drawer. The silver microwave looks and feels great" + ".\n Color Finish: Silver.\n"
        content=example+ "Review: "+ review+postfix 
         
            
        messages=[
        
            {"role": "user", "content":content},
        ]
        if is_debug:
            print(messages)
        return content
  
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
                    
                    attribute_value_list=compare_with_gold(attribute_value,candidate_attribute_list)
                    if len(attribute_value_list)==0 and has_index and option_position in candidate_attribute_list:
                        attribute_value_list=[candidate_attribute_list[option_position]]
                    if len(attribute_value_list)>0:
                        print(attribute_value_list,gold_attribute_value)
                    is_found=True
                    break
        return attribute_value_list,is_found


    def generate_per_review_attribute(self,args, prefix,review,attribute_key,candidate_attribute_list,is_constrained_beam_search,review_dataset_json,review_image_dir,
                                      total_attribute_value_candidate_list,total_confidence_score_list,mention
                                      ,attribute_value,ocr_raw,gpt_context,chatgpt_context):
        postfix="\n"+attribute_key+":"
         
        response = openai.Completion.create(
        model=self.model,
        prompt =self.build_message(mention,attribute_key,candidate_attribute_list,review,prefix),
        temperature=args.temperature,
        max_tokens=16,
        top_p=args.p,
        # presence_penalty=args.repetition_penalty,
        n=args.num_return_sequences,
        # best_of=1
        # n=2
        )
        # response= completion.choices[0].text
        # if attribute_key=="Product Title":
        #     print()
        # print(response)
        generated_sequences = []
        attribute_value_candidate_list=[]
        for one_response in  response["choices"]:
            total_sequence=one_response["text"] 
            generated_sequences.append(total_sequence)
            if len(total_sequence)>0:
                attribute_value_list=extract_attribute_from_response(total_sequence,candidate_attribute_list,attribute_value,attribute_key,mention)
                 
                attribute_value_candidate_list.extend(attribute_value_list)
            else:
                attribute_value_candidate_list=[]
            # print(total_sequence)
            # attribute_value_list,is_found=self.decode_multi_choice( total_sequence,candidate_attribute_list,attribute_value)
            # attribute_value_list=compare_with_gold(attribute_value,candidate_attribute_list)
            # attribute_value_candidate_list.extend(attribute_value_list)
        return None,None,attribute_value_candidate_list,generated_sequences,[],[]
    
    #The brand of the camera is Arlo.
def   extract_by_chat(context,candidate_attribute_list):
    is_found=False
    extracted_attribute_list=[]
    for candidate_attribute in candidate_attribute_list:
        if candidate_attribute in context.lower() or candidate_attribute in context:
            is_found=True
            extracted_attribute_list.append(  candidate_attribute)
            break
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

def extract_attribute_from_response(predicted_attribute_context,candidate_attribute_list,gold_attribute_value,attribute_key,mention):
     
    extracted_attribute_list,is_found=extract_by_chat(predicted_attribute_context,candidate_attribute_list)
    # print(f"extract by chat: {extracted_attribute_list},{gold_attribute_value}")
    # if not is_found:
        
    #     is_mention=True
    #     for no_mention_key_word in ["not mentioned","not provide","not enough information",
    #                                 "not mention","no clear","not specified","not possible","no specific","not provided","not specify","no clear indication","no indication","no information","impossible"]:
    #         if no_mention_key_word in predicted_attribute_context:
    #             is_mention=False
    #             break
    #     if is_mention:
    #         predicted_attribute_context_word_list=word_tokenize(predicted_attribute_context) 
    #         if "no" in predicted_attribute_context_word_list or "not" in predicted_attribute_context_word_list or  "none" in predicted_attribute_context_word_list:
    #             is_mention=False
            
        # if   is_mention:
        #     extracted_attribute_list_out_of_top10,is_found_out_of_top10=extract_by_template(predicted_attribute_context,mention,attribute_key,[])
        #     print(f"extract by template: {extracted_attribute_list_out_of_top10},{gold_attribute_value}")
        #     if not is_found_out_of_top10:
        #         print(f"still not find: {predicted_attribute_context},{gold_attribute_value}")
        #         print("")
    return extracted_attribute_list


class ChatGPTParser(Extractor):
  
    def __init__(self) -> None:
        super().__init__()
        self.option_id_list=["A. ","B. ","C. ","D. ","E. ","F. ","G. ","H. ","I. ","J. "]

    
    def generate_per_review_attribute(self,args, prefix,review,attribute_key,candidate_attribute_list,is_constrained_beam_search,review_dataset_json,review_image_dir,
                                      total_attribute_value_candidate_list,total_confidence_score_list,mention,gold_attribute_value):
        
        attribute_value_candidate_list=[]
        is_found=False
        extracted_attribute_list_out_of_top10=[]
        extracted_attribute_list=[]
        total_sequence_context_dict=review_dataset_json["predicted_attribute_context_chatgpt"]
        if attribute_key in total_sequence_context_dict:
            predicted_attribute_context_list=total_sequence_context_dict[attribute_key] 
            predicted_attribute_context=predicted_attribute_context_list[0]
            for option_position,option_id in enumerate(self.option_id_list):
                if predicted_attribute_context.startswith(option_id):
                    attribute_value=predicted_attribute_context[len(option_id):]
                    extracted_attribute_list=[attribute_value]
                    is_found=True
            if not is_found:
                extracted_attribute_list,is_found=extract_by_no_option_id(predicted_attribute_context,candidate_attribute_list)
                 
                if not is_found:
                    extracted_attribute_list,is_found=extract_by_chat(predicted_attribute_context,candidate_attribute_list)
                    # print(f"extract by chat: {extracted_attribute_list},{gold_attribute_value}")
                    if not is_found:
                        
                            is_mention=True
                            for no_mention_key_word in ["not mentioned","not provide","not enough information",
                                                        "not mention","no clear","not specified","not possible","no specific","not provided","not specify","no clear indication","no indication","no information","impossible"]:
                                if no_mention_key_word in predicted_attribute_context:
                                    is_mention=False
                                    break
                            if is_mention:
                                predicted_attribute_context_word_list=word_tokenize(predicted_attribute_context) 
                                if "no" in predicted_attribute_context_word_list or "not" in predicted_attribute_context_word_list or  "none" in predicted_attribute_context_word_list:
                                    is_mention=False
                                
                            if   is_mention:
                                extracted_attribute_list_out_of_top10,is_found_out_of_top10=extract_by_template(predicted_attribute_context,mention,attribute_key,self.option_id_list)
                                print(f"extract by template: {extracted_attribute_list_out_of_top10},{gold_attribute_value}")
                                if not is_found_out_of_top10:
                                    print(f"still not find: {predicted_attribute_context},{gold_attribute_value}")
                                    print("")
                
  
        else:
            predicted_attribute_context_list=[]
            attribute_value=""
            
        # attribute_value_list,is_found=self.decode_multi_choice( total_sequence,candidate_attribute_list,attribute_value)
            
        attribute_value_candidate_list=extracted_attribute_list
        return None,None,attribute_value_candidate_list,predicted_attribute_context_list,[],extracted_attribute_list_out_of_top10
        
     