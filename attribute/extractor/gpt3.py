import os
import openai
from nltk.tokenize import word_tokenize

from attribute.extractor.base import Extractor
openai.api_key = os.getenv("OPENAI_API_KEY")


class GPT3Extractor(Extractor):
    def _gen_input(self,prefix,review,postfix):
        input_text= prefix + review+postfix 
        return input_text
    def generate_per_review_attribute(self,args, prefix,prompt_text,attribute_key,candidate_attribute_list,is_constrained_beam_search,review_dataset_json,review_image_dir,total_attribute_value_candidate_list,total_confidence_score_list):
        postfix="\n"+attribute_key+":"
        input_text=self._gen_input(prefix,prompt_text,postfix )
        response = openai.Completion.create(
        model="text-davinci-003",
        prompt=input_text,
        temperature=1,
        max_tokens=20,
        top_p=0.9,
        frequency_penalty=0,
        presence_penalty=0,
        n=1
        )
        print(response)
        generated_sequences = []
        attribute_value_candidate_list=[]
        for one_response in  response["choices"]:
            total_sequence=one_response["text"]
            generated_sequences.append(total_sequence)
            # print(total_sequence)
            total_sequence_words=word_tokenize(total_sequence)
            if len(total_sequence_words)>0:
                attribute_value=total_sequence_words[0]

                attribute_value_candidate_list.append(attribute_value)
        return None,None,attribute_value_candidate_list,generated_sequences,[]
    
    
    