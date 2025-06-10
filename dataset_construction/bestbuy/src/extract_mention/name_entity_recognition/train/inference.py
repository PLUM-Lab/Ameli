 
from unittest import result
from transformers import pipeline

from bestbuy.src.extract_mention.name_entity_recognition.train.train import gen_model

def inference_one(model, tokenizer,text):
     

    pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple",device=0 ) # pass device=0 if using gpu
    results= pipe(text)
    max_score=-1
    max_token=None 
    candidate_list=[]
    for result in results:
        candidate_list.append(result["word"])
        if result["entity_group"]=="product"  :
            if result["score"]>max_score:
                max_score=result["score"]
                max_token=result["word"]
    return max_token,candidate_list
    
if __name__ == "__main__":   
    model, tokenizer=gen_model("asahi417/tner-xlm-roberta-base-ontonotes5") #asahi417/tner-xlm-roberta-base-ontonotes5 djagatiya/ner-roberta-base-ontonotesv5-englishv4
    print(inference_one(model, tokenizer,"I have had the ring doorbell for years now and it seemed quite natural that I would migrate to a Ring Alarm System. I did. I left ADT and decided upon such as it integrated well with that was already in place. I really prefer the self monitoring and the idea that it’s at such a low cost... $8.50/mthly. I love the ease of installation too. Did it myself and quick too")    )