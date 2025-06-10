from bestbuy.src.extract_mention.dependency_parser.parser import gen_mention_candidates
from bestbuy.src.extract_mention.name_entity_recognition.train.inference import inference_one
from bestbuy.src.extract_mention.name_entity_recognition.train.train import gen_model
from  bestbuy.src.extract_mention.name_entity_recognition.ner_bert import ner_bert
from  bestbuy.src.extract_mention.name_entity_recognition.ner_spacy import ner_spacy
from  bestbuy.src.extract_mention.name_entity_recognition.token_similarity import find_most_similar_word
from  bestbuy.src.extract_mention.pos.pos_spacy import pos_spacy

from nltk.tokenize import sent_tokenize, word_tokenize
 
 
#1, concanate tokens together if they are neighbors. 2, compute sbert similarity between these spans and product_name from dataset (not the product title!)  
def test_acc():
    mention_entity_review_list=[
        ["a Ring Alarm System","Ring - Alarm 8-Piece Security Kit","I have had the ring doorbell for years now and it seemed quite natural that I would migrate to a Ring Alarm System. I did. I left ADT and decided upon such as it integrated well with that was already in place. I really prefer the self monitoring and the idea that it’s at such a low cost... $8.50/mthly. I love the ease of installation too. Did it myself and quick too."],
        ["the sonos arc","Sonos - Arc Soundbar with Dolby Atmos, Google Assistant and Amazon Alexa","The sonos arc is the best sound bar. It looks beautiful and has unbelievable sounds. It sounds like there are speakers all over the house. I highly recommend this product"],
        ["The lens","Nikon - AF-S DX NIKKOR 35mm f/1.8G Standard Lens","The lens is great and light. good for portrait, group shooting and ability to shoot at a low light. Nice blur effect. I used with nikon d5200."],
        ["my new tv","TCL - 75' Class 5-Series QLED 4K UHD Smart Google TV","Best delivered and installed my new tv with professionalism. They installed it, made sure it worked properly and took the old one. I could not have asked for a better service. The tv is a 75 inch tcl 5 series and i am thoroughly happy with it"],
        ["Ring exterior spotlights","Ring - Spotlight Cam Wired (Plug-In)","We recently had 3 Ring exterior spotlights installed on our home. We were surprised at how well the clarity and response time these cameras provide. They are 1080p and the picture looks fantastically clear! We tried them on a recommendation from a friend who had 3 cameras displaying 4K pics. These are just as good and the subscription to ring for monitoring is cheap! Getcha self some!"],
        ["Battery","Rechargeable Lithium-ion Battery for Select Ring Devices","Battery is very reliable and holds for a few days for ring garage camera."],
        ["the MacBook Air","MacBook Air 13.3' Laptop - Apple M1 chip","I love the MacBook Air it is lighter than my MacBook Pro. It still is fast and does everything the MacBook Pro does, just lighter. It has very colorful screen and background pictures, and you can get 6 months free apple news, Apple Music and Webroot security protection."],
        ["the One’s","Sonos - One (Gen 2) Smart Speaker with Voice Control built-in","I added the One’s to my Sonos arc and Sub, I must say it makes a world of difference when it comes to atmos and surround sound. The arc is impressive by itself but nothing really replaces have speakers across the sound spectrum. I just returned the ambeo soundbar two days ago before switching to sonos!"],
        ["this keyboard","Razer - Huntsman Mini 60% Wired Optical Clicky Switch Gaming Keyboard with Chroma RGB Backlighting","I love the size but this keyboard is way over priced. The shift button already broke on me, keeps on getting stuck. It’s 119 for a fully built plastic keyboard. If the key board was something else besides plastic then yeah I could under stand. Just buy a ducky one 2, they care about theirs product. Positives: it’s 60% Very comfortable Usb C Negative: Cheaply made All plastic for price range To noisy, rattle is very loud(not talking about the clicky switches, I love clicky) Worst part it was built by razer Can’t put custom key caps on because of the clips they use."],
        ["mouses","Logitech - PRO X SUPERLIGHT Lightweight Wireless Optical Gaming Mouse with HERO 25K Sensor","Upgraded from a logitech g pro wireless mouse and you can tell a weight difference between the two mouses. I play a lot of FPS games and I think with the quick flick of the wrist when someone tries to shoot you from behind and you kill them has saved me a couple of times especially in Squad. If you have the money and definitely buy it and the great thing about best buy is you can try it out and if you don't like it, you can take it back. So far no double click or any defective mouse issues and it works with logitech power play wireless charging mouse pad"]
        
    ]
    acc=0
    is_first_sentence=False 
    mode="parser"
    if mode=="ner_pipeline":
        model, tokenizer=gen_model("/home/menglong/workspace/code/referred/product_scraper/dataset_construction/results/models_40_djagatiya_ner-bert-base")#asahi417/tner-xlm-roberta-base-ontonotes5   
        
    else:
        model, tokenizer=None,None
         
    for mention, entity_title, review in mention_entity_review_list:
        entity_title=entity_title.replace("- ","")
        
        if is_first_sentence:
            review_sent_list=sent_tokenize(review)
            
            named_entity_in_review_list=ner(review_sent_list[0],mode,model, tokenizer,entity_title)
        else:
            named_entity_in_review_list=ner(review ,mode,model, tokenizer,entity_title)
        print(f"review:{named_entity_in_review_list}")
        if mode !="parser":
            named_entity_in_entity_title_list=ner(entity_title,mode,model, tokenizer,entity_title)
            print(f"entity:{named_entity_in_entity_title_list}")
        print(f"mention: {mention}----------------------------------------")
        if mode=="pos":
            find_most_similar_word(entity_title,named_entity_in_review_list)
        # for named_entity_in_review  in named_entity_in_review_list:
        #     if named_entity_in_review.text== 
        
def ner(review,mode,model=None, tokenizer=None ,product_title=None ):
    
    if mode=="ner_spacy":
        results=ner_spacy(review) 
    elif mode=="ner_bert":
        results=ner_bert(review)
    elif mode=="pos":
        results=pos_spacy(review) 
    elif mode=="ner_pipeline":
        results=inference_one(model,tokenizer,review)
    elif mode=="parser":
        results=gen_mention_candidates(review,product_title,product_title,product_title)
        # print(resultntion_candidates(review,product_title,product_title,product_title)
        # print(results)
    return results 
    
    
test_acc()