import spacy
#import the phrase matcher
from spacy.matcher import PhraseMatcher

from bestbuy.src.extract_mention.name_entity_recognition.util.data_util import match_spacy
#load a model and create nlp object
nlp = spacy.load("en_core_web_lg")
#initilize the matcher with a shared vocab
matcher = PhraseMatcher(nlp.vocab)

def test_parser2():
    #create the list of words to match
    fruit_list = ['apple','Refurbished product','banana']
    #obtain doc object for each word in the list and store it in a list
    patterns = nlp("freezer")  
    print(patterns[0].pos_)
 

def test_parser_match():
    #create the list of words to match
    fruit_list = ['apple','Refurbished product','banana',]
    #obtain doc object for each word in the list and store it in a list
    patterns = [nlp(fruit) for fruit in fruit_list]
    #add the pattern to the matcher
    matcher.add("FRUIT_PATTERN", patterns)
    #process some text
    doc = nlp("An Refurbished Product contains citric acid and an apple contains oxalic acid")
    matches = matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        print(span.text)
                
        
import spacy
def test2():
    nlp = spacy.blank("en")
    ruler = nlp.add_pipe("span_ruler")
    patterns = [{"label": "ORG", "pattern": "Apple"},
                {"label": "GPE", "pattern": [{"LOWER": "san"}, {"LOWER": "francisco"}]}]
    ruler.add_patterns(patterns)

    doc = nlp("Apple is opening its Francisco first big office in San Francisco.")
    print([(span.text, span.label_) for span in doc.spans["ruler"]])
def test3():
    review_text="Maybe used. I use Philip hue bulbs all over my house.  Love them. The ones I received look like they may be used.  Packaging is completely torn apart.  I will be returning these, not because of the product, but the condition I received them."
    print(match_spacy("Bulb",review_text))
        
def test4():
    import spacy
    from spacy.matcher import PhraseMatcher
    nlp = spacy.load('en_core_web_lg')
    matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
    matcher.add("CAT", None, nlp("cats run"))
    matches = matcher(nlp("cat ran"))   
    print(matches)  
     
def test5():
    import spacy
    from spacy.matcher import PhraseMatcher
    nlp = spacy.load('en_core_web_lg')
    matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
    matcher.add("CAT", None, nlp("Refurbished Product"))
    matches = matcher(nlp("Refurbished products"))   
    print(matches)      
     
def test6():
    print("test6")
    import spacy
    from spacy.matcher import PhraseMatcher
    nlp = spacy.load('en_core_web_lg')
    matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
    matcher.add("CAT", None, nlp("Refurbished products"))
    matches = matcher(nlp("Came without its ipad box. They sent the Wrong charging cable. I Cannot verify product works because it has 40 battery and no usb-c charging cable, so it will not download iOS . I have no idea what condition the battery is in at this point, or if the product really works. Purchased for a Christmas present, I paid an extra $35 to expedite the shipping. Now there are none available, so they cannot replace it, only offering a refund on my purchase. It is 8:12 PM on Dec. 13th. Description does not mention NO BOX. NO PAPERWORK. WRONG CABLE.Will never purchase another Geek Squad Certified Refurbished Product again."))   
    print(matches)      
    
def test7():
    print("test6")
    import spacy
    from spacy.matcher import PhraseMatcher
    nlp = spacy.load('en_core_web_lg')
    matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
    matcher.add("CAT", None, nlp("Refurbished products"))
    matches = matcher(nlp("I will never purchase another Geek Squad Certified Refurbished Product again."))   
    print(matches)       
     
     
def test8():
    str1 = "we\u2019re building a world where robots intelligently communicate with each other and the rest of your connected home"
    print(str1)
    str1.encode().decode('unicode-escape')
    print(str1)
    # str.replace('•','something')
    
    
def test_root():
    string="Apple iPad (5th Generation) (2017) Wi-Fi"
    nlp = spacy.load('en_core_web_lg')
    doc=nlp(string)
    for token in doc:
        print(token.text, token.dep_, token.head.text, token.head.pos_,
            [child for child in token.children])
        
        
from pathlib import Path
import json         
def vtest_format():
    review_json_path = Path(
                f'bestbuy/data/example/bestbuy_review_200_cleaned_annotation.json' #200_cleaned_annotation
            )   
    sum=0
    with open( review_json_path, 'r', encoding='utf-8') as fp:
        review_json_array = json.load(fp)
        for review_json in review_json_array:
            noisy_product_name=review_json["product_name"]
            noisy_product_name_list=noisy_product_name.split(" - ") 
            if len(noisy_product_name_list)>3:
                print(noisy_product_name)
                sum+=1
    print(sum )
    
    

        
        
# test5()
# test6()
# test7()
# test_parser2()
test3()