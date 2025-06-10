from multiprocessing import current_process
import spacy
#https://machinelearningknowledge.ai/tutorial-on-spacy-part-of-speech-pos-tagging/#:~:text=Spacy%20provides%20a%20bunch%20of,is%20most%20likely%20a%20noun.
nlp=spacy.load("en_core_web_lg")
def pos_spacy(my_text):
    candidate_list=[]
    my_doc=nlp(my_text)
    one_name_list=[]
    current_position=-1
    for token in my_doc:
        # print(f"{token.text},'---- ',{token.pos_}")
        if token.pos_ in ["NOUN","PROPN"]:#"PRON",
            if current_position==-1 or token.i==current_position+1:
                one_name_list.append(token.text)
                current_position=token.i
            else:
                one_name=" ".join(one_name_list)
                candidate_list.append(one_name)
                one_name_list=[]
                one_name_list.append(token.text)
                current_position=token.i
    one_name=" ".join(one_name_list)
    candidate_list.append(one_name)
    return candidate_list
    
    
my_text='John plays basketball,if time permits. He played in high school too.'
# pos(my_text)