import spacy
from torch import prod
from spacy.matcher import Matcher
 
from nltk.tokenize import sent_tokenize, word_tokenize
import spacy
from spacy.matcher import PhraseMatcher
from bestbuy.src.extract_mention.name_entity_recognition.util.data_util import ProductNameCandidate, compare_two_equal_len_texts_w_base_form, find_longest_match, gen_root_by_doc, is_prediction_right, is_word_same
nlp = spacy.load("en_core_web_lg")

    
def is_match_spacy(root_text,review_doc):
    # pattern = [{"NORM": root_text} ]
    # matcher.add("HelloWorld", [pattern])
    # matches = matcher(review_doc)
    result= False 
    # if len(matches)>0:
    #     result=True 
    # else:
    real_root_doc=nlp(root_text)
    real_root_lemma_lower=real_root_doc[:].lemma_.lower()
    for token in review_doc:
        # if is_word_same(token.lemma_.lower(),nlp(token.text)[0],real_root_lemma_lower,real_root_doc[0]):
        # if compare_two_equal_len_texts_w_base_form(token.text,root_text):
        if token.lemma_.lower()==real_root_lemma_lower:
    
            result= True 
            break 
            
     
    
    # if result!=result1:
    #     print("TODO")
    return result 

def gen_mention_candidates(review,noisy_product_name,product_category,product_desc,gt_mention,
                           is_debug,parser_setting="all",parser_chunk_setting=None   ):
    mention_candidate_list=[]
    mention_candidate_list_before_review_match=gen_product_name_candidates(noisy_product_name,product_category
                                                                           ,product_desc,parser_setting,
                                                                           parser_chunk_setting)
    
    review_doc = nlp(review)
    mention_candidate_list =match_in_review(review,mention_candidate_list_before_review_match,review_doc)
    # mention_candidate_list.extend(add_mention_in_review_first_sent(review))
    return list(set(mention_candidate_list)),mention_candidate_list_before_review_match

def add_mention_in_review_first_sent(review):
    mention_candidate_list=[]
    review_sent_list=sent_tokenize(review)
    review_doc = nlp(review_sent_list[0])
    for chunk in review_doc.noun_chunks:
        if chunk.root.pos_!="PRON":
            mention_candidate_list.append(chunk.text )
    return mention_candidate_list
    
    
    
def match_in_review(review,product_name_candidate_object_dict,review_doc):
    mention_candidate_list=[]
    
    # for chunk in review_doc.noun_chunks:
    #     if chunk.root.norm_ in product_name_candidate_object_dict:
    #         mention_candidate_list.append(chunk.text )
            
    for root_text,product_name_candidate_object in product_name_candidate_object_dict.items():
        if is_match_spacy(root_text,review_doc):
  
            mention_candidate,is_found=find_longest_match(review,product_name_candidate_object,review_doc)
            if is_found:
                mention_candidate_list.append(mention_candidate)

    return mention_candidate_list 
            
            
         
     
def gen_product_name_candidates_for_one(product_name_context,parser_chunk_setting):
    
    doc = nlp(product_name_context)
    return gen_product_name_candidates_by_chunk(doc,parser_chunk_setting)
      
      
def gen_product_name_candidates_for_one_context(product_name_context,product_name_candidates,parser_chunk_setting):
    new_product_name_candidates_dict=gen_product_name_candidates_for_one(product_name_context,parser_chunk_setting)
    if new_product_name_candidates_dict is not None:
        for candidate_root, new_product_name_candidates_object in new_product_name_candidates_dict.items():
            if candidate_root in product_name_candidates:
                product_name_candidates[candidate_root].candidate_text_list.extend(new_product_name_candidates_object.candidate_text_list)
            else:
                product_name_candidates[candidate_root]=new_product_name_candidates_object
    return product_name_candidates
    
def gen_product_name_candidates(noisy_product_name,product_category,product_desc,parser_setting=None ,
                                parser_chunk_setting=None ):
    product_name_candidates ={} 
    if parser_setting=="all":
        for product_name_context in [noisy_product_name,product_category,product_desc]:
            product_name_candidates=gen_product_name_candidates_for_one_context(product_name_context,product_name_candidates)
            
    else:
        for product_name_context in [noisy_product_name,product_category]:
            product_name_candidates=gen_product_name_candidates_for_one_context(product_name_context,
                                                                                product_name_candidates,parser_chunk_setting)
        
    return product_name_candidates
 

# def gen_product_name_candidates_by_walk_tree(noisy_product_name):    
#     pass 
    
# def gen_product_name_candidates_by_chunk(doc):   
#     product_name_candidate_list={} 
    
#     for chunk in doc.noun_chunks:
#         product_name_candidate=ProductNameCandidate(chunk.root.norm_)
#         for idx in range(chunk.root.left_edge.i,chunk.root.i):
#             current_span=doc[idx: chunk.root.i+1]
#             norm_str=""
#             for token in current_span:
#                 norm_str+=" "+token.norm_
#             norm_str=norm_str[1:]
#             product_name_candidate.add(norm_str)
#         product_name_candidate.add(chunk.root.norm_)
#         product_name_candidate_list[chunk.root.norm_]=product_name_candidate
#     return product_name_candidate_list

def gen_product_name_candidates_by_chunk(doc,parser_chunk_setting):   
    product_name_candidate_list={} 
    root=gen_root_by_doc(doc)
    for chunk in doc.noun_chunks:
        if len(chunk)==1 and chunk.root.pos_ not in[ "PROPN","NOUN"]:
            continue 
        else:  
            if parser_chunk_setting=="root_only" and chunk.root.text  !=root:
                 
                continue 
            else:
                product_name_candidate=ProductNameCandidate(chunk.root.lemma_.lower())
                for idx in range(chunk.root.left_edge.i,chunk.root.i):
                    product_name_candidate.add(doc[idx: chunk.root.i+1].text  )
                product_name_candidate.add(chunk.root.text  )
                product_name_candidate_list[chunk.root.lemma_.lower()]=product_name_candidate
    return product_name_candidate_list
        # for left_node in chunk.root.lefts:
        #     product_name_candidate.add
        # product_name_candidate.add(chunk.text)
        # print(chunk.text, chunk.root.text, chunk.root.dep_,
        #         chunk.root.head.text) 

# def gen_product_name_candidates_by_edge(noisy_product_name):    
#     pass     

#Alarm 8-Piece Security Kit
# gen_mention_candidates("I have had the ring doorbell for years now and it seemed quite natural that I would migrate to a Ring Alarm System. I did. I left ADT and decided upon such as it integrated well with that was already in place. I really prefer the self monitoring and the idea that it’s at such a low cost... $8.50/mthly. I love the ease of installation too. Did it myself and quick too.",
#                        "Ring Batteries","Home Security Systems","Ring Alarm Security System packs the power of whole-home security into an easy DIY package.")