import spacy
from torch import prod
from spacy.matcher import Matcher
 
from nltk.tokenize import sent_tokenize, word_tokenize
import spacy
from spacy.matcher import PhraseMatcher

from bestbuy.src.extract_mention.name_entity_recognition.util.data_util import ProductNameCandidate 
nlp = spacy.load("en_core_web_lg")





def gen_overlap_mention_candidates(noisy_product_name,product_category,product_desc,review):
    mention_candidate_list=[]
    mention_candidate_list.extend(gen_overlap_mention(noisy_product_name,review) )
    mention_candidate_list.extend(gen_overlap_mention(product_category,review) )
    mention_candidate_list.extend(gen_overlap_mention(product_desc,review) )
    mention_candidate_list=list(set(mention_candidate_list))
    fake_root=review[-3:]
    product_name_candidate=ProductNameCandidate(fake_root)
    product_name_candidate.candidate_text_list.extend(mention_candidate_list)
    product_name_candidate_object_list={} 
    product_name_candidate_object_list[fake_root]=product_name_candidate
    return mention_candidate_list,product_name_candidate_object_list
def gen_overlap_mention(product_context,review):
    review_by_token=review.split(" ")
    overlapped = set(product_context.split(" ")).intersection(review_by_token)
    span=""
    span_list=[]
    is_start_overlap=False 
    for token in review_by_token:
        if token in overlapped:
            is_start_overlap=True 
            span+=token+" "
        else:
            if is_start_overlap:
                is_start_overlap=False 
                doc=nlp(span[:-1] )
                for chunk in doc.noun_chunks:
                    if chunk.root.pos_  in[ "PROPN","NOUN"]:
                        span_list.append(chunk.text)
                span=""
                 
    return span_list 
    
# gen_overlap_mention("Experience clear internet calls a with simple plug-and-play USB connection and a noise-canceling mic. Rigid left-sided mic boom is moveable and can be tucked out of the way when you're not using it. In-line controls let you control volume or mute without interrupting calls. Laser-tuned drivers deliver enhanced digital audio from your favorite music and games.Explore the gear you need for content creation.Learn about content creator gear for vlogging and more","I can hear clearly and speak clearly have not heard my Manager's complain about back ground noses it\u2019s comfortable because I work 8 hours with them so no pain at the end of my shift. Plug in and it works no extra I am a no fuse person I like easy !!!!")    