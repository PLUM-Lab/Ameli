


class Extractor:
    def _gen_postfix(self,attribute_key):
        postfix="\n"+attribute_key+":"
        return postfix
    
    def generate_per_review(self,args, prefix,prompt_text,attribute_key,candidate_attribute_list,is_constrained_beam_search,review_dataset_json,review_image_dir):
        return None,None,[] ,[],[] 