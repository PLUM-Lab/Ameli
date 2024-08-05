

       

class CandidateChooser:
    def __init__(self,candidate_base,candidate_mode,max_candidate_num)  :
        self.candidate_base=candidate_base
        self.candidate_mode=candidate_mode
        self.max_candidate_num=max_candidate_num
        if  self.candidate_base in ["by_gold","by_review"]:
            self.similar_image_product_num=self.max_candidate_num/2
            self.similar_text_product_num=self.max_candidate_num-self.similar_image_product_num-1
        else:
            self.similar_image_product_num=0
            self.similar_text_product_num=self.max_candidate_num-1
        self.is_count_gold=False
            
    def _gen_similar_product_list(self,product_json_dict,review_id_product_json_dict,gold_product_id,
                                    review_id):
        if  self.candidate_base=="by_gold":
            product_json= product_json_dict[gold_product_id]
            similar_products=product_json["similar_product_id"]
            product_id_with_similar_image=product_json["product_id_with_similar_image"]
        elif self.candidate_base=="by_review":
            review_product_json= review_id_product_json_dict[review_id]
            similar_products=review_product_json["product_id_with_similar_text_by_review"]
            product_id_with_similar_image=review_product_json["product_id_with_similar_image_by_review"]
        else:
            review_product_json= review_id_product_json_dict[review_id]
            similar_products=review_product_json["fused_candidate_list"]
            product_id_with_similar_image=[]
        return similar_products,product_id_with_similar_image
            
    def _add_candidate_from_one_list(self,product_id_list,max_num,product_json_dict,negative_rel_docs,gold_product_id,review_id) :
        cur_similar_product_num=0
        is_find_gold=False
        if review_id not in negative_rel_docs  :
            negative_rel_docs[review_id] = set()
        for similar_product_id in product_id_list:
            if  gold_product_id!=similar_product_id  :
                if similar_product_id not in negative_rel_docs[review_id]:
                    negative_rel_docs[review_id].add(similar_product_id)
                    cur_similar_product_num+=1
            else:
                if self.is_count_gold:
                    cur_similar_product_num+=1
                is_find_gold=True
            if cur_similar_product_num>=max_num:
                break 
        # if len(product_id_list)>0 and not is_find_gold:
        #     print("stop")
        return negative_rel_docs
    
    def choose_candidate(self, product_json_dict,review_id_product_json_dict,gold_product_id,
                                    review_id,negative_rel_docs,empty_negative_example_num):
        
        product_id_list_with_similar_text,product_id_list_with_similar_image=self._gen_similar_product_list( product_json_dict,review_id_product_json_dict,gold_product_id,
                                    review_id)
        negative_rel_docs=self._add_candidate_from_one_list(product_id_list_with_similar_text,self.similar_text_product_num,product_json_dict,negative_rel_docs,gold_product_id,review_id) 
        negative_rel_docs=self._add_candidate_from_one_list( product_id_list_with_similar_image,self.similar_image_product_num,product_json_dict,negative_rel_docs,gold_product_id,review_id)
        if len(negative_rel_docs[review_id])<=0:
            empty_negative_example_num+=1
        return negative_rel_docs,empty_negative_example_num



class End2EndCandidateChooser(CandidateChooser):    
    def __init__(self,candidate_base,candidate_mode,max_candidate_num)  :
        super().__init__(candidate_base,candidate_mode,max_candidate_num)
        self.similar_text_product_num=self.max_candidate_num
        self.is_count_gold=True
    
class Hard10easy9(CandidateChooser):
    def __init__(self,candidate_base,candidate_mode,max_candidate_num):
        super().__init__(candidate_base,candidate_mode,max_candidate_num)
        self.similar_hard_product_num=int(max_candidate_num/2)
        self.similar_easy_product_num=max_candidate_num-self.similar_hard_product_num-1
        
    def _add_candidate_from_one_list(self,product_id_list,max_num,product_json_dict,negative_rel_docs,gold_product_id,review_id) :
        gold_product_json= product_json_dict[gold_product_id]
        gold_product_category=gold_product_json["product_category"]
        cur_similar_easy_product_num=0
        cur_similar_hard_product_num=0
        if review_id not in negative_rel_docs  :
            negative_rel_docs[review_id] = set()
        for similar_product_id in product_id_list:
            if  gold_product_id!=similar_product_id:
                if similar_product_id not in negative_rel_docs[review_id]:
                    product_json= product_json_dict[similar_product_id]
                    product_category=product_json["product_category"]
                    if product_category != gold_product_category:
                        if self.similar_easy_product_num>cur_similar_easy_product_num:
                            negative_rel_docs[review_id].add(similar_product_id)
                            cur_similar_easy_product_num+=1
                    else:
                        if self.similar_hard_product_num>cur_similar_hard_product_num:
                            negative_rel_docs[review_id].add(similar_product_id)
                            cur_similar_hard_product_num+=1
            if cur_similar_easy_product_num>=self.similar_easy_product_num and cur_similar_hard_product_num>=self.similar_hard_product_num:
                break 
        return negative_rel_docs 
    
class EasyCandidateChooser( CandidateChooser):
    def _add_candidate_from_one_list(self,product_id_list,max_num,product_json_dict,negative_rel_docs,gold_product_id,review_id) :
        gold_product_json= product_json_dict[gold_product_id]
        gold_product_category=gold_product_json["product_category"]
        cur_similar_product_num=0
        if review_id not in negative_rel_docs  :
            negative_rel_docs[review_id] = set()
        for similar_product_id in product_id_list:
            if  gold_product_id!=similar_product_id:
                if similar_product_id not in negative_rel_docs[review_id]:
                    product_json= product_json_dict[similar_product_id]
                    product_category=product_json["product_category"]
                    if product_category != gold_product_category:
                        negative_rel_docs[review_id].add(similar_product_id)
                        cur_similar_product_num+=1
            if cur_similar_product_num>=max_num:
                break 
        return negative_rel_docs 