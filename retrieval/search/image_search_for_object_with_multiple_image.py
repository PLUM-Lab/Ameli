import sentence_transformers
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import glob
import torch
import pickle
import zipfile 
import os
from tqdm.autonotebook import tqdm
from retrieval.search.util.semantic_search_engine import semantic_search_by_list_attribute_of_one_query
from retrieval.utils.config import config
from util.read_example import get_father_dir
torch.set_num_threads(4)
from transformers import CLIPTokenizer
import logging 
class ImageSearcherForObjectWithMultipleImage:
    def __init__(self,image_encoder_checkpoint,logger,corpus_pickle_dir,is_only_in_category,mode)  :
        #First, we load the respective CLIP model
        self.model = SentenceTransformer(image_encoder_checkpoint)
        # self.model._first_module().max_seq_length =77
        self.mode=mode
        # self.tokenizer_for_truncation=  CLIPTokenizer.from_pretrained(image_encoder_checkpoint)#"openai/clip-vit-base-patch32"
        self.logger=logger
        self.is_only_in_category=is_only_in_category 
        self.corpus_pickle_dir=corpus_pickle_dir
        
    def encode_one_batch(self, total_img_emb,total_entity_img_path_list,total_num,total_entity_name_list_in_image_level,live_num_in_current_batch,image_in_current_batch,image_path_list_in_current_batch,entity_name_list_in_current_batch):
        is_success=True 
        try:
            img_emb = self.model.encode(image_in_current_batch, batch_size=live_num_in_current_batch, convert_to_tensor=True, show_progress_bar=True)
        except Exception as e:
            logging.info(f"encode issue: {image_path_list_in_current_batch},{entity_name_list_in_current_batch} {e}")
            # cur_entity_image_num-=1
            live_num_in_current_batch=0
            image_in_current_batch=[]
            image_path_list_in_current_batch=[]
            entity_name_list_in_current_batch=[]
            # print(f"current_image_path_list: {e}")
            is_success=False 
        total_img_emb=torch.cat([total_img_emb,img_emb],0)
        total_entity_img_path_list.extend(image_path_list_in_current_batch)
        total_num+=live_num_in_current_batch
        total_entity_name_list_in_image_level.extend(entity_name_list_in_current_batch)
        live_num_in_current_batch=0
        image_in_current_batch=[]
        image_path_list_in_current_batch=[]
        entity_name_list_in_current_batch=[]
        return  total_img_emb,total_entity_img_path_list,total_num,total_entity_name_list_in_image_level,live_num_in_current_batch,image_in_current_batch,image_path_list_in_current_batch,entity_name_list_in_current_batch

    def encode_corpus(self,entity_dict , image_dir,use_precomputed_embeddings_flag):
        # Now, we need to compute the embeddings
        # To speed things up, we destribute pre-computed embeddings
        # Otherwise you can also encode the images yourself.
        # To encode an image, you can use the following code:
        # from PIL import Image
        # img_emb = model.encode(Image.open(filepath))
        # emb_folder="/home/menglong/workspace/code/multimodal/multimodal_entity_linking/product_scraper/dataset_construction/bestbuy/data/final/v3" #  
        # emb_folder=os.path.join(get_father_dir(image_dir))
        os.makedirs(self.corpus_pickle_dir,exist_ok=True)
        emb_filename = 'corpus_image_embeddings.pkl'
        emb_dir=os.path.join(self.corpus_pickle_dir,emb_filename)
        if use_precomputed_embeddings_flag and   os.path.exists(emb_dir): 
           
            with open(emb_dir, 'rb') as fIn:
                # unpkl = pickle.Unpickler(fIn)
                # emb_file=unpkl.load()
                emb_file =  pickle.load(fIn) #torch.load(fIn,map_location={'cuda':'cpu'}   )#pickle.load(fIn) 
                self.entity_dict,self.entity_image_num_list,self.img_emb,self.entity_name_list,self.entity_img_path_list,self.entity_name_list_in_image_level=emb_file["entity_dict"],emb_file["entity_image_num_list"],emb_file["img_emb"],emb_file["entity_name_list"] ,emb_file["entity_img_path_list"],emb_file["entity_name_list_in_image_level"]
         
            print("Images:", len(self.img_emb))
        else:
            batch_size=1024
            live_num_in_current_batch=0
            image_in_current_batch=[]
            image_path_list_in_current_batch=[]
            entity_name_list_in_current_batch=[]
            
            total_num=0
            total_img_emb= torch.tensor([],device= torch.device('cuda'))
            total_entity_img_path_list=[]
            total_entity_name_list_in_image_level=[]
            
            entity_name_list_in_entity_level=[]
            entity_image_num_list_in_entity_level=[]
            # total_img_num=0
            save_step=5000
            for entity_position,(entity_id,entity)  in  tqdm(enumerate(entity_dict.items())):
                img_path_list=entity.image_path_list
                entity_image_num_list_in_entity_level.append(len(img_path_list))
                entity_name_list_in_entity_level.append(entity_id)
                
                # cur_entity_image_num=len(img_path_list)     
                # img_path_list_str=entity.img_path
                # img_path_list=img_path_list_str.split("[AND]")
                # is_img_downloaded_list=entity.is_img_downloaded.split("[AND]")
                for idx,img_path  in enumerate(img_path_list):
                    if  os.path.exists(img_path):
                        
                        try:
                            image=Image.open(img_path)
                            
                        except Exception as e:
                            logging.info(f"{e} {img_path}")
                            # cur_entity_image_num-=1
                            continue 
                        image_in_current_batch.append(image)
                        image_path_list_in_current_batch.append(image_path_list_in_current_batch)
                        entity_name_list_in_current_batch.append(entity_id)
                        live_num_in_current_batch+=1
                        if live_num_in_current_batch%batch_size==0:
                            total_img_emb,total_entity_img_path_list,total_num,total_entity_name_list_in_image_level,live_num_in_current_batch,image_in_current_batch,image_path_list_in_current_batch,entity_name_list_in_current_batch=self.encode_one_batch( total_img_emb,total_entity_img_path_list,total_num,total_entity_name_list_in_image_level,live_num_in_current_batch,image_in_current_batch,image_path_list_in_current_batch,entity_name_list_in_current_batch)
                    else:
                        logging.info(f"miss image {img_path}")
                        # cur_entity_image_num-=1
                
                if entity_position%save_step==0:
                    emb_file = {"entity_dict":entity_dict, "entity_image_num_list":entity_image_num_list_in_entity_level,"img_emb":  total_img_emb, "entity_name_list":  entity_name_list_in_entity_level ,"entity_img_path_list":total_entity_img_path_list,"entity_name_list_in_image_level":total_entity_name_list_in_image_level}            
                    pickle.dump( emb_file, open(emb_dir , "wb" ) )
                        
                # total_img_num+=cur_entity_image_num      
                # assert total_img_num==total_num+live_num_in_current_batch, f"total_img_num {total_img_num} != live_num {total_num}+{live_num_in_current_batch}"
                # if entity_id>20:
                #     break 
                print(f"finish entity_position {entity_position},{live_num_in_current_batch}")
            if len(image_in_current_batch)>0:
                total_img_emb,total_entity_img_path_list,total_num,total_entity_name_list_in_image_level,live_num_in_current_batch,image_in_current_batch,image_path_list_in_current_batch,entity_name_list_in_current_batch=self.encode_one_batch( total_img_emb,total_entity_img_path_list,total_num,total_entity_name_list_in_image_level,live_num_in_current_batch,image_in_current_batch,image_path_list_in_current_batch,entity_name_list_in_current_batch)

                # total_img_emb=self.encode_image_list(image_in_current_batch,batch_size,image_path_list_in_current_batch,total_img_emb)
                # total_num+=len(image_in_current_batch)
            # assert total_img_num==total_num, f"total_img_num {total_img_num} != live_num {total_num}" 
            self.img_emb = total_img_emb
            self.entity_name_list=entity_name_list_in_entity_level
            self.entity_img_path_list=total_entity_img_path_list
            self.entity_name_list_in_image_level=total_entity_name_list_in_image_level
            self.entity_image_num_list=entity_image_num_list_in_entity_level
            self.entity_dict=entity_dict
            # self.img_emb = self.model.encode([Image.open(os.path.join(img_folder,filepath)) for filepath in self.img_names], batch_size=128, convert_to_tensor=True, show_progress_bar=True)
            emb_file = {"entity_dict":entity_dict, "entity_image_num_list":entity_image_num_list_in_entity_level,"img_emb": self.img_emb, "entity_name_list": self.entity_name_list ,"entity_img_path_list":total_entity_img_path_list,"entity_name_list_in_image_level":total_entity_name_list_in_image_level}            
            pickle.dump( emb_file, open(emb_dir , "wb" ) )
            print("Finish encoding, Images:", len(self.img_emb))
             
    def encode_image_list(self,current_image_batch,batch_size,current_image_path_list,total_img_emb):
        try:
            img_emb = self.model.encode(current_image_batch, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True)
        except Exception as e:
            logging.info(f"encode issue: {current_image_path_list}, {e}")
            # print(f"current_image_path_list: {e}")
            return  total_img_emb
        total_img_emb=torch.cat([total_img_emb,img_emb],0)
        return total_img_emb
    
 
                
    def search(self,img_path_list, gold_entity_category,top_k=3):
        if self.is_only_in_category:
            filtered_img_emb,filtered_entity_name_list,filtered_entity_img_path_list,filtered_entity_image_num_list,filtered_entity_dict=filter_to_specific_category(self.img_emb,self.entity_name_list,self.entity_img_path_list,self.entity_image_num_list,self.entity_dict,gold_entity_category)
        else:
            filtered_img_emb,filtered_entity_name_list,filtered_entity_img_path_list,filtered_entity_image_num_list,filtered_entity_dict=self.img_emb,self.entity_name_list,self.entity_img_path_list,self.entity_image_num_list,self.entity_dict
        img_list=[]
        # total_query_img_emb= torch.tensor([],device= torch.device('cuda'))
        for img_path in img_path_list:
            try:
                image=Image.open(img_path)
                
                # total_query_img_emb=torch.cat([total_query_img_emb,query_emb],0)
            except Exception as e:
                print(f"{img_path}, {e}")
                continue 
            img_list.append(image)
            # First, we encode the query (which can either be an image or a text string)
        
        total_query_img_emb = self.model.encode(img_list, convert_to_tensor=True, show_progress_bar=False )
        # Then, we use the util.semantic_search function, which computes the cosine-similarity
        # between the query embedding and all image embeddings.
        # It then returns the top_k highest ranked images, which we output
        hits = semantic_search_by_list_attribute_of_one_query(total_query_img_emb,filtered_img_emb, top_k=top_k, 
                                                              entity_image_num_list =filtered_entity_image_num_list,mode=self.mode )[0]
        
        retrieved_entity_name_list=[]
        score_dict={}
        for idx,hit in enumerate(hits):
            corpus_id=filtered_entity_name_list[hit['corpus_id']]
            retrieved_entity_name_list.append(corpus_id)
            score_dict[corpus_id]={"idx":idx,"score":hit['score']}
        if config.verbose==True:
            print(f"Query:{img_path}")
            for hit in hits:
                # print(self.img_names[hit['corpus_id']])
                image_path_list=filtered_entity_dict[filtered_entity_name_list[hit['corpus_id']]].image_path_list
                for image_path in image_path_list:
                # image = Image.open(image_path)
                # image.show()
                    print(image_path)
        
        return None ,retrieved_entity_name_list,score_dict





def filter_to_specific_category(img_emb,entity_name_list,entity_img_path_list,entity_image_num_list,entity_dict,gold_entity_category):
        
    filtered_img_emb= torch.tensor([],device= torch.device('cuda'))
    filtered_entity_name_list=[]
    filtered_entity_img_path_list=[]
    filtered_entity_dict={}
    filtered_entity_image_num_list=[]
    start_idx=0
    for entity_relative_position,(entity_id,entity_object) in enumerate(entity_dict.items()):
        image_num=entity_image_num_list[entity_relative_position]
        end_idx=start_idx+image_num 
        
        if entity_object.entity_category==gold_entity_category:
            filtered_entity_dict[entity_id]=entity_object
            filtered_entity_image_num_list.append(entity_image_num_list[entity_relative_position])
            filtered_entity_name_list.append(entity_name_list[entity_relative_position])
            
            
            corresponding_img_emb=img_emb[start_idx:end_idx,:]
            filtered_img_emb=torch.cat([filtered_img_emb,corresponding_img_emb],0)
            corresponding_entity_img_path_list=entity_img_path_list[start_idx:end_idx]
            filtered_entity_img_path_list.extend(corresponding_entity_img_path_list)
        start_idx=end_idx 
        
    return filtered_img_emb,filtered_entity_name_list,filtered_entity_img_path_list,filtered_entity_image_num_list,filtered_entity_dict