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
class ImageSearcher:
    def __init__(self,image_encoder_checkpoint,logger)  :
        #First, we load the respective CLIP model
        self.model = SentenceTransformer(image_encoder_checkpoint)
        # self.model._first_module().max_seq_length =77
        self.tokenizer_for_truncation=  CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.logger=logger

    def encode_corpus(self,entity_dict , image_dir,use_precomputed_embeddings_flag):
        # Now, we need to compute the embeddings
        # To speed things up, we destribute pre-computed embeddings
        # Otherwise you can also encode the images yourself.
        # To encode an image, you can use the following code:
        # from PIL import Image
        # img_emb = model.encode(Image.open(filepath))

        emb_folder=os.path.join(get_father_dir(image_dir))
        emb_filename = 'corpus_image_embeddings.pkl'
        emb_dir=os.path.join(emb_folder,"embed",emb_filename)
        if use_precomputed_embeddings_flag and   os.path.exists(emb_dir): 
           
            with open(emb_dir, 'rb') as fIn:
                emb_file =  pickle.load(fIn) #torch.load(fIn,map_location={'cuda':'cpu'}   )#pickle.load(fIn) 
                self.entity_image_num_list,self.img_emb,self.entity_name_list,self.entity_img_path_list=emb_file["entity_image_num_list"],emb_file["img_emb"],emb_file["entity_name_list"] ,emb_file["entity_img_path_list"]
         
            print("Images:", len(self.img_emb))
        else:
            batch_size=256
            live_num_in_current_batch=0
            live_num=0
            
            current_image_batch=[]
            total_img_emb= torch.tensor([],device= torch.device('cuda'))
            entity_name_list=[]
            entity_img_path_list=[]
            current_image_path_list=[]
        
            entity_image_num_list=[]
            total_img_num=0
            for entity_id,entity  in  tqdm(entity_dict.items()):
                img_path_list=entity.image_path_list
                entity_image_num_list.append(len(img_path_list))
                total_img_num+=len(img_path_list)
                # img_path_list_str=entity.img_path
                # img_path_list=img_path_list_str.split("[AND]")
                # is_img_downloaded_list=entity.is_img_downloaded.split("[AND]")
                for idx,img_path  in enumerate(img_path_list):
                    if  os.path.exists(img_path):
                        
                        try:
                            image=Image.open(img_path)
                            
                        except Exception as e:
                            logging.info(f"{e} {img_path}")
                            continue 
                        current_image_batch.append(image)
                        current_image_path_list.append(img_path)
                        entity_name_list.append(entity_id)
                        entity_img_path_list.append(img_path)
                        live_num_in_current_batch+=1
                        if live_num_in_current_batch%batch_size==0:
                            try:
                                img_emb = self.model.encode(current_image_batch, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True)
                            except Exception as e:
                                logging.info(f"encode issue: {current_image_path_list}, {e}")
                                # print(f"current_image_path_list: {e}")
                                continue 
                            total_img_emb=torch.cat([total_img_emb,img_emb],0)
                            live_num_in_current_batch=0
                            current_image_batch=[]
                            current_image_path_list=[]
                            live_num+=batch_size
                            # print(live_num/img_len)
                    else:
                        logging.info(f"miss image {img_path}")
                # assert total_img_num==live_num+live_num_in_current_batch, f"total_img_num {total_img_num} != live_num {live_num}+{live_num_in_current_batch}"
                if total_img_num!=live_num+live_num_in_current_batch:
                    
                    raise Exception(f"total_img_num {total_img_num} != live_num {live_num}+{live_num_in_current_batch}")
            if len(current_image_batch)>0:
                total_img_emb=self.encode_image_list(current_image_batch,batch_size,current_image_path_list,total_img_emb)
                live_num+=len(current_image_batch)
            assert total_img_num==live_num, f"total_img_num {total_img_num} != live_num {live_num}"
            self.img_emb = total_img_emb
            self.entity_name_list=entity_name_list
            self.entity_img_path_list=entity_img_path_list
            self.entity_image_num_list=entity_image_num_list
            # self.img_emb = self.model.encode([Image.open(os.path.join(img_folder,filepath)) for filepath in self.img_names], batch_size=128, convert_to_tensor=True, show_progress_bar=True)
            emb_file = { "entity_image_num_list":entity_image_num_list,"img_emb": self.img_emb, "entity_name_list": self.entity_name_list ,"entity_img_path_list":entity_img_path_list}            
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
    
    def search(self,img_path, top_k=3):
        image=Image.open(img_path)
        
        # First, we encode the query (which can either be an image or a text string)
        query_emb = self.model.encode([image], convert_to_tensor=True, show_progress_bar=False )
        
        # Then, we use the util.semantic_search function, which computes the cosine-similarity
        # between the query embedding and all image embeddings.
        # It then returns the top_k highest ranked images, which we output
        hits = util.semantic_search(query_emb, self.img_emb, top_k=top_k)[0]
        
        retrieved_entity_name_list=[]
        for hit in hits:
            retrieved_entity_name_list.append(self.entity_name_list[hit['corpus_id']])
        
        if config.verbose==True:
            print(f"Query:{img_path}")
            for hit in hits:
                # print(self.img_names[hit['corpus_id']])
                image_path=self.entity_img_path_list[hit['corpus_id']]
                # image = Image.open(image_path)
                # image.show()
                print(image_path)
        
        return hits,retrieved_entity_name_list
                
 





