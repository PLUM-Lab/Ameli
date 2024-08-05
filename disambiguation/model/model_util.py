from disambiguation.model.layer.sequence_classifier import *
from disambiguation.model.model_contrastive import   FLAVAEL, UNITER, V2TEL, V2VELCLIP, V2VELCLIPAttributeFromSBERT, V2VELCLIPAttributeFromSBERTText, myPrinter
from disambiguation.model.model_joint_training import AMELI_MLP, AMELIJoint
from disambiguation.model.model_pairwise  import *
from transformers import CLIPProcessor, CLIPModel, BertTokenizer, BertModel,AutoConfig ,AutoTokenizer

import re

from retrieval.model.model import MultiMediaSentenceTransformer 


def gen_model(args,device,model_attribute,train_attribute):
    tokenizer=gen_tokenizer(args)
    model_special_processor=None
    print(f"use model:{model_attribute}")
    if args.model_attribute in ["C_joint","C_joint_adapt" ]:
        config = AutoConfig.from_pretrained(args.pre_trained_dir)
        classifier_trained = True
        num_labels=None
        if config.architectures is not None:
            classifier_trained = any([arch.endswith('ForSequenceClassification') for arch in config.architectures])
        if num_labels is None and not classifier_trained:
            num_labels = 1
        if "deberta-v3" in args.pre_trained_dir:
            feature_model = DebertaV2ForSequenceClassificationFeature.from_pretrained(args.pre_trained_dir, config= config )
        elif "deberta" in args.pre_trained_dir:
            feature_model = DebertaForSequenceClassificationFeature.from_pretrained(args.pre_trained_dir, config= config )
        clip_model = CLIPModel.from_pretrained(args.pre_trained_image_model_dir)
        if args.is_freeze_clip:
            clip_model=freeze(clip_model)
        model =  AMELIJoint( args,device,train_attribute,model_attribute,feature_model,clip_model,args.model_group)
    elif args.model_attribute in ["A_SBERT_ATTRIBUTE"]:
        text_encoder= MultiMediaSentenceTransformer(args.pre_trained_dir)

        tokenizer=text_encoder.tokenizer
        image_encoder= MultiMediaSentenceTransformer(args.pre_trained_image_model_dir) 
        model_special_processor=image_encoder._first_module().processor
        model =  V2VELCLIPAttributeFromSBERT( args,device,train_attribute,model_attribute,text_encoder,image_encoder,has_adapter=model_attribute.has_adapter)
    elif args.model_attribute in ["A_SBERT_ATTRIBUTE_TEXT"]:
        text_encoder= MultiMediaSentenceTransformer(args.pre_trained_dir)

        tokenizer=text_encoder.tokenizer
        
        model =  V2VELCLIPAttributeFromSBERTText( args,device,train_attribute,model_attribute,text_encoder,None,has_adapter=model_attribute.has_adapter)
        
    elif args.model_attribute in ["D_MLP","D_MLP_residual" ]:
         
        model =  AMELI_MLP( args,device,train_attribute,model_attribute,args.model_group)
    
    elif args.model_attribute in[ "B6","B10"]:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        if args.is_freeze_clip:
            clip_model=freeze(clip_model)
        model=AMELIMatch(args,  model_attribute,clip_model,device)
    elif args.model_attribute=="B7":
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        if args.is_freeze_clip:
            clip_model=freeze(clip_model)
        model=AMELIAtten(args,  model_attribute,clip_model,device)
    elif args.model_attribute=="B8":
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        if args.is_freeze_clip:
            clip_model=freeze(clip_model)
        model=AMELIMatchPooling2(args,  model_attribute,clip_model,device)
    elif args.model_attribute in [ "B9"]:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        if args.is_freeze_clip:
            clip_model=freeze(clip_model)
        model=AMELIMatchMultiImage(args,  model_attribute,clip_model,device)    
    elif args.model_attribute in [ "B11"]:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        if args.is_freeze_clip:
            clip_model=freeze(clip_model)
        model=AMELIQuery(args,  model_attribute,clip_model,device)    
    elif args.model_attribute in [ "B12"]:
        model=AMELIQuery(args,  model_attribute,None, device)    
    elif args.model_attribute in ["B13"]:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        model=AMELICross(args,  model_attribute,clip_model, device)    
    elif args.model_attribute in ["B14"]:
        config = AutoConfig.from_pretrained(args.pre_trained_dir)
        classifier_trained = True
        num_labels=None
        if config.architectures is not None:
            classifier_trained = any([arch.endswith('ForSequenceClassification') for arch in config.architectures])
        if num_labels is None and not classifier_trained:
            num_labels = 1

       
        if "deberta-v3" in args.pre_trained_dir:
            feature_model = DebertaV2ForSequenceClassificationFeature.from_pretrained(args.pre_trained_dir, config= config )
        elif "deberta" in args.pre_trained_dir:
            feature_model = DebertaForSequenceClassificationFeature.from_pretrained(args.pre_trained_dir, config= config )
        elif "roberta" in args.pre_trained_dir:
            feature_model = RobertaForSequenceClassificationFeature.from_pretrained(args.pre_trained_dir, config= config )
        else:
            feature_model=  AutoModel.from_pretrained(args.pre_trained_dir)
        model=AMELINLI(args,  model_attribute,feature_model, device)    
    elif args.model_attribute in ["B15"]:
        config = AutoConfig.from_pretrained(args.pre_trained_dir)
        classifier_trained = True
        num_labels=None
        if args.use_image:
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            if args.is_freeze_clip:
                clip_model=freeze(clip_model)
        else:
            clip_model=None
        if config.architectures is not None:
            classifier_trained = any([arch.endswith('ForSequenceClassification') for arch in config.architectures])
        if num_labels is None and not classifier_trained:
            num_labels = 1
        if "deberta-v3" in args.pre_trained_dir:
            feature_model = DebertaV2ForSequenceClassificationFeature.from_pretrained(args.pre_trained_dir, config= config )
        elif "deberta" in args.pre_trained_dir:
            feature_model = DebertaForSequenceClassificationFeature.from_pretrained(args.pre_trained_dir, config= config )
        elif "roberta" in args.pre_trained_dir:
            feature_model = RobertaForSequenceClassificationFeature.from_pretrained(args.pre_trained_dir, config= config )
        else:
            feature_model= DebertaV2ForSequenceClassificationFeature.from_pretrained(args.pre_trained_dir, config= config )#  AutoModel.from_pretrained(args.pre_trained_dir)
        model=AMELINLImage(args,  model_attribute,feature_model,clip_model, device)    
    elif args.model_attribute in ["A7","A_adapt_contrast","A_contrast"]:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        if args.is_freeze_clip:
            clip_model=freeze(clip_model)
         
        model =  V2VELCLIP( args,device,train_attribute,model_attribute,clip_model,has_adapter=model_attribute.has_adapter)
    elif args.model_attribute in ["A_V2TEL_contrast","A_V2TEL" ]:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        if args.is_freeze_clip:
            clip_model=freeze(clip_model)
        model =  V2TEL( args,device,train_attribute,model_attribute,clip_model,has_adapter=model_attribute.has_adapter)
    elif args.model_attribute in ["resnet_adapt_contrast"]:
        
        clip_model=FeatureExtractorResnet152( ) #TODO are affected by wikidiverse modification
        if args.is_freeze_clip:
            clip_model=freeze(clip_model)
        model =  V2VELCLIP( args,device,train_attribute,model_attribute,clip_model,is_resnet=True)
    elif args.model_attribute in ["Text_cross_encoder"]:
        model = TextCrossEncoder( args,device,train_attribute,model_attribute,args.is_freeze_bert,args.is_freeze_clip)
    elif args.model_attribute in ["A_FLAVA"]:
        model=FLAVAEL(args,device,train_attribute,model_attribute,args.is_freeze_bert,args.is_freeze_clip)
    elif args.model_attribute.startswith("A"):
        model =  UNITER( args,device,train_attribute,model_attribute,args.is_freeze_bert,args.is_freeze_clip)
    else:
        raise Exception("wrong model_attribute")
    # freeze(clip_model)
    
    myPrinter(args,model)
    
    
    return model ,tokenizer,model_special_processor

def  gen_output(batch,model,dataset_class,device,args,is_train,is_contrastive=False,model_attribute=None ):
    text_score_before_softmax,image_score_before_softmax=None,None
    loss,text_score,image_score=None,None,None
    if dataset_class in [ "v5","v6","v9" ]:
        processed_inputs,label,entity_mask,entity_id_list,query_id =batch 
        # pixel_values=processed_inputs['pixel_values'] 
        input_ids=processed_inputs['input_ids'] 
        attention_mask=processed_inputs['attention_mask']  
        if "token_type_ids" in processed_inputs:
            token_type_ids =processed_inputs["token_type_ids"]
        else:
            token_type_ids=attention_mask
        if  args.parallel !="data_parallel":
            token_type_ids=token_type_ids.to(device)
        else:
            token_type_ids=token_type_ids.cuda()
        if  args.parallel !="data_parallel":
            entity_mask=entity_mask.to( device)
            # pixel_values=pixel_values.to( device)
            input_ids=input_ids.to( device)
            attention_mask=attention_mask.to(device)
            label=label.to(device)
        else:
            entity_mask=entity_mask.cuda()
            # pixel_values=pixel_values.cuda()
            input_ids=input_ids.cuda()
            attention_mask=attention_mask.cuda()
            label=label.cuda()
        # processed_inputs['pixel_values'] =pixel_values
        # processed_inputs['input_ids'] =input_ids
        # processed_inputs['attention_mask'] =attention_mask
        output,labels,entity_match_score,entity_text_match_score,entity_image_match_score,entity_attribute_match_score=model( input_ids,attention_mask,label,entity_mask , is_train=is_train, 
                        is_contrastive=is_contrastive,token_type_ids=token_type_ids )
    elif dataset_class in [ "v10","select_image_with_nli"]:
        processed_inputs,label,entity_mask,entity_id_list,query_id,mention_entity_image_mask =batch 
        
        pixel_values=processed_inputs['pixel_values'] 
        input_ids=processed_inputs['input_ids'] 
        attention_mask=processed_inputs['attention_mask'] 
        if "token_type_ids" in processed_inputs:
            token_type_ids =processed_inputs["token_type_ids"]
        else:
            token_type_ids=attention_mask
        if  args.parallel !="data_parallel":
            token_type_ids=token_type_ids.to(device)
        else:
            token_type_ids=token_type_ids.cuda()
        if  args.parallel !="data_parallel":
            entity_mask=entity_mask.to( device)
            pixel_values=pixel_values.to( device)
            input_ids=input_ids.to( device)
            attention_mask=attention_mask.to(device)
            label=label.to(device)
        else:
            entity_mask=entity_mask.cuda()
            pixel_values=pixel_values.cuda()
            input_ids=input_ids.cuda()
            attention_mask=attention_mask.cuda()
            labels=label.cuda()
        # processed_inputs['pixel_values'] =pixel_values
        # processed_inputs['input_ids'] =input_ids
        # processed_inputs['attention_mask'] =attention_mask
        if model_attribute.name in["C_joint_adapt","C_joint"]:
            output,labels  ,loss,text_score,image_score,text_score_before_softmax,image_score_before_softmax=model(pixel_values,input_ids,attention_mask,label,entity_mask , is_train=is_train, 
                        is_contrastive=is_contrastive,token_type_ids=token_type_ids )#,entity_match_score,entity_text_match_score,entity_im
        elif model_attribute.name in["B15"]:
            output,text_score =model(pixel_values, input_ids,attention_mask,label,entity_mask , is_train=is_train, 
                        is_contrastive=is_contrastive,token_type_ids=token_type_ids,mention_entity_image_mask=mention_entity_image_mask )
            text_score_before_softmax,image_score,image_score_before_softmax=output,output,output
            
        else:
            output=model(pixel_values, input_ids,attention_mask,label,entity_mask , is_train=is_train, 
                        is_contrastive=is_contrastive,token_type_ids=token_type_ids,mention_entity_image_mask=mention_entity_image_mask )
    elif dataset_class in [ "select_image_with_nli_score_and_retrieval_score"]:
        processed_inputs,label,entity_mask,entity_id_list,query_id,mention_entity_image_mask,retrieval_nli_score_list_tensor =batch 
        
        # pixel_values=processed_inputs['pixel_values'] 
        input_ids=processed_inputs['input_ids'] 
        attention_mask=processed_inputs['attention_mask'] 
        if "token_type_ids" in processed_inputs:
            token_type_ids =processed_inputs["token_type_ids"]
        else:
            token_type_ids=attention_mask
        if  args.parallel !="data_parallel":
            token_type_ids=token_type_ids.to(device)
        else:
            token_type_ids=token_type_ids.cuda()
        if  args.parallel !="data_parallel":
            entity_mask=entity_mask.to( device)
            # pixel_values=pixel_values.to( device)
            input_ids=input_ids.to( device)
            attention_mask=attention_mask.to(device)
            label=label.to(device)
            retrieval_nli_score_list_tensor=retrieval_nli_score_list_tensor.to(device)
        else:
            entity_mask=entity_mask.cuda()
            # pixel_values=pixel_values.cuda()
            input_ids=input_ids.cuda()
            attention_mask=attention_mask.cuda()
            labels=label.cuda()
            retrieval_nli_score_list_tensor=retrieval_nli_score_list_tensor.cuda()
        output,labels  ,loss,text_score,image_score=model( input_ids,attention_mask,label,entity_mask ,retrieval_nli_score_list_tensor, is_train=is_train, 
                    is_contrastive=is_contrastive,token_type_ids=token_type_ids )#,entity_match_score,entity_text_match_score,entity_im
    elif dataset_class in [ "select_image_with_attribute_hash_and_retrieval_score_rich_review"]:
        processed_inputs,lapl,entity_mask,entity_id_list,query_id,mention_entity_image_mask,retrieval_nli_score_list_tensor =batch 
        
        # pixel_values=processed_inputs['pixel_values'] 
        input_ids=processed_inputs['input_ids'] 
        attention_mask=processed_inputs['attention_mask'] 
        if "token_type_ids" in processed_inputs:
            token_type_ids =processed_inputs["token_type_ids"]
        else:
            token_type_ids=attention_mask
        if  args.parallel !="data_parallel":
            token_type_ids=token_type_ids.to(device)
        else:
            token_type_ids=token_type_ids.cuda()
        if  args.parallel !="data_parallel":
            entity_mask=entity_mask.to( device)
            # pixel_values=pixel_values.to( device)
            input_ids=input_ids.to( device)
            attention_mask=attention_mask.to(device)
            label=label.to(device)
            retrieval_nli_score_list_tensor=retrieval_nli_score_list_tensor.to(device)
        else:
            entity_mask=entity_mask.cuda()
            # pixel_values=pixel_values.cuda()
            input_ids=input_ids.cuda()
            attention_mask=attention_mask.cuda()
            labels=label.cuda()
            retrieval_nli_score_list_tensor=retrieval_nli_score_list_tensor.cuda()
        output,labels  ,loss,text_score,image_score=model( input_ids,attention_mask,label,entity_mask ,retrieval_nli_score_list_tensor, is_train=is_train, 
                    is_contrastive=is_contrastive,token_type_ids=token_type_ids )#,entity_m
    elif dataset_class in[ "select_image_sbert_attribute"]:
        processed_inputs,labels,entity_mask,entity_id_list,query_id =batch 
        input_ids=processed_inputs['input_ids'] 
        attention_mask=processed_inputs['attention_mask'] 
        if args.use_image:
            pixel_values=processed_inputs['pixel_values'] 
            pixel_values=pixel_values.cuda()
        
        if  args.parallel !="data_parallel":
            entity_mask=entity_mask.to( device)
             
            input_ids=input_ids.to( device)
            attention_mask=attention_mask.to(device)
            labels=labels.to(device)
        else:
            entity_mask=entity_mask.cuda()
            
            input_ids=input_ids.cuda()
            attention_mask=attention_mask.cuda()
            labels=labels.cuda()
        if   model_attribute.name    in["A_SBERT_ATTRIBUTE"]:
           
            output,_,m_embedding,entity_embedding_list=model(pixel_values,input_ids,attention_mask,labels,entity_mask , is_train=is_train, 
                        is_contrastive=is_contrastive  )
        elif  model_attribute.name    in["A_SBERT_ATTRIBUTE_TEXT"]:
            output,_,m_embedding,entity_embedding_list=model( input_ids,attention_mask,labels,entity_mask , is_train=is_train, 
                        is_contrastive=is_contrastive  )
        if is_train and is_contrastive:
            batch_size=len(m_embedding)
            # batch_size * batch_size * n_cand * d_hid    batch_size * d_hid * 1
            simi = torch.einsum(
                'ijkl,ild->ijk', [entity_embedding_list.expand(batch_size, -1, -1, -1), m_embedding])
            simi = simi.view(batch_size, -1)  # bsz * (bsz * n_cand)
            
            output =  masked_softmax(simi, entity_mask.expand(
                batch_size, -1, -1).view(batch_size, -1))
            labels = gen_contrastive_labels(output,labels)
    elif dataset_class in[ "v3","resnet","select_image","select_image_v2tel","select_image_flava","select_image_sbert_attribute"]:
        processed_inputs,labels,entity_mask,entity_id_list,query_id =batch 
        pixel_values=processed_inputs['pixel_values'] 
        input_ids=processed_inputs['input_ids'] 
        attention_mask=processed_inputs['attention_mask'] 
        if "token_type_ids" in processed_inputs:
            token_type_ids =processed_inputs["token_type_ids"]
        else:
            token_type_ids=attention_mask
        if  args.parallel !="data_parallel":
            entity_mask=entity_mask.to( device)
            pixel_values=pixel_values.to( device)
            input_ids=input_ids.to( device)
            attention_mask=attention_mask.to(device)
            labels=labels.to(device)
            token_type_ids=token_type_ids.to(device)
        else:
            entity_mask=entity_mask.cuda()
            pixel_values=pixel_values.cuda()
            input_ids=input_ids.cuda()
            attention_mask=attention_mask.cuda()
            labels=labels.cuda()
            token_type_ids=token_type_ids.cuda()
        # processed_inputs['pixel_values'] =pixel_values
        # processed_inputs['input_ids'] =input_ids
        # processed_inputs['attention_mask'] =attention_mask
        if  model_attribute.name    in["A_adapt_contrast"]:
            output,labels,image_score=model(pixel_values,input_ids,attention_mask,labels,entity_mask , is_train=is_train, 
                        is_contrastive=is_contrastive )
            image_score_before_softmax=output
            text_score_before_softmax,text_score=output,output
        elif model_attribute.name    in["resnet_adapt_contrast"]:
            output,labels,image_score=model(pixel_values,input_ids,attention_mask,labels,entity_mask , is_train=is_train, 
                        is_contrastive=is_contrastive )#
            image_score_before_softmax=output
            text_score_before_softmax,text_score=output,output  
        elif model_attribute.name not  in[ "A_V2TEL_contrast","A_V2TEL","A_FLAVA","A_SBERT_ATTRIBUTE"]:
            output,labels=model(pixel_values,input_ids,attention_mask,labels,entity_mask , is_train=is_train, 
                        is_contrastive=is_contrastive )#,entity_match_score,entity_text_match_score,entity_image_match_score,entity_attribute_match_score
        elif   model_attribute.name    in["A_SBERT_ATTRIBUTE"]:
           
            output,_,m_embedding,entity_embedding_list=model(pixel_values,input_ids,attention_mask,labels,entity_mask , is_train=is_train, 
                        is_contrastive=is_contrastive  )
            if is_train and is_contrastive:
                batch_size=len(m_embedding)
                # batch_size * batch_size * n_cand * d_hid    batch_size * d_hid * 1
                simi = torch.einsum(
                    'ijkl,ild->ijk', [entity_embedding_list.expand(batch_size, -1, -1, -1), m_embedding])
                simi = simi.view(batch_size, -1)  # bsz * (bsz * n_cand)
                
                output =  masked_softmax(simi, entity_mask.expand(
                    batch_size, -1, -1).view(batch_size, -1))
                labels = gen_contrastive_labels(output,labels)
        else:
            output,_,m_embedding,entity_embedding_list=model(pixel_values,input_ids,attention_mask,labels,entity_mask , is_train=is_train, 
                        is_contrastive=is_contrastive ,token_type_ids=token_type_ids)
            if is_train and is_contrastive:
                batch_size=len(m_embedding)
                # batch_size * batch_size * n_cand * d_hid    batch_size * d_hid * 1
                simi = torch.einsum(
                    'ijkl,ild->ijk', [entity_embedding_list.expand(batch_size, -1, -1, -1), m_embedding])
                simi = simi.view(batch_size, -1)  # bsz * (bsz * n_cand)
                
                output =  masked_softmax(simi, entity_mask.expand(
                    batch_size, -1, -1).view(batch_size, -1))
                labels = gen_contrastive_labels(output,labels)
                
    elif dataset_class in[ "v4","v7"]:
        processed_inputs,label,entity_mask,entity_id_list,query_id ,mention_entity_image_mask=batch 
        pixel_values=processed_inputs['pixel_values'] 
        input_ids=processed_inputs['input_ids'] 
        attention_mask=processed_inputs['attention_mask'] 
        if  args.parallel !="data_parallel":
            entity_mask=entity_mask.to( device)
            pixel_values=pixel_values.to( device)
            input_ids=input_ids.to( device)
            attention_mask=attention_mask.to(device)
            label=label.to(device)
            mention_entity_image_mask=mention_entity_image_mask.to(device)
        else:
            entity_mask=entity_mask.cuda()
            pixel_values=pixel_values.cuda()
            input_ids=input_ids.cuda()
            attention_mask=attention_mask.cuda()
            label=label.cuda()
            mention_entity_image_mask=mention_entity_image_mask.cuda()
        # processed_inputs['pixel_values'] =pixel_values
        # processed_inputs['input_ids'] =input_ids
        # processed_inputs['attention_mask'] =attention_mask
        output,labels,entity_match_score,entity_text_match_score,entity_image_match_score,entity_attribute_match_score=model(pixel_values,input_ids,attention_mask,label,entity_mask , is_train=is_train, 
                        is_contrastive=is_contrastive,mention_entity_image_mask=mention_entity_image_mask )
    elif dataset_class in[ "v2","resnet_text","select_image_resnet_text","text_cross_encoder" ]:
        mention_tokens,mention_image,entity_token_tensor,entity_image_tensor,entity_mask,labels, mention_attribute_tokens, entities_attribute_tokens_list,entity_id_list,query_id=batch 
         
        mention_input_ids,mention_attention_mask,mention_token_type_ids,mention_image=one_object_to_device(mention_tokens,args,device,mention_image)
        entity_input_ids,entity_attention_mask,entity_token_type_ids,entity_image=one_object_to_device(entity_token_tensor,args,device,entity_image_tensor)
        if args.is_gpu:
            if  args.parallel !="data_parallel":
                entity_mask=entity_mask.to( device)
                labels=labels.to(device)
            else:
                entity_mask=entity_mask.cuda()
                labels=labels.cuda()
        output,_,m_embedding,entity_embedding_list=model(mention_input_ids,mention_attention_mask,mention_token_type_ids,mention_image,entity_input_ids,entity_attention_mask,entity_token_type_ids,entity_image,entity_mask, mention_attribute_tokens, entities_attribute_tokens_list, is_train=is_train, 
                            is_contrastive=is_contrastive,labels=labels)
        if is_train and is_contrastive:
            batch_size=len(m_embedding)
            # batch_size * batch_size * n_cand * d_hid    batch_size * d_hid * 1
            simi = torch.einsum(
                'ijkl,ild->ijk', [entity_embedding_list.expand(batch_size, -1, -1, -1), m_embedding])
            simi = simi.view(batch_size, -1)  # bsz * (bsz * n_cand)
            
            output =  masked_softmax(simi, entity_mask.expand(
                batch_size, -1, -1).view(batch_size, -1))
            labels = gen_contrastive_labels(output,labels)
    else:
        exit(1)
    return output ,labels,entity_id_list,query_id,loss,text_score,image_score,text_score_before_softmax,image_score_before_softmax


def gen_contrastive_labels(output,labels):
    (batch_size, batch_size_times_cand_num) = output.shape
    cand_num = int(batch_size_times_cand_num/batch_size)
    label_int_list = []
    for i in range(batch_size):
        original_label=labels[i]
        label_int_list.append(i*cand_num+original_label)
    labels = torch.tensor(label_int_list, device=output.device)
    return labels


def masked_softmax(  tensor, mask):
    input_tensor = tensor.masked_fill((~mask), value=torch.tensor(-1e9))
    
    return input_tensor

def one_object_to_device(mention_tokens,args,device,mention_image):
    input_ids=mention_tokens['input_ids'] 
    attention_mask=mention_tokens['attention_mask'] 
    if "token_type_ids" in mention_tokens:
        token_type_ids =mention_tokens["token_type_ids"]
    else:
        token_type_ids=attention_mask
    if args.is_gpu:
        if  args.parallel !="data_parallel":
            token_type_ids=token_type_ids.to(device)
            mention_image=mention_image.to( device)
            input_ids=input_ids.to( device)
            attention_mask=attention_mask.to(device)
        else:
            token_type_ids=token_type_ids.cuda()
            mention_image=mention_image.cuda()
            input_ids=input_ids.cuda()
            attention_mask=attention_mask.cuda()
    return input_ids,attention_mask,token_type_ids,mention_image


def load_specific_checkpoint_to_AMELIJoint(model,checkpoint_path,model_group):
    if model_group in ["all","text"]:
        ameli_nli_image_model=extract_module(torch.load(checkpoint_path),"module.")
        #load_711 AMELINLImage to AMELIJoint.AMELITextModel
        #  model.load_state_dict(model_dict,strict=False)
        model.module.text_model.load_state_dict(ameli_nli_image_model)
    #load 750 V2VELCLIP to AMELIJoint.AMELIImageModel 
    if model_group in ["all","image"]:
        image_checkpoint_dir= "output/runs/00750-/base.pt"
        v2velclip_model=extract_module( torch.load(image_checkpoint_dir),"module.encoder.encoder.")
        model.module.image_model.encoder.encoder.load_state_dict(v2velclip_model )
    return model

def extract_module(state_dict,model_name_key):
    model_dict = OrderedDict()
    pattern = re.compile(model_name_key)
    for k,v in state_dict.items():
        if re.search(model_name_key, k):
            model_dict[re.sub(pattern, '', k)] = v
 
    return model_dict

def gen_tokenizer(args):
    if "lxmert" in args.pre_trained_dir:
        from transformers import LxmertTokenizer
        tokenizer = LxmertTokenizer.from_pretrained(args.pre_trained_dir)
    elif args.pre_trained_dir=="all-mpnet-base-v2":
        tokenizer= None
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.pre_trained_dir)
    return tokenizer


def freeze(model):
    model.requires_grad_(False)
    return model 


# def gen_deberta_feature():