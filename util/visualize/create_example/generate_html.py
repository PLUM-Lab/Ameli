 
import webbrowser

import os 
import pandas as pd
import shutil 
import dominate
from dominate.tags import * 
from tqdm import tqdm    
import os 
import pandas as pd

from util.visualize.create_example.read_example import read_example


def get_father_dir(splited_dir):
    from pathlib import Path
    path = Path(splited_dir)
    whole_dataset_dir=path.parent.absolute()
    return whole_dataset_dir
  
    
  
  

def tar_example(tar_dir,out_path):
    
    source_dir=out_path
    import tarfile
    with tarfile.open(os.path.join(tar_dir,f"{os.path.basename(source_dir)}_example.tar"), "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


def save_one_report(i,example_num_in_one_report,example_dict,review_dict,out_path):
    key_list=list(example_dict.keys())[i*example_num_in_one_report:(i+1)*example_num_in_one_report]
    report_list=[]
    for key in key_list:
        example=example_dict[key]
        review=review_dict[example.review_id]
        report_list.append([  review.text,review.review_id, ""])
    with pd.ExcelWriter(os.path.join(out_path,f"report{i}.xlsx")) as writer:  
       
        df = pd.DataFrame(report_list, columns =['Review Text', 'Review Id', 'Target Product Position (1-21)'])
        
        df.to_excel(writer,sheet_name= "entity_linking" )                          

def generate_report_sheet( out_path,out_report_path,example_dict,review_dict,report_num,example_num_in_one_report):  
    
    report_list = []
    if report_num>1:
        for i in range(report_num):
            save_one_report(i, example_num_in_one_report,example_dict,review_dict,out_path)
    else:
        for key,value in example_dict.items():
            example=example_dict[key]
            review=review_dict[example.review_id]
            report_list.append([review.review_id, review.text,example.reshuffled_target_product_id_position+1,example.target_product_id,""])
            
        df = pd.DataFrame(report_list, columns =['review_id', 'review_text','reshuffled_target_product_id_position','target_product_id','human_predict'])
        df.to_csv(out_report_path,index=False)                          
import json
def generate_attribute_annotation_json(example_dict,review_dict,product_dict,output_attribute_annotation_json,is_add_gold_product):
    output_json={}
    for key,value in example_dict.items():
            
        review,products,gold_product_position, target_product_id=prepare_data(example_dict[key],review_dict,product_dict,is_add_gold_product)
        output_review={}
        output_review["review_id"]=review.review_id
        # output_review["text"]=review.text
        output_review["gold_attribute_for_predicted_category"]=review.predicted_attribute_dict[f"Gold Attribute"] 
        output_json[str(review.review_id)]=[output_review]
    with open(output_attribute_annotation_json, 'w', encoding='utf-8') as fp:
        json.dump(output_json, fp, indent=4)

def generate(  out_path, data_path,similar_products_path,product_image_dir,
             review_image_dir,generate_claim_level_report,example_num=None ,is_reshuffle=True ,is_need_image=True,is_tar=True,is_need_corpus=True,
             is_generate_report_for_human_evaluation=False,is_add_gold=True,is_add_gold_at_end=True,html_template=None,
             is_generate_attribute_annotation_json=False,is_add_gold_in_html=False ):
        
    needed_field=None 
    
    
      
    example_dict,review_dict,product_dict =read_example( data_path,is_reshuffle,similar_products_path,is_need_corpus,is_add_gold,is_add_gold_at_end)
    if example_num is None:
        example_num=len(example_dict)
    example_num_in_one_page=10
    page_num=example_num/example_num_in_one_page   
    if page_num<1:
        page_num=1
    out_path=os.path.join("output/HTML_to_show",out_path)
    os.makedirs(out_path,exist_ok=True)
    for i in tqdm(range(int(page_num))):
      
        create_one_html_with_scrollbar(example_num_in_one_page,i,example_dict,review_dict,product_dict ,out_path,html_template,is_add_gold_in_html)
         
        
      
    if generate_claim_level_report:
        generate_report_sheet( out_path,os.path.join(out_path,"report.csv"),example_dict,review_dict ,1,None )
        if is_generate_report_for_human_evaluation:
            generate_report_sheet( out_path,os.path.join(out_path,"human_evaluation_report.csv"),example_dict,review_dict ,12,330)
        print("begin to extract image")
        if is_need_image:
            extract_images(  out_path,example_dict,review_dict,product_dict ,product_image_dir,review_image_dir)
        tar_dir=get_father_dir(out_path)
        print("begin to tar")
        if is_tar:
            tar_example(tar_dir,out_path )        
        if is_generate_attribute_annotation_json:
            generate_attribute_annotation_json(example_dict,review_dict,product_dict,os.path.join(out_path,"annotation.json"),is_add_gold_in_html)
            
    
    return example_dict
 

def extract_images(  out_path,example_dict,review_dict,product_dict ,product_image_dir,review_image_dir):
    target_image_corpus=os.path.join(out_path ,"images") 
    for key,value in tqdm(example_dict.items()):
        example=example_dict[key]
        review,products=prepare_data(example,review_dict,product_dict)
        extract_images_for_one_list(review.img_list,review_image_dir,target_image_corpus)
        for product in products:
            extract_images_for_one_list(product.img_list,product_image_dir,target_image_corpus)
        
     
 

def split_image(image_corpus,splited_data_path,filepath,is_copy):
    source_path=os.path.join(image_corpus,filepath)
    target_path=os.path.join(splited_data_path ,filepath)
    
    # 
    if not os.path.exists(target_path):
        if os.path.exists(source_path):
            if is_copy :
                shutil.copy(source_path, target_path)
            else:
                os.rename(source_path,target_path )   
         
def extract_images_for_one_list(img_list,source_img_path,target_image_corpus):
    is_copy=True 
    for filepath in  img_list:
        
        real_image_name=filepath
            # print("ERROR! in extract_images_for_one_list")
            # continue 
            
        split_image(source_img_path,target_image_corpus,real_image_name,is_copy)
     
     
# def create_one_html(example_num_in_one_html,html_index,example_dict,review_dict,product_dict  , out_path,needed_field):
#     doc = dominate.document(title='examples'+str(html_index))
#     with doc.head:
#         link(rel='stylesheet', href='style.css')
#         script(type='text/javascript', src='script.js')

#     with doc:
#         with div() as body:
#             attr(cls='body')
#             j=-1
#             for key,value in example_dict.items():
#                 j+=1
#                 expected_beginning_num=html_index*example_num_in_one_html 
#                 if j>=(html_index+1)*example_num_in_one_html:
#                     break
#                 elif j<expected_beginning_num:
#                     continue
#                 example=example_dict[key]
#                 review_id=example.review_id  
#                 candidate_product_id_list  =example.candidate_product_id_list
#                 review=review_dict[review_id]
#                 products=[]
#                 for  product_id in candidate_product_id_list:
#                     product=product_dict[product_id]
#                     products.append(product) 
                
                 
#                 body.add(h1(f"review id: {review.review_id}"))
#                 body.add(h2( f"review text: {review.text}"  ))
#                 body.add(h2("review image"))
#                 with body.add(ol()):
#                     for img_path in review.img_list:
#                         li(img(src=img_path,alt=""))
#                 body.add(h2("candidate products"))
#                 with body.add(ol()):
#                     for product in products:
#                         with  li(h3(a("product id",href=product.product_id))) :
#                             h3(f"product name: {product.name }")
                            
#                             h3(f"product text: {product.text }")
#                             h3("product image")
#                             with  ol() :
#                                 for img_path in product.img_list:
#                                     li(img(src=img_path,alt=""))
                
                
#                 # body.add(h2(a("review text",href=example.snopes_url)))
                
                
                 
                        
#                 body.add(hr())
                
              


#     # print(doc)
#     html_name=os.path.join(out_path,'example'+str(html_index)+'.html')
#     with open(html_name, 'w') as f:
#         print(doc,file=f)
#     webbrowser.open(html_name) 


def add_into_body( key,value):
    paragraph=p()
    paragraph.add(b(f"{key}: "))
    paragraph.add(f"{value}")
    return paragraph
     

def prepare_data(example,review_dict,product_dict,is_add_gold_product):
     
    review_id=example.review_id  
    candidate_product_id_list  =example.candidate_product_id_list
    review=review_dict[review_id]
    products=[]
    for  product_id in candidate_product_id_list:
        if product_id!=-1:
            product=product_dict[product_id]
            products.append(product) 
    if  example.target_product_id in candidate_product_id_list:
        gold_product_position=candidate_product_id_list.index( example.target_product_id )
    else:
        gold_product_position=-1
        if is_add_gold_product  :
            products.append(product_dict[example.target_product_id])
        
    return review,products,gold_product_position,example.target_product_id

def attribute_annotation_review_template(review,target_product_id,gold_product_position):
    add_into_body( "Review ID",review.review_id)
    add_into_body( "Review Text",review.text)
    add_into_body( "Gold Attribute",review.predicted_attribute_dict[f"Gold Attribute"])
    b(f"Review Image:")
    with ol():
        for img_path in review.review_image_path_before_select:
            li(img(src="/~menglong/for_inner_usage/review_images/"+img_path ,alt="",  width="300" ,height="200", cls="zoomE"))


def human_experiment_review_template(review,target_product_id,gold_product_position):
    add_into_body( "Review ID",review.review_id)
    add_into_body( "Review Text",review.text)
    # add_into_body( "Gold Attribute",review.predicted_attribute_dict[f"Gold Attribute"])
    b(f"Review Image:")
    with ol():
        for img_path in review.review_image_path_before_select:
            li(img(src="/~menglong/for_inner_usage/review_images/"+img_path ,alt="",  width="300" ,height="200", cls="zoomE"))


def general_review_template(review,target_product_id,gold_product_position):
    add_into_body( "Review ID",review.review_id)
    add_into_body( "Mention",review.mention)
    add_into_body( "Review Text",review.text)
    add_into_body( "Predicted Product",1)
    add_into_body( "Gold Product",target_product_id)
    add_into_body( "Gold Product in HTML",gold_product_position+1)
    add_into_body( "Gold Product Position in Candidate List sorted by Fused Score",review.gold_product_index+1)
    # add_into_body( "Attribute",review.attribute)
    # add_into_body( "Attribute by OCR",review.predicted_attribute_ocr)
    # add_into_body( "OCR Confidence",review.confidence_score_ocr)
    predicted_attribute_dict=review.predicted_attribute_dict
    for predict_attribute_logic,predict_attribute_list in predicted_attribute_dict.items():
        if len(predict_attribute_list)>0:
            add_into_body( predict_attribute_logic,predict_attribute_list)
    # add_into_body( "Low Information?",review.is_low_quality_review)
    # add_into_body( "Predicted Low Information",review.predicted_is_low_quality_review)
    # add_into_body( "Max Image Similarity",review.image_similarity_score)
    # add_into_body( "Text Similarity",review.text_similarity_score)
    # add_into_body( "Product Title Similarity",review.product_title_similarity_score)
    # h1(f"review id: {review.review_id}")
    # h2( f"review text: {review.text}" )
    b(f"Review Image:")
    with ol():
        for img_path in review.img_list:
            li(img(src="/~menglong/for_inner_usage/review_images/"+img_path ,alt="",  width="300" ,height="200", cls="zoomE"))



def attribute_annotation_product_template(review,products,review_special_product_info,gold_product_id):
    # h3("Candidate Products")
    with div(id="product_scroller"):
        attr(cls='scroller')
        with ol():
            score_dict=review.score_dict
            for product in products:
                if   product.product_id == gold_product_id:
                    if product.product_id in score_dict:
                        product_score_dict=score_dict[product.product_id]
                    else:
                        product_score_dict={}
                    with li():
                        # button(f"{product.name}. (Click to unfold)",cls="collapsible",type="button" )
                        with div( ):
                            # add_into_body( "Product ID",product.product_id)
                            add_into_body( "Product Title",product.name)
                            # b("Specifications:")
                            with ol():
                                for key,value in product.spec_dict.items():
                                    li(f'"{key}" : "{value}",')

def general_product_template(review,products,review_special_product_info):
    with div(id="product_scroller"):
        attr(cls='scroller')
        h3("Candidate Products")
        button(f"Expand All",cls="expand_all",type="button" )
        with ol():
            score_dict=review.score_dict
            for product in products:
                if product.product_id in score_dict:
                    product_score_dict=score_dict[product.product_id]
                else:
                    product_score_dict={}
                with li():
                    button(f"{product.name}. (Click to unfold)",cls="collapsible",type="button" )
                    with div(cls="content"):
                        add_into_body( "Product ID",product.product_id)
                        add_into_body( "Product Title",product.name)
                        add_into_body( "fused_score_dict",product_score_dict)
                        
                        add_into_body( "Product Description",product.text)
                        # if str(product.product_id) in review.image_score_dict and "score" in review.image_score_dict[str(product.product_id)]:
                        #     add_into_body( "Image Similarity",round(review.image_score_dict[str(product.product_id)]["score"],4))
                        # else:
                        #     add_into_body( "Image Similarity",-100)
                        # if product.text_bi_score!=-1:
                        # if str(product.product_id) in review.desc_score_dict and "bi_score" in review.desc_score_dict[str(product.product_id)]:
                        #     add_into_body( "Description Text Bi-encoder Similarity",round(review.desc_score_dict[str(product.product_id)]["bi_score"],4))
                        # else:
                        #     add_into_body( "Description Text Bi-encoder Similarity",-100)
                        # add_into_body( "Fused Score=0.3*Description Text Bi-encoder Similarity+0.7*Image Similarity",round(product_score_dict["fused_score"],4))
                        # if str(product.product_id) in review.desc_score_dict and "cross_score" in review.desc_score_dict[str(product.product_id)]:    
                        #     add_into_body( "Description Text Cross-encoder Similarity",round(review.desc_score_dict[str(product.product_id)]["cross_score"],4))
                        # else:
                        #     add_into_body( "Description Text Cross-encoder Similarity",-100)
                        
                        # if str(product.product_id) in review.text_score_dict and  "bi_score" in review.text_score_dict[str(product.product_id)]:
                        #     add_into_body( "Title Text Bi-encoder Similarity",round(review.text_score_dict[str(product.product_id)]["bi_score"],4))
                        #     add_into_body( "Title Text Cross-encoder Similarity",round(review.text_score_dict[str(product.product_id)]["cross_score"],4))
                        # else:
                        #     add_into_body( "Title Text Bi-encoder Similarity",-100)
                        #     add_into_body( "Title Text Cross-encoder Similarity",-100)
                        
                    
                    
                        
                        # h3( f"product id: {product.product_id}" )
                        # h3(f"product name: {product.name }")
                        # h3(f"product text: {product.text }")
                        image_similarity_score_list=product.image_similarity_score_list
                        b("Product Image")
                        with ol():
                            if review_special_product_info is not None and str(product.product_id) in review_special_product_info:
                                image_list=review_special_product_info[str(product.product_id)]["image_path"]
                            else:
                                image_list=product.img_list
                            for idx,img_path in enumerate(image_list):
                                with li():
                                    img(src="/~menglong/for_inner_usage/product_images/"+ img_path,alt="",  width="200" ,height="130", cls="zoomE")
                                    if image_similarity_score_list is not None and len(image_similarity_score_list)>0:
                                        add_into_body("Image Similarity",", ".join(map(str,image_similarity_score_list[idx])))
                        b("Specifications:")
                        with ol():
                            for key,value in product.spec_dict.items():
                                li(f"{key} : {value}")




def human_experiment_product_template(review,products,review_special_product_info):
    with div(id="product_scroller"):
        attr(cls='scroller')
        # h3("Candidate Products")
        button(f"Expand All",cls="expand_all",type="button" )
        with ol():
            score_dict=review.score_dict
            for product in products:
                with li():
                    button(f"{product.name}. (Click to unfold)",cls="collapsible",type="button" )
                    with div(cls="content"):
                        # add_into_body( "Product ID",product.product_id)
                        add_into_body( "Product Title",product.name)
                        add_into_body( "Product Description",product.text)
                        # b("Product Image")
                        with ol():
                            image_list=product.img_list
                            for idx,img_path in enumerate(image_list):
                                with li():
                                    img(src="/~menglong/for_inner_usage/product_images/"+ img_path,alt="",  width="200" ,height="130", cls="zoomE")
                        b("Specifications:")
                        with ol():
                            for key,value in product.spec_dict.items():
                                li(f"{key} : {value}")



def create_one_html_with_scrollbar(example_num_in_one_html,html_index,example_dict,review_dict,product_dict  , out_path,
                                   html_template,is_add_gold_product):
    doc = dominate.document(title='examples'+str(html_index))
    with doc.head:
        link(rel='stylesheet', href='style.css')
        
        style("""\
            
            * {
                box-sizing: border-box;
            }

            /* Create two equal columns that floats next to each other */
            .column {
                float: left;
                width: 50%;
                padding:10px;
                height: 300px; /* Should be removed. Only for demonstration */
            }
            .scroller {
                float: left;
                width: 45%;
                height: 2000px;
                overflow-y: scroll;
                scrollbar-color: rebeccapurple green;
                scrollbar-width: thin;
            }
            /* Clear floats after the columns */
            .row:after {
                content: '';
                display: table;
                clear: both;
            }
            
            .collapsible {
              background-color: #777 ;
              color: white;
              cursor: pointer;
              padding: 18px;
              width: 100%;
              border: none;
              text-align: left;
              outline: none;
              font-size: 15px;
            }
            
            .active, .collapsible:hover {
              background-color: #555;
            }
            
            .content {
              padding: 0 18px;
              display: none;
              overflow: hidden;
              background-color: #f1f1f1;
            }
            
            .expand_all {
              background-color: #777 ;
              color: white;
              cursor: pointer;
              padding: 18px;
              width: 100%;
              border: none;
              text-align: left;
              outline: none;
              font-size: 15px;
            }
            
            .expand_all:hover {
              background-color: #555;
            }

     """)
    
  
    
    with doc:
        j=-1
        for key,value in example_dict.items():
            j+=1
            expected_beginning_num=html_index*example_num_in_one_html 
            if j>=(html_index+1)*example_num_in_one_html:
                break
            elif j<expected_beginning_num:
                continue
            with div(id="review_product_row"):
                attr(cls='row')
                review,products,gold_product_position, target_product_id=prepare_data(example_dict[key],review_dict,product_dict,
                                                                                      is_add_gold_product)
                with div(id="review_column"):
                    attr(cls='column')
                    if html_template=="attribute_annotation":
                        attribute_annotation_review_template(review,target_product_id,gold_product_position)
                    elif html_template=="human_experiment":
                        human_experiment_review_template(review,target_product_id,gold_product_position)
                    else:
                        general_review_template(review,target_product_id,gold_product_position)
            
                review_special_product_info=review.review_special_product_info
                
                if html_template=="attribute_annotation":
                    attribute_annotation_product_template(review,products,review_special_product_info,target_product_id)
                elif html_template=="human_experiment":
                    human_experiment_product_template(review,products,review_special_product_info)
                else:
                    general_product_template(review,products,review_special_product_info)
                    
                                            
                # hr()
            hr()
            br()
            # with div( ):
            #     attr(cls='row')
    
                
        script(type='text/javascript', src='/~menglong/for_inner_usage/js/script.js')          


    # print(doc)
    html_name=os.path.join(out_path,'example'+str(html_index)+'.html')
    with open(html_name, 'w') as f:
        print(doc,file=f)
    # webbrowser.open(html_name) 

 
# def test():
#     doc = dominate.document(title='examples')

#     with doc.head:
#         link(rel='stylesheet', href='style.css')
#         script(type='text/javascript', src='script.js')

#     with doc:
#         with div(id='header').add(ol()):
#             for i in ['home', 'about', 'contact']:
#                 li(a(i.title(), href='/%s.html' % i))

#         with div():
#             attr(cls='body')
#             p('Lorem ipsum..')

#     print(doc)



if __name__ == '__main__':    
    create_one_html_with_scrollbar(0,100,{},{},{}  , "/home/menglong/workspace/code/multimodal/multimodal_entity_linking/multimodal_entity_linking/output/example")