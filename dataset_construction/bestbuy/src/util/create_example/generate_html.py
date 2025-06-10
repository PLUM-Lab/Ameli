 
import webbrowser

import os 
import pandas as pd
import shutil 
import dominate
from dominate.tags import * 
    
import os 
import pandas as pd

from bestbuy.src.util.create_example.read_example import read_example

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

def generate_report_sheet( out_report_path,example_dict,review_dict):  
    
    report_list = []
    for key,value in example_dict.items():
        example=example_dict[key]
        review=review_dict[example.review_id]
        report_list.append([review.review_id, review.text,example.reshuffled_target_product_id_position,example.target_product_id,""])
        
    df = pd.DataFrame(report_list, columns =['review_id', 'review_text','reshuffled_target_product_id_position','target_product_id','human_predict'])
    df.to_csv(out_report_path,index=False)                          

def generate(example_path_str,  out_path,version, generate_claim_level_report,example_num=None ,is_reshuffle=True ):
        
    needed_field=None 
    
    
      
    example_dict  =read_example(example_path_str, is_reshuffle )
    if example_num is None:
        example_num=len(example_dict)
    page_num=int(example_num/100   )
    if page_num==0:
        page_num=1
    for i in range(page_num):
        create_one_html(100,i,example_dict,version, out_path,needed_field)
        
        
     
    
    return example_dict
 
def create_one_html(example_num_in_one_html,html_index,example_dict, version, out_path,needed_field):
    doc = dominate.document(title='examples'+str(html_index)+"_"+version)
    with doc.head:
        link(rel='stylesheet', href='style.css')
        script(type='text/javascript', src='script.js')

    with doc:
        style(".calendar_table{width:880px;}")
        style("body{font-family:Helvetica}")
        style("h1{font-size:x-large}")
        style("h2{font-size:large}")
        style("table{border-collapse:collapse}")
        style("th{font-size:small;border:1px solid gray;padding:4px;background-color:#DDD}")
        style("td{font-size:small;text-align:center;border:1px solid gray;padding:4px}")

        with div() as body:
            attr(cls='body')
            j=-1
            body=add_into_body(body,"Config","score threshold=0.3")
             
            body.add(hr())
            for  example in example_dict :
                j+=1
                expected_beginning_num=html_index*example_num_in_one_html 
                if j>=(html_index+1)*example_num_in_one_html:
                    break
                elif j<expected_beginning_num:
                    continue
                
                gold_mention=example.gold_mention
                predicted_mention=example.predicted_mention
                similarity_score=example.similarity_score
                mention_candidates_before_review_match=example.mention_candidates_before_review_match
                
                product_category=example.product_category
                product_title=example.product_title
                review=example.review
                review_id=example.review_id
                product_title_chunk=example.product_title_chunk
                product_category_chunk=example.product_category_chunk
                url=example.url
                body.add(h2(f"id: {review_id}"))
                # body.add(p(f"review: {review }"))
                body=gen_review_with_mention_highlight(body,review,gen_mention_list(similarity_score),gold_mention,predicted_mention)
                # body=add_into_body(body,"Review",)
                paragraph=p()
                paragraph.add(b(f"Gold Mention: "))
                paragraph.add(b(f"{gold_mention}",style="color:#f0960d"))
                body.add(paragraph)
                # with body.add(p()):
                #     b("Review:"), f"{review}"
               
                paragraph=p()
                paragraph.add(b(f"Predicted Mention: "))
                paragraph.add(b(f"{predicted_mention}",style="color:#2922ea"))
                body.add(paragraph)
                    
                if similarity_score is not None: 
                    with body.add(table()):
                        with  thead():
                            th("Candidate", style = "color:#ffffff;background-color:#6A75F2")
                            th("Product Category Score", style = "color:#ffffff;background-color:#6A75F2")
                            th("Product Category Chunk Score", style = "color:#ffffff;background-color:#6A75F2")
                            th("Product Title Score", style = "color:#ffffff;background-color:#6A75F2")
                            th("Product Title Chunk Score", style = "color:#ffffff;background-color:#6A75F2")
                        with   tbody():
                             
                            for _,score_object in similarity_score.items():
                                l = tr()
                                l.add(td( score_object["mention"] , style = "font-size:small;text-align:center;padding:4px"))
                                l.add(td(round(score_object["product_category_score"],2) , style = "font-size:small;text-align:center;padding:4px"))    
                                l.add(td(round(score_object["product_category_chunk_score"],2), style = "font-size:small;text-align:center;padding:4px" ))
                                l.add(td(round(score_object["product_title_score"],2), style = "font-size:small;text-align:center;padding:4px" ))    
                                l.add(td(round(score_object["product_title_chunk_score"],2), style = "font-size:small;text-align:center;padding:4px" ))
                
                body=add_into_body(body,"Product Category",product_category)
                body=add_into_body(body,"Product Title",product_title)
                body=add_into_body(body,"Product Category Chunk",product_category_chunk)
                body=add_into_body(body,"Product Title Chunk",product_title_chunk)
             
                body.add(p(b(f"Mention Candidates before Review Match:" )))
                with body.add(ol()):
                    for mention_candidate_before_review_match in mention_candidates_before_review_match:
                        li(p(f"{mention_candidate_before_review_match}"))
                        
                 
                body.add(h2(a("Product Url",href=url)))
                body.add(hr())
                
              


    # print(doc)
    html_name=os.path.join(out_path,'example'+str(html_index)+"_"+version+'.html')
    with open(html_name, 'w') as f:
        print(doc,file=f)
    # webbrowser.open(html_name) 

def gen_mention_list(similarity_score):
    mntion_list=[]
    for _,similarity_one_score_object in similarity_score.items():
        mention=similarity_one_score_object["mention"]
        mntion_list.append(mention )
    return mntion_list 
        
def gen_review_with_mention_highlight(body,review,mention_candiates,gold_mention_list,predicted_mention):
    paragraph=p()
    paragraph.add(b(f"Review: "))
    skip_until_idx=-1
    for i in range(len(review)):
        if i >= skip_until_idx:
            is_found=False
            for gold_mention in gold_mention_list:
                if gold_mention!="" and review.startswith(gold_mention, i):
                    is_found=True 
                    paragraph.add(b(f"{gold_mention}",style="color:#f0960d")) 
                    skip_until_idx=i+len(gold_mention) 
                    break 
            if not is_found:
                if predicted_mention!="" and  review.startswith(predicted_mention, i):
                    is_found=True 
                    paragraph.add(b(f"{predicted_mention}",style="color:#2922ea")) 
                    skip_until_idx=i+len(predicted_mention) 
                else:
                    for candidate in mention_candiates:
                        if review.startswith(candidate, i):
                            is_found=True 
                            paragraph.add(b(f"{candidate}",style="color:#05C7FA")) 
                            skip_until_idx=i+len(candidate) 
                            break 
                
                
            if not is_found:
                paragraph.add( f"{review[i]}" )         
    
    body.add(paragraph)     
    return body  
          
        

def add_into_body(body,key,value):
    paragraph=p()
    paragraph.add(b(f"{key}: "))
    paragraph.add(f"{value}")
    body.add(paragraph)
    return body 

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



    
