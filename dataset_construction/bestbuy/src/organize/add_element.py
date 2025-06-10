from pathlib import Path
import json
def add_id():
    incomplete_products_path = Path(
        f'../../data/bestbuy_products_40000_0_desc.json'
    )
    complete_products_path = Path(
        f'../../data/bestbuy_products_40000_0_desc.json'
    )
    output_list=[]
    product_id=0
    with open(incomplete_products_path, 'r', encoding='utf-8') as fp:
        incomplete_dict_list = json.load(fp)
        for product_json in incomplete_dict_list:
            product_json["id"]=product_id 
            product_id+=1
            output_list.append(product_json)
            
         
        with open(complete_products_path, 'w', encoding='utf-8') as fp:
            json.dump(output_list, fp, indent=4)


if __name__ == "__main__":  
    add_id()                