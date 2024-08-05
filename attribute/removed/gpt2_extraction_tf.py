
from transformers import pipeline, set_seed
def gp2_attribute_value_extract(review,attribute_key):
    prompt="Attribute Value Extraction:"
    input=prompt+"\n"+review+attribute_key+":"
    
    generator = pipeline('text-generation', model='gpt2-xl')
    set_seed(42)
    # generator(input, max_length=30, num_return_sequences=5)
    generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)
    
    
    


if __name__ == "__main__":
    review=""
    attribute_key="Brand"
    gp2_attribute_value_extract(review,attribute_key)
    
