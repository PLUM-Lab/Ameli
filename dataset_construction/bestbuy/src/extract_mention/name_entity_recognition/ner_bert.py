from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
def ner_bert(example):
    

    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER")

    nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    # example = "My name is Wolfgang and I live in Berlin"

    ner_results = nlp(example)
    for result in ner_results:
        print(result)
    
def inference():
    mobile_industry_article="I have had the ring doorbell for years now and it seemed quite natural that I would migrate to a Ring Alarm System. I did. I left ADT and decided upon such as it integrated well with that was already in place. I really prefer the self monitoring and the idea that it’s at such a low cost... $8.50/mthly. I love the ease of installation too. Did it myself and quick too."
    mobile_industry_article1="I recently updated to this system from an older DVR still swann system. I figured I could upgrade just the cameras and DVR since my original 8 camera system that was installed 5 years ago was BNC cable. First of all, swann is a excellent brand, however it’s older models DVR suffer from poor app supposed for the the iPhone. Customer service is pretty good, takes a while to get them on but they’re very helpful. This new model has a newer app the “swann security” app and it’s a whole lots faster with the substream so you get very little lag once you fire up the app on the iPhone and start streaming live video. Since this camera has both a LED light and sirens it’s cool that the live stream works quick because the whole point of having the light/siren is to be able to activate it when I want. So through the app there’s a light/siren activator button, however, at home on the actual DVR system the menu does NOT have an option to activate alarm/light at will, you would have to program the system to do it which doesn’t make sense to have the siren/light option and not being able to active from the DVR itself. The 2TB HD and new DVR menus are much friendly than the older systems def worth the upgrade from 1080p cameras to this one."
    ner_bert(mobile_industry_article)