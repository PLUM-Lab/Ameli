#https://tahaashtiani.com/genrating-product-niche-titles-(ner-using-spacy,-gcp-nlp,-and-bert)
# from google.cloud import language
# from google.cloud.language import enums
# from google.cloud.language import types

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="   "
# client = language.LanguageServiceClient()

# text  = products[i].split(",")[1]
# document = types.Document(
# content=text,
# type=enums.Document.Type.PLAIN_TEXT)
# entities = client.analyze_entities(document).entities

# for entity in entities:
#     entity_type = enums.Entity.Type(entity.type)
#     try :
#         if entity_type.CONSUMER_GOOD:
#             if entity_type.name == "CONSUMER_GOOD":
#                 niche = entity.mentions[0].text.content
#                 niche_words = len(niche.split(" "))                            
#                 if  niche_words >1 and niche_words <8 and not re.search(r"\d",niche):
#                     if niche not in qualified_niches:
#                         qualified_niches.append(niche)
#                         mined_niches.writelines(niche+"\n")