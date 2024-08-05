from sentence_transformers import CrossEncoder
model = CrossEncoder('cross-encoder/nli-deberta-v3-base')
scores = model.predict([('Is just perfect, excellent size thanks Marshall-Everton','The brand is Marshall'),('A man is eating pizza', 'A man eats something'), ('A black race car starts up in front of a crowd of people.', 'A man is driving down a lonely road.')])

#Convert scores to labels
label_mapping = ['contradiction', 'entailment', 'neutral']
labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]