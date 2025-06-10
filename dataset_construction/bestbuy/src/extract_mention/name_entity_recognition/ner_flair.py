from flair.data import Sentence
from flair.models import SequenceTagger

def ner_flair():
    

    # load tagger
    tagger = SequenceTagger.load("flair/ner-english-ontonotes-large")

    # make example sentence
    sentence = Sentence("On September 1st George won 1 dollar while watching Game of Thrones.")

    # predict NER tags
    tagger.predict(sentence)

    # print sentence
    print(sentence)

    # print predicted NER spans
    print('The following NER tags are found:')
    # iterate over entities and print
    for entity in sentence.get_spans('ner'):
        print(entity)
