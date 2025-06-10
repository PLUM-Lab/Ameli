import spacy
from spacy import displacy

def test0():
    nlp = spacy.load("en_core_web_lg")
    doc = nlp("Autonomous cars shift insurance liability toward manufacturers")
    # Since this is an interactive Jupyter environment, we can use displacy.render here
    displacy.render(doc, style='dep')
review="It seemed quite natural that I would migrate to a Ring Alarm System."    

def test2():
    import spacy
    nlp = spacy.load("en_core_web_lg")
    doc = nlp(review)
    for token in doc:
        print(token.text, token.dep_, token.head.text, token.head.pos_,
                [child for child in token.children])
    
    
def test5():
    import spacy

    nlp = spacy.load("en_core_web_lg")
    doc = nlp(review)

    root = [token for token in doc if token.head == token][0]
    subject = list(root.lefts)[0]
    for descendant in subject.subtree:
        assert subject is descendant or subject.is_ancestor(descendant)
        print(descendant.text, descendant.dep_, descendant.n_lefts,
                descendant.n_rights,
                [ancestor.text for ancestor in descendant.ancestors])
        

def test1():
    nlp = spacy.load("en_core_web_lg")
    doc = nlp(review)
    for chunk in doc.noun_chunks:
        print(chunk.text, chunk.root.text, chunk.root.dep_,
                chunk.root.head.text)
        
def test4():
    import spacy

    nlp = spacy.load("en_core_web_lg")
    doc = nlp(review)
    print([token.text for token in doc[12].lefts])  # ['bright', 'red']
    print([token.text for token in doc[12].rights])  # ['on']
    print(doc[12].n_lefts)  # 2
    print(doc[12].n_rights)  # 1        

def test6():
    import spacy

    # nlp = spacy.load("en_core_web_lg")
    # doc = nlp(review)
    nlp = spacy.load("en_core_web_sm")
    doc = nlp("Credit and mortgage account holders must submit their requests")
    span = doc[doc[4].left_edge.i : doc[4].right_edge.i+1]
    with doc.retokenize() as retokenizer:
        retokenizer.merge(span)
    for token in doc:
        print(token.text, token.pos_, token.dep_, token.head.text)
 
def test7():
    review="TCL 75' Class 5-Series QLED 4K UHD Smart Google TV"
    nlp = spacy.load("en_core_web_lg")
    doc = nlp(review)
    for chunk in doc.noun_chunks:
        print(chunk.text, chunk.root.text, chunk.root.dep_,
                chunk.root.head.text)
        
        
test7()