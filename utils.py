import xml.etree.ElementTree as ET
import pickle


def YNmilk_data_reform(file):
    
    sentiment_map = {
        1: "POS",
        0: "NEU",
        -1: "NEG"
    }
    
    with open(file, 'rb') as pikd:
        df = pickle.load(pikd)

    sentences = []
    aspect_term_sentiments = []

    for _, d in df.iterrows():
        sentence_text = d.search_text

        for aspect, sentiment in iter(d.label): 
            if len(d.label)!=0:
                sentences.append(sentence_text)
                aspect_term_sentiments.append([(aspect, sentiment_map[sentiment])])


    return sentences, aspect_term_sentiments