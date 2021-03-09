import xml.etree.ElementTree as ET
import pickle

def readlines(path, strip = True):
    """
    Return the text read from the filepath 'path'
    """
    try:
        file = open(path, 'r', encoding='utf-8')
        lines = file.readlines()
        file.close()
        if strip: lines = [l.strip('\n') for l in lines]
        return lines
    except Exception as e: 
        print(e)
        # print("Something went wrong when reading the file")


def YNmilk_data_reform(file):

    from meilibs import FileUtils
    tag_map = FileUtils.load_pkl("tagmap.pickle")

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
                aspect_term_sentiments.append([(tag_map[aspect], sentiment_map[sentiment])])


    return sentences, aspect_term_sentiments