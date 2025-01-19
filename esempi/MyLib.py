import os
import pandas as pd
from gensim.models import Word2Vec
import numpy as np
from transformers import BertTokenizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize 
import string
from nltk.corpus import stopwords


class LLModel:
    def __init__(self, name, corpus:str, params:dict):
        self.name = name 
        self.corpus = corpus
        self.params = params
        print ("Nome = ", name)
        print ("Corpus = ", corpus)
        print ("Parametri = ", params)
        print ("uno")
        for chiave, valore in params.items():
            print(f"Chiave: {chiave}, Valore: {valore}")
        print ("due")
 
    def creaCorpus(self):
        # Initialize lists to store filenames and contents
        filenames = []
        contents = []

        print (self.corpus)
        for root, dirs, files in os.walk(self.corpus):
          for file in files:
                if file.endswith(".txt"):  # Check if the file is a text file
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            filenames.append(filepath)  # Store the file path
                            contents.append(f.read())  # Store the file content
                    except Exception as e:
                        print(f"Error reading {filepath}: {e}")

        # Create the dataset (DataFrame)
        dataset = pd.DataFrame({
            "filename": filenames,
            "content": contents
            })
        print (dataset)

        
        with open("/home/acarugat/Testi/Manzoni/InteroContenuto.txt", "w") as file:
            file.writelines([line + "\n" for line in contents])

        model = Word2Vec(corpus_file="/home/acarugat/Testi/Manzoni/InteroContenuto.txt", vector_size=100, window=2, min_count=1, workers=4,sg=0, epochs=100)
        model.save ("/home/acarugat/ModelloManzoni.model")
        print ("Modello = ", model)
        return (model)