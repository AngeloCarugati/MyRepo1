from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

def compute_similarity(a, b): 
    tfidf = vectorizer.fit_transform([a, b])
    return ((tfidf * tfidf.T).toarray())[0,1]

file1_path = "/home/acarugat/DiscorsoMattarellaFine2023.txt"

with open(file1_path, 'r', encoding='utf-8') as file: 
    testo1 = file.read()

file2_path = "/home/acarugat/DiscorsoMattarellaFine2024.txt"

with open(file2_path, 'r', encoding='utf-8') as file: 
    testo2 = file.read()

s=compute_similarity(testo1,testo2)

print (s)

import os
import pandas as pd
import numpy as np

# Initialize lists to store filenames and contents
filenames = []
contents = []

# Walk through the home directory and read all text files
print ("Dimmi che documenti vuoi analizzare: Presidenti Giornali o Parlamento")
line = input()

for root, dirs, files in os.walk("/home/acarugat/Testi/"+line):
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

# Display the dataset
# print(dataset)

M = np.zeros((dataset.shape[0], dataset.shape[0])) # creiamo una matrice 30x30 per contenere i risultati di testo_i con testo_j

#print (M)

labels=dataset.filename.str.split('/').str[4:].str[1]
labels=labels.str.split('.').str[0]
print (labels)
similarity_df = pd.DataFrame(M, columns=labels, index=labels) # creiamo un dataframe

from tqdm import tqdm

for i, row in tqdm(dataset.iterrows(), total=dataset.shape[0], desc='1st level'): # definiamo i
    for j, next_row in dataset.iterrows(): # definiamo j
        M[i, j] = compute_similarity(row.content, next_row.content) # popoliamo la matrice con i risultati

#print (M)

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

mask = np.triu(np.ones_like(similarity_df)) # applichiamo una maschera per rimuovere la parte superiore della heatmap

# creiamo la visualizzazione
plt.figure(figsize=(12, 12))
sns.heatmap(
			similarity_df,
			square=True, 
			annot=True, 
			robust=True,
			fmt='.2f',
			annot_kws={'size': 7, 'fontweight': 'bold'},
			yticklabels=similarity_df.columns,
			xticklabels=similarity_df.columns,
			cmap="YlGnBu",
            #mask=mask
			mask=None
)

plt.title('Heatmap delle similarità tra testi', fontdict={'fontsize': 24})
plt.show()
