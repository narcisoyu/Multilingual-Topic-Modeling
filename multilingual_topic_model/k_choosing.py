import pandas as pd
import nltk
import torch
import random
import numpy as np
from nltk.corpus import stopwords as stop_words
from contextualized_topic_models.models.ctm import ZeroShotTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessingStopwords

train_data = pd.read_csv("/mnt/home/jiyu2657/Dokumente/ythales_model_training/train_set.csv")

# Use the Translated(English) text for training
documents = train_data['english_lemma_text'].tolist() 


# Prepare data
nltk.download('stopwords')
stopwords = list(stop_words.words("english"))
sp = WhiteSpacePreprocessingStopwords(documents, stopwords_list=stopwords)
preprocessed_documents, unpreprocessed_corpus, vocab, retained_indices = sp.preprocess()

# plug-in the language model
tp = TopicModelDataPreparation("paraphrase-multilingual-mpnet-base-v2")
training_dataset = tp.fit(text_for_contextual=unpreprocessed_corpus, text_for_bow=preprocessed_documents)

# Choose the number of topics

from transformers.pipelines import text_classification
from pandas.io.parsers import TextParser
from contextualized_topic_models.evaluation.measures import CoherenceNPMI, InvertedRBO
from gensim.corpora.dictionary import Dictionary

# RBO measures topic diversity
# NPMI measures topic coherence

# be sure to split sentence before feed into Dictionary
texts = [d.split() for d in preprocessed_documents]

def compute_scores(topics, text):
    rbo = InvertedRBO(topics)
    npmi = CoherenceNPMI(texts=texts, topics=topics)
    return npmi.score(), rbo.score()

# define a dictionary to store the metrics number
min_k = 4  # define the min k
max_k = 41 # define the max k

results_dictionary = {}
for metric in ["npmi", "rbo"]:
    results_dictionary[metric] = {str(i):[] for i in range(min_k, max_k, 2)}

# now compute the metrics
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

for iteration_index in range(0, 50):
    for topic_num in list(range(min_k, max_k, 2)):

        print("iteration ", iteration_index, topic_num)
        ctm = ZeroShotTM(bow_size=len(tp.vocab), contextual_size=768, n_components=topic_num, num_epochs=20)

        ctm.fit(training_dataset) 

        npmix, rbox = compute_scores(ctm.get_topic_lists(10), texts)

        results_dictionary["npmi"][str(topic_num)].append(npmix)
        results_dictionary["rbo"][str(topic_num)].append(rbox)

# Create an empty DataFrame with two columns named 'metrics' and 'k'
df = pd.DataFrame(columns=['metrics', 'k'])

for a in results_dictionary:
    for b in results_dictionary[a]:
        # Calculate the mean of the results_dictionary[a][b] values
        mean_value = np.mean(results_dictionary[a][b])
        
        # Append the values of 'metrics', 'k', and 'mean_value' as a new row in the DataFrame
        df = df.append({'metrics': a, 'k': b, 'mean_value': mean_value}, ignore_index=True)

# save the metrics in csv
df.to_csv("10iteration_npmi_rbo.csv", encoding='utf-8')

# reshape the data for plot
# pivot, re-shapt the df
df = df.pivot(index='k', columns='metrics', values='mean_value')
# reset index
df = df.reset_index()
# convert column 'k' to numeric
df['k'] = pd.to_numeric(df['k'])
# sort the column by k
df.sort_values('k', inplace=True)
# remove multilevel index names
df = df.rename_axis(None, axis=1).reset_index(drop=True)

# PLOT and save plot
import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(df['k'], df['npmi'], 'r-')
ax2.plot(df['k'], df['rbo'], 'b-')

ax1.set_xlabel('k')
ax1.set_ylabel('npmi', color='r')
ax2.set_ylabel('rbo', color='b')

fig.tight_layout()
plt.savefig('npmi_rbo_plot.png')


