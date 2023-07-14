#!/usr/bin/env python
# coding: utf-8

# # Read data

# In[1]:


import pandas as pd
train_data = pd.read_csv("train_set.csv")

# Use the Translated(English) text for training
documents = train_data['english_lemma_text'].tolist() 


# In[2]:


# Here Youtube data = title + description

train_data.head()


# In[3]:


documents[:2]


# In[4]:


import nltk
import torch
import random
import numpy as np
from contextualized_topic_models.models.ctm import ZeroShotTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessingStopwords

# set seed
# torch.manual_seed(10)
# torch.cuda.manual_seed(10)
# np.random.seed(10)
# random.seed(10)


# # Preparing the data

# In[5]:


from nltk.corpus import stopwords as stop_words

nltk.download('stopwords')

stopwords = list(stop_words.words("english"))

sp = WhiteSpacePreprocessingStopwords(documents, stopwords_list=stopwords)
preprocessed_documents, unpreprocessed_corpus, vocab, retained_indices = sp.preprocess()


# In[6]:


preprocessed_documents[:2]


# In[7]:


# plug-in the language model

tp = TopicModelDataPreparation("paraphrase-multilingual-mpnet-base-v2")


# In[8]:


training_dataset = tp.fit(text_for_contextual=unpreprocessed_corpus, text_for_bow=preprocessed_documents)


# In[9]:


tp.vocab[0:10]


# ## Use CTM to make 20 topics 

# In[26]:


# set seed
torch.manual_seed(1442)
torch.cuda.manual_seed(1442)
np.random.seed(1442)
random.seed(1442)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True

final_k = 20

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ctm20 = ZeroShotTM(bow_size=len(tp.vocab), contextual_size=768, n_components=final_k, num_epochs=100)

ctm20.fit(training_dataset) 


# In[27]:


from collections import defaultdict
topics = ctm20.get_topics(k = final_k)
df_topiclist = pd.DataFrame.from_dict(topics, orient='index').transpose()
df_topiclist


# In[28]:


# topic-term distribution
n_term = 10
top_terms = pd.DataFrame()

for i in range(0, final_k):
    # get the list of terms per topic
    list_term = ctm20.get_word_distribution_by_topic_id(i)[0:n_term]
    # convert to dataframe
    high_prob_terms = pd.DataFrame(list_term , columns=['term', 'prob'])
    # make topic index
    topicindex = [i] * n_term
    # join topic index into df
    high_prob_terms['topic_index'] = topicindex
    top_terms = pd.concat([top_terms, high_prob_terms], ignore_index=True, axis=0)
    print(top_terms)


# In[29]:


top_terms.to_csv("top_terms_k20model.csv", encoding='utf-8')


# In[30]:


# doc-topic distribution
doc_topic_distribution = ctm20.get_thetas(training_dataset, n_samples=20)


# In[31]:


doc_topic_distribution


# In[32]:


n_doc = 10 
top_docs = pd.DataFrame()

for i in range(0, final_k): 
    # compute topic doc per topic
    lst = ctm20.get_top_documents_per_topic_id(unpreprocessed_corpus, doc_topic_distribution, i, k=n_doc)
    # make it as df
    high_prob_docs = pd.DataFrame(lst, columns=['document', 'prob'])
    # make topic index
    topicindex = [i] * n_doc
    # join topic index into df
    high_prob_docs['topic_index'] = topicindex
    # merge dataframes
    top_docs = pd.concat([top_docs, high_prob_docs], ignore_index=True, axis=0)
    print(top_docs)


# In[33]:


top_docs.to_csv("top_docs_k20model.csv", encoding='utf-8')


# # Predict topics for test data (in original language)

# In[34]:


test_data = pd.read_csv("test_set.csv")

# Use the Translated(English) text for prediction
test_documents = test_data['text'].tolist() 

# transform the test set into adequate form
testing_dataset = tp.transform(test_documents)


# In[35]:


# set seed
torch.manual_seed(1442)
torch.cuda.manual_seed(1442)
np.random.seed(1442)
random.seed(1442)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True

test_set_topics_predictions = ctm20.get_thetas(testing_dataset, n_samples=100)


# In[36]:


# get the topic id of the first document
final_results = []

for i in range(0, len(testing_dataset)):
    n_topic = np.argmax(test_set_topics_predictions[i]) 
    pred_result = ctm20.get_topic_lists(10)[n_topic]
    pred_result = " ".join(pred_result)
    final_results.append(pred_result)


# In[37]:


final_results


# In[38]:


# assign new predicted column
test_data['topics'] = final_results


# In[39]:


test_data


# In[40]:


test_data.to_csv("test_data_results_k20.csv")


# # View session info

# In[41]:


import session_info
session_info.show()


# In[42]:


ctm20.save(models_dir="./")

