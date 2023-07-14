#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
from contextualized_topic_models.models.ctm import ZeroShotTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessing
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessingStopwords
import nltk


# In[8]:


general_path = "/mnt/home/jiyu2657/Dokumente/ythales_model_training/match_test/"


# In[9]:


train_data = pd.read_csv(general_path + "train_set.csv")

# Use the Translated(English) text for training
documents = train_data['english_lemma_text'].tolist() 


# In[10]:


from nltk.corpus import stopwords as stop_words

stopwords = list(stop_words.words("english"))

sp = WhiteSpacePreprocessingStopwords(documents, stopwords_list=stopwords)
preprocessed_documents, unpreprocessed_corpus, vocab, retained_indices = sp.preprocess()


# In[17]:


tp = TopicModelDataPreparation("paraphrase-multilingual-mpnet-base-v2")


# In[18]:


training_dataset = tp.fit(text_for_contextual=unpreprocessed_corpus, text_for_bow=preprocessed_documents)


# In[19]:


tp.vocab[0:10]


# In[37]:


k10model = ZeroShotTM(bow_size=len(tp.vocab), contextual_size=768, n_components=10, num_epochs=100)
k10model.load("k10model", epoch = 99)


# In[38]:


k15model = ZeroShotTM(bow_size=len(tp.vocab), contextual_size=768, n_components=15, num_epochs=100)
k15model.load("k15model", epoch = 99)


# In[39]:


k20model = ZeroShotTM(bow_size=len(tp.vocab), contextual_size=768, n_components=20, num_epochs=100)
k20model.load("k20model", epoch = 99)


# In[40]:


# Filter no English data.
noenglishdf = train_data.loc[(train_data['country'] != "uk") & (train_data['country'] != "us")]

# random sample 100 non-EN text
noenglishdf_sample = noenglishdf.sample(n=100, random_state=2012)

# have a look at the lang/country composition
noenglishdf_sample.country.value_counts()


# In[41]:


originallang = noenglishdf_sample['text'].tolist()
translatelang = noenglishdf_sample['english_text'].tolist()

originallang_transformed = tp.transform(originallang)
translatelang_transformed = tp.transform(translatelang)


# # K10 model match

# In[1]:


# set seed
import torch
import random
import numpy as np
import os

torch.manual_seed(1442)
torch.cuda.manual_seed(1442)
np.random.seed(1442)
random.seed(1442)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"

k10_original_predictions = k10model.get_thetas(originallang_transformed, n_samples=100)
k10_translate_predictions = k10model.get_thetas(translatelang_transformed, n_samples=100)


# In[ ]:


k10result_in_original_language = []
k10result_in_translate_language = []

for i in range(0, len(k10_original_predictions)):
    o_topic = np.argmax(k10_original_predictions[i]) 
    t_topic = np.argmax(k10_translate_predictions[i]) 
    opred_result = k10model.get_topic_lists(10)[o_topic]
    tpred_result = k10model.get_topic_lists(10)[t_topic]
    opred_result = " ".join(opred_result)
    tpred_result = " ".join(tpred_result)
    k10result_in_original_language.append(opred_result)
    k10result_in_translate_language.append(tpred_result)


# In[ ]:


k10counter = 0

for i in range(0, len(k10_original_predictions)):
    if k10result_in_original_language[i] == k10result_in_translate_language[i]:
        k10counter = k10counter + 1
    else:
        k10counter = k10counter + 0

print("k10 counter")
print(k10counter)


# In[ ]:

print("k10 match score")
print(k10counter/len(k10_original_predictions))


# # K15 model match

# In[ ]:


torch.manual_seed(1442)
torch.cuda.manual_seed(1442)
np.random.seed(1442)
random.seed(1442)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"

k15_original_predictions = k15model.get_thetas(originallang_transformed, n_samples=100)
k15_translate_predictions = k15model.get_thetas(translatelang_transformed, n_samples=100)


# In[ ]:


k15result_in_original_language = []
k15result_in_translate_language = []

for i in range(0, len(k15_original_predictions)):
    oo_topic = np.argmax(k15_original_predictions[i]) 
    tt_topic = np.argmax(k15_translate_predictions[i]) 
    oopred_result = k15model.get_topic_lists(10)[oo_topic]
    ttpred_result = k15model.get_topic_lists(10)[tt_topic]
    oopred_result = " ".join(oopred_result)
    ttpred_result = " ".join(ttpred_result)
    k15result_in_original_language.append(oopred_result)
    k15result_in_translate_language.append(ttpred_result)


# In[ ]:


k15counter = 0

for i in range(0, len(k15_original_predictions)):
    if k15result_in_original_language[i] == k15result_in_translate_language[i]:
        k15counter = k15counter + 1
    else:
        k15counter = k15counter + 0

print("k15 counter")
print(k15counter)


# In[2]:

print("k15 match score")
print(k15counter/len(k15_original_predictions))


# # K20 model match

# In[ ]:


torch.manual_seed(1442)
torch.cuda.manual_seed(1442)
np.random.seed(1442)
random.seed(1442)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"

k20_original_predictions = k20model.get_thetas(originallang_transformed, n_samples=100)
k20_translate_predictions = k20model.get_thetas(translatelang_transformed, n_samples=100)


# In[ ]:


k20result_in_original_language = []
k20result_in_translate_language = []

for i in range(0, len(k20_original_predictions)):
    ooo_topic = np.argmax(k20_original_predictions[i]) 
    ttt_topic = np.argmax(k20_translate_predictions[i]) 
    ooopred_result = k20model.get_topic_lists(10)[ooo_topic]
    tttpred_result = k20model.get_topic_lists(10)[ttt_topic]
    ooopred_result = " ".join(ooopred_result)
    tttpred_result = " ".join(tttpred_result)
    k20result_in_original_language.append(ooopred_result)
    k20result_in_translate_language.append(tttpred_result)


# In[ ]:


k20counter = 0

for i in range(0, len(k20_original_predictions)):
    if k20result_in_original_language[i] == k20result_in_translate_language[i]:
        k20counter = k20counter + 1
    else:
        k20counter = k20counter + 0

print("k20counter")
print(k20counter)


# In[ ]:

print("k20 match score")
print(k20counter/len(k20_original_predictions))

