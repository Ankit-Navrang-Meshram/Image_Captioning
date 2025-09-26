#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import re
import emoji
import contractions
from nltk.stem import WordNetLemmatizer
import spacy
from nltk.corpus import stopwords
import nltk
import numpy as np
# Download stopwords if not already
nltk.download("stopwords")
nltk.download("wordnet")

# Load spaCy
nlp = spacy.load("en_core_web_sm")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def text_preprocessing(text):
    text = emoji.demojize(text)
    text = contractions.fix(text)
    text = text.lower()
    text = re.sub(r'\d{4}-\d{2}-\d{2}', 'DATE', text)
    text = re.sub(r'\d+', "NUM", text)
    text = re.sub(r'\b\w+@\w+\.\w+\b', 'EMAIL', text)
    text = re.sub(r'[^\w\s]', "", text)
    # Tokenize with spaCy
    doc = nlp(text)
    # Lemmatize + remove stopwords
    tokens = [
        lemmatizer.lemmatize(token.text) 
        for token in doc 
        if token.text not in stop_words and not token.is_space
    ]

    return " ".join(tokens)


# In[2]:


import os

dataset_path = "/home/mudasir/ankit/NLP/bbc"

topic = os.listdir(dataset_path)

del topic[topic.index('README.TXT')]

file_paths = []

for t in topic:
    for file_name in os.listdir(os.path.join(dataset_path, t)):
        file_path = os.path.join(dataset_path, t, file_name)
        file_paths.append(file_path)

docs = []
for file_path in file_paths:
    doc = []
    with open(file_path, 'r', encoding="utf-8", errors="ignore") as file:
        for line in file:
            line = text_preprocessing(line)
            line = re.sub(r'\n', '', line)
            doc.append(line)
    docs.append(' '.join(doc))


# In[3]:


docs


# ---
# Task A: VSM for word similarity
# 
# a. Using the provided dataset, you will construct a word co-occurrence matrix based on a
# chosen window size (e.g., K=5) and generate word vectors using the Positive Pointwise
# Mutual Information (PPMI) weighting scheme. To identify meaningful context features,
# apply feature selection methods such as retaining the most frequent context words,
# utilizing mutual information criteria, and removing stopwords or context terms with low
# informational value. This process ensures your matrix focuses on the most useful
# semantic relationships while reducing dimensionality. Report the resulting dimensions of
# your word co-occurrence matrix.

# In[13]:


def word_cooccurence_matrix(docs , window_size):
    dic = {}
    for doc in docs:
        words = doc.split()
        for i, word in enumerate(words):
            if word not in dic:
                dic[word] = {}
            for j in range(max(0, i - window_size), min(len(words), i + window_size + 1)):
                if i != j:
                    neighbor = words[j]
                    if neighbor not in dic[word].keys():
                        dic[word][neighbor] = 0
                    dic[word][neighbor] += 1
    return dic


# In[14]:


import pandas as pd
wc_matrix = word_cooccurence_matrix(docs , 5)
df = pd.DataFrame(wc_matrix).fillna(0)
df


# In[15]:


df.shape


# In[16]:


def PPMI(wc_matrix):
    # Compute the PPMI matrix
    unigram_count = {}
    total_unigram_count = 0
    for word in wc_matrix:
        unigram_count[word] = sum(wc_matrix[word].values())
        total_unigram_count += unigram_count[word]

    bigram_count = 0
    for word1 in wc_matrix:
        for word2 in wc_matrix[word1]:
            bigram_count += wc_matrix[word1][word2]

    # Compute PPMI
    ppmi = {}
    for word1 in wc_matrix:
        ppmi[word1] = {}
        for word2 in wc_matrix[word1]:
            if word2 not in ppmi[word1]:
                ppmi[word1][word2] = {}
            pmi = (wc_matrix[word1][word2] / bigram_count) / ((unigram_count[word1] / total_unigram_count) * (unigram_count[word2] / total_unigram_count))
            ppmi[word1][word2] = max(0, np.log2(pmi))

    return ppmi


# In[17]:


df = pd.DataFrame.from_dict(PPMI(wc_matrix)).fillna(0)
df


# In[18]:


df.shape


# In[19]:


PPMI_wc_matrix = PPMI(wc_matrix)


# In[20]:


def reduce_by_threshold(ppmi_matrix, threshold=1.0):
    # Zero out small values
    reduced_matrix = {}
    for word in ppmi_matrix:
        reduced_matrix[word] = {k: (v if v >= threshold else 0) for k, v in ppmi_matrix[word].items()}
    return reduced_matrix


# In[21]:


reduced_PPMI_matrix = reduce_by_threshold(PPMI_wc_matrix, threshold=0.5)


# In[22]:


df = pd.DataFrame.from_dict(reduced_PPMI_matrix).fillna(0)
df


# In[23]:


vocab = {}
for doc in docs:
    words = doc.split()
    for word in words:
        if word not in vocab:
            vocab[word] = 1
        else:
            vocab[word] += 1

vocab = dict(sorted(vocab.items(), key=lambda item: item[1], reverse=True))
len(vocab)


# In[24]:


def retaining_top_k_words(wc_matrix, k):
    context_word_matrix = {}
    for word in wc_matrix:
        for context_word in wc_matrix[word]:
            if context_word not in context_word_matrix.keys():
                context_word_matrix[context_word] = wc_matrix[word][context_word]
            else:
                context_word_matrix[context_word] += wc_matrix[word][context_word]

    context_word_matrix = dict(sorted(context_word_matrix.items(), key=lambda item: item[1], reverse=True)[:k])
    for word in wc_matrix:
        wc_matrix[word] = {k: v for k, v in wc_matrix[word].items() if k in context_word_matrix}
    return wc_matrix


# In[25]:


reduced_wc_matrix = retaining_top_k_words(wc_matrix, 100)


# In[26]:


df = pd.DataFrame.from_dict(reduced_wc_matrix, orient="index").fillna(0)
df


# In[ ]:


df.shape


# In[34]:


from sklearn.decomposition import TruncatedSVD

def reduce_with_svd(ppmi_matrix, dim=300):
    svd = TruncatedSVD(n_components=dim)
    reduced_matrix = svd.fit_transform(ppmi_matrix)
    return reduced_matrix


# In[35]:


wc_matrix = word_cooccurence_matrix(docs , 5)
PPMI_matrix = PPMI(wc_matrix)
wc_matrix_df = pd.DataFrame.from_dict(PPMI_matrix).fillna(0)


# In[36]:


df_reduced_svd = pd.DataFrame(
    reduce_with_svd(wc_matrix_df, dim=300),
    index=wc_matrix_df.index,             # words as row labels
    columns=[f"dim{i+1}" for i in range(300)]  # nicer column names
)
df_reduced_svd


# In[30]:


import random 
Query_word = {}
k = 5
for t in topic:
    Query_word[t] = []
    for i , file_name in enumerate(os.listdir(os.path.join(dataset_path, t))):
        file_path = os.path.join(dataset_path, t, file_name)
        with open(file_path, 'r', encoding="utf-8", errors="ignore") as file:
            content = re.sub(r'\n', ' ', file.read())
            content = text_preprocessing(content).split()
            random.shuffle(content)
            Query_word[t].extend(content[:k])
        if i == 4:
            break


# In[31]:


def cosine_similarity(vec_a, vec_b):
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / ((norm_a * norm_b)+ 1e-10)


# In[37]:


report = {}

for topic in Query_word.keys():
    report[topic] = {}
    for query in Query_word[topic]:
        if query in df_reduced_svd.index:
            query_vec = df_reduced_svd.loc[query].values
            similarities = {}
            for word in df_reduced_svd.index:
                if word != query:
                    word_vec = df_reduced_svd.loc[word].values
                    sim = cosine_similarity(query_vec, word_vec)
                    similarities[word] = sim
            # Get top 5 similar words
            top_5_similar = sorted(similarities.items(), key=lambda item: item[1], reverse=True)[:5]
            report[topic][query] = top_5_similar
        else:
            report[topic][query] = "Word not in vocabulary"


# In[39]:


report


# In[42]:


# Flatten into (topic, query, similar_word, score)
rows = []
for topic, queries in report.items():
    for query, similars in queries.items():
        for word, score in similars:
            rows.append((topic, query, word, score))

# Create DataFrame
report_df = pd.DataFrame(rows, columns=["Topic", "Query", "SimilarWord", "Score"])
report_df


# ----

# ### Word2Vec

# In[4]:


import re
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import random
from tqdm import tqdm
#os.environ['CUDA_VISIBLE_DEVICES'] = '4'


# In[5]:


from collections import Counter
corpus = []
for doc in docs:
    for word in doc.split():
        corpus.append(word)

len(corpus)


# In[6]:


word_counts = Counter(corpus)
vocab = sorted(word_counts, key=word_counts.get, reverse=True)
word2idx = {w: idx for idx, w in enumerate(vocab)}
idx2word = {idx: w for w, idx in word2idx.items()}
vocab_size = len(vocab)

print("Vocab size:", vocab_size)


# In[7]:


def generate_skipgram_data(corpus, window_size=5):
    pairs = []
    for i, target_word in enumerate(corpus):
        target_idx = word2idx[target_word]
        # context words within window
        for j in range(max(0, i - window_size), min(len(corpus), i + window_size + 1)):
            if i != j:
                context_idx = word2idx[corpus[j]]
                pairs.append((target_idx, context_idx))
    return pairs

pairs = generate_skipgram_data(corpus, window_size=5)
print("Number of training pairs:", len(pairs))


# In[8]:


class SkipGramNegSampling(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramNegSampling, self).__init__()
        self.target_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Initialize embeddings
        initrange = 0.5 / embedding_dim
        self.target_embeddings.weight.data.uniform_(-initrange, initrange)
        self.context_embeddings.weight.data.uniform_(-initrange, initrange)

    def forward(self, target, context, negative_samples):
        # Embeddings
        target_emb = self.target_embeddings(target)         # (batch, dim)
        context_emb = self.context_embeddings(context)       # (batch, dim)
        neg_emb = self.context_embeddings(negative_samples)  # (batch, n_neg, dim)

        # Positive score (target dot context)
        pos_score = torch.mul(target_emb, context_emb).sum(dim=1)  # (batch)
        pos_loss = torch.log(torch.sigmoid(pos_score))

        # Negative score (target dot negative samples)
        neg_score = torch.bmm(neg_emb, target_emb.unsqueeze(2)).squeeze()  # (batch, n_neg)
        neg_loss = torch.log(torch.sigmoid(-neg_score)).sum(dim=1)

        return -(pos_loss + neg_loss).mean()


# In[9]:


def get_negative_samples(batch_size, n_neg=5):
    # Sample negatives from unigram distribution (approx)
    negative_samples = torch.randint(low=0, high=vocab_size, size=(batch_size, n_neg))
    return negative_samples


# In[10]:


# Hyperparameters
embedding_dim = 300
window_size = 5
n_neg = 5
batch_size = 512
epochs = 30
learning_rate = 0.001

# Model, optimizer
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
model = SkipGramNegSampling(vocab_size, embedding_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Convert pairs to tensor
pairs_tensor = torch.tensor(pairs, dtype=torch.long)

# Training
for epoch in tqdm(range(epochs)):
    total_loss = 0
    random.shuffle(pairs)

    for i in tqdm(range(0, len(pairs), batch_size)):
        batch = pairs[i:i+batch_size]
        if len(batch) == 0:
            continue
        target, context = zip(*batch)
        target = torch.tensor(target, dtype=torch.long).to(device)
        context = torch.tensor(context, dtype=torch.long).to(device)

        negative_samples = get_negative_samples(len(batch), n_neg).to(device)

        optimizer.zero_grad()
        loss = model(target, context, negative_samples)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


# In[11]:


embeddings = model.target_embeddings.weight.data.cpu().numpy()

# Example: get nearest words to "king"
import numpy as np

def get_similar_words(word, topn=5):
    if word not in word2idx:
        return []
    idx = word2idx[word]
    word_vec = embeddings[idx]
    similarities = np.dot(embeddings, word_vec) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(word_vec) + 1e-10
    )
    similar_indices = similarities.argsort()[-topn-1:][::-1][1:]
    return [idx2word[i] for i in similar_indices]

print(get_similar_words("king"))


# In[12]:


torch.save(model , "Ankit_model.pth")


# In[43]:


Word2Vec_report = {}

for topic in Query_word.keys():
    Word2Vec_report[topic] = {}
    for query in Query_word[topic]:
         Word2Vec_report[topic][query] = get_similar_words(query , topn = 5)




# In[45]:


report_df = pd.DataFrame.from_dict(
    {(topic, query): Word2Vec_report[topic][query]
     for topic in Word2Vec_report
     for query in Word2Vec_report[topic]},
    orient="index"
)   
report_df.head()


# In[ ]:




