#!/usr/bin/env python
# coding: utf-8

# ### loading dataset and preprocessing

# In[1]:


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
    #print([token.text for token in doc])
    # Lemmatize + remove stopwords
    tokens = [
        lemmatizer.lemmatize(token.text) 
        for token in doc 
        if token.text not in stop_words and not token.is_space
    ]

    return  tokens #" ".join(tokens)


# In[52]:


spam_data = {}

with open("/home/mudasir/ankit/NLP/Lab_06/spam.txt" , 'r') as f:
    text = []
    label = []
    for line in f:
        label.append(int(line.split()[-1]))
        text.append(" ".join(line.split()[:-1]))
    spam_data['text'] = text
    spam_data['label'] = label


# In[53]:


print(spam_data['text'][:5])
print(spam_data['label'][:5])


# In[54]:


preprocessed_spam_data = {}
preprocessed_spam_data['text'] = [text_preprocessing(t) for t in spam_data['text']]
preprocessed_spam_data['label'] = spam_data['label']


# In[55]:


print(preprocessed_spam_data['text'][:5])
print(preprocessed_spam_data['label'][:5]) 


# In[56]:


from torch.utils.data import Dataset, random_split

class SpamDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# Create dataset
dataset = SpamDataset(preprocessed_spam_data['text'], preprocessed_spam_data['label'])

# Split sizes
dataset_len = len(dataset)
train_len = int(dataset_len * 0.7)
val_len = int(dataset_len * 0.2)
test_len = dataset_len - train_len - val_len  # ensure total matches

# Split
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_len, val_len, test_len])
print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")
print("Tain, Val, Test ratio:" , len(train_dataset)/dataset_len, len(val_dataset)/dataset_len, len(test_dataset)/dataset_len)


# ---

# ###  what do we require from lab 04?
# 1. word2vec model target_embedding layer
# 2. input to this layer is the index value of each word
# 3. mapping of word->index is given by word2idx() dictionary
# 4. word2idx() requires lab 04 dataset and creating vocab fromn it and then creating word2idx() dictionary
# 5. to handle oov add <|pad|> token to voacb and then create word2idx , then train word2vec algo

# In[57]:


import os
import sys
sys.path.append('/home/mudasir/ankit/NLP')
from custom_word2vec import preprocess

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
            line = preprocess.text_preprocessing(line)
            line = re.sub(r'\n', '', line)
            doc.append(line)
    docs.append(' '.join(doc))

corpus = []
for doc in docs:
    for word in doc.split():
        corpus.append(word)

corpus.append("UNK") # for oov words



# In[58]:


from custom_word2vec import Utility
u = Utility(corpus)
vocab, word2idx, idx2word, vocab_size , embedding_dim = u.return_essentials()


# In[112]:


# getting pretrained embeddings from the saved word2vec model
import sys
sys.path.append("/home/mudasir/ankit/NLP")
import torch
from custom_word2vec import SkipGramNegSampling  # class must be importable

word2vec_model = SkipGramNegSampling(vocab_size , embedding_dim)
word2vec_model.load_state_dict(torch.load("/home/mudasir/ankit/NLP/custom_kipgram_model.pth"))


# In[113]:


pretrained_embeddings = word2vec_model.target_embeddings


# In[114]:


print(pretrained_embeddings.weight)  # should be (vocab_size, embedding_dim)


# ### RNN model

# In[115]:


import torch 
import torch.nn as nn


class RNNModel(nn.Module):
    def __init__(self, hidden_size, output_size, vocab_size=None, embed_size=None,n_layers=1, bidirectional=False, pretrained_embeddings=None):
        super(RNNModel, self).__init__()
        if pretrained_embeddings is not None:
            vocab_size, embed_size = pretrained_embeddings.weight.shape
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings.weight, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embed_size)

        self.rnn = nn.RNN(embed_size, hidden_size, num_layers=n_layers,bidirectional=bidirectional, batch_first=True)
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        rnn_out, _ = self.rnn(x)
        out = rnn_out[:, -1, :]  # last time step
        out = self.fc(out)
        out = self.softmax(out)
        return out


# In[116]:


def custom_collate_fn(batch):
    texts, labels = zip(*batch)
    lengths = [len(text) for text in texts]
    max_length = max(lengths)
    
    padded_texts = []
    for text in texts:
        # encode text
        encoded_text = [word2idx.get(word, word2idx["UNK"]) for word in text]
        encoded_tensor = torch.tensor(encoded_text, dtype=torch.long)
        # pad
        padded_text = torch.cat([encoded_tensor, torch.zeros(max_length - len(encoded_tensor), dtype=torch.long)])
        padded_texts.append(padded_text)
    
    return torch.stack(padded_texts), torch.tensor(labels, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)


# In[117]:


# Train, test, val dataloaders
from torch.utils.data import DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)


# In[118]:


train_loader.dataset[0]


# In[119]:


model = RNNModel(hidden_size=128, output_size=2, pretrained_embeddings=pretrained_embeddings, bidirectional=False)


# In[120]:


emb = model.embedding.weight.data
print("NaN in embedding:", torch.isnan(emb).any())


# #### Normal Training

# In[125]:


def train_rnn_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001 , device = None , fold = None):
    criterion = nn.CrossEntropyLoss() 
    if device is None:
        device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        total_val_loss = 0
        for texts, labels , _ in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        for texts, labels , _ in val_loader:
            texts, labels = texts.to(device), labels.to(device)
            model.eval()
            with torch.no_grad():
                outputs = model(texts)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    if fold is not None:
        torch.save(model.state_dict(), f"rnn_model_fold{fold}.pth")
        print(f"Model saved as rnn_model_fold{fold}.pth")
    else:
        torch.save(model.state_dict(), "rnn_model.pth")
        print("Model saved as rnn_model.pth")


# In[122]:


device  = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
train_rnn_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001, device=device)


# In[ ]:


import sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score , confusion_matrix

def evaluate_rnn_model(model , test_loader , device = None , fold = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for texts, labels , _ in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)
    if fold is not None:
        print(f"Confusion Matrix for {fold+1} fold:")
        print(cm)
        print(f"Test Accuracy for {fold+1} fold: {accuracy:.4f}")
        print(f"Test Precision for {fold+1} fold: {precision:.4f}")
        print(f"Test Recall for {fold+1} fold: {recall:.4f}")
        print(f"Test F1 Score for {fold+1} fold: {f1:.4f}")
    else:
        print("Confusion Matrix for")
        print(cm)
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Test F1 Score: {f1:.4f}")
    print()
    return accuracy, precision, recall, f1, cm


# In[ ]:


_ , _ , _ , _ , _ = evaluate_rnn_model(model , test_loader , device = device)


# #### K-fold Cross validation

# In[135]:


from sklearn.model_selection import KFold
from torch.utils.data import random_split , DataLoader , Subset

dataset = SpamDataset(preprocessed_spam_data['text'], preprocessed_spam_data['label'])

train_dataset_len = int(0.9 * len(dataset))


train_dataset, test_dataset = random_split(dataset, [train_dataset_len, len(dataset) - train_dataset_len])

kf = KFold(n_splits=5, shuffle=True, random_state=42)

accuracys, precisions, recalls, f1s, cms = [], [], [], [], []

for fold , (train_idx , val_idx) in enumerate(kf.split(train_dataset)):
    print(f"Fold {fold+1}")
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(train_dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)

    model = RNNModel(hidden_size=128, output_size=2, pretrained_embeddings=pretrained_embeddings, bidirectional=False)

    device  = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    train_rnn_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001, device=device , fold = fold+1)
    accuracy, precision, recall, f1, cm  = evaluate_rnn_model(model , test_loader , device = device , fold = fold+1 )
    accuracys.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)
    cms.append(cm)
    print("--------------------------------------------------")
    print("Avarage accuracy : " , sum(accuracys) / len(accuracys))
    print("Avarage precision : " , sum(precisions) / len(precisions))
    print("Avarage recall : " , sum(recalls) / len(recalls))     
    print("Avarage f1 : " , sum(f1s)/len(f1s)) 
    print()
    print("--------------------------------------------------")


# ---

# ### CNN Model

# In[146]:


import torch.nn as nn
from torch.nn import Conv1d

class CNNModel(nn.Module):
    def __init__(self, num_classes, vocab_size=None, embed_size=None, pretrained_embeddings=None):
        super(CNNModel, self).__init__()
        if pretrained_embeddings is not None:
            vocab_size, embed_size = pretrained_embeddings.weight.shape
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings.weight, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embed_size)

        self.conv1 = Conv1d(in_channels=embed_size, out_channels=200, kernel_size=3, padding=1)
        self.conv2 = Conv1d(in_channels=200, out_channels=100, kernel_size=3, padding=1)
        self.conv3 = Conv1d(in_channels=100, out_channels=10, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc = nn.Linear(10 * embed_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)  # (batch_size, seq_length, embed_size)
        x = x.permute(0, 2, 1)  # (batch_size, embed_size, seq_length)

        x1 = self.pool(self.relu(self.conv1(x)))  # (batch_size, out_channels, seq_length)
        x2 = self.pool(self.relu(self.conv2(x1)))
        x3 = self.pool(self.relu(self.conv3(x2)))

        x = self.fc(torch.flatten(x3, start_dim=1))
        x = self.softmax(x)
        return x




# In[147]:


def train_cnn_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001 , device = None , fold = None):
    criterion = nn.CrossEntropyLoss() 
    if device is None:
        device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        total_val_loss = 0
        for texts, labels , _ in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        for texts, labels , _ in val_loader:
            texts, labels = texts.to(device), labels.to(device)
            model.eval()
            with torch.no_grad():
                outputs = model(texts)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    if fold is not None:
        torch.save(model.state_dict(), f"cnn_model_fold{fold}.pth")
        print(f"Model saved as cnn_model_fold{fold}.pth")
    else:
        torch.save(model.state_dict(), "cnn_model.pth")
        print("Model saved as cnn_model.pth")


# In[148]:


import sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score , confusion_matrix

def evaluate_cnn_model(model , test_loader , device = None , fold = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for texts, labels , _ in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)
    if fold is not None:
        print(f"Confusion Matrix for {fold+1} fold:")
        print(cm)
        print(f"Test Accuracy for {fold+1} fold: {accuracy:.4f}")
        print(f"Test Precision for {fold+1} fold: {precision:.4f}")
        print(f"Test Recall for {fold+1} fold: {recall:.4f}")
        print(f"Test F1 Score for {fold+1} fold: {f1:.4f}")
    else:
        print("Confusion Matrix for")
        print(cm)
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Test F1 Score: {f1:.4f}")
    print()
    return accuracy, precision, recall, f1, cm


# In[149]:


from sklearn.model_selection import KFold
from torch.utils.data import random_split , DataLoader , Subset

dataset = SpamDataset(preprocessed_spam_data['text'], preprocessed_spam_data['label'])

train_dataset_len = int(0.9 * len(dataset))


train_dataset, test_dataset = random_split(dataset, [train_dataset_len, len(dataset) - train_dataset_len])

kf = KFold(n_splits=5, shuffle=True, random_state=42)

accuracys, precisions, recalls, f1s, cms = [], [], [], [], []

for fold , (train_idx , val_idx) in enumerate(kf.split(train_dataset)):
    print(f"Fold {fold+1}")
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(train_dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)

    model = CNNModel(num_classes=2, vocab_size=vocab_size, embed_size=embedding_dim, pretrained_embeddings=pretrained_embeddings)

    device  = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    train_cnn_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001, device=device , fold = fold+1)
    accuracy, precision, recall, f1, cm  = evaluate_cnn_model(model , test_loader , device = device , fold = fold+1 )
    accuracys.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)
    cms.append(cm)
    print("--------------------------------------------------")
    print("Avarage accuracy : " , sum(accuracys) / len(accuracys))
    print("Avarage precision : " , sum(precisions) / len(precisions))
    print("Avarage recall : " , sum(recalls) / len(recalls))     
    print("Avarage f1 : " , sum(f1s)/len(f1s)) 
    print()
    print("--------------------------------------------------")


# ---
# 

# #### pos tagging

# In[13]:


from datasets import load_dataset , get_dataset_split_names

print(get_dataset_split_names("batterydata/pos_tagging"))



# In[21]:


train_dataset = load_dataset("batterydata/pos_tagging" , split = 'train')
test_dataset = load_dataset("batterydata/pos_tagging", split="test")


# In[22]:


len(train_dataset), len(test_dataset)


# In[23]:


train_dataset


# In[24]:


import torch
from torch.utils.data import Dataset

class PosDataset(Dataset):

    def __init__(self , dataset_dic):
        self.texts = dataset_dic['words']
        self.labels = dataset_dic['labels']

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


# In[25]:


train_dataset = PosDataset(train_dataset)
test_dataset = PosDataset(test_dataset)
train_len = len(train_dataset)
test_len = len(test_dataset)


# In[ ]:


import torch.nn as nn

class RNNModelforPos(nn.Module):
    def __init__(self, text_vocab_size, hidden_dim, output_dim):
        super(RNNModelforPos, self).__init__()
        self.InputEmbedding = nn.Embedding(text_vocab_size, hidden_dim)
        self.rnn = nn.RNN(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.InputEmbedding(x)
        x, _ = self.rnn(x)
        x = self.fc(x[:, :, :])
        return nn.Softmax(dim=-1)(x)


# In[ ]:


vocabulary = set()
for text, _ in train_dataset:
    for word in text:
        vocabulary.add(word)
vocabulary.add("UNK") # for oov words
vocab_size = len(vocabulary) + 1  # +1 for padding or unknown token
text_vocab_size = vocab_size

pos_tags = set()

for _ , pos in train_dataset:
    for tag in pos:
        pos_tags.add(tag)

pos_tags.add("UNKTAG") # for oov tags
output_dim = len(pos_tags) + 1  # +1 for padding or unknown tag


# In[28]:


word2idx = {word: idx for idx, word in enumerate(vocabulary)}
idx2word = {idx: word for word, idx in word2idx.items()}
tag2idx = {tag: idx for idx, tag in enumerate(pos_tags)}
idx2tag = {idx: tag for tag, idx in tag2idx.items()}


# In[ ]:


from torch.nn.utils.rnn import pad_sequence
def custom_collate_fn_pos(batch):
    texts, pos_tags = zip(*batch)  
    texts = [torch.tensor([word2idx.get(word, word2idx["UNK"]) for word in text]) for text in texts]
    pos_tags = [torch.tensor([tag2idx.get(tag, tag2idx["UNKTAG"]) for tag in tags]) for tags in pos_tags]
    texts = pad_sequence(texts, batch_first=True, padding_value=word2idx["UNK"])
    pos_tags = pad_sequence(pos_tags, batch_first=True, padding_value=tag2idx["UNKTAG"])
    
    return texts, pos_tags


# In[39]:


from sklearn.model_selection import KFold
from torch.utils.data import random_split , DataLoader , Subset


kf = KFold(n_splits=5, shuffle=True, random_state=42)

accuracys, precisions, recalls, f1s, cms = [], [], [], [], []

for fold , (train_idx , val_idx) in enumerate(kf.split(train_dataset)):
    print(f"Fold {fold+1}")
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(train_dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn_pos)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn_pos)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn_pos)

    model = RNNModelforPos(hidden_dim=128, text_vocab_size=text_vocab_size, output_dim=output_dim)
    device  = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    train_pos_rnn_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001, device=device , fold = fold+1)
    accuracy, precision, recall, f1, cm  = evaluate_pos_rnn_model(model , test_loader , device = device , fold = fold+1 )
    accuracys.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)
    cms.append(cm)
    print("--------------------------------------------------")
    print("Avarage accuracy : " , sum(accuracys) / len(accuracys))
    print("Avarage precision : " , sum(precisions) / len(precisions))
    print("Avarage recall : " , sum(recalls) / len(recalls))     
    print("Avarage f1 : " , sum(f1s)/len(f1s)) 
    print()
    print("--------------------------------------------------")



# In[38]:


def train_pos_rnn_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001 , device = None , fold = None):
    criterion = nn.CrossEntropyLoss() 
    if device is None:
        device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        total_val_loss = 0
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        for texts, labels in val_loader:
            texts, labels = texts.to(device), labels.to(device)
            model.eval()
            with torch.no_grad():
                outputs = model(texts)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    if fold is not None:
        torch.save(model.state_dict(), f"pos_rnn_model_fold{fold}.pth")
        print(f"Model saved as pos_rnn_model_fold{fold}.pth")
    else:
        torch.save(model.state_dict(), "pos_rnn_model.pth")
        print("Model saved as pos_rnn_model.pth")


# In[36]:


import sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score , confusion_matrix

def evaluate_pos_rnn_model(model , test_loader , device = None , fold = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)
    if fold is not None:
        print(f"Confusion Matrix for {fold+1} fold:")
        print(cm)
        print(f"Test Accuracy for {fold+1} fold: {accuracy:.4f}")
        print(f"Test Precision for {fold+1} fold: {precision:.4f}")
        print(f"Test Recall for {fold+1} fold: {recall:.4f}")
        print(f"Test F1 Score for {fold+1} fold: {f1:.4f}")
    else:
        print("Confusion Matrix for")
        print(cm)
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Test F1 Score: {f1:.4f}")
    print()
    return accuracy, precision, recall, f1, cm





