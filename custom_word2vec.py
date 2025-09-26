
import torch
import torch.nn as nn
from collections import Counter
from tqdm import tqdm
import random
import argparse as parse
import os
import re
import emoji
import contractions
import spacy
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer




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
    
class Utility():
    
    def __init__(self, corpus):
        self.corpus = corpus
        # Build vocabulary
        word_counts = Counter(self.corpus)
        self.vocab = sorted(word_counts, key=word_counts.get, reverse=True)
        self.word2idx = {w: idx for idx, w in enumerate(self.vocab)}
        self.idx2word = {idx: w for w, idx in self.word2idx.items()}
        self.vocab_size = len(self.vocab)


    def get_negative_samples(self, batch_size, n_neg=5):
        # Sample negatives from unigram distribution (approx)
        negative_samples = torch.randint(low=0, high=self.vocab_size, size=(batch_size, n_neg))
        return negative_samples


    def generate_skipgram_data(self, window_size=5):
        pairs = []
        for i, target_word in enumerate(self.corpus):
            target_idx = self.word2idx[target_word]
            # context words within window
            for j in range(max(0, i - window_size), min(len(self.corpus), i + window_size + 1)):
                if i != j:
                    context_idx = self.word2idx[self.corpus[j]]
                    pairs.append((target_idx, context_idx))
        return pairs


    def train(self):
        # Hyperparameters
        embedding_dim = 300
        window_size = 5
        n_neg = 5
        batch_size = 512
        epochs = 45
        learning_rate = 0.001

        # Model, optimizer
        device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
        model = SkipGramNegSampling(self.vocab_size, embedding_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        pairs = self.generate_skipgram_data(self.corpus, window_size=5)
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

                negative_samples = self.get_negative_samples(len(batch), n_neg).to(device)

                optimizer.zero_grad()
                loss = model(target, context, negative_samples)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

        torch.save(model.state_dict(), "custom_kipgram_model.pth")

    def return_essentials(self):
        return self.vocab, self.word2idx, self.idx2word, self.vocab_size


class preprocess:

    nlp = spacy.load("en_core_web_sm")

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    @staticmethod
    def text_preprocessing(text):
        text = emoji.demojize(text)
        text = contractions.fix(text)
        text = text.lower()
        text = re.sub(r'\d{4}-\d{2}-\d{2}', 'DATE', text)
        text = re.sub(r'\d+', "NUM", text)
        text = re.sub(r'\b\w+@\w+\.\w+\b', 'EMAIL', text)
        text = re.sub(r'[^\w\s]', "", text)
        # Tokenize with spaCy
        doc = preprocess.nlp(text)
        #print([token.text for token in doc])
        # Lemmatize + remove stopwords
        tokens = [
            preprocess.lemmatizer.lemmatize(token.text) 
            for token in doc 
            if token.text not in preprocess.stop_words and not token.is_space
        ]

        return  tokens #" ".join(tokens)

if __name__ == "__main__":
    args = parse.ArgumentParser()
    args.add_argument("--data_path", type=str, default="/home/mudasir/ankit/NLP/bbc", help="Enter the path to the text data file")
    parsed_args = args.parse_args()
    dataset_path = args.get("data_path")

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

    u = Utility(corpus)
    u.train()




