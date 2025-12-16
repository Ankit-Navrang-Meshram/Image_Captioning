# Run this inside model_a_cnn_lstm/
import pandas as pd
import pickle
from dataset import Vocabulary 
df = pd.read_csv("../data/captions.txt")
vocab = Vocabulary(freq_threshold=5)
vocab.build_vocabulary(df["caption"].tolist())
with open("vocab.pkl", "wb") as f: pickle.dump(vocab, f)