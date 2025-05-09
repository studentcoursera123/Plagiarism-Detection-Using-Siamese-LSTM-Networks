import streamlit as st
import torch
import torch.nn as nn
from collections import Counter
import numpy as np

# --------------------------
# Tokenizer and Vocabulary
# --------------------------
def tokenize(text):
    return text.lower().split()

# Dummy sentences to build vocab (same as your notebook)
sent1 = ["hello world", "this is a test", "hello again"]
sent2 = ["hello universe", "this is another test", "hello once more"]

counter = Counter()
for s in sent1 + sent2:
    counter.update(tokenize(s))

word2idx = {'<pad>': 0, '<unk>': 1}
for word in counter:
    word2idx[word] = len(word2idx)

def encode(sentence):
    return [word2idx.get(w, word2idx['<unk>']) for w in tokenize(sentence)]

def pad_sequence(seq, max_len=100):
    padded = torch.zeros(max_len, dtype=torch.long)
    seq = torch.tensor(seq[:max_len])
    padded[:len(seq)] = seq
    return padded

# --------------------------
# Siamese LSTM Model
# --------------------------
class SiameseLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, max_len, dropout=0.5):
        super(SiameseLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 4, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def encode(self, x):
        embedded = self.embedding(x.unsqueeze(0))  # Add batch dim
        _, (hidden, _) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return hidden

    def forward(self, x1, x2):
        out1 = self.encode(x1)
        out2 = self.encode(x2)
        combined = torch.cat((out1, out2), dim=1)
        return self.fc(combined)

# --------------------------
# Load Trained Model
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = SiameseLSTM(vocab_size=5000, embedding_dim=128, hidden_dim=256, max_len=100)

# Try loading the model and handle potential errors
try:
    # Load the model's state dict
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.to(device)
    model.eval()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# --------------------------
# Streamlit UI
# --------------------------
st.title("Plagiarism Detection App")
st.write("Enter two sentences to check for plagiarism probability.")

# Input fields for the sentences
s1 = st.text_input("Sentence 1")
s2 = st.text_input("Sentence 2")

if st.button("Check"):
    if not s1.strip() or not s2.strip():
        st.warning("⚠️ Please enter both sentences.")
    else:
        # Process the input sentences
        with torch.no_grad():
            e1 = encode(s1)
            e2 = encode(s2)
            p1 = pad_sequence(e1).to(device)
            p2 = pad_sequence(e2).to(device)
            output = model(p1, p2)
            prob = output.item()

            # Display the plagiarism probability
            st.success(f"Plagiarism Probability: **{prob*100:.2f}%**")
