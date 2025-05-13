import numpy as np
from collections import defaultdict
from numpy import random



def build_vocab(texts):
    vocab = set()
    for sentence in texts:
        vocab.update(sentence.split())
    return sorted(vocab)

def text_to_matrix(texts, word2idx):
    mat = np.zeros((len(texts), len(word2idx)))
    for i, sentence in enumerate(texts):
        for word in sentence.split():
            if word in word2idx:
                mat[i,word2idx[word]] += 1

    return mat


if __name__=="__main__":
    # Sample data
    X_text = ["free money now", "hello friend", "win a prize", "let's catch up"]
    y = random.randint(0, 2, (len(X_text)))
    
    vocab = build_vocab(X_text)
    word2idx = {word : i for i, word in enumerate(vocab)}

    X = text_to_matrix(X_text, word2idx)

    print(X)