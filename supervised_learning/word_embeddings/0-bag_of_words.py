#!/usr/bin/env python3
"""bag of words"""
import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix.
    """
    def tokenize(sentence):
        '''lowercase, strip possessives ('s), keep only word chars'''
        sentence = sentence.lower()
        words = re.findall(r"[a-z]+(?:'[a-z]+)?", sentence)
        # drop bare possessive 's tokens, strip trailing 's from words
        cleaned = []
        for w in words:
            if w.endswith("'s"):
                w = w[:-2]
            if w:
                cleaned.append(w)
        return cleaned

    tokenized = [tokenize(s) for s in sentences]

    if vocab is None:
        vocab_set = set()
        for tokens in tokenized:
            vocab_set.update(tokens)
        features = sorted(vocab_set)
    else:
        features = list(vocab)

    feat_index = {word: i for i, word in enumerate(features)}

    s = len(sentences)
    f = len(features)
    embeddings = np.zeros((s, f), dtype=int)

    for i, tokens in enumerate(tokenized):
        for word in tokens:
            j = feat_index.get(word)
            if j is not None:
                embeddings[i, j] += 1

    return embeddings, np.array(features)
