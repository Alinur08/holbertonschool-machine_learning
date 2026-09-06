#!/usr/bin/env python3
"""Tf-idf vectorizer"""
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding.
    """
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform(sentences)

    embeddings = X.toarray()
    features = vectorizer.get_feature_names_out()

    return embeddings, features
