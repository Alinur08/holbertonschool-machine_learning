#!/usr/bin/env python3
"""Unigram bleu score"""
import numpy as np


def uni_bleu(references, sentence):
    """
    Calculates the unigram BLEU score for a sentence.
    """
    sentence_len = len(sentence)

    # Count occurrences of each word in the candidate sentence
    word_counts = {}
    for word in sentence:
        word_counts[word] = word_counts.get(word, 0) + 1

    # Clipped counts:
    max_ref_counts = {}
    for ref in references:
        ref_counts = {}
        for word in ref:
            ref_counts[word] = ref_counts.get(word, 0) + 1
        for word, count in ref_counts.items():
            max_ref_counts[word] = max(max_ref_counts.get(word, 0), count)

    clipped_count = 0
    for word, count in word_counts.items():
        clipped_count += min(count, max_ref_counts.get(word, 0))

    precision = clipped_count / sentence_len

    ref_lens = [len(ref) for ref in references]
    closest_ref_len = min(
        ref_lens,
        key=lambda ref_len: (abs(ref_len - sentence_len), ref_len)
    )

    if sentence_len > closest_ref_len:
        bp = 1
    else:
        bp = np.exp(1 - closest_ref_len / sentence_len)

    return bp * precision
