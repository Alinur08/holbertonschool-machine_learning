#!/usr/bin/env python3
"""N-gram bleu score"""
import numpy as np


def ngram_bleu(references, sentence, n):
    """
    Calculates the n-gram BLEU score for a sentence.
    """
    def get_ngrams(words, n):
        return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]

    sentence_ngrams = get_ngrams(sentence, n)
    sentence_len = len(sentence_ngrams)

    # Count occurrences of each n-gram in the candidate
    ngram_counts = {}
    for ngram in sentence_ngrams:
        ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1

    # Clipped counts
    max_ref_counts = {}
    for ref in references:
        ref_ngrams = get_ngrams(ref, n)
        ref_counts = {}
        for ngram in ref_ngrams:
            ref_counts[ngram] = ref_counts.get(ngram, 0) + 1
        for ngram, count in ref_counts.items():
            max_ref_counts[ngram] = max(max_ref_counts.get(ngram, 0), count)

    clipped_count = 0
    for ngram, count in ngram_counts.items():
        clipped_count += min(count, max_ref_counts.get(ngram, 0))

    precision = clipped_count / sentence_len

    # Brevity penalty
    sen_len = len(sentence)
    ref_lens = [len(ref) for ref in references]
    closest_ref_len = min(
        ref_lens,
        key=lambda ref_len: (abs(ref_len - sen_len), ref_len)
        )

    if sen_len > closest_ref_len:
        bp = 1
    else:
        bp = np.exp(1 - closest_ref_len / sen_len)

    return bp * precision
