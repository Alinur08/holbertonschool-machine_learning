#!/usr/bin/env python3
"""Cumulative bleu score"""
import numpy as np


def cumulative_bleu(references, sentence, n):
    """
    Calculates the cumulative n-gram BLEU score for a sentence.
    """
    def get_ngrams(words, k):
        """ngram words"""
        return [tuple(words[i:i + k]) for i in range(len(words) - k + 1)]

    def ngram_precision(k):
        """Precision calculation"""
        sentence_ngrams = get_ngrams(sentence, k)
        sentence_len = len(sentence_ngrams)

        ngram_counts = {}
        for ngram in sentence_ngrams:
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1

        max_ref_counts = {}
        for ref in references:
            ref_ngrams = get_ngrams(ref, k)
            ref_counts = {}
            for ngram in ref_ngrams:
                ref_counts[ngram] = ref_counts.get(ngram, 0) + 1
            for ngram, count in ref_counts.items():
                max_ref_counts[ngram] = max(
                    max_ref_counts.get(ngram, 0), count
                    )

        clipped_count = 0
        for ngram, count in ngram_counts.items():
            clipped_count += min(count, max_ref_counts.get(ngram, 0))

        return clipped_count / sentence_len

    precisions = [ngram_precision(k) for k in range(1, n + 1)]

    # Geometric mean of precisions, weighted evenly
    log_precisions = [
        np.log(p) if p > 0 else float('-inf') for p in precisions
        ]
    if float('-inf') in log_precisions:
        geo_mean = 0
    else:
        geo_mean = np.exp(sum(log_precisions) / n)

    # Brevity penalty based on word counts
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

    return bp * geo_mean
