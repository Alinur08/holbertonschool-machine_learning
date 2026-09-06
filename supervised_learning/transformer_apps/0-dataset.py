#!/usr/bin/env python3
"""Dataset class for machine translation"""
import transformers
from setup import load_pt2en


class Dataset:
    """Loads and preps a dataset for machine translation"""

    def __init__(self):
        """Class constructor

        Creates the instance attributes:
            data_train: the ted_hrlr_translate/pt_to_en train split as
                a tf.data.Dataset
            data_valid: the ted_hrlr_translate/pt_to_en validation
                split as a tf.data.Dataset
            tokenizer_pt: the Portuguese tokenizer created from the
                training set
            tokenizer_en: the English tokenizer created from the
                training set
        """
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """Creates sub-word tokenizers for our dataset

        Args:
            data: a tf.data.Dataset whose examples are formatted as a
                tuple (pt, en)
                pt: the tf.Tensor containing the Portuguese sentence
                en: the tf.Tensor containing the corresponding
                    English sentence

        Returns:
            tokenizer_pt, tokenizer_en
                tokenizer_pt: the Portuguese tokenizer
                tokenizer_en: the English tokenizer
        """
        pt_sentences = []
        en_sentences = []
        for pt, en in data.as_numpy_iterator():
            pt_sentences.append(pt.decode('utf-8'))
            en_sentences.append(en.decode('utf-8'))

        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased')
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased')

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            pt_sentences, vocab_size=2 ** 13)
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            en_sentences, vocab_size=2 ** 13)

        return tokenizer_pt, tokenizer_en
