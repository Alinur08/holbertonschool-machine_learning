#!/usr/bin/env python3
"""Dataset module"""
import tensorflow as tf
import transformers
from setup import load_pt2en


class Dataset:
    """Loads and tokenizes the Portuguese-English translation dataset."""

    def __init__(self):
        """Class constructor."""
        self.data_train = load_pt2en(split="train")
        self.data_valid = load_pt2en(split="validation")
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )
        # Tokenize the training and validation examples
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

    def tokenize_dataset(self, data):
        """
        Creates subword tokenizers for Portuguese and English.
        """
        pt_sentences = []
        en_sentences = []
        # Iterate over the dataset
        for pt, en in data:
            pt_sentences.append(pt.numpy().decode('utf-8'))
            en_sentences.append(en.numpy().decode('utf-8'))
        # Create tokenizers for Portuguese and English
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased', use_fast=True,
            clean_up_tokenization_spaces=True)
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased', use_fast=True,
            clean_up_tokenization_spaces=True)
        # Train the tokenizers
        tokenizer_pt = tokenizer_pt.train_new_from_iterator(pt_sentences,
                                                            vocab_size=2 ** 13)
        tokenizer_en = tokenizer_en.train_new_from_iterator(en_sentences,
                                                            vocab_size=2 ** 13)
        self.tokenizer_pt = tokenizer_pt
        self.tokenizer_en = tokenizer_en
        return self.tokenizer_pt, self.tokenizer_en

    def encode(self, pt, en):
        """
        Encodes a translation into tokens, adding start/end tokens.
        """
        # Determine start and end token indices for each language
        pt_vocab_size = self.tokenizer_pt.vocab_size
        en_vocab_size = self.tokenizer_en.vocab_size
        # Encode the sentences without the tokenizers' own special tokens
        pt_tokens = self.tokenizer_pt.encode(
            pt.numpy().decode('utf-8'), add_special_tokens=False)
        en_tokens = self.tokenizer_en.encode(
            en.numpy().decode('utf-8'), add_special_tokens=False)
        # Prepend start token and append end token
        pt_tokens = [pt_vocab_size] + pt_tokens + [pt_vocab_size + 1]
        en_tokens = [en_vocab_size] + en_tokens + [en_vocab_size + 1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        Acts as a tensorflow wrapper for the encode instance method.
        """
        # Wrap the Python encode method for use inside tf.data.Dataset.map
        pt_tokens, en_tokens = tf.py_function(
            self.encode, [pt, en], [tf.int64, tf.int64])
        # Set the shape of the returned tensors
        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])

        return pt_tokens, en_tokens
