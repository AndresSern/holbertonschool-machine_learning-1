#!/usr/bin/env python3
"""
loads and preps a dataset for machine translation
"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """loads and preps a dataset for machine translation"""
    def __init__(self, batch_size, max_len):
        """
        initialize class constructor
        -tokenizer_pt:the Portuguese tokenizer
            created from the training set
        -tokenizer_en:the English tokenizer
            created from the training set
        """
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.batch_size = batch_size
        self.max_len = max_len
        self.data_train,info = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train', as_supervised=True,with_info =True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation', as_supervised=True)
        buffer_size=info.splits['train'].num_examples

        Portuguese, English = self.tokenize_dataset(self.data_train)
        self.tokenizer_en = English
        self.tokenizer_pt = Portuguese

        self.data_train = self.data_train.map(self.tf_encode)
        
        self.data_train = self.data_train.filter(lambda x,y: tf.math.logical_and(
            tf.size(x)<=self.max_len , tf.size(y)<= self.max_len))
       

        self.data_train = self.data_train.cache().shuffle(buffer_size).padded_batch(
            self.batch_size).prefetch(buffer_size=AUTOTUNE)
        

        self.data_valid = self.data_valid.map(self.tf_encode)
        self.data_valid = self.data_valid.filter(lambda x,y: tf.math.logical_and(tf.size(
            x) <= self.max_len , tf.size(y) <= self.max_len)).padded_batch(self.batch_size)


    def tokenize_dataset(self, data):
        """
        creates sub-word tokenizers for our dataset
        ARGS:
        --data :{tf.data.Dataset} whose examples are
            formatted as a tuple (pt, en)
        --pt:{tf.Tensor} containing the Portuguese sentence
        --en:{tf.Tensor} containing the corresponding English sentence

        Returns: tokenizer_pt, tokenizer_en
        --tokenizer_pt is the Portuguese tokenizer
        --tokenizer_en is the English tokenizer
        """
        Portuguese = []
        English = []
        for pt, en in data:
            Portuguese.append(pt.numpy())
            English.append(en.numpy())
        tok_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            Portuguese, target_vocab_size=2**15)
        tok_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            English, target_vocab_size=2**15)
        return tok_pt, tok_en

    def encode(self, pt, en):
        """
        encodes a translation into tokens
        ARGS:
            -pt:{tf.Tensor} containing the
                Portuguese sentence
            -en:{tf.Tensor} containing the
                corresponding English sentence
        Returns: pt_tokens, en_tokens
            -pt_tokens:{np.ndarray} containing
                the Portuguese tokens
            -en_tokens:{np.ndarray}
                containing the English tokens
        """
        encoder = self.tokenizer_en
        encoder1 = self.tokenizer_pt
        en_tokens = [encoder.vocab_size] + encoder.encode(en.numpy()) + \
            [encoder.vocab_size+1]
        pt_tokens = [encoder1.vocab_size]+encoder1.encode(pt.numpy()) + \
            [encoder1.vocab_size+1]
        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        acts as a tensorflow wrapper for
        the encode instance method"""

        pt_tokens, en_tokens = tf.py_function(
            func=self.encode, inp=[pt, en], Tout=[tf.int64,tf.int64])
        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])
        return pt_tokens, en_tokens

