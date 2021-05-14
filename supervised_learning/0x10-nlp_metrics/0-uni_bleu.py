#!/usr/bin/env python3

""" 0. Unigram BLEU score"""
import numpy as np


def uni_bleu(references, sentence):
    """
   calculates the unigram BLEU score for a sentence
   ARGS:
        *references: {list} :reference translations
            each reference translation is a list of
            the words in the translation
        *sentence: {list}: the model proposed sentence
    Returns: the unigram BLEU score
    """
    out_put_len = len(sentence)
    ref_len = np.array([len(r) for r in references])
    ref_len_idx = np.argmin(np.abs(ref_len - out_put_len))
    ref_len = len(references[ref_len_idx])

    if out_put_len > ref_len:
        bp = 1
    else:
        bp = np.exp(1 - ref_len / out_put_len)
    flat_list = list(np.concatenate(references).flat)
    flat_list = set(flat_list)
    wordsin = list(flat_list.intersection(sentence))
    # wordsout =  list(set(flat_list) - set(sentence))
    p = len(wordsin) / out_put_len
    return bp * p
