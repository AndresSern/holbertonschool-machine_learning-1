#!/usr/bin/env python3
"""
performs semantic search on a corpus of documents
"""
import tensorflow_hub as hub
import os
import numpy as np


def semantic_search(corpus_path, sentence):
    """
    ARGS:
        -corpus_path : is the path to the corpus of reference
            documents on which to perform semantic search
        -sentence : is the sentence from which to
            perform semantic search
    Returns:
        -the reference text of the document most similar to sentence
    """
    dirs = os.listdir(corpus_path)

    # documents_list = [word, sentence, paragraph]
    documents_list = []
    documents_list.append(sentence)
    for fle in dirs:
        if not fle.endswith('.md'):
            continue
        # open the file and then call .read() to get the text
        with open(corpus_path+'/'+fle,   encoding="utf_8") as f:
            text = f.read()
            documents_list.append(text)

    embed = hub.load(
        "https://tfhub.dev/google/universal-sentence-encoder-large/5")

    embeddings = embed(documents_list)
    similarity = np.inner(embeddings, embeddings)

    indx = np.argmax(similarity[0, 1:])
    return(documents_list[indx+1])
