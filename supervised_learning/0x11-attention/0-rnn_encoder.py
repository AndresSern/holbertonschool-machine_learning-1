#!/usr/bin/env python3
"""
inherits from tensorflow.keras.layers.Layer
to encode for machine translation:"""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """
    inherits from tensorflow.keras.layers.Layer
    to encode for machine translation:
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        ARGS:
            vocab:{integer} :the size of the input vocabulary
            embedding:{integer} : the dimensionality of the embedding vector
            units:{integer} : the number of hidden units in the RNN cell
            batch:{integer} : the batch size
        """
        super().__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       kernel_initializer="glorot_uniform",
                                       return_sequences=True,
                                       return_state=True)

    def initialize_hidden_state(self):
        """
        Initializes the hidden states for the RNN cell to a tensor of zeros
        Returns:
            tensor of shape (batch, units) :the initialized hidden states
        """

        initializer = tf.keras.initializers.Zeros()
        hidden_states = initializer(shape=(self.batch, self.units))
        return hidden_states

    def call(self, x, initial):
        """
        ARGS:
            -x : {tensor} shape :(batch, input_seq_len) :the input of encoder
                layer as word indices within the vocabulary
            -initial: {tensor}:shape (batch, units) : the initial hidden state
        Returns: outputs, hidden
            -outputs: {tensor} : shape (batch, input_seq_len, units)
                the outputs of the encoder
            -hidden: {tensor} : shape (batch, units)
                the last hidden state of the encoder
        """
        embedding = self.embedding(x)
        encoder_outputs, state_h = self.gru(embedding, initial_state=initial)
        return encoder_outputs, state_h
