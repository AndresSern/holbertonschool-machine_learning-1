#!/usr/bin/env python3
"""
decode for machine translation:
"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
    decode for machine translation:

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
        self.vocab = vocab
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       kernel_initializer="glorot_uniform",
                                       return_sequences=True,
                                       return_state=True)
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """
        ARGS:
            -x : {tensor} shape :(batch, 1) : the previous word
                in the target sequence as an index of the target vocabulary
            -s_prev {tensor} shape (batch, units):
                containing the previous decoder hidden state
            -hidden_states {tensor} shape (batch, input_seq_len, units)
                containing the outputs of the
        Returns: y, s
            -y :{tensor} :shape (batch, vocab): containing the output
                word as a one hot vector in the target vocabulary
            -s :{tensor} :shape (batch, units): containing the new
                decoder hidden state
        """

        # used for attention
        attention = SelfAttention(self.units)
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = attention(s_prev, hidden_states)

        # x shape after passing through embedding
        # (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation
        # (batch_size, 1, embedding_dim + hidden_size)
        attention_vector = tf.concat([tf.expand_dims(context_vector, 1), x],
                                     axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(attention_vector)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.F(output)

        return x, state
