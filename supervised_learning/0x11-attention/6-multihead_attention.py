#!/usr/bin/env python3
"""
Multi Head Attention
src : www.tensorflow.org/tutorials/text/transformer#multi-head_attention
"""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Multi Head Attention

    """

    def __init__(self, dm, h):
        """
        ARGS:
            -dm:{integer} :the dimensionality of the model
            -h:{integer} :the number of heads
        """
        super().__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """
        -Split the last dimension into :(num_heads, depth).
        -Transpose the result such that the shape is
        (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """
        ARGS:
            -Q : {tensor} shape :(batch, seq_len_q, dk) :  the query matrix

            -K: {tensor} shape (batch, seq_len_q, dk):  the key matrix

            -V: {tensor} shape (batch, seq_len_v, dv):  the value matrix

            -mask : mask is always None
        Returns: output, weights
            -output : {tensor} shape (..., seq_len_q, dm)
                containing the scaled dot product attention
            -weights : {tensor} shape (..., h, seq_len_q, seq_len_v)
                containing the attention weights
        """

        batch_size = tf.shape(Q)[0]
        # (batch_size, seq_len, d_model)
        q = self.Wq(Q)
        # (batch_size, seq_len, d_model)
        k = self.Wk(K)
        # (batch_size, seq_len, d_model)
        v = self.Wv(V)

        # (batch_size, num_heads, seq_len_q, depth)
        q = self.split_heads(q, batch_size)
        # (batch_size, num_heads, seq_len_k, depth)
        k = self.split_heads(k, batch_size)
        # (batch_size, num_heads, seq_len_v, depth)
        v = self.split_heads(v, batch_size)

        # scaled_attention:shape(batch_size, num_heads, seq_len_q, depth)
        # attention_weights:shape(batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = sdp_attention(
            q, k, v, mask)

        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        # (batch_size, seq_len_q, d_model)
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.dm))
        # (batch_size, seq_len_q, d_model)
        output = self.linear(concat_attention)

        return output, attention_weights
