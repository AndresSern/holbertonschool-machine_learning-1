#!/usr/bin/env python3
"""
Scaled Dot Product Attention
"""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    calculates the scaled dot product attention
    Args:
        -Q :{tensor} shape= (..., seq_len_q, dk)
            containing the query matrix
        -K : {tensor} shape= (..., seq_len_v, dk)
            containing the key matrix
        -V : {tensor} shape=(..., seq_len_v, dv)
            containing the value matrix
        -mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns: output, attention_weights
        -outputa: {tensor} SHAPE =(..., seq_len_q, dv)
            containing the scaled dot product attention
        -weights:{tensor} shape= (..., seq_len_q, seq_len_v)
            containing the attention weights
    """
    matmul_qk = tf.matmul(Q, K, transpose_b=True)

    # scale matmul_qk
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    output = tf.matmul(attention_weights, V)

    return output, attention_weights
