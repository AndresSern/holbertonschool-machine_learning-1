 
#!/usr/bin/env python3
"""
Creates all masks for training/validation
"""
import tensorflow.compat.v2 as tf


def create_masks(inputs, target):
    """
    Creates all masks for training/validation
    Args:
        -inputs :{tf.Tensor} of shape (batch_size, seq_len_in)
            contains the input sentence
        -target :{tf.Tensor} of shape (batch_size, seq_len_out)
            contains the target sentence

    Returns: encoder_mask, combined_mask, decoder_mask
        -encoder_mask :{tf.Tensor} padding mask of shape
            (batch_size, 1, 1, seq_len_in) to be applied in the encoder
        -combined_mask :{tf.Tensor} of shape
            (batch_size, 1, seq_len_out, seq_len_out) used in the
            1st attention block in the decoder to pad and mask
            future tokens in the input received by the decoder.
        -decoder_mask is the tf.Tensor padding mask of shape
            (batch_size, 1, 1, seq_len_in) used in the 2nd
            attention block in the decoder.
   
    """
    batch_size, seq_len_in = inputs.shape
    batch_size, seq_len_out = target.shape

    # the encoder mask
    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]
    # the decoder mask
    decoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    decoder_mask = decoder_mask[:, tf.newaxis, tf.newaxis, :]

    # the look ahead mask
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len_out, seq_len_out)), -1, 0)

    #  the decoder target padding mask.
    dec_target_padding_mask = tf.cast(tf.math.equal(target, 0), tf.float32)
    dec_target_padding_mask = dec_target_padding_mask[:, tf.newaxis, tf.newaxis, :]
    # combined_mask
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return encoder_mask, combined_mask, decoder_mask