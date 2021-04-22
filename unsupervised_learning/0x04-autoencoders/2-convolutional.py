#!/usr/bin/env python3
""" Convolutional Autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    ARGS:
        -input_dims {integer}: containing the dimensions of the model input

        -filters {list}: containing the number of filters
            for each convolutional layer in the encoder, respectively

        -latent_dims {tuple of integers}: containing the dimensions
            of the latent space representation

    Returns: encoder, decoder, auto
        -encoder is the encoder model

        -decoder is the decoder model

        -auto is the full autoencoder model
    """

    """ ___________________Encoder_________________ """

    input_encoder = keras.Input(shape=(input_dims))

    for i in range(len(filters)):
        if i == 0:
            encode = keras.layers.Conv2D(filters[i], (3, 3),
                                         activation='relu',
                                         padding='same')(input_encoder)
            encode = keras.layers.MaxPooling2D((2, 2),
                                               padding='same')(encode)

        else:
            encode = keras.layers.Conv2D(filters[i], (3, 3),
                                         activation='relu',
                                         padding='same')(encode)

            encode = keras.layers.MaxPooling2D((2, 2),
                                               padding='same')(encode)

    # Encoder call
    encoder = keras.Model(inputs=input_encoder, outputs=encode)
    # encoder.summary()

    """ _________________Decoder____________________ """
    input_decoder = keras.Input(shape=(latent_dims))

    # hidden layers should be reversed for the decoder

    for j in range(len(filters)-1, 0, -1):
        if j == len(filters) - 1:
            decode = keras.layers.Conv2D(filters[j], (3, 3),
                                         activation='relu',
                                         padding='same')(input_decoder)
            decode = keras.layers.UpSampling2D((2, 2))(decode)
        else:
            decode = keras.layers.Conv2D(filters[j], (3, 3),
                                         activation='relu',
                                         padding='same')(decode)
            decode = keras.layers.UpSampling2D((2, 2))(decode)

    # Decoder output
    decode = keras.layers.Conv2D(filters[0], (3, 3),
                                 activation='relu',
                                 padding='valid')(decode)
    decode = keras.layers.UpSampling2D((2, 2))(decode)

    decode = keras.layers.Conv2D(input_dims[-1], (2, 2), padding='same',
                                 activation='sigmoid')(decode)

    # Decoder call
    decoder = keras.Model(inputs=input_decoder, outputs=decode)
    # decoder.summary()

    """__________autoencoder_____________ """

    Encoder_output = encoder(input_encoder)

    Decoder_output = decoder(Encoder_output)

    autoencoder = keras.Model(inputs=input_encoder, outputs=Decoder_output)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, autoencoder
