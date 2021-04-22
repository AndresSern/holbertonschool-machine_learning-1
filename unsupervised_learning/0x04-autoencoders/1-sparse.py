#!/usr/bin/env python3
""" Sparse Autoencoder """
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    ARGS:
        -input_dims {integer}: containing the dimensions of the model input

        -hidden_layers {list}: containing the number of nodes
            for each hidden layer in the encoder, respectively

        -latent_dims {integer}: containing the dimensions
            of the latent space representation

        -lambtha is the regularization parameter used
            for L1 regularization on the encoded output

    Returns: encoder, decoder, auto
        -encoder is the encoder model

        -decoder is the decoder model

        -auto is the full autoencoder model
    """

    """ Encoder """
    input_encoder = keras.Input(shape=(input_dims,))

    for i in range(len(hidden_layers)):
        if i == 0:
            # Sparce : Dense layer with a L1 activity regularizer
            encode = keras.layers.Dense(hidden_layers[i],
                                        activation='relu')(input_encoder)
        else:
            encode = keras.layers.Dense(hidden_layers[i],
                                        activation='relu')(encode)

    # Encoder output
    latent = keras.layers.Dense(latent_dims, activation='relu',
                                activity_regularizer=keras.regularizers.l1(
                                    lambtha))(encode)

    # Encoder call
    encoder = keras.Model(inputs=input_encoder, outputs=latent)

    """ Decoder """
    input_decoder = keras.Input(shape=(latent_dims, ))

    # hidden layers should be reversed for the decoder

    for j in range(len(hidden_layers)-1, -1, -1):
        if j == len(hidden_layers) - 1:
            decode = keras.layers.Dense(hidden_layers[i],
                                        activation='relu')(input_decoder)
        else:
            decode = keras.layers.Dense(hidden_layers[i],
                                        activation='relu')(decode)

    # Decoder output
    decode = keras.layers.Dense(input_dims,
                                activation='sigmoid')(decode)

    # Decoder call
    decoder = keras.Model(inputs=input_decoder, outputs=decode)

    """ autoencoder """

    # Encoder_output = encoder(input_encoder) ==> latent
    # Decoder_output = decoder(latent) ==> decode
    Decoder_output = decoder(latent)

    autoencoder = keras.Model(inputs=input_encoder, outputs=Decoder_output)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, autoencoder
