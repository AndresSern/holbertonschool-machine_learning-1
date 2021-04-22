#!/usr/bin/env python3
""" variational autoencoder """
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    ARGS:
        -input_dims {integer}: containing the dimensions of the model input

        -hidden_layers {list}: containing the number of nodes
            for each hidden layer in the encoder, respectively

        -latent_dims {integer}: containing the dimensions
            of the latent space representation

    Returns: encoder, decoder, auto
        -encoder is the encoder model

        -decoder is the decoder model

        -auto is the full autoencoder model
    """

    """ Encoder """
    input_encoder = keras.Input(shape=(input_dims,))

    for i in range(len(hidden_layers)):
        if i == 0:
            encode = keras.layers.Dense(hidden_layers[i],
                                        activation='relu')(input_encoder)
        else:
            encode = keras.layers.Dense(hidden_layers[i],
                                        activation='relu')(encode)

    z_mean = keras.layers.Dense(latent_dims)(encode)
    z_log_sigma = keras.layers.Dense(latent_dims)(encode)

    def sampling(args):
        z_mean, z_log_sigma = args

        epsilon = keras.backend.random_normal(
            shape=(keras.backend.shape(z_mean)[0], latent_dims),
            mean=0., stddev=0.1)

        return z_mean + keras.backend.exp(z_log_sigma) * epsilon

    z = keras.layers.Lambda(sampling)([z_mean, z_log_sigma])
    # Encoder call
    encoder = keras.Model(inputs=input_encoder,
                          outputs=[z_mean, z_log_sigma, z])

    """ _________________________________Decoder____________ """

    input_decoder = keras.Input(shape=(latent_dims, ))

    # hidden layers should be reversed for the decoder

    for j in range(len(hidden_layers)-1, -1, -1):
        if j == len(hidden_layers) - 1:
            decode = keras.layers.Dense(hidden_layers[j],
                                        activation='relu')(input_decoder)
        else:
            decode = keras.layers.Dense(hidden_layers[j],
                                        activation='relu')(decode)

    # Decoder output
    decode = keras.layers.Dense(input_dims,
                                activation='sigmoid')(decode)

    # Decoder call
    decoder = keras.Model(inputs=input_decoder, outputs=decode)

    """ ___________-autoencoder_____________________ """

    outputs = decoder(encoder(input_encoder,)[2])
    autoencoder = keras.Model(input_encoder, outputs, name='vae_mlp')
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, autoencoder
