##################################################
# This is the tensorflow implementation of the Variational Autoencoder.
# This is how it was tested and verifier initially.
##################################################

import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Conv2D, ReLU, BatchNormalization, Flatten, Dense,
    Reshape, Conv2DTranspose, Activation, Lambda
)
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

tf.compat.v1.disable_eager_execution()

def _calc_reconstruction_loss(y_true, y_pred):
    error = y_true - y_pred
    recon_loss = K.mean(K.square(error), axis=[1, 2, 3])
    return recon_loss

def calc_kl_loss(model):
    def _calc_kl_loss(*args):
        kl_loss = -0.5 * K.sum(
            1 + model.log_variance - K.square(model.mu) - K.exp(model.log_variance),
            axis=1
        )
        return kl_loss
    return _calc_kl_loss

class VAE:
    def __init__(self, input_shape, conv_filters, conv_kernels, conv_strides, latent_dim):
        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.latent_dim = latent_dim
        self.recon_loss_weight = 1000000

        self.encoder = None
        self.decoder = None
        self.model = None

        self.num_conv_layers = len(conv_filters)
        self.shape_before_bottleneck = None
        self.model_input = None

        self._build()

    def get_summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def compile(self, learning_rate=0.0001):
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss=self._calc_combined_loss,
            metrics=[_calc_reconstruction_loss, calc_kl_loss(self)]
        )

    def train(self, x_train, batch_size, num_epochs):
        self.model.fit(x_train, x_train, batch_size=batch_size, epochs=num_epochs, shuffle=True)

    def save(self, save_folder="."):
        self._create_folder_if_not_exist(save_folder)
        self._save_parameters(save_folder)
        self._save_weights(save_folder)

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def reconstruct(self, images):
        latent_representations = self.encoder.predict(images)
        reconstructed_images = self.decoder.predict(latent_representations)
        return reconstructed_images, latent_representations

    @classmethod
    def load(cls, save_folder="."):
        parameters_path = os.path.join(save_folder, "parameters.pkl")
        with open(parameters_path, "rb") as file:
            parameters = pickle.load(file)
        autoencoder = VariationalAutoencoder(*parameters)
        weights_path = os.path.join(save_folder, "weights.h5")
        autoencoder.load_weights(weights_path)
        return autoencoder

    def _calc_combined_loss(self, y_true, y_pred):
        recon_loss = _calc_reconstruction_loss(y_true, y_pred)
        kl_loss = calc_kl_loss(self)()
        combined_loss = self.recon_loss_weight * recon_loss + kl_loss
        return combined_loss

    def _create_folder_if_not_exist(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    def _save_parameters(self, save_folder):
        params = [self.input_shape, self.conv_filters, self.conv_kernels, self.conv_strides, self.latent_dim]
        save_path = os.path.join(save_folder, "parameters.pkl")
        with open(save_path, "wb") as file:
            pickle.dump(params, file)

    def _save_weights(self, save_folder):
        save_path = os.path.join(save_folder, "weights.h5")
        self.model.save_weights(save_path)

    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    def _build_autoencoder(self):
        model_input = self.model_input
        model_output = self.decoder(self.encoder(model_input))
        self.model = Model(model_input, model_output, name="autoencoder")

    def _build_decoder(self):
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self.decoder = Model(decoder_input, decoder_output, name="decoder")

    def _add_decoder_input(self):
        return Input(shape=self.latent_dim, name="decoder_input")

    def _add_dense_layer(self, decoder_input):
        num_neurons = np.prod(self.shape_before_bottleneck)
        dense_layer = Dense(num_neurons, name="decoder_dense")(decoder_input)
        return dense_layer

    def _add_reshape_layer(self, dense_layer):
        return Reshape(self.shape_before_bottleneck)(dense_layer)

    def _add_conv_transpose_layers(self, x):
        for layer_index in reversed(range(1, self.num_conv_layers)):
            x = self._add_conv_transpose_layer(layer_index, x)
        return x

    def _add_conv_transpose_layer(self, layer_index, x):
        layer_num = self.num_conv_layers - layer_index
        conv_transpose_layer = Conv2DTranspose(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"decoder_conv_transpose_layer_{layer_num}"
        )
        x = conv_transpose_layer(x)
        x = ReLU(name=f"decoder_relu_{layer_num}")(x)
        x = BatchNormalization(name=f"decoder_bn_{layer_num}")(x)
        return x

    def _add_decoder_output(self, x):
        conv_transpose_layer = Conv2DTranspose(
            filters=1,
            kernel_size=self.conv_kernels[0],
            strides=self.conv_strides[0],
            padding="same",
            name=f"decoder_conv_transpose_layer_{self.num_conv_layers}"
        )
        x = conv_transpose_layer(x)
        output_layer = Activation("sigmoid", name="sigmoid_layer")(x)
        return output_layer
