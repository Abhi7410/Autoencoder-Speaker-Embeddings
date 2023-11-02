############################################
# Since the lab requires us to use PyTorch, we have implemented the VAE model in PyTorch.
# This version however is not tested and verified. It is just a rough implementation/conversion from tensorflow to pytorch.
############################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pickle

class VAE(nn.Module):
    def __init__(self, input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dim):
        super(VAE, self).__init__()
        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.latent_space_dim = latent_space_dim
        self.reconstruction_loss_weight = 1000000

        self.encoder = None
        self.decoder = None
        self._shape_before_bottleneck = None
        self._model_input = None

        self._num_conv_layers = len(conv_filters)

        self._build()

    def forward(self, x):
        latent_representations, _ = self.encode(x)
        reconstructed_images = self.decode(latent_representations)
        return reconstructed_images

    def encode(self, x):
        x = self.encoder(x)
        self._shape_before_bottleneck = x.shape[1:]
        x = x.view(x.size(0), -1)
        mu = self.mu(x)
        log_variance = self.log_variance(x)
        epsilon = torch.randn_like(mu)
        sampled_point = mu + torch.exp(0.5 * log_variance) * epsilon
        return sampled_point, (mu, log_variance)

    def decode(self, x):
        x = self.decoder_input(x)
        x = x.view(x.size(0), *self._shape_before_bottleneck)
        x = self.decoder(x)
        return x

    def summary(self):
        print(self.encoder)
        print(self.decoder)
        print(self)

    def train(self, x_train, batch_size, num_epochs):
        pass  # Implement training loop here

    def save(self, save_folder="."):
        self._create_folder_if_it_doesnt_exist(save_folder)
        self._save_parameters(save_folder)
        self._save_weights(save_folder)

    @classmethod
    def load(cls, save_folder="."):
        parameters_path = os.path.join(save_folder, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        autoencoder = VAE(*parameters)
        weights_path = os.path.join(save_folder, "weights.pth")
        autoencoder.load_state_dict(torch.load(weights_path))
        return autoencoder

    def _create_folder_if_it_doesnt_exist(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    def _save_parameters(self, save_folder):
        parameters = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.latent_space_dim
        ]
        save_path = os.path.join(save_folder, "parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

    def _save_weights(self, save_folder):
        save_path = os.path.join(save_folder, "weights.pth")
        torch.save(self.state_dict(), save_path)

    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._model_input = self._add_encoder_input()

    def _build_encoder(self):
        encoder_layers = []
        for i in range(self._num_conv_layers):
            encoder_layers.append(self._add_conv_layer(i))

        self.encoder = nn.Sequential(*encoder_layers)

        # Determine the shape before the bottleneck based on the encoder's output
        dummy_input = torch.randn(1, *self.input_shape)  # Create a dummy input
        encoder_output = self.encoder(dummy_input)
        self._shape_before_bottleneck = encoder_output.shape[1:]

        self.mu = nn.Linear(np.prod(self._shape_before_bottleneck), self.latent_space_dim)
        self.log_variance = nn.Linear(np.prod(self._shape_before_bottleneck), self.latent_space_dim)



    def _add_encoder_input(self):
        return nn.Conv2d(self.input_shape[0], self.input_shape[1], self.input_shape[2])

    def _add_conv_layer(self, layer_index):
        padding = 1  # You can adjust this value as needed
        return nn.Sequential(
            nn.Conv2d(self.conv_filters[layer_index - 1] if layer_index > 0 else self.input_shape[0], 
                    self.conv_filters[layer_index], 
                    self.conv_kernels[layer_index], 
                    stride=self.conv_strides[layer_index], 
                    padding=padding),  # Change padding here
            nn.ReLU(),
            nn.BatchNorm2d(self.conv_filters[layer_index])
        )

    def _build_decoder(self):
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_dense_layer()
        reshape_layer = self._add_reshape_layer()
        conv_transpose_layers = self._add_conv_transpose_layers()
        decoder_output = self._add_decoder_output()

        self.decoder = nn.Sequential(
            decoder_input,
            dense_layer,
            reshape_layer,
            conv_transpose_layers,
            decoder_output
        )

    def _add_decoder_input(self):
        return nn.Linear(self.latent_space_dim, np.prod(self._shape_before_bottleneck))

    def _add_decoder_output(self):
        return nn.Sequential(
            nn.Conv2d(self.conv_filters[-1], 1, self.conv_kernels[0], stride=self.conv_strides[0], padding=1),
            nn.Sigmoid()
        )

    def _add_dense_layer(self):
        return nn.Linear(self.latent_space_dim, np.prod(self._shape_before_bottleneck))

    def _add_reshape_layer(self):
        return lambda x: x.view(x.size(0), *self._shape_before_bottleneck)

    def _add_conv_transpose_layers(self):
        layers = []
        for i in reversed(range(1, self._num_conv_layers)):
            layers.append(self._add_conv_transpose_layer(i))
        return nn.Sequential(*layers)

    def _add_conv_transpose_layer(self, layer_index):
        return nn.Sequential(
            nn.ConvTranspose2d(self.conv_filters[layer_index], 
                               self.conv_filters[layer_index - 1], 
                               self.conv_kernels[layer_index], 
                               stride=self.conv_strides[layer_index], 
                               padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(self.conv_filters[layer_index - 1])
        )

    def _add_bottleneck(self):
        return nn.Sequential(
            nn.Flatten(),
            self.mu,
            self.log_variance
        )
