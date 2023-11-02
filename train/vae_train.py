from utils.pretrain import Loader, Padder, LogSpectrogramExtractor, MinMaxNormaliser, Saver, PreprocessingPipeline
from models.vae_tf import VAE
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=150, help='number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.0005, help='learning rate')
parser.add_argument('--num_speakers', type=int, default=10, help='number of speakers')
parser.add_argument('--pth', type=str, default="/kaggle/input/speaker-dataset/autoencoder.pth", help='path to autoencoder')
parser.add_argument('--spectrogram_dir', type=str, default="./spectrogram", help='path to spectrogram')
parser.add_argument('--minmax_dir', type=str, default="./minmax", help='path to minmax')
parser.add_argument('--files_dir', type=str, default="./speaker_dataset/", help='path to files')
parser.add_argument('--duration', type=int, default=1, help='duration')
parser.add_argument('--sample_rate', type=int, default=22050, help='sample rate')
parser.add_argument('--mono', type=bool, default=True, help='mono')
parser.add_argument('--frame_size', type=int, default=512, help='frame size')
parser.add_argument('--hop_length', type=int, default=256, help='hop length')

BATCH_SIZE = parser.parse_args().batch_size
AUTOENCODER_PTH = parser.parse_args().pth
FRAME_SIZE = parser.parse_args().frame_size
HOP_LENGTH = parser.parse_args().hop_length
DURATION = parser.parse_args().duration
SAMPLE_RATE = parser.parse_args().sample_rate
MONO = parser.parse_args().mono
SPECTROGRAMS_SAVE_DIR = parser.parse_args().spectrogram_dir
MIN_MAX_VALUES_SAVE_DIR = parser.parse_args().minmax_dir
FILES_DIR = parser.parse_args().files_dir
LEARNING_RATE = parser.parse_args().learning_rate
BATCH_SIZE = parser.parse_args().batch_size
EPOCHS = parser.parse_args().epochs

loader = Loader(SAMPLE_RATE, DURATION, MONO)
padder = Padder()
log_spectrogram_extractor = LogSpectrogramExtractor(FRAME_SIZE, HOP_LENGTH)
min_max_normaliser = MinMaxNormaliser(0, 1)
saver = Saver(SPECTROGRAMS_SAVE_DIR, MIN_MAX_VALUES_SAVE_DIR)

preprocessing_pipeline = PreprocessingPipeline()
preprocessing_pipeline.loader = loader
preprocessing_pipeline.padder = padder
preprocessing_pipeline.extractor = log_spectrogram_extractor
preprocessing_pipeline.normaliser = min_max_normaliser
preprocessing_pipeline.saver = saver
preprocessing_pipeline.process(FILES_DIR)

class CustomDataset(Dataset):
    def __init__(self, spectrograms_path):
        self.spectrograms_path = spectrograms_path
        self.file_paths = []

        for root, _, file_names in os.walk(spectrograms_path):
            for file_name in file_names:
                file_path = os.path.join(root, file_name)
                self.file_paths.append(file_path)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        spectrogram = np.load(file_path)  # (n_bins, n_frames, 1)
        spectrogram = spectrogram[np.newaxis, ...]  # Add batch dimension
        return spectrogram
    
dataset = CustomDataset(SPECTROGRAMS_SAVE_DIR)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


##USE THE BELOW TRAIN IF TRAINING WITH TORCH
def train_vae(x_train, learning_rate, batch_size, epochs):
    dataset = CustomDataset(x_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    autoencoder = VAE(
        input_shape=(1, 256, 64),
        conv_filters=(512, 256, 128, 64, 32),
        conv_kernels=(3, 3, 3, 3, 3),
        conv_strides=(2, 2, 2, 2, (2, 1)),
        latent_space_dim=256
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder.to(device)

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
    reconstruction_criterion = nn.MSELoss()

    for epoch in range(epochs):
        autoencoder.train()
        total_loss = 0.0

        for batch in dataloader:
            batch = batch.to(device, dtype=torch.float32)
            reconstructed_batch = autoencoder(batch)
            reconstruction_loss = reconstruction_criterion(reconstructed_batch, batch)

            mu, log_variance = autoencoder.encode(batch)[1]
            kl_loss = -0.5 * torch.sum(1 + log_variance - mu.pow(2) - log_variance.exp())

            loss = autoencoder.reconstruction_loss_weight * reconstruction_loss + kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(dataloader)}")

    return autoencoder

def load_fsdd(spectrograms_path):
    x_train = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path) 
            x_train.append(spectrogram)
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis] 
    return x_train

##USE THE BELOW TRAIN IF TRAINING WITH TF
def train(x_train, learning_rate, batch_size, epochs):
    autoencoder = VAE(
        input_shape=(256, 64, 1),
        conv_filters=(512, 256, 128, 64, 32),
        conv_kernels=(3, 3, 3, 3, 3),
        conv_strides=(2, 2, 2, 2, (2, 1)),
        latent_space_dim=256
    )
    autoencoder.compile(learning_rate)
    autoencoder.train(x_train, batch_size, epochs)
    return autoencoder

if __name__ == "__main__":
    x_train = load_fsdd(SPECTROGRAMS_SAVE_DIR)
    autoencoder = train(x_train, LEARNING_RATE, BATCH_SIZE, EPOCHS)
    autoencoder.save("model")