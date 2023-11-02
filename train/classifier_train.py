import os
import random
import shutil
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from models.classifier import SpeakerClassifier
from utils.pretrain import MinMaxNormaliser, Saver, PreprocessingPipeline, Loader, Padder, LogSpectrogramExtractor
import librosa
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

folders = os.listdir('/kaggle/input/speaker-dataset/speaker_dataset')
print("Folders: ", folders)
#map each folder to a number
folder_to_number = {}
number_to_folder = {}
for i, folder in enumerate(folders):
    folder_to_number[folder] = float(i)
    number_to_folder[i] = folder
num_speakers = len(folders)
print("Number of speakers: ", num_speakers)

autoencoder = torch.load(AUTOENCODER_PTH)
encoder = autoencoder.encoder

model = SpeakerClassifier(encoder, num_speakers, (1, 128, 44), 0.1)

#now for each folder, separate 70% as src_train, 20% as target_train and 10% as target_test
src_train, tar_train, tar_test = [], [], []
for folder in folders:
    files = os.listdir(os.path.join('/kaggle/input/speaker-dataset/speaker_dataset', folder))
    src_train.extend([(os.path.join('/kaggle/input/speaker-dataset/speaker_dataset', folder, file), folder_to_number[folder]) for file in files[:int(0.7*len(files))]])
    tar_train.extend([(os.path.join('/kaggle/input/speaker-dataset/speaker_dataset', folder, file), folder_to_number[folder]) for file in files[int(0.7*len(files)):int(0.9*len(files))]])
    tar_test.extend([(os.path.join('/kaggle/input/speaker-dataset/speaker_dataset', folder, file), folder_to_number[folder]) for file in files[int(0.9*len(files)):]])

print(len(src_train), len(tar_train), len(tar_test))
encoder_input_shape = (1, 256, 64)
def get_spect(src):
    audio_src, _ = librosa.load(src, sr=SAMPLE_RATE, mono=MONO, duration=DURATION)
    audio_src = padder.right_pad(audio_src, preprocessing_pipeline._num_expected_samples - len(audio_src))
    audio_src = log_spectrogram_extractor.extract(audio_src)
    audio_src = min_max_normaliser.normalise(audio_src)
    audio_src = audio_src.reshape(encoder_input_shape)
    return audio_src

optimizer = nn.optim.Adam(model.parameters(), lr=LEARNING_RATE)
def train(num_epochs):
    model.train()
    total_steps = num_epochs * len(src_train)
    i = 0
    for epoch in range(num_epochs):
        print("Epoch: ", epoch)
        for i, (src, tar) in enumerate(zip(src_train, tar_train)):
            s_data, s_label = get_spect(src[0]), src[1]
            t_data, t_label = get_spect(tar[0]), tar[1]

            optimizer.zero_grad()
            s_data = torch.from_numpy(s_data).float()
            s_label = torch.tensor([s_label]).long()
            t_data = torch.from_numpy(t_data).float()
            t_label = torch.tensor([t_label]).long()

            s_label_pred, s_domain_pred = model(s_data)
            t_label_pred, t_domain_pred = model(t_data)

            s_label_loss = nn.CrossEntropyLoss()(s_label_pred, s_label)
            s_domain_loss = nn.CrossEntropyLoss()(s_domain_pred, torch.zeros_like(s_domain_pred))
            t_domain_loss = nn.CrossEntropyLoss()(t_domain_pred, torch.ones_like(t_domain_pred))

            loss = s_label_loss + s_domain_loss + t_domain_loss
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(epoch+1, num_epochs, i+1, total_steps, loss.item()))
    torch.save(model.state_dict(), 'model.pth')

if __name__ == "__main__":
    train(EPOCHS)