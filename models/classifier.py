#############################################################3
# The speaker classifier mdoel, combined with domain adaptation is implemented here
#############################################################3

import torch
import torch.nn as nn
from torch.autograd import Function
from gradient_reveral import GRL

class SpeakerClassifier(nn.Module):
    def __init__(self, encoder, num_speakers, input_shape, alpha):
        super(SpeakerClassifier, self).__init__()
        self.encoder = encoder
        self.num_speakers = num_speakers
        self.input_shape = input_shape
        self.alpha = alpha
        self.feature_extractor = encoder

        self.speaker_classifier = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_speakers)
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(256, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 2),
        )

    def forward(self, x, alpha):
        if(alpha == None):
            alpha = self.alpha
        x = self.feature_extractor(x)
        reverse_x = GRL.apply(x, alpha)
        classes = self.speaker_classifier(x)
        domain = self.domain_classifier(reverse_x)
        return classes, domain