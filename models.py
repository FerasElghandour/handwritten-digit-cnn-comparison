"""Model classes extracted from analysis.ipynb; see source hash in protocol."""
import torch
from torch import nn

class BaselineCNN(nn.Module):
    """
    Shallow two-block CNN for 10-class digit classification.
    Input:  (B, 1, 8, 8) normalized grayscale images.
    Output: (B, 10) raw logits for CrossEntropyLoss.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(32 * 4 * 4, 64), nn.ReLU(), nn.Linear(64, 10))

    def forward(self, x):
        return self.classifier(self.features(x))

class DeepCNN(nn.Module):
    """
    Deeper CNN with BatchNorm and Dropout regularization.
    Changes vs. BaselineCNN:
      - Three convolutional blocks (vs. two)
      - BatchNorm2d after each Conv2d
      - Dropout(0.4) before the final classifier layer
      - Wider channels: 32 → 64 → 64
    Input:  (B, 1, 8, 8)
    Output: (B, 10) raw logits
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU())
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.4), nn.Linear(128, 10))

    def forward(self, x):
        return self.classifier(self.features(x))
