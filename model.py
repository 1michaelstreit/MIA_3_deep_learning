"""CNN architecture for breast lesion classification.

TODO: Modify BreastLesionCNN to improve test set accuracy on BreastMNIST.

Rules:
- Keep the class name BreastLesionCNN.
- Keep __init__ signature: def __init__(self, num_classes: int = 1)
- Keep forward signature: def forward(self, x: torch.Tensor) -> torch.Tensor
- Input tensor shape: (N, 1, 28, 28)  — grayscale images normalised to [0, 1]
- Output tensor shape: (N, 1)  — raw logits for binary classification
- The saved model file must be ≤ 10 MB.

You may change:
- Number of layers and channels
- Kernel sizes, strides, padding
- Activation functions
- Pooling strategies
- Batch normalisation, dropout, etc.
- Anything else inside the class body
"""

import torch
import torch.nn as nn

class BreastLesionCNN(nn.Module):
    """Define your CNN architecture here."""

    def __init__(self, num_classes: int = 1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Single conv block — 28×28 → 14×14


            nn.Conv2d(1, 24, kernel_size=5, padding=2),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # compress features before classifier
            nn.Conv2d(96, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        
        self.head = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(48, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))
