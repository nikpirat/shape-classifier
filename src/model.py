import torch.nn as nn
import torch.nn.functional as F

# Define a neural network model class for shape classification
class ShapeClassifier(nn.Module):
    def __init__(self):
        super(ShapeClassifier, self).__init__()  # Initialize the parent nn.Module class

        # First fully connected (dense) layer
        # Input: 48x48 grayscale image = 2304 pixels (flattened)
        # Output: 128 features
        self.fc1 = nn.Linear(48 * 48, 128)

        # Second fully connected layer
        # Input: 128 features, Output: 64 features
        self.fc2 = nn.Linear(128, 64)

        # Final output layer
        # Input: 64 features, Output: 4 neurons (one for each class: circle, square, triangle, Unknown)
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
        # Flatten the image tensor: [batch_size, 1, 48, 48] → [batch_size, 2304]
        x = x.view(-1, 48 * 48)

        # Apply first linear layer + ReLU activation
        x = F.relu(self.fc1(x))

        # Apply second linear layer + ReLU activation
        x = F.relu(self.fc2(x))

        # Output layer (raw logits); don't apply softmax here — handled by CrossEntropyLoss
        return self.fc3(x)
